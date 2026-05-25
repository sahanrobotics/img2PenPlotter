import cv2
import numpy as np
import math
from utils import check_time


def process_algorithms(img_gray, mode, spacing, density, start_t):
    h, w = img_gray.shape
    paths = []
    spacing = max(4, int(spacing))
    density = min(5000, max(100, int(density)))

    if mode == "Edge Contour":
        edges = cv2.Canny(img_gray, 50, 150)
        cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.arcLength(c, False) > 5:
                paths.append([(float(pt[0][0]), float(pt[0][1])) for pt in c])

    elif mode == "Hatching":
        _, binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
        for y in range(0, h, spacing):
            row = binary[y, :]
            whites = np.where(row > 0)[0]
            if len(whites) == 0: continue
            splits = np.split(whites, np.where(np.diff(whites) > 1)[0] + 1)
            for split in splits:
                if len(split) > 2:
                    paths.append([(float(split[0]), float(y)), (float(split[-1]), float(y))])

    elif mode == "Artistic Cross-Hatching":
        for angle in [0, 45, 90]:
            center = (w // 2, h // 2)
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img_gray, rot_mat, (w, h), borderValue=255)

            thresh_val = 150 if angle == 0 else (100 if angle == 45 else 60)
            _, binary = cv2.threshold(rotated, thresh_val, 255, cv2.THRESH_BINARY_INV)
            inv_mat = cv2.getRotationMatrix2D(center, -angle, 1.0)

            for y in range(0, h, spacing * 2):
                whites = np.where(binary[y, :] > 0)[0]
                if len(whites) == 0: continue
                splits = np.split(whites, np.where(np.diff(whites) > 1)[0] + 1)
                for split in splits:
                    if len(split) > 2:
                        p1 = np.array([split[0], y, 1]) @ inv_mat.T
                        p2 = np.array([split[-1], y, 1]) @ inv_mat.T
                        paths.append([(float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))])

    elif mode == "Flow Field":
        blur = cv2.GaussianBlur(img_gray, (15, 15), 0)
        gx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5)
        gy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)
        angles = np.arctan2(gy, gx) + (math.pi / 2)
        Y, X = np.mgrid[0:h:spacing, 0:w:spacing]
        seeds = np.column_stack([X.ravel(), Y.ravel()])
        valid_seeds = seeds[img_gray[np.clip(seeds[:, 1], 0, h - 1), np.clip(seeds[:, 0], 0, w - 1)] <= 220]

        for idx, (x, y) in enumerate(valid_seeds):
            if idx % 1000 == 0: check_time(start_t)
            cx, cy = float(x), float(y)
            path = [(cx, cy)]
            for _ in range(20):
                if not (0 <= int(cx) < w and 0 <= int(cy) < h) or img_gray[int(cy), int(cx)] > 220: break
                ang = angles[int(cy), int(cx)]
                cx += math.cos(ang) * 3
                cy += math.sin(ang) * 3
                path.append((cx, cy))
            if len(path) > 2: paths.append(path)

    elif mode == "Small Circles":
        y_coords = np.arange(spacing, h - spacing, spacing)
        x_coords = np.arange(spacing, w - spacing, spacing)
        if len(y_coords) > 0 and len(x_coords) > 0:
            X, Y = np.meshgrid(x_coords, y_coords)
            vals = img_gray[Y, X]
            darkness = 1.0 - (vals / 255.0)
            R = (spacing / 2) * darkness

            valid = R > 0.8
            v_X, v_Y, v_R = X[valid], Y[valid], R[valid]

            sides = 6  # Hexagons are drastically faster than 8 sides
            angles = np.linspace(0, 2 * math.pi, sides + 1)
            cos_a, sin_a = np.cos(angles), np.sin(angles)

            for cx, cy, r in zip(v_X, v_Y, v_R):
                px = cx + r * cos_a
                py = cy + r * sin_a
                paths.append([(float(x), float(y)) for x, y in zip(px, py)])

    elif mode == "Stippling (Dots)":
        prob = (255 - img_gray.astype(np.float64)).flatten()
        prob[prob < 20] = 0
        prob_sum = prob.sum()
        if prob_sum > 0:
            idx = np.random.choice(h * w, min(density, len(prob[prob > 0])), replace=False, p=prob / prob_sum)
            ys, xs = np.unravel_index(idx, (h, w))
            for i, (x, y) in enumerate(zip(xs, ys)):
                if i % 1000 == 0: check_time(start_t)
                r = max(0.5, 2.0 * (1.0 - img_gray[y, x] / 255.0))
                paths.append([(float(x - r), float(y)), (float(x + r), float(y))])

    elif mode == "Sine Waves":
        y_coords = np.arange(spacing, h - spacing, spacing)
        x_coords = np.arange(0, w, max(2, int(spacing / 2)))

        if len(y_coords) > 0 and len(x_coords) > 0:
            X, Y = np.meshgrid(x_coords, y_coords)
            Y_idx = np.clip(Y, 0, h - 1)
            X_idx = np.clip(X, 0, w - 1)
            vals = img_gray[Y_idx, X_idx]

            darkness = 1.0 - (vals / 255.0)
            amplitude = (spacing * 0.9) * darkness
            freq = (2 * np.pi) / (spacing * 2.5)

            Y_new = Y + amplitude * np.sin(freq * X)

            for r in range(Y_new.shape[0]):
                paths.append([(float(X[r, c]), float(Y_new[r, c])) for c in range(Y_new.shape[1])])

    return paths