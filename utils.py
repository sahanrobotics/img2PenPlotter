import time
import cv2
import numpy as np

# --- PERFORMANCE SETTINGS ---
MAX_EXECUTION_TIME = 15.0
MAX_IMAGE_DIMENSION = 800
TSP_SEARCH_WINDOW = 250


class TimeoutException(Exception):
    pass


def check_time(start_t):
    """Raises an exception if processing takes too long."""
    if time.time() - start_t > MAX_EXECUTION_TIME:
        raise TimeoutException("Processing Timeout: Image too complex.")


def simplify_paths(paths, simplify_value):
    """Reduces the number of nodes in paths to save processing/plotter time."""
    if simplify_value <= 0:
        return paths

    simped = []
    for p in paths:
        if len(p) < 3:
            simped.append(p)
            continue
        approx = cv2.approxPolyDP(np.array(p, dtype=np.float32), simplify_value, False)
        simped.append([(float(pt[0][0]), float(pt[0][1])) for pt in approx])
    return simped


def optimize_paths_tsp(paths, start_time):
    """Vectorized Nearest Neighbor Algorithm to minimize pen-up travel time."""
    if not paths:
        return paths

    N = len(paths)
    starts = np.array([p[0] for p in paths])
    ends = np.array([p[-1] for p in paths])
    active = np.ones(N, dtype=bool)

    opt = [paths[0]]
    active[0] = False
    last_pt = ends[0]

    for _ in range(1, N):
        if _ % 2000 == 0: check_time(start_time)

        active_idx = np.nonzero(active)[0][:TSP_SEARCH_WINDOW]

        w_starts = starts[active_idx]
        w_ends = ends[active_idx]

        ds = np.sum((w_starts - last_pt) ** 2, axis=1)
        de = np.sum((w_ends - last_pt) ** 2, axis=1)

        ms_idx = np.argmin(ds)
        me_idx = np.argmin(de)

        if ds[ms_idx] <= de[me_idx]:
            best_idx = active_idx[ms_idx]
            rev = False
            last_pt = ends[best_idx]
        else:
            best_idx = active_idx[me_idx]
            rev = True
            last_pt = starts[best_idx]

        next_p = paths[best_idx]
        if rev: next_p = next_p[::-1]

        opt.append(next_p)
        active[best_idx] = False

    return opt


def generate_outputs(paths, img_w, img_h, target_w_mm, target_h_mm, mode, invert):
    """Converts the processed paths into G-code and an SVG preview.
    Ensures A4 bounding box constraints and removes ALL negative coordinates."""
    
    # 1. Define Standard A4 bounds with a 5mm safety margin (200 x 287 mm printable area)
    is_landscape = img_w > img_h
    a4_w = 287.0 if is_landscape else 200.0
    a4_h = 200.0 if is_landscape else 287.0
    
    # Use requested targets, but NEVER exceed A4 paper dimensions
    fit_w = min(target_w_mm, a4_w)
    fit_h = min(target_h_mm, a4_h)

    # 2. Compute a uniform scaling factor to preserve aspect ratio (prevents stretching)
    scale = min(fit_w / img_w, fit_h / img_h)
    
    # Calculate actual physical dimensions
    actual_w = img_w * scale
    actual_h = img_h * scale
    
    # 3. Calculate offset to perfectly center the plot on the page
    offset_x = (fit_w - actual_w) / 2.0
    offset_y = (fit_h - actual_h) / 2.0

    gcode = [f"; Mode: {mode}", f"; Inverted: {invert}", "G21", "G90", "G0 Z5"]
    svg_p = []

    def format_coord(px, py):
        """Scales, centers, and rigidly clamps coordinates to forbid negatives."""
        # Scale and add centering offset
        gx = (px * scale) + offset_x
        
        # G-Code origin (0,0) is bottom-left, SVG origin is top-left. So we invert Y.
        gy = fit_h - ((py * scale) + offset_y)
        
        # STRICT CLAMPING: Forces value to be exactly between 0.0 and the max paper dimension.
        gx = max(0.0, min(gx, fit_w))
        gy = max(0.0, min(gy, fit_h))
        
        return f"X{gx:.2f} Y{gy:.2f}"

    for path in paths:
        if len(path) < 2: continue

        # --- G-Code Generation ---
        gcode.append(f"G0 {format_coord(path[0][0], path[0][1])}")
        gcode.append("G1 Z0 F3000")

        # --- SVG Generation ---
        svg_d = f"M {path[0][0]:.2f},{path[0][1]:.2f} "

        for px, py in path[1:]:
            gcode.append(f"G1 {format_coord(px, py)}")
            svg_d += f"L {px:.2f},{py:.2f} "

        gcode.append("G0 Z5")
        svg_p.append(f'<path d="{svg_d}" fill="none" stroke="black" stroke-width="1.5"/>')

    svg_final = f'<svg viewBox="0 0 {img_w} {img_h}" style="width:100%; height:100%;" xmlns="http://www.w3.org/2000/svg">{"".join(svg_p)}</svg>'
    gcode_final = "\n".join(gcode)

    return gcode_final, svg_final