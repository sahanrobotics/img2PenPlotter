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

        # Tiny search window ensures calculation takes practically zero CPU power
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
    """Converts the processed paths into G-code and an SVG preview."""
    scale_x, scale_y = target_w_mm / img_w, target_h_mm / img_h
    gcode = [f"; Mode: {mode}", f"; Inverted: {invert}", "G21", "G90", "G0 Z5"]
    svg_p = []

    for path in paths:
        if len(path) < 2: continue

        # G-Code
        gcode.append(f"G0 X{path[0][0] * scale_x:.2f} Y{target_h_mm - (path[0][1] * scale_y):.2f}")
        gcode.append("G1 Z0 F3000")

        # SVG
        svg_d = f"M {path[0][0]},{path[0][1]} "

        for x, y in path[1:]:
            gcode.append(f"G1 X{x * scale_x:.2f} Y{target_h_mm - (y * scale_y):.2f}")
            svg_d += f"L {x},{y} "

        gcode.append("G0 Z5")
        svg_p.append(f'<path d="{svg_d}" fill="none" stroke="black" stroke-width="1.5"/>')

    svg_final = f'<svg viewBox="0 0 {img_w} {img_h}" style="width:100%; height:100%;" xmlns="http://www.w3.org/2000/svg">{"".join(svg_p)}</svg>'
    gcode_final = "\n".join(gcode)

    return gcode_final, svg_final