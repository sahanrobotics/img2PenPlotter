import os
import cv2
import numpy as np
import time
import gc

from fastapi import FastAPI, UploadFile, Form, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from utils import (
    MAX_IMAGE_DIMENSION,
    TimeoutException,
    simplify_paths,
    optimize_paths_tsp,
    generate_outputs
)

from algorithms import process_algorithms

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def format_error_svg(msg):
    return f'<svg xmlns="http://www.w3.org/2000/svg"><text x="10" y="20">{msg}</text></svg>'


@app.post("/api/generate")
async def generate(
    file: UploadFile = File(...),
    invert: str = Form("false"),
    mode: str = Form(...),
    spacing: float = Form(...),
    density: float = Form(...),
    simplify: float = Form(...),
    target_w_mm: float = Form(...),
    target_h_mm: float = Form(...)
):

    start_time = time.time()

    img = None
    nparr = None
    contents = None
    paths = None

    try:
        # -------------------------
        # Load image
        # -------------------------
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        contents = None

        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        del nparr

        if img is None:
            return JSONResponse({"gcode": "", "svg": format_error_svg("Invalid image")}, status_code=400)

        # -------------------------
        # Resize (no quality loss in grayscale processing)
        # -------------------------
        h, w = img.shape
        if max(h, w) > MAX_IMAGE_DIMENSION:
            scale = MAX_IMAGE_DIMENSION / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        # -------------------------
        # Invert (in-place safe)
        # -------------------------
        if invert.lower() == "true":
            cv2.bitwise_not(img, img)

        # -------------------------
        # Processing
        # -------------------------
        paths = process_algorithms(img, mode, spacing, density, start_time)

        del img  # CRITICAL: free raw image immediately

        paths = simplify_paths(paths, simplify)
        paths = optimize_paths_tsp(paths, start_time)

        # -------------------------
        # Output
        # -------------------------
        gcode, svg = generate_outputs(
            paths,
            0, 0,
            target_w_mm,
            target_h_mm,
            mode,
            invert
        )

        del paths  # CRITICAL

        return JSONResponse({"gcode": gcode, "svg": svg})

    except TimeoutException as e:
        return JSONResponse({"gcode": "", "svg": format_error_svg(str(e))})

    except Exception as e:
        return JSONResponse({"gcode": "", "svg": format_error_svg(str(e))})

    finally:
        # -------------------------
        # HARD MEMORY CLEANUP (this is the real fix)
        # -------------------------
        try:
            if img is not None:
                del img
        except:
            pass

        try:
            if paths is not None:
                del paths
        except:
            pass

        gc.collect()
        cv2.destroyAllWindows()