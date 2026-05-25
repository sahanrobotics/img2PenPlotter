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

MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB limit


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

    try:
        # -----------------------------
        # 1. Read file safely (limit memory spike)
        # -----------------------------
        contents = await file.read()

        if len(contents) > MAX_UPLOAD_SIZE:
            return JSONResponse(
                {"gcode": "", "svg": format_error_svg("File too large")},
                status_code=413
            )

        nparr = np.frombuffer(contents, np.uint8)
        contents = None  # free raw buffer early

        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        nparr = None  # free memory

        if img is None:
            return JSONResponse({"gcode": "", "svg": format_error_svg("Invalid image")}, status_code=400)

        # -----------------------------
        # 2. Resize early (big memory saver)
        # -----------------------------
        h, w = img.shape
        max_dim = MAX_IMAGE_DIMENSION

        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        # -----------------------------
        # 3. Invert without extra copy
        # -----------------------------
        if invert.lower() == "true":
            cv2.bitwise_not(img, img)

        # -----------------------------
        # 4. Processing pipeline
        # -----------------------------
        paths = process_algorithms(img, mode, spacing, density, start_time)

        img = None  # release early

        paths = simplify_paths(paths, simplify)
        paths = optimize_paths_tsp(paths, start_time)

        # -----------------------------
        # 5. Output generation
        # -----------------------------
        img_h, img_w = 0, 0  # already freed image, avoid reuse
        gcode, svg = generate_outputs(
            paths,
            0, 0,
            target_w_mm,
            target_h_mm,
            mode,
            invert
        )

        # -----------------------------
        # 6. Force cleanup (important on Railway)
        # -----------------------------
        paths = None
        gc.collect()

        return JSONResponse({"gcode": gcode, "svg": svg})

    except TimeoutException as e:
        gc.collect()
        return JSONResponse({"gcode": "", "svg": format_error_svg(str(e))})

    except Exception as e:
        gc.collect()
        return JSONResponse({"gcode": "", "svg": format_error_svg(str(e))})


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))

    uvicorn.run("main:app", host="0.0.0.0", port=port)