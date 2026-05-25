import os
import cv2
import numpy as np
import time

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

# CORS (safe for frontend apps)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def format_error_svg(error_msg: str):
    return f"""
    <svg viewBox="0 0 400 100" xmlns="http://www.w3.org/2000/svg">
        <text x="20" y="50" fill="red">{error_msg}</text>
    </svg>
    """


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
        # 1. Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)

        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        if img is None:
            return JSONResponse(
                content={"gcode": "", "svg": format_error_svg("Invalid image")},
                status_code=400
            )

        # 2. Resize if too large
        if max(img.shape) > MAX_IMAGE_DIMENSION:
            scale = MAX_IMAGE_DIMENSION / max(img.shape)
            img = cv2.resize(
                img,
                (int(img.shape[1] * scale), int(img.shape[0] * scale))
            )

        # 3. Invert if needed
        if invert.lower() == "true":
            img = cv2.bitwise_not(img)

        # 4. Generate paths
        paths = process_algorithms(img, mode, spacing, density, start_time)

        # 5. Simplify
        paths = simplify_paths(paths, simplify)

        # 6. Optimize (TSP)
        paths = optimize_paths_tsp(paths, start_time)

        # 7. Output generation
        img_h, img_w = img.shape

        gcode, svg = generate_outputs(
            paths,
            img_w,
            img_h,
            target_w_mm,
            target_h_mm,
            mode,
            invert
        )

        return JSONResponse(content={
            "gcode": gcode,
            "svg": svg
        })

    except TimeoutException as e:
        return JSONResponse(content={
            "gcode": "",
            "svg": format_error_svg(f"Timeout: {str(e)}")
        })

    except Exception as e:
        return JSONResponse(content={
            "gcode": "",
            "svg": format_error_svg(f"Error: {str(e)}")
        })


# -----------------------------
# Railway / Production Entry
# -----------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port
    )