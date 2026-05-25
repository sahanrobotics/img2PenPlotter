import cv2
import numpy as np
import time
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import our separated logic modules
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


def format_error_svg(error_msg):
    return f'<svg viewBox="0 0 400 100" xmlns="http://www.w3.org/2000/svg"><text x="20" y="50" fill="red">{error_msg}</text></svg>'


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
        # 1. Read & Resize Image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        if max(img.shape) > MAX_IMAGE_DIMENSION:
            scale = MAX_IMAGE_DIMENSION / max(img.shape)
            img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))

        # 2. Process Image Inversion
        if invert.lower() == "true":
            img = cv2.bitwise_not(img)

        # 3. Generate Paths based on Algorithm selection
        paths = process_algorithms(img, mode, spacing, density, start_time)

        # 4. Simplify Paths (Remove unnecessary nodes)
        paths = simplify_paths(paths, simplify)

        # 5. Optimize Pen Movements (TSP)
        paths = optimize_paths_tsp(paths, start_time)

        # 6. Generate outputs
        img_h, img_w = img.shape
        gcode, svg = generate_outputs(paths, img_w, img_h, target_w_mm, target_h_mm, mode, invert)

        return JSONResponse(content={"gcode": gcode, "svg": svg})

    except TimeoutException as e:
        return JSONResponse(content={"gcode": "", "svg": format_error_svg(f"Error: {str(e)}")})
    except Exception as e:
        return JSONResponse(content={"gcode": "", "svg": format_error_svg(f"Error: {str(e)}")})


if __name__ == "__main__":
    import uvicorn

    # Make sure to run 'main:app' since the file is now called main.py
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)