import os
import cv2
import numpy as np
import time

from fastapi import FastAPI, UploadFile, Form, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional

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

# -------------------------------------------------------------------------
# GLOBAL STATE: Acts as a gateway between Flutter App and ESP32
# -------------------------------------------------------------------------
class PlotterState:
    command: str = "stop"       # start, stop, pause, return_to_zero, reset_zero
    command_id: int = 0         # Increments on new command (tells ESP32 to execute)
    feedrate: int = 3000        # Current drawing speed
    file_version: int = 0       # Increments on new image generation
    latest_file: str = "last_generated.gcode"

plotter_state = PlotterState()

class CommandPayload(BaseModel):
    command: Optional[str] = None
    feedrate: Optional[int] = None

VALID_COMMANDS = ["start", "stop", "pause", "return_to_zero", "reset_zero"]


def format_error_svg(error_msg: str):
    return f"""
    <svg viewBox="0 0 400 100" xmlns="http://www.w3.org/2000/svg">
        <text x="20" y="50" fill="red">{error_msg}</text>
    </svg>
    """

# -------------------------------------------------------------------------
# GATEWAY ENDPOINTS (Flutter App <--> ESP32)
# -------------------------------------------------------------------------

@app.get("/api/plotter/status")
async def get_plotter_status():
    """
    ESP32 calls this endpoint every ~1-2 seconds.
    It checks 'command_id' and 'file_version' to know if it needs to act.
    """
    return {
        "command": plotter_state.command,
        "command_id": plotter_state.command_id,
        "feedrate": plotter_state.feedrate,
        "file_version": plotter_state.file_version,
        "has_file": os.path.exists(plotter_state.latest_file)
    }


@app.post("/api/plotter/command")
async def set_plotter_command(payload: CommandPayload):
    """
    Flutter App calls this to send commands or change Feedrate.
    """
    state_changed = False

    if payload.command is not None:
        cmd = payload.command.lower()
        if cmd in VALID_COMMANDS:
            plotter_state.command = cmd
            plotter_state.command_id += 1  # ESP32 will see this increment and execute
            state_changed = True
        else:
            return JSONResponse(
                status_code=400, 
                content={"error": f"Invalid command. Must be one of: {VALID_COMMANDS}"}
            )
    
    if payload.feedrate is not None:
        if payload.feedrate > 0:
            plotter_state.feedrate = payload.feedrate
            # We don't necessarily increment command_id for feedrate alone, 
            # ESP32 can just silently update its speed variable.
        else:
            return JSONResponse(status_code=400, content={"error": "Feedrate must be positive"})

    return {
        "status": "success",
        "current_state": {
            "command": plotter_state.command,
            "command_id": plotter_state.command_id,
            "feedrate": plotter_state.feedrate
        }
    }


@app.get("/api/latest-gcode")
async def get_latest_gcode():
    """
    ESP32 downloads the file from here when 'file_version' increments.
    """
    if os.path.exists(plotter_state.latest_file):
        headers = {"Cache-Control": "no-cache, no-store, must-revalidate"}
        return FileResponse(
            plotter_state.latest_file, 
            media_type="text/plain", 
            filename="latest_plot.gcode",
            headers=headers
        )
    return JSONResponse(
        content={"error": "No G-code file found. Please generate one first."}, 
        status_code=404
    )


# -------------------------------------------------------------------------
# GENERATION ENDPOINT
# -------------------------------------------------------------------------

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
            return JSONResponse(content={"gcode": "", "svg": format_error_svg("Invalid image")}, status_code=400)

        # 2. Resize
        if max(img.shape) > MAX_IMAGE_DIMENSION:
            scale = MAX_IMAGE_DIMENSION / max(img.shape)
            img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))

        # 3. Invert
        if invert.lower() == "true":
            img = cv2.bitwise_not(img)

        # 4. Generate & Optimize paths
        paths = process_algorithms(img, mode, spacing, density, start_time)
        paths = simplify_paths(paths, simplify)
        paths = optimize_paths_tsp(paths, start_time)

        # 5. Output generation
        img_h, img_w = img.shape
        gcode, svg = generate_outputs(paths, img_w, img_h, target_w_mm, target_h_mm, mode, invert)

        # 6. Save G-code
        with open(plotter_state.latest_file, "w", encoding="utf-8") as f:
            f.write(gcode)

        # -----------------------------------------------------------------
        # GATEWAY UPDATE: Notify ESP32 of new file & safely halt the plotter
        # -----------------------------------------------------------------
        plotter_state.file_version += 1
        
        # Automatically stop the plotter so it doesn't draw randomly while downloading
        plotter_state.command = "stop"
        plotter_state.command_id += 1  

        return JSONResponse(content={"gcode": gcode, "svg": svg})

    except TimeoutException as e:
        return JSONResponse(content={"gcode": "", "svg": format_error_svg(f"Timeout: {str(e)}")})
    except Exception as e:
        return JSONResponse(content={"gcode": "", "svg": format_error_svg(f"Error: {str(e)}")})


# -----------------------------
# Railway / Production Entry
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)