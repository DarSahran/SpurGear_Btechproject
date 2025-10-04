#!/usr/bin/env python3
"""
FastAPI Backend for Gear Analyzer
Run: python api_server.py
API Docs: http://localhost:8000/docs
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
import tempfile
from pathlib import Path
from gear_analyzer import analyze_gear, Config
import base64
import uuid
import traceback

app = FastAPI(
    title="Gear Analyzer API",
    description="RESTful API for gear image analysis",
    version="1.0.0"
)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = Path("api_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


@app.get("/")
async def root():
    return {
        "service": "Gear Analyzer API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "analyze": "/api/analyze",
            "health": "/api/health",
            "docs": "/docs"
        }
    }


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Gear Analyzer API"
    }


@app.post("/api/analyze")
async def analyze_gear_endpoint(
    file: UploadFile = File(...),
    pixels_per_mm: float = Form(None),
    min_teeth: int = Form(12),
    max_teeth: int = Form(120)
):
    """
    Analyze uploaded gear image
    
    Parameters:
    - file: Gear image file (PNG, JPG, JPEG)
    - pixels_per_mm: Optional calibration value
    - min_teeth: Minimum expected tooth count
    - max_teeth: Maximum expected tooth count
    
    Returns:
    - JSON with analysis results, overlay image, and diagnostic plots
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read uploaded image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid or corrupted image file")
        
        # Save temporarily
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            cv2.imwrite(str(tmp_path), img)
        
        # Create unique output directory for this analysis
        session_id = str(uuid.uuid4())[:8]
        out_dir = OUTPUT_DIR / session_id
        out_dir.mkdir(exist_ok=True)
        
        # Configure analysis
        cfg = Config(
            input_image=str(tmp_path),
            out_dir=str(out_dir),
            show_plots=False,
            save_plots=True,
            save_overlay=True,
            expected_teeth_min=min_teeth,
            expected_teeth_max=max_teeth,
            pixels_per_mm=pixels_per_mm if pixels_per_mm and pixels_per_mm > 0 else None
        )
        
        # Run analysis
        results = analyze_gear(cfg)
        
        # Read overlay image and encode to base64
        overlay_path = out_dir / "gear_teeth_overlay.png"
        overlay_b64 = None
        if overlay_path.exists():
            with open(overlay_path, "rb") as img_file:
                overlay_b64 = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Read radii overlay
        radii_path = out_dir / "gear_radii_overlay.png"
        radii_b64 = None
        if radii_path.exists():
            with open(radii_path, "rb") as img_file:
                radii_b64 = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Read diagnostic plots
        plots = {}
        for plot_file in out_dir.glob("*.png"):
            if plot_file.name not in ["gear_teeth_overlay.png", "gear_radii_overlay.png", "gear_mask.png"]:
                with open(plot_file, "rb") as pf:
                    plots[plot_file.stem] = base64.b64encode(pf.read()).decode('utf-8')
        
        # Cleanup temp file
        tmp_path.unlink()
        
        return {
            "success": True,
            "session_id": session_id,
            "results": results,
            "images": {
                "teeth_overlay": overlay_b64,
                "radii_overlay": radii_b64,
                "diagnostic_plots": plots
            },
            "summary": {
                "tooth_count": results.get('teeth_estimate'),
                "addendum_radius_px": results.get('r_add_px'),
                "pitch_radius_px": results.get('r_pitch_px'),
                "module_px": results.get('module_px'),
                "has_calibration": results.get('r_add_mm') is not None
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        # Log full traceback for debugging
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Starting Gear Analyzer FastAPI Server")
    print("="*60)
    print("\n📍 API Server: http://localhost:8000")
    print("📖 API Docs: http://localhost:8000/docs")
    print("🔬 Interactive Docs: http://localhost:8000/redoc")
    print("\n💡 Press Ctrl+C to stop the server\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
