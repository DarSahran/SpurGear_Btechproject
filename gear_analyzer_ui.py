#!/usr/bin/env python3
"""
Gradio Web UI for Gear Analyzer
Run: python gear_analyzer_ui.py
Access: http://localhost:7860
"""

import gradio as gr
import cv2
import numpy as np
import tempfile
from pathlib import Path
from gear_analyzer import analyze_gear, Config


def process_gear_image(image, pixels_per_mm=None, min_teeth=12, max_teeth=120):
    """Process uploaded gear image and return results"""
    if image is None:
        return None, None, None, "⚠️ Please upload an image first"
    
    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = Path(tmp.name)
        cv2.imwrite(str(tmp_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    # Create output directory
    out_dir = Path("Output_UI")
    out_dir.mkdir(exist_ok=True)
    
    # Configure and run analysis
    cfg = Config(
        input_image=str(tmp_path),
        out_dir=str(out_dir),
        show_plots=False,
        save_plots=True,
        save_overlay=True,
        expected_teeth_min=int(min_teeth) if min_teeth else 12,
        expected_teeth_max=int(max_teeth) if max_teeth else 120,
        pixels_per_mm=float(pixels_per_mm) if pixels_per_mm and pixels_per_mm > 0 else None
    )
    
    try:
        results = analyze_gear(cfg)
        
        # Load overlay image
        overlay_path = out_dir / "gear_teeth_overlay.png"
        overlay_img = None
        if overlay_path.exists():
            overlay_img = cv2.imread(str(overlay_path))
            overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)
        
        # Load radii diagram
        radii_path = out_dir / "gear_radii_overlay.png"
        radii_img = None
        if radii_path.exists():
            radii_img = cv2.imread(str(radii_path))
            radii_img = cv2.cvtColor(radii_img, cv2.COLOR_BGR2RGB)
        
        # Format results for display
        results_text = format_results(results)
        
        # Cleanup temp file
        tmp_path.unlink()
        
        return overlay_img, radii_img, results_text, "✅ Analysis completed successfully!"
        
    except Exception as e:
        return None, None, None, f"❌ Error: {str(e)}"


def format_results(results):
    """Format results dictionary into readable text"""
    output = "🔧 GEAR ANALYSIS RESULTS\n" + "="*60 + "\n\n"
    
    # Tooth count - most important result
    if results.get('teeth_estimate'):
        output += f"⚙️  TOOTH COUNT: {results['teeth_estimate']} teeth\n\n"
    else:
        output += "⚙️  TOOTH COUNT: Unable to determine\n\n"
    
    # Basic measurements in pixels
    output += "📏 MEASUREMENTS (Pixels)\n"
    output += f"   Center Position: ({results['center_px'][0]:.1f}, {results['center_px'][1]:.1f})\n"
    output += f"   Addendum Radius (outer): {results['r_add_px']:.2f} px\n"
    output += f"   Dedendum Radius (root): {results['r_ded_px']:.2f} px\n"
    output += f"   Pitch Radius (effective): {results['r_pitch_px']:.2f} px\n"
    output += f"   Tooth Height: {(results['r_add_px'] - results['r_ded_px']):.2f} px\n"
    
    if results['r_hole_px']:
        output += f"   Bore/Hole Radius: {results['r_hole_px']:.2f} px\n"
    
    # Derived parameters
    if results.get('module_px'):
        output += f"\n📐 GEAR PARAMETERS (Pixels)\n"
        output += f"   Module: {results['module_px']:.3f} px\n"
        output += f"   Circular Pitch: {results['circular_pitch_px']:.3f} px\n"
        output += f"   Outer Diameter: {(2 * results['r_add_px']):.2f} px\n"
    
    # Physical measurements if calibrated
    if results.get('r_add_mm'):
        output += f"\n📏 PHYSICAL MEASUREMENTS (Millimeters)\n"
        output += f"   Addendum Radius: {results['r_add_mm']:.2f} mm\n"
        output += f"   Dedendum Radius: {results['r_ded_mm']:.2f} mm\n"
        output += f"   Pitch Radius: {results['r_pitch_mm']:.2f} mm\n"
        output += f"   Tooth Height: {(results['r_add_mm'] - results['r_ded_mm']):.2f} mm\n"
        output += f"   Outer Diameter: {(2 * results['r_add_mm']):.2f} mm\n"
        
        if results.get('module_mm'):
            output += f"   Module: {results['module_mm']:.3f} mm\n"
            output += f"   Circular Pitch: {results['circular_pitch_mm']:.3f} mm\n"
    
    # Detection confidence and method
    tooth_meta = results['debug']['tooth_meta']
    output += f"\n🎯 DETECTION QUALITY\n"
    output += f"   Method Used: {tooth_meta['primary']['method']}\n"
    output += f"   Signal Quality: {(1 - tooth_meta['primary']['noise_level']) * 100:.1f}%\n"
    output += f"   Conservative Count: {tooth_meta['primary']['cons_count']}\n"
    output += f"   Liberal Count: {tooth_meta['primary']['lib_count']}\n"
    
    # Threshold info
    thresh_info = results['debug']['threshold']
    output += f"\n🔬 IMAGE PROCESSING\n"
    output += f"   Method: {thresh_info['method_used']}\n"
    output += f"   Foreground/Background Gap: {thresh_info['intensity_gap']:.1f}\n"
    
    return output


# Create Gradio Interface with tabs
with gr.Blocks(title="Gear Analyzer Pro", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # ⚙️ Professional Gear Analysis System
    Upload a gear image to automatically detect tooth count, measurements, and parameters
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Upload & Configure")
            input_image = gr.Image(label="Upload Gear Image", type="numpy", height=400)
            
            with gr.Accordion("⚙️ Advanced Settings", open=False):
                calibration_input = gr.Number(
                    label="Calibration (pixels per mm)", 
                    value=None,
                    info="Optional: Enter pixels/mm for physical measurements"
                )
                min_teeth_input = gr.Slider(
                    minimum=4, maximum=60, value=12, step=1,
                    label="Minimum Expected Teeth",
                    info="Lower bound for tooth count detection"
                )
                max_teeth_input = gr.Slider(
                    minimum=20, maximum=200, value=120, step=1,
                    label="Maximum Expected Teeth",
                    info="Upper bound for tooth count detection"
                )
            
            analyze_btn = gr.Button("🔍 Analyze Gear", variant="primary", size="lg")
            status_text = gr.Textbox(label="Status", interactive=False, lines=2)
        
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Analysis Results")
            
            with gr.Tabs():
                with gr.Tab("🎯 Tooth Detection"):
                    output_image = gr.Image(label="Detected Teeth (Red Markers)")
                
                with gr.Tab("📏 Measurements"):
                    radii_image = gr.Image(label="Radii Visualization")
                
                with gr.Tab("📋 Detailed Report"):
                    results_output = gr.Textbox(
                        label="Complete Analysis Report", 
                        lines=25,
                        max_lines=30
                    )
    
    # Instructions
    with gr.Accordion("📖 How to Use", open=False):
        gr.Markdown("""
        ### Quick Start Guide:
        
        1. **Upload Image**: Click the upload box and select a clear gear image
           - Best results with white/light background
           - Ensure gear is fully visible and centered
           - Supported formats: PNG, JPG, JPEG
        
        2. **Configure (Optional)**:
           - **Calibration**: If you know pixels-per-mm, enter it for real measurements
           - **Tooth Range**: Adjust if you know approximate tooth count
        
        3. **Analyze**: Click the "Analyze Gear" button and wait 5-10 seconds
        
        4. **Review Results**:
           - **Tooth Detection Tab**: See marked teeth on your gear
           - **Measurements Tab**: View radius measurements
           - **Report Tab**: Get complete numerical analysis
        
        ### Tips for Best Results:
        - Use high-resolution images (1000x1000 px or higher)
        - Ensure good contrast between gear and background
        - Avoid shadows, reflections, or obstructions
        - Gear should occupy most of the image frame
        
        ### Calibration Help:
        To get pixels-per-mm: Measure a known distance on your image in pixels,
        divide by actual distance in mm. Example: 500 px = 25 mm → 500/25 = 20 px/mm
        """)
    
    # Examples section
    gr.Markdown("### 📚 Example Results")
    gr.Markdown("""
    **Typical Output**: 24 teeth, Module: 2.5mm, Pitch Radius: 30mm  
    **Processing Time**: 5-10 seconds depending on image size  
    **Accuracy**: ±1 tooth for clean images with good contrast
    """)
    
    # Button click handler
    analyze_btn.click(
        fn=process_gear_image,
        inputs=[input_image, calibration_input, min_teeth_input, max_teeth_input],
        outputs=[output_image, radii_image, results_output, status_text]
    )


# Launch the app
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Starting Gear Analyzer Web Interface")
    print("="*60)
    print("\n📍 Server will start at: http://localhost:7860")
    print("🌐 Access from other devices: http://YOUR_IP:7860")
    print("\n💡 Press Ctrl+C to stop the server\n")
    
    demo.launch(
        server_name="0.0.0.0",  # Allow network access
        server_port=7860,
        share=False,  # Set True to get public Gradio link
        show_error=True
    )
