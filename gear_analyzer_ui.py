#!/usr/bin/env python3
"""
Spur Gear Dimensional Analysis using Computer Vision
Professional-grade gear measurement and tooth counting system
"""

import gradio as gr
import cv2
import numpy as np
import tempfile
from pathlib import Path
from gear_analyzer import analyze_gear, Config
import time


def process_gear_image(image, pixels_per_mm=None, min_teeth=12, max_teeth=120):
    """Process uploaded gear image and return results with timing"""
    if image is None:
        empty_html = """
        <div style='text-align: center; padding: 60px; background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%); border-radius: 16px;'>
            <div style='font-size: 4em; margin-bottom: 16px;'>📤</div>
            <h3 style='color: #475569; margin: 0;'>No Image Uploaded</h3>
            <p style='color: #94a3b8; margin-top: 8px;'>Please upload a gear image to begin analysis</p>
        </div>
        """
        return None, None, empty_html, "⚠️ Upload an image first"
    
    start_time = time.time()
    
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
        processing_time = time.time() - start_time
        
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
        results_html = format_results_html(results, processing_time)
        
        # Cleanup temp file
        tmp_path.unlink()
        
        status_msg = f"✅ Analysis Complete • {processing_time:.2f}s"
        return overlay_img, radii_img, results_html, status_msg
        
    except Exception as e:
        error_html = f"""
        <div style='background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); 
                    border-left: 4px solid #ef4444; padding: 24px; border-radius: 12px;'>
            <h3 style='color: #991b1b; margin: 0 0 8px 0;'>❌ Analysis Failed</h3>
            <p style='color: #7f1d1d; margin: 0;'>{str(e)}</p>
        </div>
        """
        return None, None, error_html, f"❌ Error: {str(e)}"


def format_results_html(results, processing_time):
    """Format results into ultra-modern HTML dashboard"""
    
    # Extract values
    tooth_count = results.get('teeth_estimate', 'N/A')
    cx, cy = results['center_px']
    r_add_px = results['r_add_px']
    r_ded_px = results['r_ded_px']
    r_pitch_px = results['r_pitch_px']
    r_hole_px = results.get('r_hole_px')
    module_px = results.get('module_px')
    
    # Physical measurements
    r_add_mm = results.get('r_add_mm')
    r_ded_mm = results.get('r_ded_mm')
    r_pitch_mm = results.get('r_pitch_mm')
    module_mm = results.get('module_mm')
    
    # Quality metrics
    tooth_meta = results['debug']['tooth_meta']
    quality = (1 - tooth_meta['primary']['noise_level']) * 100
    method = tooth_meta['primary']['method'].replace('_', ' ').title()
    
    # Determine quality color
    if quality >= 90:
        quality_color = "#10b981"
        quality_label = "Excellent"
    elif quality >= 75:
        quality_color = "#3b82f6"
        quality_label = "Good"
    elif quality >= 60:
        quality_color = "#f59e0b"
        quality_label = "Fair"
    else:
        quality_color = "#ef4444"
        quality_label = "Poor"
    
    html = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        
        .dashboard {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            padding: 0;
            margin: 0;
            animation: fadeIn 0.5s ease-in;
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .hero-section {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 20px;
            padding: 40px;
            text-align: center;
            color: white;
            margin-bottom: 24px;
            box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
            position: relative;
            overflow: hidden;
        }}
        
        .hero-section::before {{
            content: '';
            position: absolute;
            top: -50%;
            right: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
            animation: pulse 4s ease-in-out infinite;
        }}
        
        @keyframes pulse {{
            0%, 100% {{ transform: scale(1); opacity: 0.5; }}
            50% {{ transform: scale(1.1); opacity: 0.8; }}
        }}
        
        .hero-content {{
            position: relative;
            z-index: 1;
        }}
        
        .hero-count {{
            font-size: 5em;
            font-weight: 800;
            margin: 0;
            line-height: 1;
            text-shadow: 0 4px 12px rgba(0,0,0,0.2);
            letter-spacing: -2px;
        }}
        
        .hero-label {{
            font-size: 1.4em;
            margin: 12px 0 0 0;
            opacity: 0.95;
            font-weight: 500;
            letter-spacing: 0.5px;
        }}
        
        .stats-bar {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 12px;
            margin-bottom: 24px;
        }}
        
        .stat-chip {{
            background: white;
            border-radius: 12px;
            padding: 16px 20px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            border: 1px solid #f1f5f9;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }}
        
        .stat-chip:hover {{
            transform: translateY(-4px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.12);
            border-color: #667eea;
        }}
        
        .stat-label {{
            font-size: 0.75em;
            color: #64748b;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 6px;
        }}
        
        .stat-value {{
            font-size: 1.8em;
            color: #0f172a;
            font-weight: 700;
            line-height: 1;
        }}
        
        .section {{
            background: white;
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            border: 1px solid #f1f5f9;
            transition: all 0.3s ease;
        }}
        
        .section:hover {{
            box-shadow: 0 8px 24px rgba(0,0,0,0.12);
        }}
        
        .section-header {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 20px;
            padding-bottom: 16px;
            border-bottom: 2px solid #f1f5f9;
        }}
        
        .section-icon {{
            font-size: 1.8em;
            filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1));
        }}
        
        .section-title {{
            font-size: 1.3em;
            font-weight: 700;
            color: #0f172a;
            margin: 0;
            letter-spacing: -0.5px;
        }}
        
        .params-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 14px;
        }}
        
        .param-card {{
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            border-radius: 12px;
            padding: 18px;
            border: 2px solid transparent;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }}
        
        .param-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transform: scaleX(0);
            transition: transform 0.3s ease;
        }}
        
        .param-card:hover {{
            border-color: #667eea;
            transform: translateY(-2px);
            box-shadow: 0 8px 16px rgba(102, 126, 234, 0.15);
        }}
        
        .param-card:hover::before {{
            transform: scaleX(1);
        }}
        
        .param-label {{
            font-size: 0.8em;
            color: #64748b;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }}
        
        .param-value {{
            font-size: 2em;
            color: #0f172a;
            font-weight: 700;
            line-height: 1;
        }}
        
        .param-unit {{
            font-size: 0.6em;
            color: #94a3b8;
            font-weight: 500;
            margin-left: 4px;
        }}
        
        .quality-section {{
            background: linear-gradient(135deg, {quality_color}15 0%, {quality_color}05 100%);
            border: 2px solid {quality_color}40;
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
        }}
        
        .quality-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }}
        
        .quality-title {{
            font-size: 1.2em;
            font-weight: 700;
            color: #0f172a;
        }}
        
        .quality-badge {{
            background: {quality_color};
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: 700;
            box-shadow: 0 4px 12px {quality_color}40;
        }}
        
        .quality-bar-container {{
            width: 100%;
            height: 12px;
            background: #e2e8f0;
            border-radius: 6px;
            overflow: hidden;
            margin-bottom: 16px;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .quality-bar-fill {{
            height: 100%;
            background: linear-gradient(90deg, {quality_color}, {quality_color}dd);
            width: {quality}%;
            border-radius: 6px;
            transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 0 10px {quality_color}80;
        }}
        
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px;
        }}
        
        .info-item {{
            background: white;
            padding: 14px 18px;
            border-radius: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border: 1px solid #e2e8f0;
            transition: all 0.2s ease;
        }}
        
        .info-item:hover {{
            background: #f8fafc;
            border-color: #cbd5e1;
        }}
        
        .info-label {{
            color: #64748b;
            font-weight: 600;
            font-size: 0.9em;
        }}
        
        .info-value {{
            color: #0f172a;
            font-weight: 700;
            font-size: 1.1em;
        }}
        
        .badge {{
            display: inline-flex;
            align-items: center;
            padding: 6px 14px;
            border-radius: 8px;
            font-size: 0.85em;
            font-weight: 600;
            background: #e0e7ff;
            color: #4338ca;
            gap: 6px;
        }}
        
        .processing-time {{
            text-align: center;
            padding: 16px;
            background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
            border-radius: 12px;
            margin-top: 20px;
            border: 2px solid #fbbf24;
        }}
        
        .processing-time strong {{
            color: #78350f;
            font-size: 1.2em;
        }}
    </style>
    
    <div class="dashboard">
        <!-- Hero Section -->
        <div class="hero-section">
            <div class="hero-content">
                <div class="hero-count">⚙️ {tooth_count}</div>
                <div class="hero-label">Teeth Detected</div>
            </div>
        </div>
        
        <!-- Quick Stats Bar -->
        <div class="stats-bar">
            <div class="stat-chip">
                <div class="stat-label">Processing</div>
                <div class="stat-value">{processing_time:.2f}s</div>
            </div>
            <div class="stat-chip">
                <div class="stat-label">Quality</div>
                <div class="stat-value">{quality:.0f}%</div>
            </div>
            <div class="stat-chip">
                <div class="stat-label">Method</div>
                <div class="stat-value" style="font-size: 1.2em;">{method.split()[0]}</div>
            </div>
        </div>
        
        <!-- Quality Section -->
        <div class="quality-section">
            <div class="quality-header">
                <div class="quality-title">🎯 Analysis Quality</div>
                <div class="quality-badge">{quality_label}</div>
            </div>
            <div class="quality-bar-container">
                <div class="quality-bar-fill"></div>
            </div>
            <div class="info-grid">
                <div class="info-item">
                    <span class="info-label">Signal Quality</span>
                    <span class="info-value">{quality:.1f}%</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Detection Method</span>
                    <span class="badge">{method}</span>
                </div>
            </div>
        </div>
    """
    
    # Pixel Measurements Section
    html += f"""
        <div class="section">
            <div class="section-header">
                <span class="section-icon">📏</span>
                <h3 class="section-title">Dimensional Measurements (Pixels)</h3>
            </div>
            <div class="params-grid">
                <div class="param-card">
                    <div class="param-label">Addendum Radius</div>
                    <div class="param-value">{r_add_px:.2f}<span class="param-unit">px</span></div>
                </div>
                <div class="param-card">
                    <div class="param-label">Dedendum Radius</div>
                    <div class="param-value">{r_ded_px:.2f}<span class="param-unit">px</span></div>
                </div>
                <div class="param-card">
                    <div class="param-label">Pitch Radius</div>
                    <div class="param-value">{r_pitch_px:.2f}<span class="param-unit">px</span></div>
                </div>
                <div class="param-card">
                    <div class="param-label">Tooth Height</div>
                    <div class="param-value">{(r_add_px - r_ded_px):.2f}<span class="param-unit">px</span></div>
                </div>
                <div class="param-card">
                    <div class="param-label">Outer Diameter</div>
                    <div class="param-value">{(2 * r_add_px):.2f}<span class="param-unit">px</span></div>
                </div>
    """
    
    if r_hole_px:
        html += f"""
                <div class="param-card">
                    <div class="param-label">Bore Radius</div>
                    <div class="param-value">{r_hole_px:.2f}<span class="param-unit">px</span></div>
                </div>
        """
    
    if module_px:
        html += f"""
                <div class="param-card">
                    <div class="param-label">Module</div>
                    <div class="param-value">{module_px:.3f}<span class="param-unit">px</span></div>
                </div>
        """
    
    html += """
            </div>
        </div>
    """
    
    # Physical Measurements if calibrated
    if r_add_mm:
        html += f"""
        <div class="section">
            <div class="section-header">
                <span class="section-icon">📐</span>
                <h3 class="section-title">Physical Dimensions (Millimeters)</h3>
            </div>
            <div class="params-grid">
                <div class="param-card">
                    <div class="param-label">Addendum Radius</div>
                    <div class="param-value">{r_add_mm:.2f}<span class="param-unit">mm</span></div>
                </div>
                <div class="param-card">
                    <div class="param-label">Dedendum Radius</div>
                    <div class="param-value">{r_ded_mm:.2f}<span class="param-unit">mm</span></div>
                </div>
                <div class="param-card">
                    <div class="param-label">Pitch Radius</div>
                    <div class="param-value">{r_pitch_mm:.2f}<span class="param-unit">mm</span></div>
                </div>
                <div class="param-card">
                    <div class="param-label">Tooth Height</div>
                    <div class="param-value">{(r_add_mm - r_ded_mm):.2f}<span class="param-unit">mm</span></div>
                </div>
                <div class="param-card">
                    <div class="param-label">Outer Diameter</div>
                    <div class="param-value">{(2 * r_add_mm):.2f}<span class="param-unit">mm</span></div>
                </div>
        """
        
        if module_mm:
            html += f"""
                <div class="param-card">
                    <div class="param-label">Module</div>
                    <div class="param-value">{module_mm:.3f}<span class="param-unit">mm</span></div>
                </div>
            """
        
        html += """
            </div>
        </div>
        """
    
    html += """
    </div>
    """
    
    return html


# Ultra-modern CSS with animations
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.gradio-container {
    max-width: 1800px !important;
}

.upload-area {
    border: 3px dashed #667eea !important;
    border-radius: 20px !important;
    background: linear-gradient(135deg, #f8fafc 0%, #e0e7ff 100%) !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    position: relative !important;
    overflow: hidden !important;
}

.upload-area::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
    transition: left 0.5s ease;
}

.upload-area:hover::before {
    left: 100%;
}

.upload-area:hover {
    border-color: #764ba2 !important;
    background: linear-gradient(135deg, #e0e7ff 0%, #ddd6fe 100%) !important;
    transform: scale(1.01) !important;
    box-shadow: 0 12px 32px rgba(102, 126, 234, 0.2) !important;
}

.gr-button-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    font-weight: 700 !important;
    font-size: 1.15rem !important;
    padding: 14px 32px !important;
    border-radius: 12px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
    letter-spacing: 0.5px !important;
}

.gr-button-primary:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 12px 28px rgba(102, 126, 234, 0.5) !important;
}

.gr-button-primary:active {
    transform: translateY(-1px) !important;
}

.gr-box {
    border-radius: 16px !important;
}

.gr-input, .gr-dropdown {
    border-radius: 10px !important;
    border: 2px solid #e2e8f0 !important;
    transition: all 0.3s ease !important;
}

.gr-input:focus, .gr-dropdown:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
}

.gr-form {
    gap: 16px !important;
}

.gr-panel {
    border: none !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
}

/* Smooth scrolling */
html {
    scroll-behavior: smooth;
}

/* Loading animation */
@keyframes spin {
    to { transform: rotate(360deg); }
}

.loading {
    animation: spin 1s linear infinite;
}
"""

# Create the ultimate Gradio interface
with gr.Blocks(
    title="Spur Gear Analysis Pro",
    theme=gr.themes.Soft(
        primary_hue="violet",
        secondary_hue="purple",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
    css=custom_css,
    analytics_enabled=False
) as demo:
    
    # Header with gradient
    gr.HTML("""
        <div style='text-align: center; padding: 40px 20px; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 20px; margin-bottom: 30px; 
                    box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);'>
            <h1 style='color: white; font-size: 3em; margin: 0; font-weight: 800; 
                       text-shadow: 0 2px 10px rgba(0,0,0,0.2); letter-spacing: -1px;'>
                ⚙️ Spur Gear Analysis Pro
            </h1>
            <p style='color: rgba(255,255,255,0.95); font-size: 1.3em; margin: 12px 0 0 0; font-weight: 500;'>
                AI-Powered Dimensional Analysis & Tooth Counting
            </p>
        </div>
    """)
    
    # Main layout
    with gr.Row():
        # Left panel - Upload
        with gr.Column(scale=5):
            input_image = gr.Image(
                label="📤 Upload Gear Image",
                type="numpy",
                height=500,
                elem_classes="upload-area"
            )
            
            with gr.Row():
                with gr.Column(scale=2):
                    calibration_input = gr.Number(
                        label="🔧 Calibration",
                        placeholder="px/mm (optional)",
                        value=1
                    )
                with gr.Column(scale=1):
                    min_teeth_input = gr.Slider(
                        minimum=4,
                        maximum=60,
                        value=12,
                        step=1,
                        label="Min"
                    )
                with gr.Column(scale=1):
                    max_teeth_input = gr.Slider(
                        minimum=20,
                        maximum=200,
                        value=120,
                        step=1,
                        label="Max"
                    )
            
            analyze_btn = gr.Button(
                "🚀 Analyze Gear",
                variant="primary",
                size="lg"
            )
            
            status_text = gr.Textbox(
                show_label=False,
                interactive=False,
                placeholder="Ready • Upload an image to start",
                container=False
            )
        
        # Right panel - Results
        with gr.Column(scale=7):
            with gr.Tabs():
                with gr.Tab("📊 Dashboard"):
                    results_output = gr.HTML()
                
                with gr.Tab("🎯 Detection"):
                    output_image = gr.Image(
                        label="Tooth Detection Result",
                        height=500
                    )
                
                with gr.Tab("📏 Measurements"):
                    radii_image = gr.Image(
                        label="Radius Overlay",
                        height=500
                    )
    
    # Quick help
    with gr.Accordion("💡 Quick Tips", open=False):
        gr.Markdown("""
        **🖼️ Image:** High resolution (1000×1000+ px), white background  
        **💡 Lighting:** Even, no shadows or reflections  
        **🔧 Calibration:** Known distance (px) ÷ actual distance (mm)  
        **⚡ Speed:** Typical analysis: 3-8 seconds
        """)
    
    # Event handler
    analyze_btn.click(
        fn=process_gear_image,
        inputs=[input_image, calibration_input, min_teeth_input, max_teeth_input],
        outputs=[output_image, radii_image, results_output, status_text],
        api_name="analyze"
    )
    
    # Footer
    gr.HTML("""
        <div style='text-align: center; padding: 24px; margin-top: 30px; 
                    color: #64748b; border-top: 2px solid #f1f5f9;'>
            <strong style='color: #0f172a;'>Spur Gear Analysis Pro</strong> • 
            Computer Vision & AI • 
            <span style='color: #667eea;'>v2.0</span>
        </div>
    """)


if __name__ == "__main__":
    print("\n" + "="*75)
    print("⚙️  SPUR GEAR ANALYSIS PRO - ULTIMATE EDITION")
    print("="*75)
    print("\n🚀 Launching ultra-modern interface...")
    print("\n📍 Local:    http://localhost:7860")
    print("🌐 Network:  http://YOUR_IP:7860")
    print("\n✨ Features: AI Detection • Real-time Processing • Dashboard Analytics")
    print("💡 Press Ctrl+C to stop\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
        inbrowser=True  # Auto-open browser
    )
