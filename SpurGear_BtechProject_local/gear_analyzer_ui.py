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
        <div style='text-align: center; padding: 60px; background: var(--background-fill-secondary); 
                    border-radius: 16px; border: 2px dashed var(--border-color-primary);'>
            <div style='font-size: 4em; margin-bottom: 16px;'>📤</div>
            <h3 style='color: var(--body-text-color); margin: 0;'>No Image Uploaded</h3>
            <p style='color: var(--body-text-color-subdued); margin-top: 8px;'>Please upload a gear image to begin analysis</p>
        </div>
        """
        return None, None, empty_html, "⚠️ Upload an image first"
    
    start_time = time.time()
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = Path(tmp.name)
        cv2.imwrite(str(tmp_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    out_dir = Path("Output_UI")
    out_dir.mkdir(exist_ok=True)
    
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
        
        overlay_path = out_dir / "gear_teeth_overlay.png"
        overlay_img = None
        if overlay_path.exists():
            overlay_img = cv2.imread(str(overlay_path))
            overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)
        
        radii_path = out_dir / "gear_radii_overlay.png"
        radii_img = None
        if radii_path.exists():
            radii_img = cv2.imread(str(radii_path))
            radii_img = cv2.cvtColor(radii_img, cv2.COLOR_BGR2RGB)
        
        results_html = format_results_html(results, processing_time)
        
        tmp_path.unlink()
        
        status_msg = f"✅ Analysis Complete • {processing_time:.2f}s"
        return overlay_img, radii_img, results_html, status_msg
        
    except Exception as e:
        error_html = f"""
        <div style='background: rgba(239, 68, 68, 0.1); 
                    border-left: 4px solid #ef4444; padding: 24px; border-radius: 12px;'>
            <h3 style='color: #ef4444; margin: 0 0 8px 0; font-weight: 700;'>❌ Analysis Failed</h3>
            <p style='color: var(--body-text-color); margin: 0; font-weight: 500;'>{str(e)}</p>
        </div>
        """
        return None, None, error_html, f"❌ Error: {str(e)}"


def format_results_html(results, processing_time):
    """Format results into ultra-modern HTML dashboard with dark mode support"""
    
    tooth_count = results.get('teeth_estimate', 'N/A')
    cx, cy = results['center_px']
    r_add_px = results['r_add_px']
    r_ded_px = results['r_ded_px']
    r_pitch_px = results['r_pitch_px']
    r_hole_px = results.get('r_hole_px')
    module_px = results.get('module_px')
    
    r_add_mm = results.get('r_add_mm')
    r_ded_mm = results.get('r_ded_mm')
    r_pitch_mm = results.get('r_pitch_mm')
    module_mm = results.get('module_mm')
    
    tooth_meta = results['debug']['tooth_meta']
    quality = (1 - tooth_meta['primary']['noise_level']) * 100
    method = tooth_meta['primary']['method'].replace('_', ' ').title()
    
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
            color: var(--body-text-color);
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
            box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
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
            text-shadow: 0 4px 12px rgba(0,0,0,0.3);
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
            background: var(--background-fill-secondary);
            border-radius: 12px;
            padding: 16px 20px;
            text-align: center;
            box-shadow: 0 2px 12px rgba(0,0,0,0.08);
            border: 1px solid var(--border-color-primary);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }}
        
        .stat-chip:hover {{
            transform: translateY(-4px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.12);
            border-color: #667eea;
        }}
        
        .stat-label {{
            font-size: 0.75em;
            color: var(--body-text-color-subdued);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 6px;
        }}
        
        .stat-value {{
            font-size: 1.8em;
            color: var(--body-text-color);
            font-weight: 700;
            line-height: 1;
        }}
        
        .section {{
            background: var(--background-fill-secondary);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.08);
            border: 1px solid var(--border-color-primary);
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
            border-bottom: 2px solid var(--border-color-primary);
        }}
        
        .section-icon {{
            font-size: 1.8em;
            filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1));
        }}
        
        .section-title {{
            font-size: 1.3em;
            font-weight: 700;
            color: var(--body-text-color);
            margin: 0;
            letter-spacing: -0.5px;
        }}
        
        .params-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 14px;
        }}
        
        .param-card {{
            background: var(--background-fill-primary);
            border-radius: 12px;
            padding: 18px;
            border: 2px solid var(--border-color-primary);
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
            box-shadow: 0 8px 24px rgba(102, 126, 234, 0.2);
        }}
        
        .param-card:hover::before {{
            transform: scaleX(1);
        }}
        
        .param-label {{
            font-size: 0.8em;
            color: var(--body-text-color-subdued);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }}
        
        .param-value {{
            font-size: 2em;
            color: var(--body-text-color);
            font-weight: 700;
            line-height: 1;
        }}
        
        .param-unit {{
            font-size: 0.6em;
            color: var(--body-text-color-subdued);
            font-weight: 500;
            margin-left: 4px;
        }}
        
        .quality-section {{
            background: var(--background-fill-secondary);
            border: 2px solid {quality_color};
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.1);
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
            color: var(--body-text-color);
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
            background: var(--border-color-primary);
            border-radius: 6px;
            overflow: hidden;
            margin-bottom: 16px;
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
            background: var(--background-fill-primary);
            padding: 14px 18px;
            border-radius: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border: 1px solid var(--border-color-primary);
            transition: all 0.2s ease;
        }}
        
        .info-item:hover {{
            background: var(--background-fill-secondary);
            border-color: var(--border-color-accent);
        }}
        
        .info-label {{
            color: var(--body-text-color-subdued);
            font-weight: 600;
            font-size: 0.9em;
        }}
        
        .info-value {{
            color: var(--body-text-color);
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
            background: rgba(102, 126, 234, 0.15);
            color: #667eea;
            gap: 6px;
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
    
    html += "</div></div>"
    
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
        
        html += "</div></div>"
    
    html += "</div>"
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
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.upload-area:hover {
    border-color: #764ba2 !important;
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
    border: 2px solid var(--border-color-primary) !important;
    transition: all 0.3s ease !important;
}

.gr-input:focus, .gr-dropdown:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
}

.gr-form {
    gap: 16px !important;
}

html {
    scroll-behavior: smooth;
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
                        value=None
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
                    color: var(--body-text-color-subdued); border-top: 2px solid var(--border-color-primary);'>
            <strong style='color: var(--body-text-color);'>Spur Gear Analysis Pro</strong> • 
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
        share=False,
        show_error=True,
        inbrowser=True
    )
