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
            <h3 style='color: #1e293b; margin: 0;'>No Image Uploaded</h3>
            <p style='color: #64748b; margin-top: 8px;'>Please upload a gear image to begin analysis</p>
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
        <div style='background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); 
                    border-left: 4px solid #dc2626; padding: 24px; border-radius: 12px;'>
            <h3 style='color: #7f1d1d; margin: 0 0 8px 0; font-weight: 700;'>❌ Analysis Failed</h3>
            <p style='color: #991b1b; margin: 0; font-weight: 500;'>{str(e)}</p>
        </div>
        """
        return None, None, error_html, f"❌ Error: {str(e)}"


def format_results_html(results, processing_time):
    """Format results into ultra-modern HTML dashboard with improved contrast"""
    
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
        quality_color = "#059669"
        quality_label = "Excellent"
    elif quality >= 75:
        quality_color = "#2563eb"
        quality_label = "Good"
    elif quality >= 60:
        quality_color = "#d97706"
        quality_label = "Fair"
    else:
        quality_color = "#dc2626"
        quality_label = "Poor"
    
    html = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        
        .dashboard {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            padding: 0;
            margin: 0;
            animation: fadeIn 0.5s ease-in;
            max-width: 100%;
            box-sizing: border-box;
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .hero-section {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: clamp(12px, 3vw, 20px);
            padding: clamp(24px, 5vw, 40px);
            text-align: center;
            color: white;
            margin-bottom: clamp(16px, 3vw, 24px);
            box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
        }}
        
        .hero-count {{
            font-size: clamp(2.5em, 8vw, 5em);
            font-weight: 800;
            margin: 0;
            line-height: 1.1;
            text-shadow: 0 4px 12px rgba(0,0,0,0.25);
        }}
        
        .hero-label {{
            font-size: clamp(1em, 2.5vw, 1.4em);
            margin: clamp(8px, 2vw, 12px) 0 0 0;
            opacity: 0.95;
            font-weight: 500;
        }}
        
        .stats-bar {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: clamp(8px, 2vw, 12px);
            margin-bottom: clamp(16px, 3vw, 24px);
        }}
        
        .stat-chip {{
            background: white;
            border-radius: clamp(8px, 2vw, 12px);
            padding: clamp(12px, 2.5vw, 16px) clamp(14px, 3vw, 20px);
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border: 1px solid #e2e8f0;
        }}
        
        .stat-label {{
            font-size: clamp(0.65em, 1.5vw, 0.75em);
            color: #475569;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 6px;
        }}
        
        .stat-value {{
            font-size: clamp(1.4em, 3vw, 1.8em);
            color: #0f172a;
            font-weight: 800;
        }}
        
        .section {{
            background: white;
            border-radius: clamp(12px, 2.5vw, 16px);
            padding: clamp(16px, 3vw, 24px);
            margin-bottom: clamp(12px, 2.5vw, 20px);
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            border: 1px solid #e2e8f0;
        }}
        
        .section-header {{
            display: flex;
            align-items: center;
            gap: clamp(8px, 2vw, 12px);
            margin-bottom: clamp(12px, 2.5vw, 20px);
            padding-bottom: clamp(10px, 2vw, 16px);
            border-bottom: 2px solid #e2e8f0;
        }}
        
        .section-title {{
            font-size: clamp(1.1em, 2.5vw, 1.3em);
            font-weight: 800;
            color: #0f172a;
            margin: 0;
        }}
        
        .params-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(clamp(160px, 25vw, 200px), 1fr));
            gap: clamp(10px, 2vw, 14px);
        }}
        
        .param-card {{
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            border-radius: clamp(8px, 2vw, 12px);
            padding: clamp(14px, 2.5vw, 18px);
            border: 2px solid transparent;
        }}
        
        .param-card:hover {{
            border-color: #667eea;
            transform: translateY(-2px);
            box-shadow: 0 8px 16px rgba(102, 126, 234, 0.2);
        }}
        
        .param-label {{
            font-size: clamp(0.7em, 1.5vw, 0.8em);
            color: #475569;
            font-weight: 700;
            text-transform: uppercase;
            margin-bottom: clamp(6px, 1.5vw, 8px);
        }}
        
        .param-value {{
            font-size: clamp(1.5em, 3vw, 2em);
            color: #0f172a;
            font-weight: 800;
        }}
        
        .param-unit {{
            font-size: 0.6em;
            color: #64748b;
            font-weight: 600;
        }}
        
        .quality-section {{
            background: linear-gradient(135deg, {quality_color}15 0%, {quality_color}05 100%);
            border: 2px solid {quality_color}40;
            border-radius: clamp(12px, 2.5vw, 16px);
            padding: clamp(16px, 3vw, 24px);
            margin-bottom: clamp(12px, 2.5vw, 20px);
        }}
        
        .quality-badge {{
            background: {quality_color};
            color: white;
            padding: clamp(6px, 1.5vw, 8px) clamp(12px, 2.5vw, 16px);
            border-radius: 20px;
            font-size: clamp(0.8em, 1.8vw, 0.9em);
            font-weight: 700;
        }}
        
        .quality-bar-container {{
            width: 100%;
            height: clamp(8px, 2vw, 12px);
            background: #cbd5e1;
            border-radius: 6px;
            overflow: hidden;
            margin-bottom: 16px;
        }}
        
        .quality-bar-fill {{
            height: 100%;
            background: linear-gradient(90deg, {quality_color}, {quality_color}dd);
            width: {quality}%;
            border-radius: 6px;
        }}
    </style>
    
    <div class="dashboard">
        <div class="hero-section">
            <div class="hero-count">⚙️ {tooth_count}</div>
            <div class="hero-label">Teeth Detected</div>
        </div>
        
        <div class="stats-bar">
            <div class="stat-chip">
                <div class="stat-label">Time</div>
                <div class="stat-value">{processing_time:.2f}s</div>
            </div>
            <div class="stat-chip">
                <div class="stat-label">Quality</div>
                <div class="stat-value">{quality:.0f}%</div>
            </div>
            <div class="stat-chip">
                <div class="stat-label">Method</div>
                <div class="stat-value" style="font-size: clamp(1em, 2vw, 1.2em);">{method.split()[0]}</div>
            </div>
        </div>
        
        <div class="quality-section">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; margin-bottom: 16px;">
                <h3 style="font-size: clamp(1em, 2.5vw, 1.2em); font-weight: 800; color: #0f172a; margin: 0;">🎯 Analysis Quality</h3>
                <div class="quality-badge">{quality_label}</div>
            </div>
            <div class="quality-bar-container">
                <div class="quality-bar-fill"></div>
            </div>
        </div>
        
        <div class="section">
            <div class="section-header">
                <span style="font-size: clamp(1.4em, 3vw, 1.8em);">📏</span>
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
    
    if r_add_mm:
        html += f"""
        <div class="section">
            <div class="section-header">
                <span style="font-size: clamp(1.4em, 3vw, 1.8em);">📐</span>
                <h3 class="section-title">Physical Dimensions (mm)</h3>
            </div>
            <div class="params-grid">
                <div class="param-card">
                    <div class="param-label">Addendum Radius</div>
                    <div class="param-value">{r_add_mm:.2f}<span class="param-unit">mm</span></div>
                </div>
                <div class="param-card">
                    <div class="param-label">Pitch Radius</div>
                    <div class="param-value">{r_pitch_mm:.2f}<span class="param-unit">mm</span></div>
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


# Custom CSS
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.gradio-container {
    max-width: 100% !important;
    padding: clamp(8px, 2vw, 20px) !important;
}

.gr-button-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    font-weight: 700 !important;
    font-size: clamp(0.95rem, 2.5vw, 1.15rem) !important;
    padding: clamp(10px, 2vw, 14px) clamp(20px, 4vw, 32px) !important;
    border-radius: clamp(8px, 2vw, 12px) !important;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
}
"""

# Create interface
with gr.Blocks(
    title="Spur Gear Analysis Pro",
    theme=gr.themes.Soft(
        primary_hue="violet",
        secondary_hue="purple",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
    css=custom_css
) as demo:
    
    gr.HTML("""
        <div style='text-align: center; padding: clamp(24px, 5vw, 40px) clamp(12px, 3vw, 20px); 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: clamp(12px, 3vw, 20px); margin-bottom: clamp(16px, 3vw, 30px); 
                    box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);'>
            <h1 style='color: white; font-size: clamp(1.8em, 5vw, 3em); margin: 0; font-weight: 800; 
                       text-shadow: 0 2px 10px rgba(0,0,0,0.2);'>
                ⚙️ Spur Gear Analysis Pro
            </h1>
            <p style='color: rgba(255,255,255,0.95); font-size: clamp(0.9em, 2.5vw, 1.3em); 
                      margin: clamp(8px, 2vw, 12px) 0 0 0; font-weight: 500;'>
                AI-Powered Dimensional Analysis & Tooth Counting
            </p>
        </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=5, min_width=300):
            input_image = gr.Image(
                label="📤 Upload Gear Image",
                type="numpy",
                height=500
            )
            
            with gr.Row():
                with gr.Column(scale=2, min_width=120):
                    calibration_input = gr.Number(
                        label="🔧 Calibration (px/mm)",
                        value=1
                    )
                with gr.Column(scale=1, min_width=80):
                    min_teeth_input = gr.Slider(4, 60, 12, step=1, label="Min")
                with gr.Column(scale=1, min_width=80):
                    max_teeth_input = gr.Slider(20, 200, 120, step=1, label="Max")
            
            analyze_btn = gr.Button("🚀 Analyze Gear", variant="primary", size="lg")
            status_text = gr.Textbox(show_label=False, interactive=False, placeholder="Ready", container=False)
        
        with gr.Column(scale=7, min_width=400):
            with gr.Tabs():
                with gr.Tab("📊 Dashboard"):
                    results_output = gr.HTML()
                with gr.Tab("🎯 Detection"):
                    output_image = gr.Image(height=500)
                with gr.Tab("📏 Measurements"):
                    radii_image = gr.Image(height=500)
    
    analyze_btn.click(
        fn=process_gear_image,
        inputs=[input_image, calibration_input, min_teeth_input, max_teeth_input],
        outputs=[output_image, radii_image, results_output, status_text]
    )

if __name__ == "__main__":
    demo.launch()
