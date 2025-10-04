---
title: Spur Gear Dimensional Analysis
emoji: ⚙️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
---

# ⚙️ Spur Gear Dimensional Analysis

## Overview
AI-powered computer vision system for automated spur gear analysis, tooth counting, and dimensional measurement.

## Features
- **Automated Tooth Counting**: Accurate detection using multiple algorithms
- **Dimensional Measurements**: Addendum, dedendum, pitch radius, module
- **Calibration Support**: Convert pixel measurements to millimeters
- **Real-time Processing**: 3-8 second analysis time
- **Interactive Dashboard**: Beautiful, responsive web interface

## How to Use
1. Upload a clear image of your spur gear (white/light background recommended)
2. Optional: Enter calibration value (pixels per mm) for physical measurements
3. Adjust min/max tooth range if needed
4. Click "Analyze Gear" and view results

## Technology Stack
- **Computer Vision**: OpenCV
- **Signal Processing**: SciPy
- **Web Interface**: Gradio
- **Analysis**: Peak detection, FFT, morphological operations

## Academic Project
Developed as a B.Tech project for automated gear inspection and quality control.

## Citation
If you use this tool in research, please cite: [Your Name], "Spur Gear Dimensional Analysis using Computer Vision", 2025.
