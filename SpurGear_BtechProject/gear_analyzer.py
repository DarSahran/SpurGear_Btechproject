#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gear Analysis System - Core Module
Analyzes gear images to extract measurements and tooth count
"""

import os
import json
import math
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for web deployment
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter, medfilt


# =========================
# 1) Configuration & CLI
# =========================

@dataclass
class Config:
    # Paths
    input_image: str = "gear_3.png"
    out_dir: str = "Output"

    # Preprocessing
    clahe_clip: float = 2.0
    clahe_grid: Tuple[int, int] = (8, 8)
    gaussian_kernel: Tuple[int, int] = (3, 3)

    # Thresholding
    min_thresh: int = 50
    max_thresh: int = 200
    otsu_fallback: bool = True

    # Morphology
    open_kernel: Tuple[int, int] = (3, 3)
    close_kernel: Tuple[int, int] = (5, 5)
    open_iter: int = 1
    close_iter: int = 2

    # Segmentation
    min_gear_area_ratio: float = 0.05
    min_circularity: float = 0.4

    # Radial sampling
    radial_samples: int = 2048
    add_quantile: float = 0.90
    ded_quantile: float = 0.10

    # Tooth counting
    expected_teeth_min: int = 12
    expected_teeth_max: int = 120
    peak_prominence_factor: float = 0.4
    peak_window_clean: int = 15
    peak_window_moderate: int = 25
    peak_window_noisy: int = 35

    # FFT fallback
    fft_enabled: bool = True

    # Visualization
    show_plots: bool = False
    save_plots: bool = True
    fig_dpi: int = 120

    # Output
    save_json: bool = True
    save_overlay: bool = True
    save_csv_profile: bool = True

    # Calibration
    pixels_per_mm: Optional[float] = None


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Gear analysis pipeline")
    p.add_argument("--input", "-i", type=str, default="gear_3.png", help="Path to input image")
    p.add_argument("--out", "-o", type=str, default="Output", help="Output directory")
    p.add_argument("--show", action="store_true", help="Show plots interactively")
    p.add_argument("--save-plots", action="store_true", help="Save plots to output")
    p.add_argument("--ppm", type=float, default=None, help="Pixels per millimeter (calibration)")

    args = p.parse_args()
    cfg = Config(
        input_image=args.input,
        out_dir=args.out,
        show_plots=args.show,
        save_plots=(args.save_plots or True),
        pixels_per_mm=args.ppm,
    )
    return cfg


# ================================
# 2) Utilities: IO & Visualization
# ================================

def ensure_out_dir(path: str) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_plot(fig: plt.Figure, out_dir: Path, name: str, dpi: int = 120):
    out_path = out_dir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=dpi)
    plt.close(fig)


def show_or_save(fig: plt.Figure, cfg: Config, out_dir: Path, name: str):
    if cfg.save_plots:
        save_plot(fig, out_dir, name, dpi=cfg.fig_dpi)
    if cfg.show_plots:
        fig.show()


# ================================
# 3) Image Loading & Validation
# ================================

def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


# ====================================
# 4) Preprocessing & Auto Thresholding
# ====================================

def auto_threshold(gray_img: np.ndarray, cfg: Config) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
    h, w = gray_img.shape
    cx, cy = w // 2, h // 2

    center_size = min(40, h // 8, w // 8)
    border_size = min(30, h // 10, w // 10)

    center_region = gray_img[cy - center_size:cy + center_size, cx - center_size:cx + center_size]

    border_samples = [
        gray_img[:border_size, :].ravel(),
        gray_img[-border_size:, :].ravel(),
        gray_img[:, :border_size].ravel(),
        gray_img[:, -border_size:].ravel(),
    ]
    corner_size = max(1, border_size // 2)
    border_samples.extend([
        gray_img[:corner_size, :corner_size].ravel(),
        gray_img[:corner_size, -corner_size:].ravel(),
        gray_img[-corner_size:, :corner_size].ravel(),
        gray_img[-corner_size:, -corner_size:].ravel(),
    ])
    all_border = np.concatenate(border_samples)

    border_clean = all_border[
        (all_border >= np.percentile(all_border, 5)) &
        (all_border <= np.percentile(all_border, 95))
    ]
    center_mean = float(np.mean(center_region))
    center_std = float(np.std(center_region))
    border_mean = float(np.mean(border_clean))
    border_std = float(np.std(border_clean))

    invert = center_mean < border_mean
    intensity_gap = abs(border_mean - center_mean)

    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    manual_threshold = max(cfg.min_thresh, min(cfg.max_thresh, int((center_mean + border_mean) / 2))) \
        if intensity_gap > 20 else None

    bin_manual = None
    if manual_threshold is not None:
        _, bin_manual = cv2.threshold(gray_img, manual_threshold, 255, flag)

    if cfg.otsu_fallback:
        _, bin_otsu = cv2.threshold(gray_img, 0, 255, flag + cv2.THRESH_OTSU)
    else:
        bin_otsu = None

    if bin_manual is not None and intensity_gap > 30:
        final_binary = bin_manual
        method_used = f"Manual (thresh={manual_threshold})"
    elif bin_otsu is not None:
        final_binary = bin_otsu
        method_used = "Otsu"
    else:
        t = manual_threshold if manual_threshold is not None else (cfg.min_thresh + cfg.max_thresh) // 2
        _, final_binary = cv2.threshold(gray_img, t, 255, flag)
        method_used = f"Fallback (thresh={t})"

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, cfg.open_kernel)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, cfg.close_kernel)
    final_binary = cv2.morphologyEx(final_binary, cv2.MORPH_OPEN, k_open, iterations=cfg.open_iter)
    final_binary = cv2.morphologyEx(final_binary, cv2.MORPH_CLOSE, k_close, iterations=cfg.close_iter)

    debug = dict(center_mean=center_mean, center_std=center_std,
                 border_mean=border_mean, border_std=border_std,
                 intensity_gap=intensity_gap, invert=invert, method_used=method_used)
    return final_binary, invert, debug


def preprocess(img: np.ndarray, cfg: Config, out_dir: Path) -> Dict[str, Any]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=cfg.clahe_grid)
    gray_eq = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray_eq, cfg.gaussian_kernel, 0)
    binary, inverted, th_debug = auto_threshold(blurred, cfg)

    fig = plt.figure(figsize=(10, 3))
    plt.subplot(1, 3, 1); plt.title("Gray"); plt.axis("off"); plt.imshow(gray, cmap="gray")
    plt.subplot(1, 3, 2); plt.title("CLAHE"); plt.axis("off"); plt.imshow(gray_eq, cmap="gray")
    plt.subplot(1, 3, 3); plt.title(f"Binary (invert={inverted})"); plt.axis("off"); plt.imshow(binary, cmap="gray")
    show_or_save(fig, cfg, out_dir, "01_preprocess")

    return dict(gray=gray, gray_eq=gray_eq, binary=binary, inverted=inverted, debug=th_debug)


# ==========================================
# 5) Gear Segmentation, Contour & Validation
# ==========================================

def select_gear_mask(binary: np.ndarray, cfg: Config, out_dir: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    mask = binary.copy()
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return mask, None

    areas = [cv2.contourArea(c) for c in cnts]
    largest = cnts[int(np.argmax(areas))]

    canvas = np.zeros_like(mask)
    cv2.drawContours(canvas, [largest], -1, 255, -1)
    mask = canvas

    h, w = mask.shape
    area = float(cv2.contourArea(largest))
    area_ratio = area / float(h * w)
    perim = float(cv2.arcLength(largest, True))
    circularity = (4 * math.pi * area) / (perim * perim + 1e-9)

    if area_ratio < cfg.min_gear_area_ratio or circularity < cfg.min_circularity:
        pass

    cv2.imwrite(str(out_dir / "gear_mask.png"), mask)

    fig = plt.figure(figsize=(4, 4))
    plt.title("Mask (Largest Component)"); plt.axis("off"); plt.imshow(mask, cmap="gray")
    show_or_save(fig, cfg, out_dir, "02_mask")
    return mask, largest


# ====================================
# 6) Center & Radii
# ====================================

def robust_radius(contour: np.ndarray, center: Tuple[float, float], top_k: int = 200, mode: str = "max") -> float:
    cx, cy = center
    pts = contour[:, 0, :]
    d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    vals = np.sort(d)
    k = min(top_k, len(vals))
    if k <= 0:
        return float(np.mean(d)) if d.size > 0 else 0.0
    if mode == "max":
        return float(np.mean(vals[-k:]))
    return float(np.mean(vals[:k]))


def center_from_moments(mask: np.ndarray) -> Tuple[float, float]:
    M = cv2.moments(mask, binaryImage=True)
    if M["m00"] == 0:
        ys, xs = np.where(mask > 0)
        return float(np.mean(xs)), float(np.mean(ys))
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    return float(cx), float(cy)


def estimate_radii(mask: np.ndarray, outer_contour: np.ndarray, cfg: Config, out_dir: Path, img: np.ndarray) -> Dict[str, Any]:
    cx, cy = center_from_moments(mask)
    r_add = robust_radius(outer_contour, (cx, cy), mode="max")
    r_ded = robust_radius(outer_contour, (cx, cy), mode="min")
    r_pitch = 0.5 * (r_add + r_ded)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    r_hole = None
    if hierarchy is not None:
        hierarchy = hierarchy[0]
        outer_idx = int(np.argmax([cv2.contourArea(c) for c in contours]))
        child_idxs = [i for i, h in enumerate(hierarchy) if h[3] == outer_idx]
        if child_idxs:
            inner_idx = max(child_idxs, key=lambda i: cv2.contourArea(contours[i]))
            r_hole = robust_radius(contours[inner_idx], (cx, cy), mode="max")

    diag = img.copy()
    for r in [r_ded, r_pitch, r_add]:
        cv2.circle(diag, (int(cx), int(cy)), int(max(1, r)), (0, 255, 0), 2)
    if r_hole:
        cv2.circle(diag, (int(cx), int(cy)), int(max(1, r_hole)), (255, 0, 0), 2)
    cv2.circle(diag, (int(cx), int(cy)), 4, (0, 0, 255), -1)
    cv2.imwrite(str(out_dir / "gear_radii_overlay.png"), diag)

    fig = plt.figure(figsize=(5, 5))
    plt.title("Center & Radii"); plt.axis("off"); plt.imshow(cv2.cvtColor(diag, cv2.COLOR_BGR2RGB))
    show_or_save(fig, cfg, out_dir, "03_radii")

    return dict(cx=cx, cy=cy, r_add=r_add, r_ded=r_ded, r_pitch=r_pitch, r_hole=r_hole)


# ======================================
# 7) Radial Profile & Tooth Counting
# ======================================

def radial_profile_from_contour(outer_contour: np.ndarray, cx: float, cy: float, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    pts = outer_contour[:, 0, :]
    theta = np.mod(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx), 2 * np.pi)
    r = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    idx = np.argsort(theta)
    theta_sorted, r_sorted = theta[idx], r[idx]
    tgrid = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    theta2 = np.concatenate([theta_sorted, theta_sorted + 2 * np.pi])
    r2 = np.concatenate([r_sorted, r_sorted])
    rgrid = np.interp(tgrid, theta2, r2)
    return tgrid, rgrid


def tooth_count_peaks(tgrid: np.ndarray, rgrid: np.ndarray, cfg: Config, out_dir: Path, label: str, do_plot: bool) -> Tuple[int, Dict[str, Any]]:
    r_centered = rgrid - np.mean(rgrid)
    r_median = medfilt(r_centered, kernel_size=15) + np.mean(rgrid)

    signal_std = float(np.std(r_median))
    signal_range = float(np.max(r_median) - np.min(r_median))
    noise_level = signal_std / (signal_range + 1e-9)

    if noise_level < 0.05:
        window_length = cfg.peak_window_clean
    elif noise_level < 0.10:
        window_length = cfg.peak_window_moderate
    else:
        window_length = cfg.peak_window_noisy
    if window_length % 2 == 0:
        window_length += 1

    r_smooth = savgol_filter(r_median, window_length=window_length, polyorder=3)

    N = len(tgrid)
    min_dist_cons = max(1, N // max(cfg.expected_teeth_max, 3))
    min_dist_lib = max(1, N // max(cfg.expected_teeth_min, 3))

    prom = max(1e-6, cfg.peak_prominence_factor * float(np.std(r_smooth - np.mean(r_smooth))))

    peaks_cons, props_cons = find_peaks(r_smooth, distance=min_dist_cons, prominence=prom, width=2)
    peaks_lib, props_lib = find_peaks(r_smooth, distance=min_dist_lib, prominence=0.6 * prom, width=1)

    def validate_regular(peaks: np.ndarray, tol: float = 0.3) -> Tuple[np.ndarray, bool]:
        if peaks is None or len(peaks) < 4:
            return peaks, False
        angles = tgrid[peaks]
        ang_sorted = np.sort(angles)
        diffs = np.diff(ang_sorted)
        diffs = np.append(diffs, 2 * np.pi - (ang_sorted[-1] - ang_sorted[0]))
        expected = 2 * np.pi / max(1, len(peaks))
        dev = np.abs(diffs - expected) / (expected + 1e-9)
        regular_idx = [i for i in range(len(peaks)) if (i < len(dev) and dev[i] < tol)]
        is_regular = (len(regular_idx) >= 0.8 * len(peaks))
        return peaks if is_regular else peaks, is_regular

    cons_valid, cons_regular = validate_regular(peaks_cons)
    lib_valid, lib_regular = validate_regular(peaks_lib)

    cons_count = len(cons_valid)
    lib_count = len(lib_valid)

    if cons_regular and cfg.expected_teeth_min <= cons_count <= cfg.expected_teeth_max:
        final_peaks = cons_valid
        method = "conservative_regular"
    elif lib_regular and cfg.expected_teeth_min <= lib_count <= cfg.expected_teeth_max:
        final_peaks = lib_valid
        method = "liberal_regular"
    elif cons_regular:
        final_peaks = cons_valid
        method = "conservative_extended"
    elif lib_regular:
        final_peaks = lib_valid
        method = "liberal_extended"
    else:
        target = math.sqrt(cfg.expected_teeth_min * cfg.expected_teeth_max)
        final_peaks = peaks_cons if abs(cons_count - target) < abs(lib_count - target) else peaks_lib
        method = "fallback_closest"

    count = int(len(final_peaks))

    if do_plot:
        fig = plt.figure(figsize=(14, 6))
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(tgrid, rgrid, alpha=0.4, label="Raw")
        ax1.plot(tgrid, r_median, alpha=0.7, label="Median")
        ax1.plot(tgrid, r_smooth, label="Smoothed")
        ax1.set_title("Signal processing")
        ax1.legend(); ax1.grid(alpha=0.3)

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(tgrid, r_smooth, label="Smoothed")
        ax2.plot(tgrid[peaks_cons], r_smooth[peaks_cons], "bo", label=f"Conservative {len(peaks_cons)}")
        ax2.plot(tgrid[peaks_lib], r_smooth[peaks_lib], "go", label=f"Liberal {len(peaks_lib)}")
        ax2.plot(tgrid[final_peaks], r_smooth[final_peaks], "ro", label=f"Final {len(final_peaks)}")
        ax2.set_title("Peak detection"); ax2.legend(); ax2.grid(alpha=0.3)

        ax3 = fig.add_subplot(2, 2, 3, projection="polar")
        ax3.plot(tgrid, r_smooth)
        if len(final_peaks) > 0:
            ax3.plot(tgrid[final_peaks], r_smooth[final_peaks], "ro")
        ax3.set_title(f"Polar view - {count} teeth")

        ax4 = fig.add_subplot(2, 2, 4)
        if len(final_peaks) > 1:
            peak_angles = np.sort(tgrid[final_peaks]) * 180 / np.pi
            diffs = np.diff(peak_angles)
            diffs = np.append(diffs, 360 - (peak_angles[-1] - peak_angles[0]))
            expected_deg = 360 / max(1, len(final_peaks))
            ax4.bar(range(len(diffs)), diffs, alpha=0.7)
            ax4.axhline(expected_deg, color="r", linestyle="--", label=f"Expected {expected_deg:.1f}°")
            ax4.legend()
        ax4.set_title("Angular spacing")

        show_or_save(fig, cfg, out_dir, f"04_peaks_{label}")

    info = dict(method=method, cons_count=cons_count, lib_count=lib_count,
                noise_level=float(noise_level), window_length=int(window_length),
                prominence=float(prom))
    return count, info


def tooth_count_fft(tgrid: np.ndarray, rgrid: np.ndarray, cfg: Config, out_dir: Path, label: str, do_plot: bool) -> Tuple[Optional[int], Dict[str, Any]]:
    kernel = np.ones(51) / 51.0
    r_det = rgrid - np.convolve(rgrid, kernel, mode="same")

    F = np.fft.rfft(r_det)
    freqs = np.fft.rfftfreq(len(tgrid), d=(2 * np.pi / len(tgrid)))
    power = np.abs(F)

    fmin = max(3, cfg.expected_teeth_min // 2)
    fmax = min(len(freqs) - 1, cfg.expected_teeth_max * 2)
    band = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(band):
        return None, dict(note="no_band")

    idx = np.argmax(power[band])
    peak_freq = float(freqs[band][idx])
    teeth_candidate = int(round(peak_freq))

    if do_plot:
        fig = plt.figure(figsize=(10, 4))
        plt.plot(freqs[band], power[band])
        plt.title(f"FFT magnitude (peak ~ {teeth_candidate})")
        plt.xlabel("Frequency (cycles/rev)"); plt.ylabel("Magnitude")
        show_or_save(fig, cfg, out_dir, f"05_fft_{label}")

    return teeth_candidate, dict(peak_freq=peak_freq)


def consensus_tooth_count(tgrid: np.ndarray, rgrid: np.ndarray, cfg: Config, out_dir: Path) -> Tuple[Optional[int], Dict[str, Any]]:
    count_peaks, info_peaks = tooth_count_peaks(tgrid, rgrid, cfg, out_dir, "primary", cfg.save_plots or cfg.show_plots)

    candidates = [count_peaks]
    meta = {"primary": info_peaks}

    if cfg.fft_enabled:
        fft_count, info_fft = tooth_count_fft(tgrid, rgrid, cfg, out_dir, "fft", cfg.save_plots or cfg.show_plots)
        meta["fft"] = info_fft
        if fft_count is not None:
            candidates.append(fft_count)

    valid = [c for c in candidates if c is not None and c > 0]

    if not valid:
        return None, meta

    if len(valid) == 1:
        return int(valid[0]), meta

    a, b = int(valid[0]), int(valid[1])
    if abs(a - b) <= 1:
        return int(round(0.5 * (a + b))), meta

    return int(a), meta


# ==========================
# 8) Overlay & CSV Exports
# ==========================

def overlay_teeth_markers(img: np.ndarray, cx: float, cy: float, tgrid: np.ndarray, peaks_count: int, r_add: float, out_dir: Path, cfg: Config):
    if peaks_count is None or peaks_count <= 0:
        return
    angles = np.linspace(0, 2 * np.pi, peaks_count, endpoint=False)
    for a in angles:
        x = int(cx + r_add * np.cos(a))
        y = int(cy + r_add * np.sin(a))
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
    cv2.circle(img, (int(cx), int(cy)), 4, (255, 0, 0), -1)
    cv2.imwrite(str(out_dir / "gear_teeth_overlay.png"), img)


def export_profile_csv(out_dir: Path, tgrid: np.ndarray, rgrid: np.ndarray):
    arr = np.vstack([tgrid, rgrid]).T
    header = "angle_rad, radius_px"
    np.savetxt(str(out_dir / "radial_profile.csv"), arr, delimiter=",", header=header, comments="")


# ==========================
# 9) Orchestrator (Main)
# ==========================

def analyze_gear(cfg: Config) -> Dict[str, Any]:
    out_dir = ensure_out_dir(cfg.out_dir)
    img = load_image(cfg.input_image)

    pre = preprocess(img, cfg, out_dir)
    gray = pre["gray"]
    binary = pre["binary"]

    mask, outer = select_gear_mask(binary, cfg, out_dir)
    if outer is None:
        raise RuntimeError("Could not find a valid gear contour")

    radii = estimate_radii(mask, outer, cfg, out_dir, img.copy())
    cx, cy = radii["cx"], radii["cy"]
    r_add, r_ded, r_pitch, r_hole = radii["r_add"], radii["r_ded"], radii["r_pitch"], radii["r_hole"]

    tgrid, rgrid = radial_profile_from_contour(outer, cx, cy, cfg.radial_samples)

    teeth_count, tooth_meta = consensus_tooth_count(tgrid, rgrid, cfg, out_dir)

    module_px = (2.0 * r_pitch / teeth_count) if (teeth_count is not None and teeth_count > 0) else None
    circular_pitch_px = (math.pi * module_px) if module_px is not None else None

    to_mm = (lambda v: (v / cfg.pixels_per_mm) if (v is not None and cfg.pixels_per_mm) else None)
    results = dict(
        center_px=[cx, cy],
        r_add_px=r_add,
        r_ded_px=r_ded,
        r_pitch_px=r_pitch,
        r_hole_px=None if r_hole is None else float(r_hole),
        teeth_estimate=None if teeth_count is None else int(teeth_count),
        module_px=None if module_px is None else float(module_px),
        circular_pitch_px=None if circular_pitch_px is None else float(circular_pitch_px),
        debug=dict(
            threshold=pre["debug"],
            tooth_meta=tooth_meta
        )
    )

    results_mm = dict(
        r_add_mm=to_mm(r_add),
        r_ded_mm=to_mm(r_ded),
        r_pitch_mm=to_mm(r_pitch),
        r_hole_mm=to_mm(r_hole) if r_hole is not None else None,
        module_mm=to_mm(module_px) if module_px is not None else None,
        circular_pitch_mm=to_mm(circular_pitch_px) if circular_pitch_px is not None else None
    )
    results.update(results_mm)

    if cfg.save_overlay:
        overlay_img = img.copy()
        overlay_teeth_markers(overlay_img, cx, cy, tgrid, teeth_count if teeth_count else 0, r_add, out_dir, cfg)

    if cfg.save_csv_profile:
        export_profile_csv(out_dir, tgrid, rgrid)

    if cfg.save_json:
        with open(out_dir / "gear_results.json", "w") as f:
            json.dump(results, f, indent=2)

    print("\n=== Gear Analysis Summary ===")
    print(json.dumps({k: v for k, v in results.items() if k != "debug"}, indent=2))
    print(f"\nArtifacts saved to: {out_dir.resolve()}")

    return results


def main():
    cfg = parse_args()
    analyze_gear(cfg)


if __name__ == "__main__":
    main()
