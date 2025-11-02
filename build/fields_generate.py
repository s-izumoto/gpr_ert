# -*- coding: utf-8 -*-
"""
build_ert_ml_dataset_nosplit.py
--------------------------------
Single-file ERT dataset generator **without** any train/val/test split.
It creates one NPZ containing all samples. Splitting is expected to be
handled later (e.g., inside your training code).

Outputs (under --out):
  - dataset.npz
      * X_log10: (N_total, NZ, NX) float32
      * y:       (N_total,) int64  (class ids)
      * cases:   (N_total,) object (case names)
  - label_map.json   (case -> int id)
  - meta.json        (shapes, mins/maxes, counts)
  - previews/        (optional PNGs; aspect='equal' to match grid)

Example:
  python build_ert_ml_dataset_nosplit.py --out ./ert_ds --nz 100 --nx 200 \
      --per-class 200 --save-png --previews-per-class 4
"""
from __future__ import annotations

# -------------------------
# Standard libs & 3rd party
# -------------------------
import argparse, os, json, random
from pathlib import Path
from typing import List, Dict, Tuple, Sequence

import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# Section A — Generators (ported from ert_gen_smooth.py)
# =====================================================

# Concentrate structure in the upper SHALLOW_FRAC of the model (0<frac≤1).
SHALLOW_FRAC = 0.55     # e.g., top 55% has most structure
DEEP_SMOOTH_THICK = (8.0, 16.0)  # logistic thickness (cells) for deep homogenization

# Optional SciPy smoothing (faster/nicer). Fallback included.
try:
    from scipy.ndimage import gaussian_filter  # type: ignore
except Exception:
    def gaussian_filter(arr, sigma=(1.0,1.0), mode='reflect'):
        import numpy as _np
        def kernel(s):
            s = float(max(1e-6, s))
            k = _np.arange(-2, 3)
            w = _np.exp(-0.5*(k/s)**2)
            w /= w.sum()
            return w
        kz = kernel(sigma[0] if _np.iterable(sigma) else sigma)
        kx = kernel(sigma[1] if _np.iterable(sigma) else sigma)
        def pad_reflect(x, r):
            if r == 0: return x
            left  = x[1:r+1][::-1] if x.shape[0] > 1 else _np.repeat(x, r, axis=0)
            right = x[-r-1:-1][::-1] if x.shape[0] > 1 else _np.repeat(x, r, axis=0)
            return _np.concatenate([left, x, right], axis=0)
        rz = len(kz)//2
        tmp = _np.zeros_like(arr, dtype=float)
        padz = pad_reflect(arr, rz)
        for i in range(arr.shape[0]):
            tmp[i] = _np.tensordot(kz, padz[i:i+2*rz+1], axes=([0],[0]))
        rx = len(kx)//2
        out = _np.zeros_like(arr, dtype=float)
        padx = pad_reflect(tmp.swapaxes(0,1), rx)
        for j in range(arr.shape[1]):
            out[:, j] = _np.tensordot(kx, padx[j:j+2*rx+1], axes=([0],[0]))
        return out

# -----------------------------
# Facies priors (log10 space)
# μ, σ in dex; lo/hi are log10 clamps; global clamp to [0.2, 1e4] Ω·m at end
# -----------------------------
FACIES = {
    "seawater":          (-0.70, 0.15, np.log10(0.15),   np.log10(0.6)),
    "saline_sat":        ( 0.00, 0.20, np.log10(0.5),    np.log10(3.0)),
    "brackish_sat":      ( 0.70, 0.25, np.log10(2.0),    np.log10(15.0)),
    "fresh_sat":         ( 1.48, 0.25, np.log10(10.0),   np.log10(80.0)),
    "moist_soil":        ( 1.78, 0.20, np.log10(20.0),   np.log10(150.0)),
    "dry_soil":          ( 2.00, 0.20, np.log10(50.0),   np.log10(200.0)),
    "moist_limestone":   ( 2.30, 0.20, np.log10(80.0),   np.log10(500.0)),
    "limestone":         ( 2.48, 0.20, np.log10(150.0),  np.log10(600.0)),
    "moist_sandstone":   ( 2.08, 0.25, np.log10(50.0),   np.log10(350.0)),
    "sandstone":         ( 3.00, 0.25, np.log10(400.0),  np.log10(2500.0)),
    "coarse_sand":       ( 3.48, 0.25, np.log10(1000.0), np.log10(6000.0)),
    "bedrock":           ( 4.00, 0.25, np.log10(5000.0), np.log10(20000.0)),
}

def draw_log10_rho(facies_name: str) -> float:
    mu, sig, lo, hi = FACIES[facies_name]
    x = np.random.normal(mu, sig)
    return float(np.clip(x, lo, hi))

def _sharpen_gate(A: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """
    Sharpen a [0,1] gate A by scaling its logit with gamma >= 1.
    Supports broadcasting gamma over (NZ, NX) or (1, NX).
    """
    eps = 1e-6
    A = np.clip(A, eps, 1 - eps)
    logit = np.log(A) - np.log(1 - A)
    return 1.0 / (1.0 + np.exp(-gamma * logit))

# -----------------------------
# Wavy, smoothly blended layers
# -----------------------------
def make_wavy_interfaces(
    NX:int, NZ:int, depths:List[float],
    amp_frac=(0.03,0.10), wavelen_frac=(0.25,0.9),
    min_sep=3, zmin:int=1, zmax:int|None=None
) -> np.ndarray:
    """Return per-column interface depths D (L-1,NX) w/ waviness, monotonic, clipped within [zmin, zmax]."""
    if zmax is None: zmax = NZ-2
    x = np.arange(NX, dtype=np.float32)
    D = []
    for d0 in depths:
        a  = np.random.uniform(*amp_frac) * NZ
        wl = np.random.uniform(*wavelen_frac) * NX
        f  = 2*np.pi/wl
        phase = np.random.uniform(0, 2*np.pi)
        D.append(d0 + a * np.sin(f*x + phase))
    D = np.stack(D, axis=0).astype(np.float32)

    # Clip shallow window
    D = np.clip(D, max(1, zmin), min(zmax, NZ-2))

    # Enforce monotonic non-crossing per column
    for j in range(NX):
        col = np.sort(D[:, j])
        for i in range(1, col.shape[0]):
            col[i] = max(col[i], col[i-1] + min_sep)
        D[:, j] = np.clip(col, max(1, zmin), min(zmax, NZ-2))
    return D

def smooth_layer_weights(
    D: np.ndarray, NZ: int,
    t_range=None,
    t_soft_range=(7.0, 15.0),
    t_crisp_range=(0.8, 2.2),     # thinner → crisper
    crisp_fraction=0.55,          # ~55% of each interface is crisp on average
    gate_gamma_range=(1.0, 2.0),  # extra sharpening on crisp spans
    t_smooth_x=20.0
) -> np.ndarray:
    Lm1, NX = D.shape
    L = Lm1 + 1
    z = np.arange(NZ, dtype=np.float32)[:, None]

    # Map legacy t_range -> t_soft_range
    if t_range is not None:
        if isinstance(t_range, (tuple, list)) and len(t_range) == 2:
            t_soft_range = (float(t_range[0]), float(t_range[1]))
        else:
            t_soft_range = (float(t_range), float(t_range))

    # Per-interface thickness & crispness masks
    T = np.zeros((Lm1, NX), dtype=np.float32)
    crisp_masks = []  # store for sharpening
    x01 = np.linspace(0, 1, NX, dtype=np.float32)
    for i in range(Lm1):
        t_soft = np.random.uniform(*t_soft_range, size=(NX,)).astype(np.float32)
        t_crsp = np.random.uniform(*t_crisp_range, size=(NX,)).astype(np.float32)
        if t_smooth_x > 0:
            t_soft = gaussian_filter(t_soft, sigma=t_smooth_x, mode='reflect')
            t_crsp = gaussian_filter(t_crsp, sigma=t_smooth_x, mode='reflect')

        # RBF blobs → crispness mask c_i(x) in [0,1]
        K = np.random.choice([2,3,4], p=[0.5,0.35,0.15])
        centers = np.random.uniform(0.1, 0.9, size=K)
        widths  = 1.0 / np.random.uniform(6, 15, size=K)
        c = np.zeros_like(x01)
        for cx, w in zip(centers, widths):
            c += np.exp(-(x01 - cx)**2 / (2*w*w))
        c = (c - c.min()) / (c.max() - c.min() + 1e-8)
        c *= (crisp_fraction / (np.mean(c) + 1e-8))
        c = np.clip(c, 0.0, 1.0)
        crisp_masks.append(c.astype(np.float32))

        # mix crisp vs soft thickness
        T[i, :] = c * t_crsp + (1.0 - c) * t_soft

    # logistic gates
    A = 1.0 / (1.0 + np.exp(-(D[None, ...] - z[:, None, :]) / T[None, ...]))  # (NZ, L-1, NX)

    # extra sharpening on crisp spans
    for i in range(Lm1):
        c = crisp_masks[i]  # (NX,)
        gamma_boost = np.random.uniform(*gate_gamma_range)
        Gamma = 1.0 + c[None, :] * (gamma_boost - 1.0)  # shape (1,NX)
        A[:, i, :] = _sharpen_gate(A[:, i, :], Gamma)

    # gates -> weights
    W = []
    prod = np.ones((NZ, NX), dtype=np.float32)
    for i in range(Lm1):
        Wi = prod * A[:, i, :]
        W.append(Wi)
        prod *= (1.0 - A[:, i, :])
    W.append(prod)
    W = np.stack(W, axis=0)
    S = np.sum(W, axis=0, keepdims=False)
    return (W / (S[None, ...] + 1e-8)).astype(np.float32)

def layered_background_logR(
    NZ:int, NX:int, L:int,
    return_weights:bool=False,
    shallow_frac:float=SHALLOW_FRAC,
    amp_frac=(0.03, 0.10),
    wavelen_frac=(0.25, 0.90)
):
    """Layered background where all interfaces are in the top shallow_frac of the depth."""
    zmax = int(np.clip(shallow_frac, 0.1, 1.0) * NZ)
    zmin = int(0.08 * NZ)

    cuts = [] if L <= 1 else np.sort(
        np.random.randint(max(1, zmin), max(zmin+1, zmax), size=L-1)
    ).tolist()

    D = make_wavy_interfaces(
        NX, NZ, cuts,
        amp_frac=amp_frac,               # << use passed waviness
        wavelen_frac=wavelen_frac,       # << use passed waviness
        min_sep=3, zmin=zmin, zmax=zmax
    )
    W = smooth_layer_weights(D, NZ, t_range=(6.0, 14.0), t_smooth_x=20.0)

    facies = []
    for k in range(L):
        if k == L-1 and np.random.rand() < 0.5:
            facies.append("bedrock")
        else:
            facies.append(np.random.choice(
                ["dry_soil","moist_soil","fresh_sat","sandstone","limestone","coarse_sand"],
                p=[0.15,0.20,0.15,0.20,0.20,0.10]
            ))
    vals = np.array([draw_log10_rho(f) for f in facies], dtype=np.float32)[:, None, None]
    logR = np.sum(W * vals, axis=0).astype(np.float32)
    return (logR, W) if return_weights else logR

# -----------------------------
# Smooth anomaly masks (signed-distance ramp)
# -----------------------------
def smooth_ellipse_mask(NZ, NX, cx, cz, ax, az, theta_rad, edge_cells=8.0):
    """Smooth [0,1] mask from ellipse signed-distance; edge_cells controls transition width."""
    X, Z = np.meshgrid(np.arange(NX, dtype=np.float32),
                       np.arange(NZ, dtype=np.float32))
    ct, st = np.cos(theta_rad), np.sin(theta_rad)
    x0, z0 = X - cx, Z - cz
    xr =  ct*x0 + st*z0
    zr = -st*x0 + ct*z0
    r  = np.sqrt((xr/ax)**2 + (zr/az)**2)
    sd = (1.0 - r) * min(ax, az)                       # signed distance in "cells"
    m  = 1.0 / (1.0 + np.exp(-sd / max(1.0, edge_cells)))
    return m.astype(np.float32)

# -----------------------------
# Structure-dominant heterogeneity (tiny, masked, capped)
# -----------------------------
def add_heterogeneity(logR_base: np.ndarray,
                      W: np.ndarray|None=None,
                      anomaly_masks: List[np.ndarray]|None=None,
                      edge_field: np.ndarray|None=None,
                      sigma_cap_dex: float = 0.02,
                      rel_cap: float = 0.08,
                      corr=(6, 12),
                      depth_decay_cells: float|None = 120) -> np.ndarray:
    NZ, NX = logR_base.shape
    noise = np.random.normal(0, 1, size=(NZ, NX)).astype(np.float32)
    n = gaussian_filter(noise, sigma=[corr[0], corr[1]], mode='reflect')
    n = n / (np.std(n) + 1e-8)

    dyn = float(np.percentile(logR_base, 95) - np.percentile(logR_base, 5))
    amp = min(sigma_cap_dex, rel_cap * max(dyn, 1e-6))

    A = np.ones((NZ, NX), dtype=np.float32)
    if W is not None:
        gz = np.sum(np.abs(np.gradient(W, axis=1)), axis=0)
        gx = np.sum(np.abs(np.gradient(W, axis=2)), axis=0)
        g  = gx + gz
        g  = g / (np.max(g) + 1e-8)
        A *= np.exp(-3.0 * g)            # suppress near interfaces

    if anomaly_masks:
        M = np.clip(np.sum(np.stack(anomaly_masks, axis=0), axis=0), 0.0, 1.0)
        A *= (1.0 - 0.7*M)               # keep anomalies clean

    if edge_field is not None:
        ex = np.abs(np.gradient(edge_field, axis=1))
        ez = np.abs(np.gradient(edge_field, axis=0))
        e  = (ex + ez); e /= (np.max(e) + 1e-8)
        A *= np.exp(-2.0 * e)            # suppress around that transition

    if depth_decay_cells:
        z = np.arange(NZ, dtype=np.float32)[:, None]
        A *= np.exp(-z / float(depth_decay_cells))

    hetero = amp * n * A
    return (logR_base + hetero).astype(np.float32)

def clamp_global(logR: np.ndarray) -> np.ndarray:
    return np.clip(logR, np.log10(0.2), np.log10(1e4)).astype(np.float32)

def homogenize_deep(
    logR: np.ndarray, z0: int, facies_choices=None, t_cells: float|None=None
) -> np.ndarray:
    """
    Blend the model to a nearly homogeneous deep value below depth z0.
    Uses a logistic ramp with thickness t_cells for smoothness.
    """
    NZ, NX = logR.shape
    if facies_choices is None:
        facies_choices = ["sandstone","limestone","moist_sandstone","moist_limestone","bedrock"]
    deep_val = draw_log10_rho(np.random.choice(facies_choices))

    if t_cells is None:
        t_cells = np.random.uniform(*DEEP_SMOOTH_THICK)

    Z = np.arange(NZ, dtype=np.float32)[:, None]
    wdeep = 1.0 / (1.0 + np.exp(-(Z - float(z0)) / float(t_cells)))  # 0 above, 1 below
    wdeep = gaussian_filter(wdeep, sigma=[1.0, 2.0], mode='reflect')
    out = (1.0 - wdeep)*logR + wdeep*deep_val
    return out.astype(np.float32)

# -----------------------------
# Generators (all smooth backgrounds; gentle heterogeneity)
# -----------------------------
def gen_tracer(NZ=100, NX=200):
    logR, W = layered_background_logR(NZ, NX, L=2, return_weights=True)

    masks = []
    A = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
    z_shallow_max = int(SHALLOW_FRAC * NZ)
    for _ in range(A):
        cx = np.random.uniform(0.15*NX, 0.85*NX)
        cz = np.random.uniform(0.10*NZ, 0.90*z_shallow_max)

        ax = np.random.uniform(0.03*NX, 0.10*NX)
        az = np.random.uniform(0.05*NZ, 0.15*NZ)

        theta = np.deg2rad(np.random.uniform(-20, 20))

        edge_cells = (np.random.uniform(1.0, 3.0) if np.random.rand() < 0.4
                      else np.random.uniform(8.0, 14.0))

        m = smooth_ellipse_mask(NZ, NX, cx, cz, ax, az, theta, edge_cells=edge_cells)
        masks.append(m)

        logRa = draw_log10_rho(np.random.choice(["saline_sat", "brackish_sat"]))
        logR  = (1.0 - m)*logR + m*logRa

    logR = add_heterogeneity(
        logR, W=W, anomaly_masks=masks,
        sigma_cap_dex=0.02, rel_cap=0.08, depth_decay_cells=120
    )
    logR = homogenize_deep(logR, z0=z_shallow_max)
    return clamp_global(logR)

def gen_geology(NZ=100, NX=200):
    shallow_here = float(np.random.uniform(0.70, 0.85))
    logR, W = layered_background_logR(
        NZ, NX,
        L=np.random.choice([2,3,4,5,6], p=[0.30, 0.30, 0.20, 0.15, 0.05]),
        return_weights=True,
        shallow_frac=shallow_here,
        amp_frac=(0.004, 0.020),
        wavelen_frac=(0.60, 1.20)
    )
    logR = add_heterogeneity(
        logR, W=W,
        sigma_cap_dex=0.015,
        rel_cap=0.06,
        depth_decay_cells=120
    )
    logR = homogenize_deep(logR, z0=int(shallow_here * NZ))
    return clamp_global(logR)

def gen_surface(NZ=100, NX=200):
    logR, W = layered_background_logR(NZ, NX, L=2, return_weights=True)
    X, Z = np.meshgrid(np.arange(NX), np.arange(NZ))
    P = np.random.choice([1,2,3,4], p=[0.3,0.35,0.25,0.1])
    for _ in range(P):
        cx, rx = np.random.uniform(0.1*NX,0.9*NX), np.random.uniform(0.05*NX,0.2*NX)
        depth_decay = np.exp(-Z/np.random.uniform(0.05*NZ,0.2*NZ))
        w = np.exp(-0.5*((X-cx)**2/rx**2)) * depth_decay
        logRa = draw_log10_rho(np.random.choice(["fresh_sat","moist_soil"]))
        logR  = (1-w)*logR + w*logRa
    logR = add_heterogeneity(logR, W=W, sigma_cap_dex=0.02, rel_cap=0.08, depth_decay_cells=120)
    logR = homogenize_deep(logR, z0=int(SHALLOW_FRAC * NZ),
                           facies_choices=["fresh_sat","brackish_sat","moist_sandstone","moist_limestone"])
    return clamp_global(logR)

def gen_seawater(NZ=100, NX=200, shallow_override: float | None = None):
    shallow_here = float(shallow_override if shallow_override is not None
                         else np.random.uniform(0.75, 0.90))
    logR, W = layered_background_logR(
        NZ, NX, L=np.random.choice([2, 3], p=[0.6, 0.4]),
        return_weights=True,
        shallow_frac=shallow_here,
    )
    coast = np.random.choice(["left", "right"])
    x = np.arange(NX, dtype=np.float32)
    if coast == "left":
        u = x / max(1, NX - 1)
    else:
        u = (NX - 1 - x) / max(1, NX - 1)
    base_depth = np.random.uniform(0.24 * NZ, 0.40 * NZ)
    tilt_span_frac = np.random.uniform(0.14, 0.24)
    tilt = tilt_span_frac * NZ * u
    wav_amp = np.random.uniform(0.02 * NZ, 0.06 * NZ)
    wav_len = np.random.uniform(0.25 * NX, 0.75 * NX)
    phase   = np.random.rand() * 2 * np.pi
    wav     = wav_amp * np.sin(2 * np.pi * x / wav_len + phase)
    z_if = base_depth + tilt + wav
    z_if = np.clip(z_if, 0.15 * NZ, min(0.70 * NZ, shallow_here * NZ))
    t_soft = np.random.uniform(8.0, 18.0, size=(NX,)).astype(np.float32)
    t_crsp = np.random.uniform(1.5, 3.5, size=(NX,)).astype(np.float32)
    t_soft = gaussian_filter(t_soft, sigma=12.0, mode="reflect")
    t_crsp = gaussian_filter(t_crsp, sigma=12.0, mode="reflect")
    x01 = np.linspace(0, 1, NX, dtype=np.float32)
    K = np.random.choice([2, 3, 4], p=[0.5, 0.35, 0.15])
    centers = np.random.uniform(0.1, 0.9, size=K)
    widths  = 1.0 / np.random.uniform(6, 15, size=K)
    c = np.zeros_like(x01)
    for cx, w in zip(centers, widths):
        c += np.exp(-(x01 - cx) ** 2 / (2 * w * w))
    c = (c - c.min()) / (c.max() - c.min() + 1e-8)
    crisp_fraction = 0.45
    c *= crisp_fraction / (np.mean(c) + 1e-8)
    c = np.clip(c, 0.0, 1.0)
    T = c * t_crsp + (1.0 - c) * t_soft
    gamma_boost = np.random.uniform(1.0, 2.2)
    Gamma = 1.0 + c * (gamma_boost - 1.0)
    T_eff = T / Gamma
    Z = np.arange(NZ, dtype=np.float32)[:, None]
    s_deep = 1.0 / (1.0 + np.exp(-(Z - z_if[None, :]) / T_eff[None, :]))
    s_deep = gaussian_filter(s_deep, sigma=[1.2, 2.5], mode="reflect")
    hi  = draw_log10_rho("seawater")
    mid = draw_log10_rho(np.random.choice(["saline_sat", "brackish_sat"]))
    lo  = draw_log10_rho(np.random.choice(["fresh_sat", "moist_sandstone", "moist_limestone"]))
    w   = np.clip(s_deep ** np.random.uniform(1.0, 1.5), 0.0, 1.0)
    logR = (1.0 - w) * ((1.0 - w) * lo + w * logR) + w * ((1.0 - w) * mid + w * hi)
    logR = add_heterogeneity(
        logR, W=W, edge_field=s_deep,
        sigma_cap_dex=0.02, rel_cap=0.08, depth_decay_cells=120,
        corr=(8, 14)
    )
    return clamp_global(logR)

def gen_watertable(NZ=100, NX=200):
    shallow_here = float(np.random.uniform(0.62, 0.75))
    logR, W = layered_background_logR(NZ, NX, L=2, return_weights=True, shallow_frac=shallow_here)
    d0    = np.random.uniform(0.22*NZ, 0.42*NZ)
    slope = np.random.uniform(-0.04, 0.04)
    x     = np.arange(NX, dtype=np.float32)
    wav   = np.random.uniform(0.02*NZ, 0.06*NZ) * np.sin(2*np.pi*x/np.random.uniform(0.3*NX,0.9*NX) + np.random.rand()*2*np.pi)
    zWT   = d0 + slope*x + wav
    t_x   = np.random.uniform(6.0, 12.0, size=(NX,)).astype(np.float32)
    t_x   = gaussian_filter(t_x, sigma=10.0, mode='reflect')
    Z = np.arange(NZ, dtype=np.float32)[:, None]
    s = 1.0 / (1.0 + np.exp((Z - zWT[None, :]) / t_x[None, :]))
    s = gaussian_filter(s, sigma=[1.0, 2.0], mode='reflect')
    top = draw_log10_rho(np.random.choice(["dry_soil","moist_soil"]))
    bot = draw_log10_rho(np.random.choice(["fresh_sat","brackish_sat"]))
    logR = (1-s)*((1-s)*bot + s*logR) + s*((1-s)*top + s*logR)
    logR = add_heterogeneity(
        logR, W=W, edge_field=s,
        sigma_cap_dex=0.02, rel_cap=0.08, depth_decay_cells=120,
        corr=(6, 11)
    )
    return clamp_global(logR)

def gen_resistive(NZ=100, NX=200):
    logR, W = layered_background_logR(NZ, NX, L=2, return_weights=True)
    cap_applied = False
    s_shallow = None
    if np.random.rand() < 0.7:
        x = np.arange(NX, dtype=np.float32)
        base_cap = np.random.uniform(0.03*NZ, 0.10*NZ)
        slope    = np.random.uniform(-0.03, 0.03)
        wav      = np.random.uniform(0.02*NZ, 0.05*NZ) * np.sin(
                      2*np.pi*x/np.random.uniform(0.3*NX,0.9*NX) + np.random.rand()*2*np.pi
                  )
        z_cap = base_cap + slope*x + wav
        t_cap = np.random.uniform(3.0, 8.0, size=(NX,)).astype(np.float32)
        t_cap = gaussian_filter(t_cap, sigma=10.0, mode='reflect')
        Z = np.arange(NZ, dtype=np.float32)[:, None]
        s_shallow = 1.0 / (1.0 + np.exp((Z - z_cap[None, :]) / t_cap[None, :]))
        s_shallow = gaussian_filter(s_shallow, sigma=[1.0, 2.0], mode='reflect')
        cap_val = draw_log10_rho(np.random.choice(["dry_soil","coarse_sand"]))
        logR = (1.0 - s_shallow)*logR + s_shallow*cap_val
        cap_applied = True
    masks = []
    A = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
    z_shallow_max = int(SHALLOW_FRAC * NZ)
    for _ in range(A):
        cx = np.random.uniform(0.15*NX, 0.85*NX)
        cz = np.random.uniform(0.10*NZ, 0.90*z_shallow_max)
        ax = np.random.uniform(0.03*NX, 0.12*NX)
        az = np.random.uniform(0.04*NZ, 0.12*NZ)
        if np.random.rand() < 0.7:
            theta = np.deg2rad(np.random.uniform(-20, 20))
            ax *= np.random.uniform(1.5, 3.0)
        else:
            theta = np.deg2rad(np.random.uniform(60, 120))
            az *= np.random.uniform(2.0, 4.0)
            ax = max(ax, 2.0)
        edge_cells = (np.random.uniform(1.0, 3.0) if np.random.rand() < 0.4
                      else np.random.uniform(8.0, 14.0))
        m = smooth_ellipse_mask(NZ, NX, cx, cz, ax, az, theta, edge_cells=edge_cells)
        masks.append(m)
        fac = np.random.choice(
            ["dry_soil","coarse_sand","sandstone","bedrock","moist_limestone","moist_sandstone"],
            p=[0.25,0.25,0.20,0.10,0.10,0.10]
        )
        logRa = draw_log10_rho(fac)
        logR  = (1.0 - m)*logR + m*logRa
    logR = add_heterogeneity(
        logR, W=W, anomaly_masks=masks, edge_field=(s_shallow if cap_applied else None),
        sigma_cap_dex=0.02, rel_cap=0.08, depth_decay_cells=120
    )
    logR = homogenize_deep(logR, z0=z_shallow_max)
    return clamp_global(logR)

# Public registry used by the dataset builder section
GENS = {
    "TRACER":     gen_tracer,
    "GEOLOGY":    gen_geology,
    "SURFACE":    gen_surface,
    "SEAWATER":   gen_seawater,
    "WATERTABLE": gen_watertable,
    "RESISTIVE":  gen_resistive,
}

# ===================================================
# Section B — Dataset builder (no splitting)
# ===================================================

CASES = list(GENS.keys())
LABEL_MAP = {k: i for i, k in enumerate(CASES)}  # case -> int id

def gen_one(case: str, nz: int, nx: int) -> np.ndarray:
    """Generate one log10 resistivity field for a given case."""
    return GENS[case](NZ=nz, NX=nx).astype(np.float32)  # shape (NZ, NX)


def save_all(out_dir: Path, X_list: List[np.ndarray], y_list: List[int], cases_list: List[str]):
    """Save the whole dataset to a compressed NPZ."""
    X = np.stack(X_list, axis=0).astype(np.float32)  # (N, NZ, NX)
    y = np.array(y_list, dtype=np.int64)
    cases_arr = np.array(cases_list, dtype=object)
    np.savez_compressed(out_dir / "dataset.npz", X_log10=X, y=y, cases=cases_arr)
    return X, y, cases_arr


def save_previews(preview_dir: Path, records: List[Tuple[str, np.ndarray]], dpi: int):
    """Save PNG previews with aspect identical to grid (square cells)."""
    preview_dir.mkdir(parents=True, exist_ok=True)
    for name, logR in records:
        nz, nx = logR.shape
        ar = nx / max(1, nz)
        plt.figure(figsize=(6*ar, 6))
        plt.imshow(logR, origin="upper", aspect="equal")  # aspect matches grid
        plt.colorbar(label="log10(ρ [Ω·m])")
        plt.title(name)
        plt.savefig(preview_dir / f"{name}.png", dpi=dpi, bbox_inches="tight")
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="./ert_ds", help="output folder")
    ap.add_argument("--nx", type=int, default=200)
    ap.add_argument("--nz", type=int, default=100)
    ap.add_argument("--per-class", type=int, default=200, help="samples per case (pure, no mixture)")
    ap.add_argument("--save-png", action="store_true", help="also save PNG previews (few per class)")
    ap.add_argument("--previews-per-class", type=int, default=4, help="how many PNGs to save per class (if --save-png)")
    ap.add_argument("--dpi", type=int, default=140, help="DPI for PNG previews")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate *pure* samples per class
    print(f"Generating dataset: {args.per_class} per class, nz={args.nz}, nx={args.nx}")
    X_all: List[np.ndarray] = []
    y_all: List[int] = []
    cases_all: List[str] = []

    for case in CASES:
        print(f"  - {case}: ", end="", flush=True)
        for i in range(args.per_class):
            logR = gen_one(case, args.nz, args.nx)
            X_all.append(logR)
            y_all.append(LABEL_MAP[case])
            cases_all.append(case)
            if (i + 1) % max(1, args.per_class // 10) == 0:
                print(".", end="", flush=True)
        print(" done")

    # Save all-in-one file
    X_arr, y_arr, cases_arr = save_all(out_dir, X_all, y_all, cases_all)

    # Optional previews (few per class)
    if args.save_png:
        previews = []
        for case in CASES:
            want = args.previews_per_class
            picked = 0
            for i, c in enumerate(cases_all):
                if c == case:
                    previews.append((f"preview_{case}_{picked+1}", X_all[i]))
                    picked += 1
                    if picked >= want:
                        break
        save_previews(out_dir / "previews", previews, dpi=args.dpi)

    # Save maps & meta
    with open(out_dir / "label_map.json", "w") as f:
        json.dump(LABEL_MAP, f, indent=2)

    meta = {
        "cases": CASES,
        "label_map": LABEL_MAP,
        "nz": args.nz, "nx": args.nx,
        "per_class": args.per_class,
        "count_total": int(len(X_all)),
        "global_minmax": [float(X_arr.min()), float(X_arr.max())],
        "dtype": "float32",
        "note": "Values are log10(resistivity in Ω·m). No mixture; pure cases only. No split.",
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Done. Saved to: {out_dir.resolve()}")
    print(f"Total samples: {len(X_all)}  (per class: {args.per_class})")
    print(f"Classes: {LABEL_MAP}")
    print(f"Example shapes: X = {X_arr.shape}, y = {y_arr.shape}")
    print("Tip: load with np.load('dataset.npz') and read arrays 'X_log10', 'y', 'cases'.")


if __name__ == "__main__":
    main()
