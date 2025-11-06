#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot True vs GPR vs Wenner — triptych per selected field
=======================================================

This standalone script loads:
- PCA metadata (joblib) and Z coefficients (npz) to reconstruct the "true" 2D log10 resistivity map.
- Two inversion bundles (NPZ): one for Wenner, one for GPR/OTHER.
  Each bundle must contain keys like "inv_rho_cells__fieldNNN" plus geometry:
    * cell_centers  (Nc,2) in world coords (x,z), with +z upward
    * L_world, Lz, world_xmin, world_zmax

For a list of field indices, it creates one PNG per field with three panels:
    [ True (log10 ρ) | GPR inversion (log10 ρ) | Wenner inversion (log10 ρ) ]
with a shared color scale.

Usage
-----
    python plot_true_gpr_wenner_triptych.py \
        --pca ../data/interim/pca/pca_joint.joblib \
        --Z ../data/interim/pca/Z.npz \
        --wenner-bundle ../outputs_wenner/inversion.npz \
        --gpr-bundle ../path/to/gpr/inversion_bundle.npz \
        --fields 7 23 102 \
        --out ./triptych_out

Notes
-----
* Images are shown in (row=z down, col=x right). The z-axis is flipped for display
  so that the top of the domain is at the top of the image.
* Inversion vectors (irregular cells) are binned to the regular truth image grid
  by nearest-neighbor assignment.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def reconstruct_log10_field(field_idx: int, pca_joblib_path: Path, z_path: Path) -> np.ndarray:
    """Reconstruct 2D log10-ρ for a given field index from PCA metadata + Z coefficients."""
    from joblib import load as joblib_load
    meta = joblib_load(pca_joblib_path)  # expects keys: mean(D,), components(k,D), nz, nx
    mean = np.asarray(meta["mean"], dtype=np.float32)
    comps = np.asarray(meta["components"], dtype=np.float32)  # (k,D)
    nz, nx = int(meta["nz"]), int(meta["nx"])

    Z = np.load(z_path, allow_pickle=True)["Z"]  # (N_fields, k)
    if not (0 <= field_idx < Z.shape[0]):
        raise IndexError(f"field_index {field_idx} is out of range [0, {Z.shape[0]-1}]")
    k = comps.shape[0]
    z = Z[field_idx, :k].astype(np.float32)      # (k,)

    x_flat = (z @ comps + mean).astype(np.float32)  # (D,)
    if x_flat.size != nz * nx:
        raise RuntimeError(f"Size mismatch: flat={x_flat.size} vs nz*nx={nz*nx}")
    arr2d = x_flat.reshape(nz, nx)
    return arr2d


def inv_vector_to_log_image(inv_rho_cells: np.ndarray,
                            cell_centers: np.ndarray,
                            L_world: float, Lz: float, xmin: float, zmax: float,
                            target_shape: tuple[int, int]) -> np.ndarray:
    """Map an inversion vector (linear ρ per cell) onto a regular (NZ, NX) grid as log10 ρ.

    Improvements:
    - When multiple cells fall into the same pixel, fill it with their **average**.
    - For pixels that receive no hits, apply light interpolation using local averaging (2 passes).
      (A simple interpolation that works without scipy. Replace with distance-based interpolation if available.)
    """
    NZ, NX = target_shape

    # Image coordinate system (rows: z top→bottom, columns: x left→right)
    xs = np.linspace(xmin, xmin + L_world, NX)
    zs = np.linspace(zmax, zmax - Lz, NZ)  # world z: top (large) → bottom (small)

    cx = cell_centers[:, 0].astype(float)
    cz = cell_centers[:, 1].astype(float)

    # Round into 0..NX-1, 0..NZ-1 (z is reversed relative to image rows, so we flip later)
    ix = np.clip(np.round((cx - xs[0]) / (xs[-1] - xs[0]) * (NX - 1)).astype(int), 0, NX - 1)
    iz = np.clip(np.round((cz - zs[-1]) / (zs[0] - zs[-1]) * (NZ - 1)).astype(int), 0, NZ - 1)
    iz = (NZ - 1) - iz  # +z points upward → image rows increase downward

    log_vals = np.log10(np.clip(np.asarray(inv_rho_cells, dtype=float), 1e-12, None))

    # ---- (1) Averaging accumulation ----
    img_sum = np.zeros((NZ, NX), dtype=float)
    img_cnt = np.zeros((NZ, NX), dtype=int)
    np.add.at(img_sum, (iz, ix), log_vals)
    np.add.at(img_cnt, (iz, ix), 1)

    out = np.full((NZ, NX), np.nan, dtype=float)
    hit = img_cnt > 0
    out[hit] = img_sum[hit] / img_cnt[hit]

    # ---- (2) Simple neighborhood interpolation (2 passes) ----
    mask = np.isnan(out)
    if np.any(mask):
        try:
            # Nearest-neighbor interpolation: copy the value of the nearest valid pixel for each NaN
            from scipy.ndimage import distance_transform_edt
            # Compute distance field for True=NaN pixels; obtain indices (iy, ix2) of nearest valid pixels
            dist, (iy, ix2) = distance_transform_edt(mask, return_indices=True)
            out[mask] = out[iy[mask], ix2[mask]]
        except Exception:
            # Fallback: run multiple rounds of local averaging to fill gaps
            for _ in range(50):  # increase iteration count to fill remaining holes
                up    = np.roll(out, -1, axis=0)
                down  = np.roll(out,  1, axis=0)
                left  = np.roll(out, -1, axis=1)
                right = np.roll(out,  1, axis=1)
                neigh = np.nanmean(np.dstack([up, down, left, right]), axis=2)
                fill = np.isnan(out) & np.isfinite(neigh)
                if not np.any(fill):
                    break
                out[fill] = neigh[fill]
            # If NaNs still remain, fill them with the global median (final cleanup of white dots)
            if np.isnan(out).any():
                out[np.isnan(out)] = np.nanmedian(out)
    return out



def load_bundle_meta(npz) -> tuple[np.ndarray, float, float, float, float]:
    """Read geometry metadata from an inversion bundle npz."""
    cell_centers = np.asarray(npz["cell_centers"], dtype=np.float32)
    L_world = float(np.asarray(npz["L_world"]).reshape(-1)[0])
    Lz      = float(np.asarray(npz["Lz"]).reshape(-1)[0])
    xmin    = float(np.asarray(npz["world_xmin"]).reshape(-1)[0])
    zmax    = float(np.asarray(npz["world_zmax"]).reshape(-1)[0])
    return cell_centers, L_world, Lz, xmin, zmax


def field_label(idx: int) -> str:
    return f"field{idx:03d}"


def extract_inv(npz, idx: int):
    candidates = [
        f"inv_rho_cells__field{idx:03d}",
        f"inv_rho_cells__field{idx}",
    ]
    for k in candidates:
        if k in npz.files:
            return np.asarray(npz[k], dtype=float)

    suffixes = {f"field{idx:03d}", f"field{idx}"}
    for k in npz.files:
        if k.startswith("inv_rho_cells__") and any(k.endswith(suf) for suf in suffixes):
            return np.asarray(npz[k], dtype=float)

    return None

def main():
    ap = argparse.ArgumentParser(description="Plot triptych: True vs GPR vs Wenner for selected fields.")
    ap.add_argument("--pca", default="data/fields/pca/pca_joint.joblib", help="Path to PCA joblib (with mean, components, nz, nx).")
    ap.add_argument("--Z", default="data/fields/pca/Z.npz", help="Path to Z.npz containing PCA coefficients (key: Z).")
    ap.add_argument("--wenner-bundle", default="data/GPR/inversion_Wenner/inversion_bundle_Wenner.npz", help="Path to Wenner inversion bundle (npz).")
    ap.add_argument("--gpr-bundle", default="data/GPR/inversion_GPR/inversion_bundle_GPR.npz", help="Path to GPR/OTHER inversion bundle (npz).")
    ap.add_argument("--fields", type=int, nargs="+", default=None,
                    help="Field indices (0-based). If omitted, process all fields in Z.npz.")

    ap.add_argument("--out", default="data/images_example", help="Output directory for PNGs.")
    args = ap.parse_args()

    pca = Path(args.pca); Zp = Path(args.Z)
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    Z_arr = np.load(Zp, allow_pickle=True)["Z"]
    n_total = int(Z_arr.shape[0])
    fields = range(n_total) if args.fields is None else args.fields

    npz_w = np.load(args.wenner_bundle, allow_pickle=False)
    npz_g = np.load(args.gpr_bundle, allow_pickle=False)

    pca = Path(args.pca); Zp = Path(args.Z)
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    npz_w = np.load(args.wenner_bundle, allow_pickle=False)
    npz_g = np.load(args.gpr_bundle, allow_pickle=False)

    # Load geometry (assume each bundle contains consistent meta for its own cells)
    cc_w, Lx_w, Lz_w, xmin_w, zmax_w = load_bundle_meta(npz_w)
    cc_g, Lx_g, Lz_g, xmin_g, zmax_g = load_bundle_meta(npz_g)

    for fidx in fields:
        # --- reconstruct truth ---
        truth_log = reconstruct_log10_field(int(fidx), pca, Zp)  # (NZ,NX)
        NZ, NX = truth_log.shape

        # --- load inversions for this field ---
        inv_w = extract_inv(npz_w, fidx)
        inv_g = extract_inv(npz_g, fidx)
        if inv_w is None and inv_g is None:
            print(f"[warn] No inversions found for field {fidx}; skipping.")
            continue

        gpr_img = inv_vector_to_log_image(inv_g, cc_g, Lx_g, Lz_g, xmin_g, zmax_g, (NZ, NX)) if inv_g is not None else None
        wen_img = inv_vector_to_log_image(inv_w, cc_w, Lx_w, Lz_w, xmin_w, zmax_w, (NZ, NX)) if inv_w is not None else None

        # Determine shared color range from available images
        imgs = [truth_log]
        if gpr_img is not None: imgs.append(gpr_img)
        if wen_img is not None: imgs.append(wen_img)
        vmin = np.nanmin([np.nanmin(im) for im in imgs])
        vmax = np.nanmax([np.nanmax(im) for im in imgs])

        # --- plot ---
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        titles = ["True (log10 ρ)", "GPR inversion (log10 ρ)", "Wenner inversion (log10 ρ)"]
        data_list = [truth_log, gpr_img, wen_img]

        for ax, data, title in zip(axes, data_list, titles):
            if data is None:
                ax.set_visible(False)
                continue
            im = ax.imshow(data, origin="upper", vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("x"); ax.set_ylabel("z")

        # Add a single colorbar spanning visible axes
        visible_axes = [ax for ax, data in zip(axes, data_list) if data is not None]
        if visible_axes:
            fig.colorbar(im, ax=visible_axes, shrink=0.9, location="right", label="log10(ρ)")

        out_path = outdir / f"triptych_field{fidx:03d}.png"
        fig.suptitle(f"Field {fidx:03d}", fontsize=12)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"[ok] wrote {out_path}")

if __name__ == "__main__":
    main()
