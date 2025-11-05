"""
fit_pca.py — PCA fitter for 2D log10 images (unsplit datasets)

Overview
--------
This script fits a Principal Component Analysis (PCA) basis to a single
"unsplit" dataset of log10‑transformed 2D images (e.g., subsurface
resistivity maps). It optionally performs:

- Vertical cropping to keep only the shallow (top) rows
- Joint PCA over all samples **or** separate PCA per class label
- Standard PCA or memory‑friendly IncrementalPCA
- Saving metadata, learned components, and (optionally) per‑sample projections
- Quick visual sanity checks by writing a few reconstruction preview PNGs

Expected input
--------------
`--ds` should point to either:
  1) A single `.npz` file, or
  2) A directory containing *exactly one* target `.npz` file.

The `.npz` must contain at least:
  - `X_log10`: float array of shape `(N, NZ, NX)`
Optional arrays:
  - `y`: integer labels of shape `(N,)` (required when using `--per-class`)
  - `cases`: arbitrary per‑sample identifiers (unused by the fitter; passed through only when saving Z)

Key options
-----------
- `--crop-frac`: keep only the top fraction of rows in Z (0.0–1.0). Row 0 is the surface.
- `--max-components`: fit up to this many principal components.
- `--target-var`: report the smallest k reaching this cumulative explained variance.
- `--solver`: PCA SVD solver ("randomized" or "full").
- `--incremental` + `--batch`: use IncrementalPCA for large datasets.
- `--per-class`: fit an independent PCA for each class value in `y`.
- `--save-projections`: save per‑sample low‑dimensional coordinates `Z`.
- `--save-recon`: save side‑by‑side (true vs recon) PNGs for quick inspection.

Outputs (written under `--out`)
-------------------------------
Joint PCA (default):
  - `pca_joint.joblib` : dict with keys {mean, components, explained_variance_ratio, nz, nx, crop_frac}
  - `pca_joint_meta.json` : summary (dimensions, k*, target variance, etc.)
  - `Z.npz` (optional) : low‑dim projections (and `y` if available)
  - `previews_joint_all/` (optional) : a few recon PNGs (true vs recon)

Per‑class PCA (`--per-class`):
  - `pca_class{c}.joblib` per class : same structure as above but class‑specific
  - `pca_per_class_meta.json` : class‑wise k* and counts
  - `Z_per_class.npz` (optional) : stacked projections with labels

Usage examples
--------------
# Joint PCA, keep top 70% rows, save projections and preview PNGs
python fit_pca.py \
  --ds ./datasets/log10_images/all.npz \
  --out ./pca_latent \
  --crop-frac 0.70 \
  --max-components 256 \
  --target-var 0.99 \
  --solver randomized \
  --save-projections \
  --save-recon

# IncrementalPCA for a large dataset
python fit_pca.py --ds ./datasets/huge.npz --incremental --batch 512

# Per‑class PCA (requires y in the dataset)
python fit_pca.py --ds ./datasets/with_labels.npz --per-class --save-projections

Notes
-----
- Cropping convention: the surface is at Z‑row 0; `--crop-frac 0.70` keeps the
  top 70% of rows. Set 1.0 to disable cropping.
- We center the data by subtracting the dataset (or class) mean before fitting.
- `k*` (reported in the metadata) is the minimal dimension that reaches
  `--target-var` cumulative explained variance, capped by `--max-components`.
- Reconstructions in previews use the first `k*` components.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump
from sklearn.decomposition import PCA, IncrementalPCA

# ---------- helpers ----------

def ensure_dir(p: Path) -> None:
    """Create directory *p* and parents if missing (no error if exists)."""
    p.mkdir(parents=True, exist_ok=True)


def crop_top(X: np.ndarray, crop_frac: float) -> np.ndarray:
    """Return the top (shallow) fraction of rows along axis=1.

    Parameters
    ----------
    X : (N, NZ, NX) array
        Input stack of 2D images.
    crop_frac : float in [0, 1]
        Fraction of Z‑rows to keep from the top. 1.0 keeps all rows.
    """
    if crop_frac >= 1.0:
        return X
    NZ = X.shape[1]
    nz_keep = max(1, int(round(crop_frac * NZ)))
    return X[:, :nz_keep, :]


def flatten_images(X: np.ndarray):
    """Flatten (N, NZ, NX) into (N, D), also return (NZ, NX).

    Useful because sklearn PCA expects 2D samples. We keep NZ and NX to
    reshape reconstructions later.
    """
    N, NZ, NX = X.shape
    return X.reshape(N, NZ * NX), NZ, NX


def preview_recons(out_dir: Path, title_prefix: str, X_true: np.ndarray,
                   X_rec: np.ndarray, n: int = 6, dpi: int = 140) -> None:
    """Save side‑by‑side (true vs recon) preview PNGs for *n* samples."""
    ensure_dir(out_dir)
    n = min(n, X_true.shape[0])
    idxs = np.linspace(0, X_true.shape[0] - 1, n, dtype=int)
    for k, i in enumerate(idxs, 1):
        t = X_true[i]
        r = X_rec[i]
        nz, nx = t.shape
        # Keep square‑ish figures while preserving pixel aspect
        ar = nx / max(1, nz)
        plt.figure(figsize=(6 * ar, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(t, origin="upper", aspect="equal")
        plt.title("true")
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.imshow(r, origin="upper", aspect="equal")
        plt.title("recon")
        plt.colorbar()
        plt.suptitle(f"{title_prefix} sample {i}")
        plt.savefig(out_dir / f"{title_prefix}_{k:02d}.png", dpi=dpi, bbox_inches="tight")
        plt.close()


def fit_pca(X_flat: np.ndarray, n_components: int, solver: str,
            incremental: bool, batch: int):
    """Construct a PCA/IncrementalPCA object and (for IPCA) pre‑fit in chunks.

    For IncrementalPCA we run a pass of `partial_fit` across the dataset to
    establish the model; the final `.transform` / `.components_` then reflect
    these passes. For standard PCA, the caller is responsible for `pca.fit`.
    """
    if incremental:
        ipca = IncrementalPCA(n_components=n_components)
        N = X_flat.shape[0]
        b = batch if batch > 0 else 256
        for s in range(0, N, b):
            ipca.partial_fit(X_flat[s : s + b])
        return ipca
    else:
        return PCA(
            n_components=n_components,
            svd_solver=solver,
            whiten=False,
            random_state=0,
        )


def explained_k(pca) -> np.ndarray:
    """Cumulative explained variance curve from a fitted PCA‑like object."""
    r = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)
    return np.cumsum(r)


def load_npz_single(npz_path: Path):
    """Load a single `.npz` file with required/optional arrays.

    Returns
    -------
    X : float32 (N, NZ, NX)
    y : int64 (N,) or None
    cases : any or None
    """
    d = np.load(npz_path, allow_pickle=True)
    X = d["X_log10"].astype(np.float32)  # (N, NZ, NX)
    y = d["y"].astype(np.int64) if "y" in d else None  # optional
    cases = d["cases"] if "cases" in d else None
    return X, y, cases


def find_single_npz(ds: Path) -> Path:
    """Resolve *ds* to a single input file; error if ambiguous.

    Accept either a direct `.npz` path or a directory containing exactly one
    target `.npz`. If multiple exist, prefer common names (all.npz,
    dataset.npz, data.npz); otherwise ask the user to point to the file.
    """
    if ds.is_file() and ds.suffix == ".npz":
        return ds
    cands = sorted([p for p in ds.glob("*.npz")])
    if len(cands) == 0:
        raise FileNotFoundError(f"No .npz found in {ds}")
    # Prefer common names if multiple exist, otherwise force explicit selection
    for preferred in ("all.npz", "dataset.npz", "data.npz"):
        for p in cands:
            if p.name.lower() == preferred:
                return p
    if len(cands) > 1:
        raise RuntimeError(
            "Multiple .npz files found in {ds}. Please point --ds directly to the intended file. "
            f"Found: {[p.name for p in cands]}"
        )
    return cands[0]


# ---------- main ----------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit PCA on a single (unsplit) dataset file containing X_log10 (and optional y)."
        )
    )
    parser.add_argument(
        "--ds",
        type=str,
        required=True,
        help=(
            "Path to a single .npz file OR a directory containing exactly one target .npz (unsplit)."
        ),
    )
    parser.add_argument("--out", type=str, default="./pca_latent", help="output folder")
    parser.add_argument(
        "--crop-frac", type=float, default=0.70, help="keep top fraction of rows"
    )
    parser.add_argument(
        "--max-components", type=int, default=256, help="fit up to this many PCs"
    )
    parser.add_argument(
        "--target-var",
        type=float,
        default=0.99,
        help="report minimal k to reach this variance",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="randomized",
        choices=["randomized", "full"],
        help="PCA SVD solver",
    )
    parser.add_argument(
        "--incremental", action="store_true", help="use IncrementalPCA (chunked)"
    )
    parser.add_argument(
        "--batch", type=int, default=256, help="mini-batch size for IncrementalPCA"
    )
    parser.add_argument(
        "--per-class",
        action="store_true",
        help="fit separate PCA per class (requires y)",
    )
    parser.add_argument(
        "--save-projections", action="store_true", help="save Z.npz"
    )
    parser.add_argument(
        "--save-recon", action="store_true", help="save a few reconstruction PNGs"
    )
    parser.add_argument("--dpi", type=int, default=140)
    args = parser.parse_args()

    ds_path = Path(args.ds)
    out_dir = Path(args.out)
    ensure_dir(out_dir)

    # ===== Unsplit mode only =====
    npz_path = find_single_npz(ds_path)
    print(f"[info] Using dataset: {npz_path}")
    X, y, cases = load_npz_single(npz_path)

    # Optionally crop shallow rows (surface is at row 0)
    X_c = crop_top(X, args.crop_frac)

    # Flatten to (N, D) for sklearn PCA
    X_f, nz, nx = flatten_images(X_c)
    D = nz * nx
    print(
        f"Fitting PCA on {X_f.shape[0]} samples, D={D} ({nz}x{nx}), crop-frac={args.crop_frac:.2f}"
    )

    if not args.per_class:
        # ---- Joint PCA over all samples ----
        mean = X_f.mean(axis=0, dtype=np.float64)
        X0 = (X_f - mean).astype(np.float32)  # centered copies for fitting

        pca = fit_pca(
            X0,
            n_components=min(args.max_components, X0.shape[0], D),
            solver=args.solver,
            incremental=args.incremental,
            batch=args.batch,
        )
        if not args.incremental:
            pca.fit(X0)

        cum = explained_k(pca)
        k_star = int(np.searchsorted(cum, args.target_var) + 1)
        k_star = min(k_star, args.max_components)
        print(
            f"Explained variance @k={k_star}: {cum[k_star-1]:.4f} (target {args.target_var})"
        )

        # Project all samples to the first k* PCs
        Z = pca.transform((X_f - mean).astype(np.float32))[:, :k_star]

        # Persist learned basis (float32 to save space) and metadata
        dump(
            {
                "mean": mean.astype(np.float32),
                "components": pca.components_[:k_star, :].astype(np.float32),
                "explained_variance_ratio": pca.explained_variance_ratio_[:k_star].astype(
                    np.float32
                ),
                "nz": nz,
                "nx": nx,
                "crop_frac": args.crop_frac,
            },
            out_dir / "pca_joint.joblib",
        )

        with open(out_dir / "pca_joint_meta.json", "w") as f:
            json.dump(
                {
                    "n_total": int(X.shape[0]),
                    "nz_crop": int(nz),
                    "nx": int(nx),
                    "D": int(D),
                    "max_components_fit": int(getattr(pca, "n_components", args.max_components)),
                    "k_star": int(k_star),
                    "target_var": float(args.target_var),
                    "cumvar_k_star": float(cum[k_star - 1]),
                    "per_class": False,
                },
                f,
                indent=2,
            )

        if args.save_projections:
            if y is None:
                np.savez_compressed(out_dir / "Z.npz", Z=Z)
            else:
                np.savez_compressed(out_dir / "Z.npz", Z=Z, y=y)

        if args.save_recon:
            comps = pca.components_[:k_star, :]

            def recon(Zm):
                return (Zm @ comps + mean).reshape((-1, nz, nx)).astype(np.float32)

            X_rec = recon(Z)
            preview_recons(
                out_dir / "previews_joint_all",
                "joint_all",
                X_c,
                X_rec,
                n=6,
                dpi=args.dpi,
            )

    else:
        # ---- Per‑class PCA (requires labels) ----
        if y is None:
            raise ValueError("--per-class requires 'y' in the dataset.")
        classes = np.unique(y)
        all_meta = {}
        Z_list, y_list = [], []

        for cls in classes:
            m = y == cls
            X_cls = X_f[m]
            print(f"[class {cls}] N={X_cls.shape[0]}")
            if X_cls.shape[0] == 0:
                continue

            # Class‑specific centering
            mean = X_cls.mean(axis=0, dtype=np.float64)
            X0 = (X_cls - mean).astype(np.float32)

            ncomp = min(args.max_components, X0.shape[0], D)
            pca = fit_pca(
                X0,
                n_components=ncomp,
                solver=args.solver,
                incremental=args.incremental,
                batch=args.batch,
            )
            if not args.incremental:
                pca.fit(X0)

            cum = explained_k(pca)
            k_star = int(np.searchsorted(cum, args.target_var) + 1)
            k_star = min(k_star, ncomp)

            dump(
                {
                    "mean": mean.astype(np.float32),
                    "components": pca.components_[:k_star, :].astype(np.float32),
                    "explained_variance_ratio": pca.explained_variance_ratio_[:k_star].astype(
                        np.float32
                    ),
                    "nz": nz,
                    "nx": nx,
                    "crop_frac": args.crop_frac,
                },
                out_dir / f"pca_class{int(cls)}.joblib",
            )

            all_meta[int(cls)] = {
                "k_star": int(k_star),
                "cumvar_k_star": float(cum[k_star - 1]),
                "n_class": int(X_cls.shape[0]),
            }

            # Faster than calling pca.transform again just for the first k*
            Zc = ((X_cls - mean) @ pca.components_[:k_star, :].T).astype(np.float32)
            Z_list.append(Zc)
            y_list.append(np.full((Zc.shape[0],), cls, dtype=np.int64))

        with open(out_dir / "pca_per_class_meta.json", "w") as f:
            json.dump(
                {
                    "nz_crop": int(nz),
                    "nx": int(nx),
                    "D": int(D),
                    "target_var": float(args.target_var),
                    "per_class": True,
                    "classes": all_meta,
                },
                f,
                indent=2,
            )

        if args.save_projections and Z_list:
            Z_all = np.concatenate(Z_list, axis=0)
            y_all = np.concatenate(y_list, axis=0)
            np.savez_compressed(out_dir / "Z_per_class.npz", Z=Z_all, y=y_all)


if __name__ == "__main__":
    main()
