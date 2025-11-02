# build_pca_latent.py (unsplit-only)
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump
from sklearn.decomposition import PCA, IncrementalPCA

# ---------- helpers ----------
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def crop_top(X: np.ndarray, crop_frac: float):
    """Keep the top (shallow) fraction of rows. Surface is at row 0."""
    if crop_frac >= 1.0: return X
    NZ = X.shape[1]
    nz_keep = max(1, int(round(crop_frac * NZ)))
    return X[:, :nz_keep, :]

def flatten_images(X: np.ndarray):
    N, NZ, NX = X.shape
    return X.reshape(N, NZ * NX), NZ, NX

def preview_recons(out_dir: Path, title_prefix: str, X_true: np.ndarray,
                   X_rec: np.ndarray, n=6, dpi=140):
    ensure_dir(out_dir)
    n = min(n, X_true.shape[0])
    idxs = np.linspace(0, X_true.shape[0]-1, n, dtype=int)
    for k, i in enumerate(idxs, 1):
        t = X_true[i]; r = X_rec[i]
        nz, nx = t.shape
        ar = nx / max(1, nz)
        plt.figure(figsize=(6*ar, 6))
        plt.subplot(1,2,1); plt.imshow(t, origin="upper", aspect="equal"); plt.title("true"); plt.colorbar()
        plt.subplot(1,2,2); plt.imshow(r, origin="upper", aspect="equal"); plt.title("recon"); plt.colorbar()
        plt.suptitle(f"{title_prefix} sample {i}")
        plt.savefig(out_dir / f"{title_prefix}_{k:02d}.png", dpi=dpi, bbox_inches="tight")
        plt.close()

def fit_pca(X_flat: np.ndarray, n_components: int, solver: str, incremental: bool, batch: int):
    if incremental:
        ipca = IncrementalPCA(n_components=n_components)
        N = X_flat.shape[0]
        b = batch if batch > 0 else 256
        for s in range(0, N, b):
            ipca.partial_fit(X_flat[s:s+b])
        return ipca
    else:
        return PCA(n_components=n_components, svd_solver=solver, whiten=False, random_state=0)

def explained_k(pca) -> np.ndarray:
    r = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)
    return np.cumsum(r)

def load_npz_single(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    X = d["X_log10"].astype(np.float32)   # (N, NZ, NX)
    y = d["y"].astype(np.int64) if "y" in d else None  # optional
    cases = d["cases"] if "cases" in d else None
    return X, y, cases

def find_single_npz(ds: Path) -> Path:
    """Accept either a single .npz path or a directory containing exactly one target .npz (no train/val/test)."""
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
        raise RuntimeError(f"Multiple .npz files found in {ds}. Please point --ds directly to the intended file. Found: {[p.name for p in cands]}")
    return cands[0]

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Fit PCA on a single (unsplit) dataset file containing X_log10 (and optional y).")
    ap.add_argument("--ds", type=str, required=True,
                    help="Path to a single .npz file OR a directory containing exactly one target .npz (unsplit).")
    ap.add_argument("--out", type=str, default="./pca_latent", help="output folder")
    ap.add_argument("--crop-frac", type=float, default=0.70, help="keep top fraction of rows")
    ap.add_argument("--max-components", type=int, default=256, help="fit up to this many PCs")
    ap.add_argument("--target-var", type=float, default=0.99, help="report minimal k to reach this variance")
    ap.add_argument("--solver", type=str, default="randomized", choices=["randomized","full"], help="PCA SVD solver")
    ap.add_argument("--incremental", action="store_true", help="use IncrementalPCA (chunked)")
    ap.add_argument("--batch", type=int, default=256, help="mini-batch size for IncrementalPCA")
    ap.add_argument("--per-class", action="store_true", help="fit separate PCA per class (requires y)")
    ap.add_argument("--save-projections", action="store_true", help="save Z.npz")
    ap.add_argument("--save-recon", action="store_true", help="save a few reconstruction PNGs")
    ap.add_argument("--dpi", type=int, default=140)
    args = ap.parse_args()

    ds_path = Path(args.ds)
    out_dir = Path(args.out)
    ensure_dir(out_dir)

    # ===== Unsplit mode only =====
    npz_path = find_single_npz(ds_path)
    print(f"[info] Using dataset: {npz_path}")
    X, y, cases = load_npz_single(npz_path)
    X_c = crop_top(X, args.crop_frac)
    X_f, nz, nx = flatten_images(X_c)
    D = nz * nx
    print(f"Fitting PCA on {X_f.shape[0]} samples, D={D} ({nz}x{nx}), crop-frac={args.crop_frac:.2f}")

    if not args.per_class:
        mean = X_f.mean(axis=0, dtype=np.float64)
        X0 = (X_f - mean).astype(np.float32)
        pca = fit_pca(X0, n_components=min(args.max_components, X0.shape[0], D),
                      solver=args.solver, incremental=args.incremental, batch=args.batch)
        if not args.incremental:
            pca.fit(X0)
        cum = explained_k(pca)
        k_star = int(np.searchsorted(cum, args.target_var) + 1)
        k_star = min(k_star, args.max_components)
        print(f"Explained variance @k={k_star}: {cum[k_star-1]:.4f} (target {args.target_var})")

        Z = pca.transform((X_f - mean).astype(np.float32))[:, :k_star]

        dump({"mean": mean.astype(np.float32),
              "components": pca.components_[:k_star, :].astype(np.float32),
              "explained_variance_ratio": pca.explained_variance_ratio_[:k_star].astype(np.float32),
              "nz": nz, "nx": nx, "crop_frac": args.crop_frac},
             out_dir / "pca_joint.joblib")

        with open(out_dir / "pca_joint_meta.json", "w") as f:
            json.dump({
                "n_total": int(X.shape[0]),
                "nz_crop": int(nz), "nx": int(nx),
                "D": int(D),
                "max_components_fit": int(getattr(pca, "n_components", args.max_components)),
                "k_star": int(k_star),
                "target_var": float(args.target_var),
                "cumvar_k_star": float(cum[k_star-1]),
                "per_class": False
            }, f, indent=2)

        if args.save_projections:
            if y is None:
                np.savez_compressed(out_dir / "Z.npz", Z=Z)
            else:
                np.savez_compressed(out_dir / "Z.npz", Z=Z, y=y)

        if args.save_recon:
            comps = pca.components_[:k_star, :]
            def recon(Zm): return (Zm @ comps + mean).reshape((-1, nz, nx)).astype(np.float32)
            X_rec = recon(Z)
            preview_recons(out_dir / "previews_joint_all", "joint_all",
                           X_c, X_rec, n=6, dpi=args.dpi)

    else:
        if y is None:
            raise ValueError("--per-class requires 'y' in the dataset.")
        classes = np.unique(y)
        all_meta = {}
        Z_list, y_list = [], []
        for cls in classes:
            m = (y == cls)
            X_cls = X_f[m]
            print(f"[class {cls}] N={X_cls.shape[0]}")
            if X_cls.shape[0] == 0:
                continue
            mean = X_cls.mean(axis=0, dtype=np.float64)
            X0 = (X_cls - mean).astype(np.float32)
            ncomp = min(args.max_components, X0.shape[0], D)
            pca = fit_pca(X0, n_components=ncomp, solver=args.solver, incremental=args.incremental, batch=args.batch)
            if not args.incremental:
                pca.fit(X0)
            cum = explained_k(pca)
            k_star = int(np.searchsorted(cum, args.target_var) + 1)
            k_star = min(k_star, ncomp)
            dump({"mean": mean.astype(np.float32),
                  "components": pca.components_[:k_star, :].astype(np.float32),
                  "explained_variance_ratio": pca.explained_variance_ratio_[:k_star].astype(np.float32),
                  "nz": nz, "nx": nx, "crop_frac": args.crop_frac},
                 out_dir / f"pca_class{int(cls)}.joblib")
            all_meta[int(cls)] = {
                "k_star": int(k_star),
                "cumvar_k_star": float(cum[k_star-1]),
                "n_class": int(X_cls.shape[0])
            }
            Zc = ((X_cls - mean) @ pca.components_[:k_star, :].T).astype(np.float32)
            Z_list.append(Zc); y_list.append(np.full((Zc.shape[0],), cls, dtype=np.int64))

        with open(out_dir / "pca_per_class_meta.json", "w") as f:
            json.dump({
                "nz_crop": int(nz), "nx": int(nx), "D": int(D),
                "target_var": float(args.target_var),
                "per_class": True,
                "classes": all_meta
            }, f, indent=2)

        if args.save_projections and Z_list:
            Z_all = np.concatenate(Z_list, axis=0)
            y_all = np.concatenate(y_list, axis=0)
            np.savez_compressed(out_dir / "Z_per_class.npz", Z=Z_all, y=y_all)

if __name__ == "__main__":
    main()
