"""
cluster_pca.py — Cluster PCA latent vectors (Z) and optionally export reconstructions
-------------------------------------------------------------------------------
Purpose
    Cluster a matrix of PCA latent vectors Z (shape: N × z_dim) using either
    KMeans or MiniBatchKMeans. Optionally, the script can:
      • whiten Z by the square-root of the PCA explained-variance ratio
      • sweep k to make an elbow (inertia vs k) plot and CSV
      • save per-sample cluster labels and basic stats (meta.json)
      • reconstruct and save cluster *centroids* as images via the PCA decoder
      • select and reconstruct *medoids* (closest real samples) per cluster

I/O
    Inputs
      --Z      Path to an .npz that contains the array "Z"  (N × z_dim)
      --pca    Path to a joblib saved dict with at least keys:
               {"mean": (D,), "components": (k_fit, D), "nz": int, "nx": int}
               Optionally: "explained_variance_ratio" (length ≥ z_dim) for
               whitening.

    Outputs (under --out directory, created if missing)
      meta.json                 : clustering settings, inertia, sizes, etc.
      kmeans_labels.npz         : labels (N,)
      centroids/*.png           : reconstructed centroids (if enabled)
      medoids/*.png             : reconstructed medoid samples (if enabled)
      medoids_index.npz         : table of medoid metadata (cluster/order/src)
      elbow_inertia.csv/.png    : k sweep results (if --elbow)
      elbow_meta.json           : auto-selected elbow k and sweep vectors

Key ideas & notes
    • Whitening (default ON) divides Z[:, i] by sqrt(explained_variance_ratio[i]).
      This rescales PCA axes to unit variance and usually makes clusters less
      dominated by the first principal axes. If the PCA dict does not have
      "explained_variance_ratio", the script will silently skip whitening.
    • Centroids are cluster centers in *Z*-space projected back to X via the
      PCA decoder. Medoids are actual samples whose whitened Z is closest to the
      center; these are also reconstructed to X for visualization.
    • The PCA inverse-transform used here is: X = Z @ components + mean.

Examples
    # Basic k-means with k=50 and all exports enabled
    python cluster_pca.py --Z ../data/interim/pca/Z.npz \
                          --pca ../data/interim/pca/pca_joint.joblib \
                          --k 50 --out ./pca_clusters_k50

    # Mini-batch k-means, elbow sweep only (keeps the main k=60 result too)
    python cluster_pca.py --algo minibatch --k 60 --elbow --k-min 10 --k-max 120

    # Disable whitening and skip medoids/centroids to run fast
    python cluster_pca.py --no-whiten --no-save-medoids --no-save-centroids

Caveats
    • Silhouette score can be slow for large N; keep it off unless needed.
    • Elbow sweep does not change the main clustering result; it only writes
      diagnostics (CSV/PNG/JSON) so you can decide a good k later.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless, safe for servers
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score


def _auto_elbow_k(k_list: Iterable[int], j_list: Iterable[float]) -> int:
    """Return the elbow k (largest distance from the end-to-end line).

    The method is a simple Kneedle-like geometry:
      • Consider points (k, inertia)
      • Compute the distance from each point to the straight line joining
        the endpoints (k_min, J_min) and (k_max, J_max)
      • Pick k with the maximum distance

    Assumptions
      k increases monotonically; J (inertia) decreases monotonically.
    """
    ks = np.asarray(k_list, dtype=float)
    js = np.asarray(j_list, dtype=float)
    x1, y1 = ks[0], js[0]
    x2, y2 = ks[-1], js[-1]
    # Twice the triangle area equals the numerator in the point–line distance
    num = np.abs((y2 - y1) * ks - (x2 - x1) * js + (x2 * y1 - y2 * x1))
    den = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    d = num / (den + 1e-12)
    idx = int(np.argmax(d))
    return int(ks[idx])


# ===== Helpers =====

def load_Z(Z_path: str | Path) -> np.ndarray:
    """Load latent matrix Z from an .npz file containing a key "Z".

    Raises
        KeyError: if key "Z" is missing.
    """
    with np.load(Z_path, allow_pickle=False) as z:
        if "Z" not in z.files:
            raise KeyError(f"'Z' not found in {Z_path}. Available: {z.files}")
        Z = np.asarray(z["Z"], dtype=np.float32)
    return Z


def load_pca(pca_path: str | Path) -> Dict[str, Any]:
    """Load a PCA dict saved via joblib.

    Required keys: "mean", "components", "nz", "nx".
    Optionally: "explained_variance_ratio" for whitening.
    """
    obj = joblib.load(pca_path)
    required = ["mean", "components", "nz", "nx"]
    for k in required:
        if k not in obj:
            raise KeyError(
                f"'{k}' not found in PCA joblib: {pca_path} (keys={list(obj.keys())})"
            )
    return obj


def inverse_transform_Z_to_X(Z: np.ndarray, pca_dict: Dict[str, Any], use_k: Optional[int] = None) -> np.ndarray:
    """Decode Z back to X using PCA components and mean.

    Args
        Z: (N, z_dim) latent vectors
        pca_dict: dict with keys "components" (k_fit × D) and "mean" (D,)
        use_k: if provided, truncate components to the first use_k PCs

    Returns
        X: (N, D) reconstructed vectors
    """
    comps = pca_dict["components"]  # (k_fit, D)
    mean = pca_dict["mean"]         # (D,)
    if use_k is not None:
        comps = comps[:use_k]
        if Z.shape[1] != use_k:
            raise ValueError(f"Z has {Z.shape[1]} dims but use_k={use_k}")
    X = (Z @ comps) + mean
    return X


def save_recon_image(
    arr_1d: np.ndarray,
    nz: int,
    nx: int,
    out_path: str | Path,
    title: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """Save a (nz×nx) image reconstructed from a flattened vector."""
    img = arr_1d.reshape(nz, nx)
    fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.imshow(img, origin="upper", aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title or Path(out_path).stem)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    # ---- CLI ----
    ap = argparse.ArgumentParser(
        description=(
            "Cluster PCA latent matrix Z (from Z.npz) and save labels/meta/"
            "optional images (centroids/medoids); optionally run an elbow sweep."
        )
    )
    ap.add_argument("--Z", default=Path("../data/interim/pca/Z.npz"), help="Path to Z.npz (contains 'Z')")
    ap.add_argument("--pca", default=Path("../data/interim/pca/pca_joint.joblib"), help="Path to pca_joint.joblib")
    ap.add_argument("--k", type=int, default=50, help="Number of clusters")
    ap.add_argument("--algo", choices=["kmeans", "minibatch"], default="kmeans", help="Clustering algorithm")
    ap.add_argument("--n-init", type=int, default=10, help="k-means n_init")
    ap.add_argument("--max-iter", type=int, default=300, help="k-means max_iter")
    ap.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility")
    ap.add_argument("--out", default="./pca_clusters_k50", help="Output directory")
    ap.add_argument("--silhouette", action="store_true", help="Compute silhouette score (can be slow)")

    # True-by-default flags (provide --no-... to disable)
    ap.add_argument("--no-whiten", dest="no_whiten", action="store_true", help="Disable whitening (default: enabled)")
    ap.add_argument("--whiten", dest="no_whiten", action="store_false", help=argparse.SUPPRESS)
    ap.set_defaults(no_whiten=False)

    ap.add_argument("--no-save-centroids", dest="no_save_centroids", action="store_true", help="Don't save centroids")
    ap.add_argument("--save-centroids", dest="no_save_centroids", action="store_false", help=argparse.SUPPRESS)
    ap.set_defaults(no_save_centroids=False)

    ap.add_argument("--no-save-medoids", dest="no_save_medoids", action="store_true", help="Don't save medoids")
    ap.add_argument("--save-medoids", dest="no_save_medoids", action="store_false", help=argparse.SUPPRESS)
    ap.set_defaults(no_save_medoids=False)

    ap.add_argument("--samples-per-cluster", type=int, default=1, help="How many medoids to export per cluster")

    # Elbow sweep controls
    ap.add_argument("--elbow", action="store_true", help=(
        "Run an inertia sweep over k and save elbow plot/CSV/JSON (does not override the main --k result)."
    ))
    ap.add_argument("--k-min", type=int, default=10, help="Minimum k for elbow sweep")
    ap.add_argument("--k-max", type=int, default=120, help="Maximum k for elbow sweep")
    ap.add_argument("--k-step", type=int, default=10, help="Step of k for elbow sweep")

    # ---- Parse & normalize booleans ----
    args = ap.parse_args()
    args.whiten = not args.no_whiten
    args.save_centroids = not args.no_save_centroids
    args.save_medoids = not args.no_save_medoids

    print(
        "[debug] flags:",
        "whiten=", args.whiten,
        "save_centroids=", args.save_centroids,
        "save_medoids=", args.save_medoids,
        "samples_per_cluster=", args.samples_per_cluster,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1) Load Z and PCA ----
    Z = load_Z(args.Z)  # (N, z_dim)
    pca = load_pca(args.pca)
    nz, nx = int(pca["nz"]), int(pca["nx"])  # image geometry for recon
    z_dim = Z.shape[1]

    # ---- 2) Optional whitening ----
    Z_input = Z.copy()
    scale: Optional[np.ndarray] = None
    if args.whiten:
        evr = pca.get("explained_variance_ratio", None)
        if evr is not None and len(evr) >= z_dim:
            # divide by sqrt(variance) so each axis has unit variance
            scale = np.sqrt(np.maximum(evr[:z_dim], 1e-12)).astype(np.float32)
            Z_input = Z_input / scale[None, :]
        # If EVR is missing, skip whitening silently.

    # ---- 3) Clustering ----
    if args.algo == "kmeans":
        km = KMeans(
            n_clusters=args.k,
            init="k-means++",
            n_init=args.n_init,
            max_iter=args.max_iter,
            random_state=args.random_state,
            verbose=0,
        )
    else:
        km = MiniBatchKMeans(
            n_clusters=args.k,
            init="k-means++",
            n_init=args.n_init,
            max_iter=args.max_iter,
            random_state=args.random_state,
            batch_size=512,
            verbose=0,
        )

    labels = km.fit_predict(Z_input)            # (N,)
    centers_input = km.cluster_centers_         # (k, z_dim) in *whitened* space if whitening was used

    # ---- 3b) Optional: Elbow sweep (diagnostics only) ----
    if args.elbow:
        k_min, k_max, k_step = int(args.k_min), int(args.k_max), int(args.k_step)
        k_vals = [k for k in range(k_min, k_max + 1, k_step) if k >= 2]
        inertias: List[float] = []
        for k_ in k_vals:
            if args.algo == "kmeans":
                km_tmp = KMeans(
                    n_clusters=k_,
                    init="k-means++",
                    n_init=args.n_init,
                    max_iter=args.max_iter,
                    random_state=args.random_state,
                    verbose=0,
                )
            else:
                km_tmp = MiniBatchKMeans(
                    n_clusters=k_,
                    init="k-means++",
                    n_init=args.n_init,
                    max_iter=args.max_iter,
                    random_state=args.random_state,
                    batch_size=512,
                    verbose=0,
                )
            km_tmp.fit(Z_input)
            inertias.append(float(getattr(km_tmp, "inertia_", float("nan"))))

        # Save sweep as CSV
        with (out_dir / "elbow_inertia.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["k", "inertia"]) 
            for k_, j_ in zip(k_vals, inertias):
                w.writerow([k_, j_])

        # Save figure
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(k_vals, inertias, marker="o")
        ax.set_xlabel("k")
        ax.set_ylabel("inertia (J)")
        ax.set_title("Elbow (inertia vs k)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "elbow.png", dpi=200)
        plt.close(fig)

        # Auto elbow selection (saved as JSON)
        elbow_k = _auto_elbow_k(k_vals, inertias)
        (out_dir / "elbow_meta.json").write_text(
            json.dumps({"k_vals": k_vals, "inertia": inertias, "elbow_k": elbow_k}, indent=2),
            encoding="utf-8",
        )
        print(f"[elbow] saved elbow_inertia.csv & elbow.png (elbow_k≈{elbow_k})")

    # ---- 4) Save labels & meta ----
    np.savez_compressed(out_dir / "kmeans_labels.npz", labels=labels)
    sizes = np.bincount(labels, minlength=args.k).tolist()
    meta: Dict[str, Any] = {
        "algo": args.algo,
        "k": args.k,
        "n_init": args.n_init,
        "max_iter": args.max_iter,
        "random_state": args.random_state,
        "whiten": bool(args.whiten),
        "inertia": float(getattr(km, "inertia_", float("nan"))),
        "sizes": sizes,
    }
    if args.silhouette:
        try:
            sil = float(silhouette_score(Z_input, labels, metric="euclidean"))
        except Exception as e:  # e.g., single-element clusters
            sil = float("nan")
            print(f"[warn] silhouette failed: {e}")
        meta["silhouette_euclidean"] = sil

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # ---- 5) Reconstruct & save cluster centroids (optional) ----
    if args.save_centroids:
        # Bring centers back to *original Z space* before decoding
        centers_Z = centers_input.copy()
        if args.whiten and (scale is not None):
            centers_Z = centers_Z * scale[None, :]
        X_centers = inverse_transform_Z_to_X(centers_Z, pca, use_k=z_dim)  # (k, D)

        # Shared color scale heuristic for consistency across PNGs
        vmin = float(np.percentile(X_centers, 1))
        vmax = float(np.percentile(X_centers, 99))
        cent_dir = out_dir / "centroids"
        cent_dir.mkdir(exist_ok=True)
        for c in range(args.k):
            save_recon_image(
                X_centers[c], nz, nx,
                cent_dir / f"centroid_{c:03d}.png",
                title=f"Cluster {c} centroid (recon)",
                vmin=vmin, vmax=vmax,
            )

    # ---- 6) Reconstruct & save medoids (optional) ----
    if args.save_medoids and args.samples_per_cluster > 0:
        med_dir = out_dir / "medoids"
        med_dir.mkdir(exist_ok=True)
        medoid_records: List[Tuple[int, int, int, str]] = []  # (cluster, order, src_idx, png_relpath)

        # For each cluster, take the closest real samples to the *whitened* center
        for c in range(args.k):
            idx = np.where(labels == c)[0]
            if idx.size == 0:
                continue
            Zc = Z_input[idx]                 # whitened space
            center = centers_input[c][None, :]
            d2 = np.sum((Zc - center) ** 2, axis=1)  # squared Euclidean
            order = np.argsort(d2)[: args.samples_per_cluster]
            for j, oi in enumerate(order):
                src_idx = int(idx[int(oi)])
                # Decode the *original* Z (not whitened) for a real sample
                z_vec = Z[src_idx][None, :]
                x_vec = inverse_transform_Z_to_X(z_vec, pca, use_k=z_dim)[0]
                out_png = med_dir / f"cluster{c:03d}_medoid{j:02d}_idx{src_idx:05d}.png"
                save_recon_image(
                    x_vec, nz, nx, out_png, title=f"C{c} medoid{j} (idx={src_idx})"
                )
                medoid_records.append((int(c), int(j), int(src_idx), str(out_png.relative_to(out_dir))))

        # Save medoid table as NPZ (columns: cluster, order, src_idx, png_relpath)
        if len(medoid_records) > 0:
            rec = np.array(medoid_records, dtype=object)
            np.savez_compressed(
                out_dir / "medoids_index.npz",
                cluster=rec[:, 0].astype(int),
                order=rec[:, 1].astype(int),
                src_index=rec[:, 2].astype(int),
                png_relpath=rec[:, 3],
            )

    print(f"[done] labels, meta and (optional) images saved to: {out_dir}")


if __name__ == "__main__":
    main()
