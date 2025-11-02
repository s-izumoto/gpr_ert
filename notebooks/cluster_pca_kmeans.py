# ===== 完全版（コピペ用） =====
import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


def load_Z(Z_path: str) -> np.ndarray:
    with np.load(Z_path, allow_pickle=False) as z:
        if "Z" not in z.files:
            raise KeyError(f"'Z' not found in {Z_path}. Available: {z.files}")
        Z = np.asarray(z["Z"], dtype=np.float32)
    return Z


def load_pca(pca_path: str):
    obj = joblib.load(pca_path)
    required = ["mean", "components", "nz", "nx"]
    for k in required:
        if k not in obj:
            raise KeyError(f"'{k}' not found in PCA joblib: {pca_path} (keys={list(obj.keys())})")
    return obj


def inverse_transform_Z_to_X(Z, pca_dict, use_k=None):
    comps = pca_dict["components"]  # (k_fit, D)
    mean = pca_dict["mean"]         # (D,)
    if use_k is not None:
        comps = comps[:use_k]
        if Z.shape[1] != use_k:
            raise ValueError(f"Z has {Z.shape[1]} dims but use_k={use_k}")
    X = (Z @ comps) + mean
    return X  # (N, D)


def save_recon_image(arr_1d, nz, nx, out_path, title=None, vmin=None, vmax=None):
    img = arr_1d.reshape(nz, nx)
    fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.imshow(img, origin="upper", aspect="auto",
                   vmin=vmin, vmax=vmax)
    ax.set_title(title or Path(out_path).stem)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Z", default=Path("../data/interim/pca/Z.npz"), help="Path to Z.npz (contains 'Z')")
    ap.add_argument("--pca", default=Path("../data/interim/pca/pca_joint.joblib"), help="Path to pca_joint.joblib")
    ap.add_argument("--k", type=int, default=50, help="Number of clusters")
    ap.add_argument("--algo", choices=["kmeans", "minibatch"], default="kmeans")
    ap.add_argument("--n-init", type=int, default=10)
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--out", default="./pca_clusters_k50")
    ap.add_argument("--silhouette", action="store_true",
                    help="Compute silhouette score (can be slow)")

    ap.add_argument("--no-whiten",    dest="no_whiten",    action="store_true",
                    help="Disable whitening (default: enabled)")
    ap.add_argument("--whiten",       dest="no_whiten",    action="store_false",
                    help=argparse.SUPPRESS)
    ap.set_defaults(no_whiten=False)

    ap.add_argument("--no-save-centroids", dest="no_save_centroids", action="store_true",
                    help="Don't save centroids (default: save)")
    ap.add_argument("--save-centroids",    dest="no_save_centroids", action="store_false",
                    help=argparse.SUPPRESS)
    ap.set_defaults(no_save_centroids=False)

    ap.add_argument("--no-save-medoids",   dest="no_save_medoids", action="store_true",
                    help="Don't save medoids (default: save)")
    ap.add_argument("--save-medoids",      dest="no_save_medoids", action="store_false",
                    help=argparse.SUPPRESS)
    ap.set_defaults(no_save_medoids=False)

    ap.add_argument("--samples-per-cluster", type=int, default=1)

    # ★ parse_args() の直後
    args = ap.parse_args()
    args.whiten         = not args.no_whiten
    args.save_centroids = not args.no_save_centroids
    args.save_medoids   = not args.no_save_medoids

    print("[debug] flags:",
        "whiten=", args.whiten,
        "save_centroids=", args.save_centroids,
        "save_medoids=", args.save_medoids,
        "samples_per_cluster=", args.samples_per_cluster)


    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) load Z and PCA
    Z = load_Z(args.Z)                # (N, z_dim)
    pca = load_pca(args.pca)
    nz, nx = int(pca["nz"]), int(pca["nx"])
    z_dim = Z.shape[1]

    # 2) optional whitening
    Z_input = Z.copy()
    scale = None
    if args.whiten:
        evr = pca.get("explained_variance_ratio", None)
        if evr is not None and len(evr) >= z_dim:
            scale = np.sqrt(np.maximum(evr[:z_dim], 1e-12)).astype(np.float32)
            Z_input = Z_input / scale[None, :]
        # evr が無ければそのまま

    # 3) clustering
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
    labels = km.fit_predict(Z_input)       # (N,)
    centers_input = km.cluster_centers_    # (k, z_dim)

    # 4) save labels & stats
    np.savez_compressed(out_dir / "kmeans_labels.npz", labels=labels)
    sizes = np.bincount(labels, minlength=args.k).tolist()
    meta = {
        "algo": args.algo,
        "k": args.k,
        "n_init": args.n_init,
        "max_iter": args.max_iter,
        "random_state": args.random_state,
        "whiten": bool(args.whiten),
        "inertia": float(getattr(km, "inertia_", np.nan)),
        "sizes": sizes,
    }
    if args.silhouette:
        try:
            sil = float(silhouette_score(Z_input, labels, metric="euclidean"))
        except Exception as e:
            sil = f"ERROR: {e}"
        meta["silhouette_euclidean"] = sil

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # 5) reconstruct and save cluster centroids (optional)
    if args.save_centroids:
        # whiten を戻す（Z空間の中心を“元のZ空間“に戻す）
        centers_Z = centers_input.copy()
        if args.whiten and scale is not None:
            centers_Z = centers_Z * scale[None, :]
        X_centers = inverse_transform_Z_to_X(centers_Z, pca, use_k=z_dim)  # (k, D)

        # 目安のカラースケールを共有
        vmin = float(np.percentile(X_centers, 1))
        vmax = float(np.percentile(X_centers, 99))
        cent_dir = out_dir / "centroids"
        cent_dir.mkdir(exist_ok=True)
        for c in range(args.k):
            save_recon_image(
                X_centers[c], nz, nx,
                cent_dir / f"centroid_{c:03d}.png",
                title=f"Cluster {c} centroid (recon)",
                vmin=vmin, vmax=vmax
            )

    # 6) reconstruct and save medoids (optional)
    if args.save_medoids and args.samples_per_cluster > 0:
        med_dir = out_dir / "medoids"
        med_dir.mkdir(exist_ok=True)
        # 各クラスタで中心に最も近い（or 上位 k 個の）実サンプルを選ぶ
        for c in range(args.k):
            idx = np.where(labels == c)[0]
            if idx.size == 0:
                continue
            Zc = Z_input[idx]                       # whiten後空間
            center = centers_input[c][None, :]
            d2 = np.sum((Zc - center)**2, axis=1)   # ユークリッド距離^2
            order = np.argsort(d2)[:args.samples_per_cluster]
            for j, oi in enumerate(order):
                src_idx = int(idx[int(oi)])
                # 元のZ空間に戻して再構成
                z_vec = Z[src_idx][None, :]         # 元のZ（whitenなし）
                x_vec = inverse_transform_Z_to_X(z_vec, pca, use_k=z_dim)[0]
                out_png = med_dir / f"cluster{c:03d}_medoid{j:02d}_idx{src_idx:05d}.png"
                save_recon_image(
                    x_vec, nz, nx, out_png,
                    title=f"C{c} medoid{j} (idx={src_idx})"
                )

    print(f"[done] labels, meta and (optional) images saved to: {out_dir}")


if __name__ == "__main__":
    main()
# ===== 完全版 ここまで =====
