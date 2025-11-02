
"""
Thin runner that reads a YAML config and dispatches to build/cluster_pca_kmeans.py

Usage:
  python -m scripts.03_cluster_pca_kmeans --config configs/pca/cluster_pca_kmeans.yml
"""
from __future__ import annotations
import argparse, sys, subprocess
from pathlib import Path
import yaml

def kv_to_cli(cfg: dict) -> list[str]:
    """Map YAML keys to build/cluster_pca_kmeans.py CLI flags.
       For booleans:
         - whiten/save_centroids/save_medoids default True. If False, pass the *negative* flag.
         - silhouette default False. If True, pass the flag.
    """
    args: list[str] = []

    # Simple (non-bool) mappings
    pairs = {
        "Z": "--Z",
        "pca": "--pca",
        "k": "--k",
        "algo": "--algo",
        "n_init": "--n-init",
        "max_iter": "--max-iter",
        "random_state": "--random-state",
        "out": "--out",
        "samples_per_cluster": "--samples-per-cluster",
    }
    for k, flag in pairs.items():
        if k in cfg and cfg[k] is not None:
            args.extend([flag, str(cfg[k])])

    # Booleans
    # Positive-by-default booleans: if set False in YAML, we pass the negative flag.
    if cfg.get("whiten") is False:
        args.append("--no-whiten")
    # If True, omit (default=True).

    if cfg.get("save_centroids") is False:
        args.append("--no-save-centroids")

    if cfg.get("save_medoids") is False:
        args.append("--no-save-medoids")

    # silhouette is default False: pass if True
    if cfg.get("silhouette") is True:
        args.append("--silhouette")

    return args

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    ap.add_argument("--script", default=str(Path("./build/cluster_pca_kmeans.py")),
                    help="Path to build script (default: build/cluster_pca_kmeans.py)")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise SystemExit("Config must be a YAML mapping (key: value pairs)")

    cli_args = [sys.executable, args.script] + kv_to_cli(cfg)
    print("[runner] Exec:", " ".join(cli_args))
    proc = subprocess.run(cli_args)
    raise SystemExit(proc.returncode)

if __name__ == "__main__":
    main()
