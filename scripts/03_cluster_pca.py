"""
Thin command-line runner that reads a YAML config and dispatches to a
lower-level clustering script (default: ``build/cluster_pca.py``).

The goal of this runner is to keep experiment configuration in a readable
YAML file, translate that mapping into CLI flags, and invoke the actual
implementation script with those flags. This keeps experiments reproducible
and versionable while avoiding long shell commands.

# How it works
1) Parse ``--config`` (path to a YAML file) and optional ``--script`` (path to
   the target Python script to execute; defaults to ``build/cluster_pca.py``).
2) Load the YAML as a dictionary (key → value).
3) Convert the dictionary into a flat list of CLI arguments using ``kv_to_cli``
   based on a fixed key→flag mapping and a few boolean/feature flags.
4) Invoke the target script via ``subprocess.run([sys.executable, script, ...])``
   and exit with the same return code as the child process.

# Expected YAML keys → CLI flags
- "Z"                      → ``--Z`` (path to latent/feature matrix or bundle)
- "pca"                   → ``--pca`` (path to PCA basis or config)
- "k"                     → ``--k`` (number of clusters)
- "algo"                  → ``--algo`` (clustering algorithm name)
- "n_init"                → ``--n-init`` (initializations for KMeans-like algos)
- "max_iter"              → ``--max-iter``
- "random_state"          → ``--random-state``
- "out"                   → ``--out`` (output directory)
- "samples_per_cluster"   → ``--samples-per-cluster`` (optional downsampling)
- "k_min"                 → ``--k-min``  (elbow/sweep lower bound)
- "k_max"                 → ``--k-max``  (elbow/sweep upper bound)
- "k_step"                → ``--k-step`` (elbow/sweep stride)

Boolean feature switches (presence/absence in YAML toggles flags):
- ``whiten: false``            → append ``--no-whiten``
- ``save_centroids: false``    → append ``--no-save-centroids``
- ``save_medoids: false``      → append ``--no-save-medoids``
- ``silhouette: true``         → append ``--silhouette``
- ``elbow: true``              → append ``--elbow``

Notes:
- Keys omitted or explicitly set to ``null`` are skipped; nothing is passed.
- ``whiten``, ``save_centroids``, and ``save_medoids`` are *negative* flags —
  they only emit a CLI flag when set to ``false`` (opt-out). This mirrors many
  scripts that default to enabling these behaviors.
- ``silhouette`` and ``elbow`` behave as standard feature toggles — flags are
  added only when the YAML value is ``true``.

# Example
YAML (``configs/pca/cluster_pca_kmeans.yml``):

    Z: data/pca/z_latent.npz
    pca: data/pca/pca_basis.npz
    k: 12
    algo: kmeans
    n_init: 10
    max_iter: 300
    random_state: 42
    out: outputs/pca_kmeans
    whiten: true
    save_centroids: true
    save_medoids: false
    silhouette: true
    elbow: false

Run:

    python -m scripts.03_cluster_pca --config configs/pca/cluster_pca_kmeans.yml

This will execute roughly:

    python build/cluster_pca.py \
      --Z data/pca/z_latent.npz \
      --pca data/pca/pca_basis.npz \
      --k 12 --algo kmeans --n-init 10 --max-iter 300 \
      --random-state 42 --out outputs/pca_kmeans \
      --no-save-medoids --silhouette

Exit status: this runner propagates the child process return code.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import yaml


def kv_to_cli(cfg: Dict) -> List[str]:
    """Translate a YAML config mapping into a flat list of CLI arguments.

    Only keys present *and* not ``None`` are converted. The mapping below is
    intentionally explicit to avoid leaking unexpected keys to the child script.

    Parameters
    ----------
    cfg : dict
        YAML-derived configuration mapping (key → value).

    Returns
    -------
    list[str]
        A list suitable to be concatenated after ``[python, script]``.
    """
    args: List[str] = []

    # 1) One-to-one key → flag mapping for scalar/string options.
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
        # Elbow/sweep parameters
        "k_min": "--k-min",
        "k_max": "--k-max",
        "k_step": "--k-step",
    }
    for key, flag in pairs.items():
        if key in cfg and cfg[key] is not None:
            args.extend([flag, str(cfg[key])])

    # 2) Boolean toggles with inverted semantics (opt-out flags)
    if cfg.get("whiten") is False:
        args.append("--no-whiten")
    if cfg.get("save_centroids") is False:
        args.append("--no-save-centroids")
    if cfg.get("save_medoids") is False:
        args.append("--no-save-medoids")

    # 3) Boolean toggles (opt-in flags)
    if cfg.get("silhouette") is True:
        args.append("--silhouette")
    if cfg.get("elbow") is True:
        args.append("--elbow")

    return args


def main() -> None:
    """Entry point: parse CLI, load YAML, dispatch child process."""
    ap = argparse.ArgumentParser(
        description=(
            "Read a YAML config and dispatch to build/cluster_pca.py with the "
            "equivalent command-line flags."
        )
    )
    ap.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file (key: value mapping).",
    )
    ap.add_argument(
        "--script",
        default=str(Path("./build/cluster_pca.py")),
        help=(
            "Path to the underlying build script to execute. "
            "Default: build/cluster_pca.py"
        ),
    )
    args = ap.parse_args()

    # Load YAML → dict
    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise SystemExit("Config must be a YAML mapping (key: value pairs)")

    # Convert mapping → CLI list and run the child script.
    cli_args = [sys.executable, args.script] + kv_to_cli(cfg)
    print("[runner] Exec:", " ".join(cli_args))
    proc = subprocess.run(cli_args)

    # Propagate child return code so callers (e.g., CI) can assert success/failure.
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
