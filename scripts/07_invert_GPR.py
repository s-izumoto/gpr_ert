# scripts/07_invert_GPR.py


"""
Script overview
---------------
This thin runner reads a YAML configuration and delegates the actual inversion
to `build.invert_GPR.run_inversion(...)`. It focuses on I/O and orchestration.

Responsibilities:
- Make the repository root importable so internal modules (e.g., build.invert_GPR) are found.
- Parse CLI arguments (required: --config; optional: --workers override).
- Load & validate a YAML mapping, ensuring at least `npz` is provided.
- Forward configuration values to `run_inversion(...)`.
- Print simple timing logs and the final bundle path for downstream scripts.

Typical YAML keys
-----------------
Required:
  npz:              Path to an input .npz bundle containing data to invert.

Optional (forwarded as-is; sensible defaults applied downstream if omitted):
  out:              Path to a single output image (e.g., a summary PNG).
  out_log:          Path to a text/CSV log file.
  out_dir:          Directory for per-field images or artifacts.
  bundle_out:       Path to save a consolidated inversion bundle (.npz).
  field_index:      Integer index of a single field to process (0-based).
  all_fields:       Bool; if True, process all fields (overrides field_index).
  images_all:       Bool; if True, save images for all fields.
  n_elec:           Number of electrodes used by the forward/inversion model.
  dx_elec:          Electrode spacing in model coordinates.
  world_Lx:         Model domain width (x-direction) used by meshing.
  margin:           Additional lateral margin around electrode span for the mesh.
  nx_full:          Target horizontal cell count for the full mesh.
  nz_full:          Target vertical   cell count for the full mesh.
  mesh_area:        Target triangle (or cell) area for meshing.
  workers:          Parallel worker processes for inversion/IO (int).

Example usage
-------------
  python scripts/07_invert_GPR.py --config configs/inversion.yml

Example YAML (minimal)
----------------------
  npz: ./data/GPR/inversions_bundle.npz

Example YAML (extended)
-----------------------
  npz: ./data/GPR/inversions_bundle.npz
  out: ./outputs/preview.png
  out_log: ./outputs/inversion.log
  out_dir: ./outputs/per_field
  bundle_out: ./outputs/inversion_bundle.npz
  all_fields: true
  images_all: true
  n_elec: 32
  dx_elec: 1.0
  world_Lx: 31.0
  margin: 3.0
  nx_full: 400
  nz_full: 100
  mesh_area: 0.1
  workers: 4

Notes
-----
- This file does NOT implement the inversion algorithm. See `build.invert_GPR`.
- If both `all_fields: true` and `field_index` are given, `all_fields` wins.
- CLI `--workers` overrides the YAML `workers` if provided.
"""

"""
Compute repository root (one directory above this file, i.e., scripts/ -> repo root)
and ensure it's on sys.path so internal imports resolve correctly.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import yaml
import os, sys
from time import perf_counter
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

"""
Import the actual inversion entry point. The heavy lifting lives in build/.
"""
from build.invert_GPR import run_inversion  # noqa: E402

"""
Record start timing (wall clock + high-resolution timer) for simple logging.
"""
_start_wall_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
_start_t = perf_counter()
print(f"[time] start: {_start_wall_ts}")


def main():
    """
    Parse CLI, load YAML config, validate keys, and delegate to run_inversion().

    CLI:
      --config  (required) YAML path
      --workers (optional) overrides YAML 'workers' if provided
    """
    ap = argparse.ArgumentParser(
        description="Read a YAML config and call build.invert_GPR.run_inversion()."
    )
    ap.add_argument("--config", required=True, help="Path to YAML config file.")
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes (1 = single process).",
    )
    args = ap.parse_args()

    """
    Load YAML as a mapping (dict). Abort if the structure is not a key-value map.
    """
    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise SystemExit("Config must be a YAML mapping (key: value pairs).")

    """
    Fetch required & optional keys. Only 'npz' is mandatory at this layer.
    Defaults here are minimal—`run_inversion` may set more robust defaults.
    """
    npz = cfg.get("npz")
    if not npz:
        raise SystemExit("YAML must contain 'npz' (path to input .npz).")

    out = cfg.get("out")
    out_log = cfg.get("out_log")
    out_dir = cfg.get("out_dir")
    bundle_out = cfg.get("bundle_out")
    field_index = cfg.get("field_index")
    all_fields = bool(cfg.get("all_fields", False))
    images_all = bool(cfg.get("images_all", False))

    """
    Geometry / mesh / inversion controls (lightweight defaults).
    These are forwarded directly; implementation decides how to interpret them.
    """
    n_elec = int(cfg.get("n_elec", 32))
    dx_elec = float(cfg.get("dx_elec", 1.0))
    world_Lx = float(cfg.get("world_Lx", 31.0))
    margin = float(cfg.get("margin", 3.0))
    nx_full = int(cfg.get("nx_full", 400))
    nz_full = int(cfg.get("nz_full", 100))
    mesh_area = float(cfg.get("mesh_area", 0.1))

    """
    Workers: CLI overrides YAML if provided. Fall back to CLI default=1.
    """
    workers = int(cfg.get("workers", args.workers))

    """
    Delegate to the inversion implementation. The returned value is typically
    the path to the saved inversion bundle (if any).
    """
    bundle = run_inversion(
        npz_path=npz,
        out=out,
        out_log=out_log,
        out_dir=out_dir,
        bundle_out=bundle_out,
        field_index=field_index,
        all_fields=all_fields,
        images_all=images_all,
        n_elec=n_elec,
        dx_elec=dx_elec,
        world_Lx=world_Lx,
        margin=margin,
        nx_full=nx_full,
        nz_full=nz_full,
        mesh_area=mesh_area,
        workers=workers,
    )

    """
    Print a concise summary for downstream tooling and human inspection.
    """
    print(f"[runner] bundle saved at: {bundle}")

    """
    End timing and print elapsed seconds for quick performance checks.
    """
    _end_t = perf_counter()
    _end_wall_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _elapsed = _end_t - _start_t
    print(f"[time] end:   {_end_wall_ts}  elapsed: {_elapsed:.3f}s")


if __name__ == "__main__":
    main()
