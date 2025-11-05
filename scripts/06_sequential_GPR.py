# scripts/06_gpr_sequential_design.py
"""
Sequential GPR Runner & Bundler
================================

This script orchestrates **sequential design runs** for Gaussian Process
Regression (GPR) across one or more "fields" (e.g., different datasets or
spatial realizations). For each field, it calls `run_from_cfg` from
`build.sequential_GPR`, then **bundles** the per-field outputs into a single
folder:

- Per-field outputs (under `<out_dir>/fieldXXX/`):
  - `candidate_stats.csv` : per-step candidate pool statistics and acquisition values
  - `gpr_params.csv`      : fitted kernel hyperparameters (e.g., length scales, noise)
  - `seq_log.npz`         : arrays logging the sequential run (chosen indices, metrics, etc.)

- Aggregated outputs (under `<out_dir>/`):
  - `bundle_candidate_stats.csv` : concatenated per-field candidate stats
  - `bundle_gpr_params.csv`      : concatenated per-field GPR parameter logs
  - `GPR_bundle.npz`             : all `seq_log.npz` keys namespaced as `<key>__fieldXXX`

Key features
------------
- Reads a YAML config and normalizes common alias keys (see `KEYMAP`).
- Field selection:
  1) by medoid mapping file (`--fields-from-medoids` or YAML `fields_from_medoids`)
  2) by explicit indices (`--field-index` or YAML `fields`)
- Parallel execution via `ProcessPoolExecutor` (configurable with `--workers`).
- Robust bundling of CSV and NPZ logs with deterministic (sorted) field order.

Typical usage
-------------
    python scripts/06_gpr_sequential_design.py \\
        --config configs/gpr_seq_example.yml \\
        --field-index 0 2 5 \\
        --workers 3

Or, if your YAML includes `fields_from_medoids: path/to/medoids_bundle.npz`,
you can simply run:

    python scripts/06_gpr_sequential_design.py --config configs/your.yml

Notes
-----
- The YAML file is parsed with PyYAML if available; otherwise it falls back to
  JSON-compatible parsing.
- Unknown keys in YAML are warned and then **dropped** before creating
  `GPRSeqConfig`, so minor config drift won’t crash the run.
"""

import argparse
import csv
import json
import os
import sys
import warnings
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np

# Optional dependency: PyYAML is recommended; fallback to JSON-compatible parsing.
try:
    import yaml  # PyYAML
except Exception:
    yaml = None

# --- Add repo root to sys.path (so `build.*` is importable when run from scripts/) ---
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from build.sequential_GPR import GPRSeqConfig, run_from_cfg  # project-provided module

# Print start time and initialize wall/CPU timers
_start_wall_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
_start_t = perf_counter()
print(f"[time] start: {_start_wall_ts}")

# --- KEYMAP: normalize common alias keys to canonical names used by GPRSeqConfig ---
KEYMAP = {
    # field selectors
    "field": "fields",
    "fields": "fields",
    "field-index": "fields",
    "field_index": "fields",
    # frequent config aliasing
    "n_elecs": "n_elec",
    "nElec": "n_elec",
    "minGap": "min_gap",
    "ddKmax": "dd_kmax",
    "dd-kmax": "dd_kmax",
    "schlAMax": "schl_a_max",
    "schl-a-max": "schl_a_max",
}


def _run_one_field_worker(cfg_dict: dict, fid: int, base_bundle_dir: str) -> tuple[int, str]:
    """
    Run a single field in a separate process-safe function.

    Parameters
    ----------
    cfg_dict : dict
        A plain dict version of `GPRSeqConfig` (via dataclasses.asdict).
    fid : int
        Field index to process.
    base_bundle_dir : str
        Aggregation directory (parent) where `fieldXXX/` will be created.

    Returns
    -------
    (field_id, out_dir_str) : tuple[int, str]
        The processed field id and the output directory path used by `run_from_cfg`.
    """
    sub_dir = Path(base_bundle_dir) / f"field{int(fid):03d}"
    sub_dir.mkdir(parents=True, exist_ok=True)

    # Override out_dir on a per-field basis to keep outputs separated.
    cfg_i = GPRSeqConfig(**{**cfg_dict, "out_dir": str(sub_dir)})
    out_dir = run_from_cfg(cfg_i, field_index=int(fid))
    return int(fid), str(out_dir)


def _normalize_config_keys(d: dict) -> dict:
    """Apply KEYMAP aliases so downstream code sees canonical names only."""
    out = {}
    for k, v in (d or {}).items():
        out[KEYMAP.get(k, k)] = v
    return out


def _normalize_fields(val) -> list[int]:
    """
    Normalize `fields` declarations to a list[int].
    Accepts None, int, list/tuple of ints, or a comma-separated string.
    """
    if val is None:
        return []
    if isinstance(val, int):
        return [int(val)]
    if isinstance(val, (list, tuple)):
        return [int(x) for x in val]
    if isinstance(val, str):
        # Allow "0,1,3" style strings
        return [int(x.strip()) for x in val.split(",") if x.strip() != ""]
    raise TypeError(f"Invalid type for fields: {type(val)}")


def _append_csv(src_csv: Path, dst_csv: Path):
    """
    Append the data rows of `src_csv` to `dst_csv` (skipping src header).
    Creates destination (and its parent) if needed.
    """
    if not src_csv.exists():
        return
    dst_csv.parent.mkdir(parents=True, exist_ok=True)
    with src_csv.open("r", encoding="utf-8", newline="") as fr:
        reader = csv.reader(fr)
        rows = list(reader)
    if not rows:
        return
    header, data = rows[0], rows[1:]
    write_header = not dst_csv.exists()
    with dst_csv.open("a", encoding="utf-8", newline="") as fw:
        w = csv.writer(fw)
        if write_header:
            w.writerow(header)
        if data:
            w.writerows(data)


def _npz_to_dict(npz_path: Path) -> dict:
    """
    Load an npz (e.g., per-field `seq_log.npz`) as a regular dict[str, np.ndarray].
    """
    out = {}
    with np.load(npz_path, allow_pickle=False) as z:
        for k in z.files:
            out[k] = np.asarray(z[k])
    return out


def _fields_from_medoids(npz_path: str | Path) -> list[int]:
    """
    Extract field indices from a medoids bundle (expects a `src_index` array).

    The `src_index` array is expected to be 0-based and non-negative.
    Returned list is de-duplicated and sorted.
    """
    with np.load(npz_path, allow_pickle=False) as z:
        if "src_index" not in z.files:
            raise KeyError(f"'src_index' not found in {npz_path}. available={list(z.files)}")
        idx = np.asarray(z["src_index"], dtype=np.int64).ravel()
    if (idx < 0).any():
        bad = np.unique(idx[idx < 0])[:8]
        raise ValueError(f"src_index contains negative values: {bad}")
    return sorted(set(int(i) for i in idx.tolist()))


def main():
    # ---- CLI argument parsing ------------------------------------------------
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True,
                   help="Path to YAML config (JSON-compatible is okay if PyYAML is missing).")
    p.add_argument(
        "--field-index", type=int, nargs="+", default=[0, 2],
        help="Target field indices. Example: --field-index 0 3 7"
    )
    p.add_argument("--fields-from-medoids", type=str, default=None,
                   help="Path to an npz with 'src_index' to derive field indices.")

    # Choose a sensible default for parallelism (cap at 4 by default).
    default_workers = max(1, min(os.cpu_count() or 1, 4))
    p.add_argument("--workers", type=int, default=None,
                   help="Number of parallel workers. If omitted, auto-choose a safe default.")

    ns = p.parse_args()

    # ---- Load YAML/JSON config ----------------------------------------------
    cfg_path = Path(ns.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        if yaml is not None:
            yaml_cfg = yaml.safe_load(f) or {}
        else:
            # Fallback: only valid if the YAML is also strict JSON.
            yaml_cfg = json.load(f)

    # Normalize keys (aliases -> canonical) early.
    yaml_cfg_norm = _normalize_config_keys(yaml_cfg if isinstance(yaml_cfg, dict) else {})

    # ---- Resolve field list (priority: medoids-from-YAML > medoids-from-CLI > fields/CLI) ---
    fields: list[int] = []

    yaml_medoids = yaml_cfg_norm.get("fields_from_medoids", None)
    cli_medoids = getattr(ns, "fields_from_medoids", None)

    if yaml_medoids:
        fields = _fields_from_medoids(yaml_medoids)
        print(f"[run] fields_from_medoids (YAML): {fields}")
    elif cli_medoids:
        fields = _fields_from_medoids(cli_medoids)
        print(f"[run] fields_from_medoids (CLI): {fields}")
    else:
        # Fall back to explicit fields (YAML or CLI)
        fields_yaml = _normalize_fields(yaml_cfg_norm.get("fields")) if "fields" in yaml_cfg_norm else []
        fields_cli = _normalize_fields(ns.field_index) if getattr(ns, "field_index", None) is not None else []
        fields = fields_yaml if fields_yaml else fields_cli
        print(f"[run] fields from YAML/CLI: {fields}")

    if not fields:
        fields = [0]

    fields = sorted(set(int(x) for x in fields))
    print(f"[run] fields to process: {fields}")

    # Prepare a config payload for GPRSeqConfig (drop keys it doesn't accept).
    cfg_payload = dict(yaml_cfg_norm)
    cfg_payload.pop("fields_from_medoids", None)
    cfg_payload.pop("fields", None)  # we decide fields here; don't pass into the dataclass

    try:
        cfg = GPRSeqConfig(**cfg_payload)
    except TypeError as e:
        # If unknown keys are present, warn and retry after filtering.
        warnings.warn(f"[warn] Unknown keys in YAML may be ignored by GPRSeqConfig: {e}")
        allowed = set(GPRSeqConfig.__annotations__.keys())
        cfg = GPRSeqConfig(**{k: v for k, v in cfg_payload.items() if k in allowed})

    # ---- Ensure bundle directory exists -------------------------------------
    bundle_dir = Path(cfg.out_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # Bundle CSV paths (we will append per-field CSVs to these).
    bundle_cand_csv = bundle_dir / "bundle_candidate_stats.csv"
    bundle_gprp_csv = bundle_dir / "bundle_gpr_params.csv"

    # Convert dataclass -> dict for picklable workers.
    cfg_base_dict = asdict(cfg)

    # Decide worker count.
    workers = ns.workers if ns.workers is not None else min(len(fields), default_workers)
    print(f"[parallel] workers = {workers}")

    # ---- Launch runs (parallel or sequential) --------------------------------
    results: list[tuple[int, str]] = []  # (fid, out_dir_str)
    if workers == 1:
        for fid in fields:
            fid_i, out_dir_str = _run_one_field_worker(cfg_base_dict, int(fid), str(bundle_dir))
            print(f"[done] field={fid_i} -> {out_dir_str}")
            results.append((fid_i, out_dir_str))
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=int(workers)) as ex:
            futs = {
                ex.submit(_run_one_field_worker, cfg_base_dict, int(fid), str(bundle_dir)): int(fid)
                for fid in fields
            }
            for fut in as_completed(futs):
                fid_i, out_dir_str = fut.result()
                print(f"[done] field={fid_i} -> {out_dir_str}")
                results.append((fid_i, out_dir_str))

    # From here on, bundle deterministically in ascending field order.
    results.sort(key=lambda x: x[0])

    # Buffers for NPZ aggregation
    bundle_npz_dict: dict[str, np.ndarray] = {}
    fields_done: list[int] = []

    for fid, out_dir_str in results:
        out_dir = Path(out_dir_str)

        # Append per-field CSVs into bundle CSVs.
        _append_csv(out_dir / "candidate_stats.csv", bundle_cand_csv)
        _append_csv(out_dir / "gpr_params.csv", bundle_gprp_csv)

        # Aggregate per-field seq_log.npz into a namespaced `GPR_bundle.npz`.
        seq_npz = out_dir / "seq_log.npz"
        if seq_npz.exists():
            d = _npz_to_dict(seq_npz)
            tag = f"field{int(fid):03d}"
            for k, arr in d.items():
                bundle_npz_dict[f"{k}__{tag}"] = arr
            fields_done.append(int(fid))
        else:
            print(f"[warn] seq_log.npz not found for field={fid}")

    # Save aggregated NPZ with a compact, consistent key scheme.
    bundle_npz_dict["fields"] = np.asarray(fields_done, dtype=np.int32)
    np.savez(bundle_dir / "GPR_bundle.npz", **bundle_npz_dict)
    print(f"[bundle] saved: {bundle_dir/'GPR_bundle.npz'}")

    print("[bundle] all done.")
    print("  dir :", bundle_dir)
    print("  csv :", bundle_cand_csv.name, ",", bundle_gprp_csv.name)
    print("  npz :", "GPR_bundle.npz")

    # ---- Final timing --------------------------------------------------------
    _end_t = perf_counter()
    _end_wall_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _elapsed = _end_t - _start_t
    print(f"[time] end:   {_end_wall_ts}  elapsed: {_elapsed:.3f}s")


if __name__ == "__main__":
    main()
