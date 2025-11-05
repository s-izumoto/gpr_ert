"""
invert_GPR.py — Batch ERT inversion with pyGIMLi (multi-field support, optional parallelism)

Overview
--------
This script loads an NPZ bundle containing ERT measurement series (ABMN designs and log10-apparent
resistivities) and runs pyGIMLi inversions to reconstruct subsurface resistivity for one or more
"fields" (i.e., independent series). It can process the first field found, a specific field, or all
fields, and optionally renders PNG figures (linear and log-colored) for quick inspection. Results
for each field are saved into a single compressed NPZ "bundle" that also stores mesh/world metadata.

Key features
------------
- Robust field-key detection: supports `y__field7`, `y__field007`, etc., and legacy single-series input
- Automatic base detection for ABMN indexing (0- or 1-based) with strict range checking
- Deterministic geometry/mesh creation (no randomness), safe for parallel processing
- Optional parallel execution for remaining fields while keeping a stable bundle layout/order
- Offscreen plotting for headless/HPC environments

Expected inputs (NPZ)
---------------------
- Multi-field mode: pairs of keys per field, e.g. `ABMN__field###` (int, Nx4) and `y__field###` (float, N)
- Legacy mode: `designs` or `ABMN` (int, Nx4) and `y` (float, N)
  * `y` must be log10(apparent resistivity in ohm·m).
  * ABMN indices may be 0- or 1-based; the script will normalize them.

Outputs
-------
- PNG images for the first field (or for all fields if requested)
- A compressed NPZ bundle containing:
  * Inverted cell resistivities per field: `inv_rho_cells__field###`
  * ABMN used and rhoa per field: `abmn__field###`, `rhoa__field###`
  * Min/max of inverted values: `cmin__field###`, `cmax__field###`
  * Mesh/world metadata (cell centers, extents)

CLI usage (examples)
--------------------
# invert the first field found and save default images + bundle:
python invert_GPR.py --npz path/to/seq_log.npz --out-dir ./outputs

# invert a specific field and put images at custom paths:
python invert_GPR.py --npz seq_log.npz --field-index 12 --out ./figs/inv_linear.png --out-log ./figs/inv_log.png --out-dir ./outputs

# invert all fields in parallel and save one bundle:
python invert_GPR.py --npz seq_log.npz --all-fields --out-dir ./outputs --bundle-out ./outputs/inversion_bundle_GPR.npz
"""

from __future__ import annotations
import os
# Use offscreen backends for headless/HPC environments
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
from pathlib import Path
from typing import Iterable, Any, Optional, Tuple, List, Dict
import re
import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed


def _invert_job(payload):
    """Child-process worker: run inversion for a single field and return arrays only (no figures).

    Notes
    -----
    - Suppresses internal BLAS threading for stability and to avoid oversubscription.
    - Rebuilds geometry/mesh deterministically from the provided parameters.
    - Converts log10(rhoa) to linear ohm·m before inversion.
    - Returns only the arrays needed for bundling; figures are produced in the main process.
    """
    # Kill internal BLAS/OpenMP threads (reproducibility & stability under multiprocessing).
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    # ---- unpack payload ----
    label          = payload["label"]
    designs        = payload["designs"]
    y              = payload["y"]
    n_elec         = payload["n_elec"]
    L_world        = payload["L_world"]
    margin         = payload["margin"]
    mesh_area      = payload["mesh_area"]

    # Geometry/mesh is rebuilt identically in each process for deterministic behavior.
    sensors, xs, dx_eff, L_inner = make_sensor_positions(n_elec, L_world, margin)
    mesh = build_mesh_world(
        L_world, payload["Lz"], sensors,
        dz_under=0.05,
        area=(mesh_area if mesh_area and mesh_area > 0 else None),
        quality=34
    )

    # Normalize ABMN indices to 0-based
    a, b, m, n, base = to_zero_based(designs, n_elec)

    # Convert log10(apparent resistivity) to linear ohm·m and avoid zeros
    rhoa = np.power(10.0, y).astype(float)
    rhoa = np.maximum(rhoa, 1e-12)

    # Build ERT scheme and compute geometric factors
    scheme = make_scheme(sensors, a, b, m, n)
    scheme.createGeometricFactors()

    # Inversion
    inv_arr, cmin, cmax, _mgr = invert_core(mesh, scheme, rhoa)

    # Return only the essentials (manager/figures are handled in the parent for the first field).
    res = {
        "label": label,
        "inv_arr": np.asarray(inv_arr, dtype=np.float64),
        "cmin": float(cmin),
        "cmax": float(cmax),
        # Store ABMN back in canonical (a,b,m,n) order (note: scheme stores fields separately)
        "abmn": np.stack([scheme['a'], scheme['b'], scheme['n'], scheme['m']], axis=1).astype(np.int32)[:, [0, 1, 3, 2]],
        "rhoa": np.asarray(rhoa, dtype=np.float64),
    }
    return res


# --------- pyGIMLi helpers (lazy import to keep top-level import light & fork-safe) ----------
def _import_pg():
    """Lazy import for pyGIMLi and ERT utilities."""
    import pygimli as pg
    import pygimli.meshtools as mt
    from pygimli.physics import ert
    return pg, mt, ert


def make_sensor_positions(n_elec: int, L_world: float, margin: float):
    """Create equally spaced sensor positions along the surface within [margin, L_world - margin].

    Returns
    -------
    sensors : list[pg.Pos]
        pyGIMLi positions for electrodes (y=0).
    xs : np.ndarray
        Electrode x-coordinates.
    dx_eff : float
        Effective electrode spacing.
    L_inner : float
        Usable line length after margins.
    """
    pg, _, _ = _import_pg()
    L_inner = L_world - 2.0 * margin
    if L_inner <= 0:
        raise ValueError(f"Margin too large relative to world length (L_world={L_world}, margin={margin}).")
    dx = L_inner / float(n_elec - 1)
    xs = margin + dx * np.arange(n_elec, dtype=np.float64)
    sensors = [pg.Pos(float(x), 0.0) for x in xs]
    return sensors, xs.astype(np.float64), float(dx), float(L_inner)


def build_mesh_world(
    L_world: float,
    Lz: float,
    sensors: Iterable[Any],
    dz_under: float = 0.05,
    area: float | None = None,
    quality: int = 34
):
    """Construct a 2D world mesh with a shallow sub-surface extension under electrodes.

    Parameters
    ----------
    dz_under : float
        Fraction of Lz to drop virtual nodes under each electrode to stabilize inversion near surface.
    area : float | None
        Target triangle area (if provided) to control mesh density.
    quality : int
        Triangle quality parameter passed to pyGIMLi mesher.

    Returns
    -------
    pg.Mesh
    """
    pg, mt, _ = _import_pg()
    world = mt.createWorld(start=[0.0, 0.0], end=[L_world, -Lz], worldMarker=True)
    # Add nodes at sensor x and slightly below for better near-surface behavior
    for p in sensors:
        world.createNode(p)
        world.createNode(pg.Pos(p[0], -dz_under * Lz))
    kwargs = dict(quality=quality)
    if isinstance(area, (int, float)) and area > 0:
        kwargs["area"] = float(area)
    return mt.createMesh(world, **kwargs)


def make_scheme(sensors: Iterable[Any], a: np.ndarray, b: np.ndarray, m: np.ndarray, n: np.ndarray):
    """Build a pyGIMLi DataContainerERT with ABMN indices and registered sensors."""
    pg, _, _ = _import_pg()
    data = pg.DataContainerERT()
    for p in sensors:
        data.createSensor(p)
    nrows = int(len(a))
    data.resize(nrows)
    data.add("a", a.astype(int))
    data.add("b", b.astype(int))
    data.add("m", m.astype(int))
    data.add("n", n.astype(int))
    return data


def invert_core(mesh, scheme, rhoa: np.ndarray):
    """Run pyGIMLi inversion and return (inv_arr, cmin, cmax, manager).

    Notes
    -----
    - Uses 3% relative error (`err=0.03`) and a moderate regularization (`lam=20`) with `robust=True`.
    - Returns min/max over inverted model for color scaling and the manager for plotting.
    """
    _, _, ert = _import_pg()
    mgr = ert.ERTManager(verbose=False)
    data = scheme.copy()
    data["rhoa"] = rhoa.astype(float)
    data["err"] = np.full_like(rhoa, 0.03, dtype=float)  # 3% relative error
    inv_res = mgr.invert(data, mesh=mesh, lam=20, robust=True, verbose=False)
    inv_arr = np.asarray(inv_res, dtype=float)
    cmin = float(np.nanmin(inv_arr))
    cmax = float(np.nanmax(inv_arr))
    return inv_arr, cmin, cmax, mgr


def _ensure_parent(path_like):
    """Create parent directories if missing (idempotent)."""
    Path(path_like).parent.mkdir(parents=True, exist_ok=True)


def save_images(mgr, inv_arr: np.ndarray, out_png_linear: str, out_png_log: Optional[str]):
    """Render and save (linear and optional log-scale) inversion result figures using mgr.showResult()."""
    cmin = float(np.nanmin(inv_arr))
    cmax = float(np.nanmax(inv_arr))

    _ensure_parent(out_png_linear)
    _ensure_parent(out_png_log)

    # Linear color figure
    fig, ax = plt.subplots(figsize=(6, 3))
    _ = mgr.showResult(ax=ax, cMin=cmin, cMax=cmax, cMap="Spectral_r", logScale=False)
    ax.set_title("Inverted resistivity (linear color)")
    fig.tight_layout()
    fig.savefig(out_png_linear, dpi=200)
    plt.close(fig)

    # Log color figure (optional)
    if out_png_log:
        fig, ax = plt.subplots(figsize=(6, 3))
        _ = mgr.showResult(ax=ax, cMin=max(cmin, 1e-12), cMax=cmax, cMap="Spectral_r", logScale=True)
        ax.set_title("Inverted resistivity (log color)")
        fig.tight_layout()
        fig.savefig(out_png_log, dpi=200)
        plt.close(fig)


# ---------------- utilities ----------------
def to_zero_based(designs: np.ndarray, n_elec: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Normalize ABMN indices to 0-based, supporting inputs that may be 0- or 1-based.

    Returns
    -------
    a, b, m, n : np.ndarray
        0-based integer indices.
    base : int
        Detected original base (0 or 1).
    """
    designs = np.asarray(designs, dtype=int)
    if designs.ndim != 2 or designs.shape[1] != 4:
        raise SystemExit(f"ABMN must have shape [T,4], got {designs.shape}")

    A, B, M, N = designs.T
    mn = designs.min()
    mx = designs.max()

    # Heuristics: detect base by inspecting min/max and whether any index equals n_elec (signals 1-based).
    if mn >= 0 and mx <= (n_elec - 1):
        base = 0
        a, b, m, n = A.copy(), B.copy(), M.copy(), N.copy()
    elif mn >= 1 and mx <= n_elec:
        base = 1
        a, b, m, n = A - 1, B - 1, M - 1, N - 1
    else:
        if (designs == n_elec).any():
            base = 1
            a, b, m, n = A - 1, B - 1, M - 1, N - 1
        else:
            base = 0 if mx <= (n_elec - 1) else 1
            if base == 1:
                a, b, m, n = A - 1, B - 1, M - 1, N - 1
            else:
                a, b, m, n = A.copy(), B.copy(), M.copy(), N.copy()

    # Final sanity check
    for name, arr in zip(["a", "b", "m", "n"], [a, b, m, n]):
        if arr.min() < 0 or arr.max() > (n_elec - 1):
            raise SystemExit(
                f"Index out of range after base detection for {name}: "
                f"min={arr.min()} max={arr.max()} n_elec={n_elec}"
            )
    return a.astype(int), b.astype(int), m.astype(int), n.astype(int), base


def find_field_indices(npz: np.lib.npyio.NpzFile) -> List[int]:
    """Collect field indices by scanning keys like `y__field*` and `ABMN__field*` (any zero-padding).

    Fallback: if not found but a `fields` array exists, use it.
    """
    pat = re.compile(r"^(?:y|ABMN)__field(\d+)$")  # \d+ => any digits; supports both y/ABMN
    fields: List[int] = []
    for k in npz.files:
        m = pat.match(k)
        if m:
            s = m.group(1)                  # '7', '007', '0012', ...
            v = int(s.lstrip("0") or "0")   # allow leading zeros; '000' -> 0
            fields.append(v)

    # De-duplicate and sort
    fields = sorted(set(fields))

    # Fallback: explicit 'fields' array
    if not fields and "fields" in npz.files:
        arr = np.asarray(npz["fields"]).reshape(-1).astype(int).tolist()
        fields = arr

    return fields


def _pick_field_key(npz: np.lib.npyio.NpzFile, base: str, field_idx: int) -> str:
    """Resolve a concrete key for the given base and field index.

    Tries in order:
      1) zero-padded 3-digit
      2) non-padded
      3) regex that allows arbitrary leading zeros
    """
    candidates = [f"{base}{field_idx:04d}", f"{base}{field_idx}"]
    for k in candidates:
        if k in npz.files:
            return k
    # Finally, allow 0*<idx> using regex
    pat = re.compile(rf"^{re.escape(base)}0*{field_idx}$")
    for k in npz.files:
        if pat.match(k):
            return k
    raise SystemExit(
        f"Missing key for base={base!r} field_idx={field_idx}. "
        f"Tried {candidates} and regex {pat.pattern}."
    )


def load_field_npz(npz: np.lib.npyio.NpzFile, field_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load ABMN (Nx4) and y (N,) for a specific field, raising on length mismatch."""
    k_abmn = _pick_field_key(npz, "ABMN__field", field_idx)
    k_y    = _pick_field_key(npz, "y__field",    field_idx)

    ABMN = np.asarray(npz[k_abmn]).astype(int)
    y    = np.asarray(npz[k_y]).astype(float)
    if ABMN.shape[0] != y.shape[0]:
        raise SystemExit(f"Mismatched lengths for field {field_idx}: ABMN={ABMN.shape[0]} vs y={y.shape[0]}")
    return ABMN, y


def legacy_single_series(npz: np.lib.npyio.NpzFile) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Support legacy NPZ: accept ('designs' or 'ABMN') + 'y'. Return None if not present."""
    keys = set(npz.files)
    if ("designs" in keys or "ABMN" in keys) and ("y" in keys):
        designs = np.asarray(npz["designs"] if "designs" in keys else npz["ABMN"]).astype(int)
        y = np.asarray(npz["y"]).astype(float)
        if designs.shape[0] != y.shape[0]:
            raise SystemExit(f"Mismatched lengths: designs/ABMN={designs.shape[0]} vs y={y.shape[0]}")
        return designs, y
    return None


# -------------- public API --------------
def run_inversion(
    npz_path: str | Path,
    out: str | None = None,
    out_log: str | None = None,
    out_dir: str | None = None,
    bundle_out: str | None = None,
    field_index: int | None = None,
    all_fields: bool = False,
    images_all: bool = False,
    n_elec: int = 32,
    dx_elec: float = 1.0,
    world_Lx: float = 31.0,
    margin: float = 3.0,
    nx_full: int = 400,
    nz_full: int = 100,
    mesh_area: float = 0.1,
    workers: int = 1,
) -> Path:
    """Main inversion pipeline.

    Steps
    -----
    1) Load NPZ and determine target fields (multi-field or legacy).
    2) Build geometry and mesh deterministically from CLI parameters.
    3) Invert the first field in the main process (also used for figure rendering).
    4) Invert remaining fields sequentially or in parallel (same bundle layout/order).
    5) Save a compressed NPZ bundle containing per-field results and mesh/world metadata.

    Parallelism
    -----------
    If `workers` is None/0/-1 or <1, the number of workers is chosen from:
      $SLURM_CPUS_PER_TASK  ->  $OMP_NUM_THREADS  ->  os.cpu_count()
    and capped by the remaining number of tasks.

    Returns
    -------
    Path
        Path to the saved bundle NPZ.
    """
    npz_path = Path(npz_path)
    z = np.load(npz_path, allow_pickle=False)

    # Detect whether this is a multi-field NPZ (preferred) or a legacy single-series NPZ.
    targets: List[Tuple[str, np.ndarray, np.ndarray, int]] = []
    field_ids = find_field_indices(z)
    if field_ids:
        if all_fields:
            chosen = field_ids
        elif field_index is not None:
            if field_index not in field_ids:
                raise SystemExit(f"--field-index {field_index} not found. Available fields: {field_ids}")
            chosen = [field_index]
        else:
            chosen = [field_ids[0]]
        for f in chosen:
            ABMN, y = load_field_npz(z, f)
            targets.append((f"field{f:04d}", ABMN, y, f))
    else:
        legacy = legacy_single_series(z)
        if legacy is None:
            raise SystemExit("NPZ is neither multi-field nor legacy('designs'/'ABMN' + 'y').")
        designs, y = legacy
        targets.append(("single", designs, y, -1))

    # Geometry / world extents
    if dx_elec and dx_elec > 0:
        L_world = 2.0 * margin + (n_elec - 1) * dx_elec
    else:
        L_world = float(world_Lx)
    hy = L_world / float(nx_full)
    Lz = hy * float(nz_full)

    sensors, xs, dx_eff, L_inner = make_sensor_positions(n_elec, L_world, margin)
    mesh = build_mesh_world(
        L_world, Lz, sensors, dz_under=0.05,
        area=(mesh_area if mesh_area and mesh_area > 0 else None),
        quality=34
    )

    # Extract world extents and cell centers (saved to the bundle for downstream plotting/analysis).
    xs_nodes = np.array([v[0] for v in mesh.nodes()], dtype=np.float32)
    zs_nodes = np.array([v[1] for v in mesh.nodes()], dtype=np.float32)
    xmin, xmax = float(xs_nodes.min()), float(xs_nodes.max())
    zmin, zmax = float(zs_nodes.min()), float(zs_nodes.max())
    L_world_eff = xmax - xmin
    Lz_eff = abs(zmax - zmin)
    cx = np.array([c.center()[0] for c in mesh.cells()], dtype=np.float32)
    cz = np.array([c.center()[1] for c in mesh.cells()], dtype=np.float32)
    cell_centers = np.stack([cx, cz], axis=1)

    bundle: Dict[str, np.ndarray] = {
        "cell_centers": cell_centers,
        "world_xmin": np.array([xmin], dtype=np.float64),
        "world_xmax": np.array([xmax], dtype=np.float64),
        "world_zmin": np.array([zmin], dtype=np.float64),
        "world_zmax": np.array([zmax], dtype=np.float64),
        "L_world":    np.array([L_world_eff], dtype=np.float64),
        "Lz":         np.array([Lz_eff], dtype=np.float64),
    }

    # Derive default image paths (keep single-thread behavior: by default, only the first field produces images).
    base_linear = Path(out).with_suffix("") if out else (Path(out_dir) / "inversion_first")
    base_log = Path(out_log).with_suffix("") if out_log else None

    # === Prepare execution: do the first field in the parent (keeps behavior identical to single-thread runs) ===
    first_label, first_designs, first_y, _ = targets[0]
    a0, b0, m0, n0, base0 = to_zero_based(first_designs, n_elec)
    rhoa0 = np.maximum(np.power(10.0, first_y).astype(float), 1e-12)
    scheme0 = make_scheme(sensors, a0, b0, m0, n0)
    scheme0.createGeometricFactors()
    inv_arr0, cmin0, cmax0, mgr0 = invert_core(mesh, scheme0, rhoa0)

    # Figures (by default, only for the first field; enabling --images-all does not affect NPZ content)
    if images_all:
        out_png_linear = Path(str(base_linear) + f"_{first_label}.png")
        out_png_log = Path(str(base_log) + f"_{first_label}.png") if base_log else (Path(out_dir) / f"inversion_log_{first_label}.png")
    else:
        out_png_linear = Path(str(base_linear) + ".png")
        out_png_log = Path(str(base_log) + ".png") if base_log else (Path(out_dir) / "inversion_first_log.png")
    save_images(mgr0, inv_arr0, str(out_png_linear), str(out_png_log))

    # Store the first field into the bundle (keep order identical to single-thread runs).
    suffix0 = f"__{first_label}"
    bundle[f"inv_rho_cells{suffix0}"] = np.asarray(inv_arr0, dtype=np.float64)
    bundle[f"cmin{suffix0}"]          = np.array([float(cmin0)], dtype=np.float64)
    bundle[f"cmax{suffix0}"]          = np.array([float(cmax0)], dtype=np.float64)
    bundle[f"abmn{suffix0}"]          = np.stack([scheme0['a'], scheme0['b'], scheme0['n'], scheme0['m']], axis=1).astype(np.int32)[:, [0, 1, 3, 2]]
    bundle[f"rhoa{suffix0}"]          = np.asarray(rhoa0, dtype=np.float64)

    # Parallelize the rest (if any). If workers == 1, it stays sequential for full determinism.
    rest = targets[1:]
    results = []

    n_total = len(targets)
    n_tasks = len(rest)
    if n_tasks > 0:
        # Auto-select worker count if unspecified; be conservative and never exceed the remaining task count.
        if workers in (None, 0, -1) or (isinstance(workers, int) and workers < 1):
            cpu_env = (
                os.environ.get("SLURM_CPUS_PER_TASK")
                or os.environ.get("OMP_NUM_THREADS")
                or os.cpu_count()
                or 1
            )
            try:
                n_cpu = int(cpu_env)
            except Exception:
                n_cpu = 1
            workers = max(1, min(n_cpu, n_tasks))   # <= remaining tasks; conservative

            # If you want to cap workers further, uncomment and set a limit:
            # workers = min(workers, 8)

        print(f"[info] Total fields: {n_total} (first handled in main)")
        print(f"[info] Remaining fields: {n_tasks}")
        print(f"[info] Using {workers} parallel worker(s) for inversion")

        if workers == 1:
            # Sequential processing (identical order to single-thread behavior)
            for (label, designs, y, _) in rest:
                payload = dict(
                    label=label, designs=designs, y=y,
                    n_elec=n_elec, L_world=L_world, margin=margin,
                    mesh_area=mesh_area, Lz=Lz,
                )
                results.append(_invert_job(payload))
        else:
            # Parallel processing
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = []
                for (label, designs, y, _) in rest:
                    payload = dict(
                        label=label, designs=designs, y=y,
                        n_elec=n_elec, L_world=L_world, margin=margin,
                        mesh_area=mesh_area, Lz=Lz,
                    )
                    futs.append(ex.submit(_invert_job, payload))
                for f in as_completed(futs):
                    results.append(f.result())

        # Sort by label to ensure deterministic insertion order identical to single-thread runs.
        results_sorted = sorted(results, key=lambda d: d["label"])
        # targets is also label-sorted, so the final bundle order matches sequential behavior.
        for d in results_sorted:
            suffix = f"__{d['label']}"
            bundle[f"inv_rho_cells{suffix}"] = d["inv_arr"]
            bundle[f"cmin{suffix}"]          = np.array([d["cmin"]], dtype=np.float64)
            bundle[f"cmax{suffix}"]          = np.array([d["cmax"]], dtype=np.float64)
            bundle[f"abmn{suffix}"]          = d["abmn"]
            bundle[f"rhoa{suffix}"]          = d["rhoa"]

    bundle_path = Path(bundle_out) if bundle_out else (Path(out_dir) / "inversion_bundle_GPR.npz")
    # Persist the bundle in a compressed NPZ format.
    np.savez_compressed(bundle_path, **bundle)
    print(f"[bundle saved] {bundle_path}")
    print("[all done]")
    return bundle_path


# -------------- CLI --------------
def _build_argparser():
    """Define CLI options. See the module top docstring for usage examples."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Path to a seq_log .npz file")
    ap.add_argument("--out", help="PNG path (linear color). If omitted, uses <out-dir>/inversion_first.png")
    ap.add_argument("--out-log", help="PNG path (log color). If omitted, uses <out-dir>/inversion_first_log.png")
    ap.add_argument("--out-dir", help="Directory for outputs (images and bundle if --bundle-out is not set)")
    ap.add_argument("--bundle-out", help="Output path for the bundled inversion NPZ (one file for all fields)")
    ap.add_argument("--field-index", type=int, help="Invert only the specified field index")
    ap.add_argument("--all-fields", action="store_true", help="Invert all detected fields")
    ap.add_argument("--images-all", action="store_true", help="Save images for every field instead of only the first")
    ap.add_argument("--n-elec", type=int, default=32, help="Number of electrodes")
    ap.add_argument("--dx-elec", type=float, default=1.0, help="Electrode spacing (controls world length if >0)")
    ap.add_argument("--world-Lx", type=float, default=31.0, help="World length used when --dx-elec <= 0")
    ap.add_argument("--margin", type=float, default=3.0, help="Margin at both ends of the line (meters)")
    ap.add_argument("--nx-full", type=int, default=400, help="Horizontal nominal grid (used to size Lz)")
    ap.add_argument("--nz-full", type=int, default=100, help="Vertical nominal grid (used to size Lz)")
    ap.add_argument("--mesh-area", type=float, default=0.1, help="Target triangle area for mesher (<=0: default)")
    return ap


def main():
    """Parse CLI args and dispatch to the inversion pipeline."""
    ap = _build_argparser()
    args = ap.parse_args()
    run_inversion(
        npz_path=args.npz,
        out=args.out,
        out_log=args.out_log,
        out_dir=args.out_dir,
        bundle_out=args.bundle_out,
        field_index=args.field_index,
        all_fields=args.all_fields,
        images_all=args.images_all,
        n_elec=args.n_elec,
        dx_elec=args.dx_elec,
        world_Lx=args.world_Lx,
        margin=args.margin,
        nx_full=args.nx_full,
        nz_full=args.nz_full,
        mesh_area=args.mesh_area,
    )


if __name__ == "__main__":
    main()
