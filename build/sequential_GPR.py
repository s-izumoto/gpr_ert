"""
Sequential Gaussian Process Regression (GPR) for ERT design selection
====================================================================

This module implements a **sequential experimental design** loop using a
Gaussian Process surrogate to choose the next ERT (Electrical Resistivity
Tomography) measurement configuration (ABMN). It is intended to operate on a
precomputed dataset (NPZ) that contains candidate designs and their responses
(e.g., log10 apparent resistivity), filtered by a latent field code `Z` so the
active phase always queries the *same* field as the warm‑up.

High‑level flow
---------------
1. **Load field descriptor (`Z`)** and subset both the warm‑up NPZ and the full
   response NPZ (`y_dataset`) to the selected field.
2. **Enumerate allowed ABMN designs** (Wenner / Schlumberger / Dipole‑Dipole /
   Gradient) and intersect with the available designs in the dataset.
3. **Warm‑up fit:** build features (distance/position metrics), optionally map
   them through a design‑space metric transform (e.g., "cdf+whiten"), and fit a
   scikit‑learn `GaussianProcessRegressor` on the first `warmup_steps` samples.
4. **Active loop:** at each step, sample a candidate pool from the *remaining*
   designs, compute posterior mean/std and an acquisition score (LCB/UCB/EI/
   MAXVAR/MI), pick the best candidate, append its observation, and re‑fit the GP.
5. **Logging:** write per‑step GP diagnostics (kernel, log‑marginal likelihood,
   train RMSE/MAE/R², recovered length scales and noise level) to
   `gpr_params.csv`, and candidate‑pool summary statistics plus the selected
   ABMN and its features to `candidate_stats.csv`. The ABMN/feature/response
   history is also saved to `seq_log.npz` along with the used config
   (`config_used.json`).

Key ideas
---------
- **Design features:** we use [dAB, dMN, mAB, mMN] where d = dipole length in
  unit‑box coordinates and m = dipole midpoint; these separate *distance* from
  *position* and work well with separable kernels.
- **Separable kernel:** a pair of RBFs acting on distance vs position slices
  (sum or product), plus `WhiteKernel` noise. Length‑scales are recovered and
  exported on every step.
- **Acquisitions:** LCB, UCB, EI, MAXVAR, and Mutual Information (MI) are
  available. MI is implemented as `0.5 * log(1 + sigma^2 / noise^2)`.

Inputs / Outputs
----------------
Inputs are configured via `GPRSeqConfig`. The most important fields are:
- `pca_joblib` : path to PCA/joblib meta (optional; used to infer latent size)
- `Z_path`     : NPZ with `Z` array (field descriptor per row)
- `warmup_npz` : NPZ filtered by Z; must include keys `ABMN` and `y` (log10)
- `y_dataset`  : full NPZ with candidate responses (must include `Z` and `ABMN`
                 plus `y` or `rhoa`); used as the oracle in the active loop
- `out_dir`    : directory to write CSV/NPZ logs

Outputs:
- `{out_dir}/gpr_params.csv`
- `{out_dir}/candidate_stats.csv`
- `{out_dir}/seq_log.npz`
- `{out_dir}/config_used.json`

Usage
-----
    from sequential_GPR import GPRSeqConfig, run_from_cfg
    cfg = GPRSeqConfig(
        pca_joblib="pca_meta.joblib",
        Z_path="bundle_Z.npz",
        warmup_npz="warmup_field.npz",
        y_dataset="responses_all.npz",
        out_dir="gpr_seq_logs/run001",
        arrays=("wenner","schlumberger","dipole"),  # enable/disable here
        random_candidates=2048,
        warmup_steps=35,
        total_steps=155,
        acquisition="MI",
        kernel_compose="sum",
    )
    run_from_cfg(cfg, field_index=0)

Notes
-----
- ABMN indices are internally normalized to **1‑based** canonical form to avoid
  duplicates due to reciprocity/swap symmetries and to catch 0‑based inputs.
- The code assumes **maximization** of the response when using EI/UCB/MI; if
  your target is to be minimized, adapt the `y_best` logic accordingly.
- This file removes non‑essential legacy comments and replaces them with compact
  English explanations for clarity.
"""

# build/gpr_seq_core.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os, csv, json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, List, Optional, Dict

import numpy as np
from scipy.stats import norm

# sklearn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# local deps (user repo)
# ERTDesignSpace is used when available; otherwise we fall back to a simple builder.
import os, sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from build.design import ERTDesignSpace  # 標準経路
except Exception as e1:
    try:
        from .design import ERTDesignSpace  # type: ignore[import-not-found]
    except Exception as e2:
        raise ImportError(
            "Failed to import ERTDesignSpace. Check that:\n"
            f"Details:\n  e1={e1}\n  e2={e2}\n  REPO_ROOT={REPO_ROOT}\n  sys.path[0]={sys.path[0]}"
        )

# --------------------------
# Configuration (YAML-driven)
# --------------------------
@dataclass
class GPRSeqConfig:
    """ Dataclass holding all configuration parameters for the sequential GPR run. """
    # IO
    pca_joblib: str
    Z_path: str
    warmup_npz: str
    out_dir: str

    y_dataset: str
    y_dataset_col: str = "y"

    # Design space
    n_elec: int = 32
    min_gap: int = 1
    arrays: Tuple[str, ...] = ("wenner", "schlumberger", "dipole")
    dd_kmax: int = 6
    schl_a_max: int = 8
    grad_kmax: Optional[int] = None

    # GP / acquisition
    acquisition: str = "MI"    # LCB | UCB | EI | MAXVAR | MI
    kernel_compose: str = "sum" # sum | prod
    kappa: float = 2.0          # for LCB/UCB
    xi: float = 0.01            # for EI
    noise_level: float = 1e-4
    normalize_y: bool = True
    use_metric_feats: bool = True
    n_restarts: int = 10

    # Candidates / steps
    random_candidates: int = 2048
    warmup_steps: int = 35
    total_steps: int = 155

    # Run
    seed: int = 42
    log_every_step: bool = True

def _collect_kernel_hyperparams(kernel):
    """
    Recursively traverse a scikit-learn Kernel tree and collect:
      - RBF (including SliceKernel) length scales (supports ARD form)
      - WhiteKernel noise level

    Returns:
        ls_items : List[Tuple[str, np.ndarray]]
            Example: [("rbf[0,1]", array([.., ..])), ("rbf[2,3]", array([.., ..]))]
        noise : float or np.nan
            Noise variance extracted from WhiteKernel, or NaN if not found.
    """
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    ls_items = []
    noise = np.nan

    def walk(k):
        """Recursive helper function to explore nested kernel components."""
        nonlocal noise
        # Extract noise level if this component is a WhiteKernel
        if isinstance(k, WhiteKernel):
            try:
                noise = float(k.noise_level)
            except Exception:
                pass  # Continue even if extraction fails

        # Extract length scales if this component is an RBF or SliceKernel
        if isinstance(k, RBF):
            try:
                ls = np.ravel(k.length_scale)
                if hasattr(k, "idx"):
                    idx = np.atleast_1d(getattr(k, "idx"))
                    tag = f"rbf[{','.join(map(str, idx.tolist()))}]"
                else:
                    tag = "rbf"
                ls_items.append((tag, ls))
            except Exception:
                pass  # Skip on any unexpected structure

        # Recursively visit sub-kernels if they exist
        if hasattr(k, "k1") and hasattr(k, "k2"):
            walk(k.k1)
            walk(k.k2)

    walk(kernel)
    return ls_items, noise


# --------------------------
# Allowed-pairs builders (old-behavior)
# --------------------------
def _enumerate_wenner_pairs(n: int, min_gap: int = 1):
    """ Enumerate 1-based Wenner ABMN quadruples for n electrodes with a minimum electrode gap. Returns a list of (A,B,M,N). """
    pairs = []
    for a in range(min_gap, n//3 + 1):
        for A in range(1, n - 3*a + 2):
            B = A + a
            M = A + 2*a
            N = A + 3*a
            if len({A,B,M,N}) == 4:
                pairs.append((A,B,M,N))
    return pairs

def _enumerate_schlumberger_pairs(n: int, min_gap: int = 1, a_min: int = 1, a_max: int | None = None):
    """ Enumerate Schlumberger arrays with center symmetry using half-widths L (AB) and a (MN). Returns 1-based (A,B,M,N). """
    pairs = []
    L_max = (n-1)//2
    if a_max is None:
        a_max = L_max
    for L in range(min_gap, L_max+1):
        for a in range(max(a_min, min_gap), min(a_max, L)+1):
            for O in range(1+L, n-L+1):
                A, B, M, N = O-L, O+L, O-a, O+a
                if 1 <= A < B <= n and 1 <= M < N <= n and len({A,B,M,N})==4:
                    pairs.append((A,B,M,N))
    return pairs

def _enumerate_dipole_dipole_pairs(n: int, min_gap: int = 1, k_min: int = 1, k_max: int | None = None):
    """ Enumerate dipole–dipole arrays with |AB|=|MN|=a and spacing k*a between dipoles. Returns 1-based (A,B,M,N). """
    pairs = []
    if k_max is None:
        k_max = n  # 実質的には制約で絞られる
    for a in range(min_gap, n//2 + 1):
        for A in range(1, n - a + 1):
            B = A + a
            for k in range(k_min, k_max+1):
                M = B + k*a
                N = M + a
                if N <= n and len({A,B,M,N})==4:
                    pairs.append((A,B,M,N))
    return pairs

def _enumerate_gradient_pairs(n: int, mn_k_min: int = 1, mn_k_max: int | None = None):
    """ Enumerate simplified 'gradient' patterns by fixing (A,B) and sliding (M,N) with spacing k. Returns 1-based (A,B,M,N). """
    pairs = []
    if mn_k_max is None:
        mn_k_max = n
    for A in range(1, n-2):
        for B in range(A+2, n+1):
            for k in range(mn_k_min, mn_k_max+1):
                a = 1
                for M in range(A+1, B-1):
                    N = M + k*a
                    if N < B and N <= n and len({A,B,M,N})==4:
                        pairs.append((A,B,M,N))
    return pairs

def _build_allowed_pairs(n: int,
                         arrays: list[str],
                         min_gap: int = 1,
                         dd_kmax: int | None = None,
                         schl_a_max: int | None = None,
                         grad_kmax: int | None = None) -> list[tuple[int,int,int,int]]:
    """ Compose the union of enabled array families, canonicalize (reciprocity-aware), and deduplicate designs. Filters by min_gap and family-specific limits. """
    allowed = []
    aset = {a.lower() for a in arrays}
    if 'wenner' in aset:
        allowed += _enumerate_wenner_pairs(n, min_gap=min_gap)
    if 'schlumberger' in aset or 'schl' in aset:
        allowed += _enumerate_schlumberger_pairs(n, min_gap=min_gap, a_min=1, a_max=schl_a_max)
    if 'dipole' in aset or 'dipole-dipole' in aset or 'dd' in aset:
        allowed += _enumerate_dipole_dipole_pairs(n, min_gap=min_gap, k_min=1, k_max=dd_kmax)
    if 'gradient' in aset:
        allowed += _enumerate_gradient_pairs(n, mn_k_min=1, mn_k_max=grad_kmax)
    def _orient(p):
        A,B,M,N = p
        if A > B: A,B = B,A
        if M > N: M,N = N,M
        return (A,B,M,N)

    def _canon(p):
        A,B,M,N = _orient(p)
        dip1 = (A,B); dip2 = (M,N)
        if dip2 < dip1:
            dip1, dip2 = dip2, dip1
        return (dip1[0], dip1[1], dip2[0], dip2[1])

    uniq = []
    seen = set()
    for p in allowed:
        q = _canon(p)
        if q not in seen:
            uniq.append(q); seen.add(q)
    return uniq

# --------------------------
# Utilities
# --------------------------
def _to_unit_box(abmn_1based: np.ndarray, n_elec: int) -> np.ndarray:
    """ Convert 1-based ABMN indices to [0,1] unit-box coordinates given the total number of electrodes. """
    return (abmn_1based.astype(float) - 1.0) / float(n_elec - 1)

def _abmn_unit_to_dmm(x: np.ndarray) -> np.ndarray:
    """ Map unit-box ABMN to 4D feature vector [dAB, dMN, mAB, mMN] (dipole lengths and midpoints). """
    # x: [A, B, M, N] in [0,1]
    A, B, M, N = [float(v) for v in x]
    dAB = abs(B - A)
    dMN = abs(N - M)
    mAB = 0.5 * (A + B)
    mMN = 0.5 * (M + N)
    return np.array([dAB, dMN, mAB, mMN], dtype=np.float64)

def _canonical_pair(p):
    """ Return the canonical 1-based ABMN with reciprocity normalization so that (A,B) <= (M,N) lexicographically. """
    A,B,M,N = map(int, p)
    dip1 = (A,B); dip2 = (M,N)
    if dip2 < dip1:
        dip1, dip2 = dip2, dip1
    return (dip1[0], dip1[1], dip2[0], dip2[1])

class SliceKernel(RBF):
    """Wrap RBF to operate on selected columns (0-based)."""
    def __init__(self, length_scale, idx, length_scale_bounds=(1e-4, 1e3)):
        super().__init__(length_scale=length_scale, length_scale_bounds=length_scale_bounds)
        self.idx = np.atleast_1d(np.asarray(idx, dtype=int))

    def __call__(self, X, Y=None, eval_gradient=False):
        Xs = X[:, self.idx]
        Ys = None if Y is None else Y[:, self.idx]
        return super().__call__(Xs, Ys, eval_gradient=eval_gradient)

# feature scaler following your DSPACE transform
class MetricFeatsScaler:
    """ Feature mapper that applies the design-space's metric transform and standardizes features based on warm-up statistics. """
    def __init__(self, dspace):
        self.dspace = dspace
        self.mu = None; self.sd = None
    def _map(self, X):
        return np.vstack([self.dspace._transform_query(x).reshape(-1) for x in X]).astype(np.float64)
    def fit(self, X):
        X2 = self._map(X); self.mu = X2.mean(0, keepdims=True); self.sd = np.clip(X2.std(0, keepdims=True), 1e-8, None); return self
    def transform(self, X):
        X2 = self._map(X); return (X2 - self.mu)/self.sd

def _kernel_separable_rbf_white(noise_level: float, compose: str = "sum"):
    """ Build a separable kernel: RBF over distances + RBF over positions (sum or product) plus WhiteKernel noise. """
    k_dist = SliceKernel([1.0,1.0], idx=[0,1])
    k_pos  = SliceKernel([1.0,1.0], idx=[2,3])
    k_core = k_dist + k_pos if compose.lower() in ("sum","+","add") else k_dist * k_pos
    return k_core + WhiteKernel(noise_level=noise_level, noise_level_bounds=(1e-6, 1.0))

def _acquisition_values(cfg: GPRSeqConfig, mu: np.ndarray, sigma: np.ndarray, y_best: float, observed_noise_var: float | None = None) -> np.ndarray:
    """ Compute acquisition values for LCB, UCB, EI, MAXVAR, or MI given posterior mean/std and current best observation. """
    A = cfg.acquisition.upper()
    if A == "LCB":
        return mu - cfg.kappa * sigma
    if A == "UCB":
        return mu + cfg.kappa * sigma
    if A == "EI":
        z = (mu - y_best - cfg.xi) / (sigma + 1e-12)
        return (mu - y_best - cfg.xi) * norm.cdf(z) + sigma * norm.pdf(z)
    if A == "MAXVAR":
        return sigma**2
    if A == "MI":
        sn2 = float(observed_noise_var) if (observed_noise_var is not None) else cfg.noise_level
        sn2 = max(sn2, 1e-12)
        return 0.5*np.log1p((sigma**2)/sn2)
    raise ValueError(f"Unknown acquisition: {cfg.acquisition}")

# --------------------------
# Data loading (warmup filter by Z)
# --------------------------
def _load_Z_row(z_npz_path: Path, field_index: int, k_lat_limit: int | None = None) -> np.ndarray:
    """ Load a single latent field descriptor (Z-row) from an NPZ and optionally truncate to k_lat components. """
    d = np.load(z_npz_path, allow_pickle=True)
    Z = d["Z"].astype(np.float32)
    z = Z[field_index]
    return z[:k_lat_limit] if (k_lat_limit is not None) else z

def _subset_npz_by_Z(npz_path: Path, z_sel: np.ndarray, *, rtol=1e-5, atol=1e-7) -> dict:
    """ Subset an NPZ dict to rows whose Z matches the provided vector (within tolerances). """
    d = np.load(npz_path, allow_pickle=True)
    if "Z" not in d.files:
        raise RuntimeError(f"{npz_path} requires key 'Z' for field filtering")
    Z_all = d["Z"].astype(np.float32)
    m = np.all(np.isclose(Z_all, z_sel[None,:], rtol=rtol, atol=atol), axis=1)
    if not np.any(m):
        raise RuntimeError("No matching Z row in warmup NPZ (check PCA and field-index alignment).")
    return {k: (d[k][m] if (hasattr(d[k],'shape') and d[k].shape[0]==Z_all.shape[0]) else d[k]) for k in d.files}

# --------------------------
# Build candidates from DSPACE
# --------------------------
def _pick_from_pool(pairs: list[tuple[int,int,int,int]],
                    idx_pool: np.ndarray,
                    n_cand: int,
                    rng: np.random.Generator,
                    n_elec: int) -> tuple[np.ndarray, np.ndarray, list[tuple[int,int,int,int]]]:
    """ Randomly sample candidate indices from the available pool and return their raw features and ABMN tuples. """

    if idx_pool.size == 0:
        raise RuntimeError("No available candidates remain in the pool.")

    if n_cand >= idx_pool.size:
        idx_cand = idx_pool.astype(np.int32, copy=True)
    else:
        idx_cand = rng.choice(idx_pool, size=n_cand, replace=False).astype(np.int32)

    X_rows = []
    pairs_cand = []
    for gi in idx_cand:
        abmn = np.array(pairs[gi], dtype=float)
        x_dmm = _abmn_unit_to_dmm(_to_unit_box(abmn, n_elec))
        X_rows.append(x_dmm.astype(np.float32))
        pairs_cand.append(tuple(map(int, abmn.tolist())))

    X_cand_raw = np.vstack(X_rows).astype(np.float32)
    return idx_cand, X_cand_raw, pairs_cand

# --------------------------
# Main runner
# --------------------------
def run_from_cfg(cfg: GPRSeqConfig, *, field_index: int = 0) -> Path:
    """ Main entry point: performs warm-up GP fit, then an active-learning loop selecting designs by acquisition; writes CSV+NPZ logs and returns the output directory path. """
    out_dir = Path(cfg.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    k_lat = None
    try:
        from joblib import load as joblib_load
        meta = joblib_load(cfg.pca_joblib)
        if isinstance(meta, dict) and "components" in meta:
            k_lat = int(meta["components"].shape[0])
    except Exception as e:
        print(f"[warn] Could not load PCA meta from {cfg.pca_joblib}: {e}  (fallback: use full Z)")

    z_row = _load_Z_row(Path(cfg.Z_path), field_index, k_lat_limit=k_lat).astype(np.float32)

    def _canon(p: tuple[int,int,int,int]) -> tuple[int,int,int,int]:
        A,B,M,N = p
        if A > B: A,B = B,A
        if M > N: M,N = N,M
        d1, d2 = (A,B), (M,N)
        if d2 < d1:
            d1, d2 = d2, d1
        return (d1[0], d1[1], d2[0], d2[1])

    def _maybe_to_one_based(abmn: "np.ndarray", n_elec: int) -> "np.ndarray":
        abmn = abmn.astype(np.int64, copy=False)
        if abmn.min() == 0 or abmn.max() == (n_elec - 1):
            print("[warn] y_dataset ABMN appears 0-based. Converting to 1-based for lookup.")
            abmn = abmn + 1
        return abmn

    warm = _subset_npz_by_Z(Path(cfg.warmup_npz), z_row)
    ABMN = warm["ABMN"].astype(np.int64)
    ABMN = _maybe_to_one_based(ABMN, cfg.n_elec)   # ← 0→1補正を追加
    yall = warm["y"].astype(np.float32)
    if ABMN.shape[0] < cfg.warmup_steps or yall.shape[0] < cfg.warmup_steps:
        raise RuntimeError(f"Warmup data is shorter than {cfg.warmup_steps} rows.")

    data_view = _subset_npz_by_Z(Path(cfg.y_dataset), z_row)
    ABMN_all = data_view["ABMN"].astype(np.int64)
    ABMN_all = _maybe_to_one_based(ABMN_all, cfg.n_elec)  # ← 0→1補正を追加
    if cfg.y_dataset_col == "y":
        Y_all = data_view["y"].astype(np.float32)        # 既に log10(ρa)
    else:
        Y_all = np.log10(data_view["rhoa"].astype(np.float32))

    y_lut = { _canon(tuple(map(int, p))) : float(v) for p, v in zip(ABMN_all, Y_all) }
    print("[debug] y_lut size:", len(y_lut))
    print("[debug] has (1,11,21,31)?", _canon((1,11,21,31)) in y_lut)

    allowed_pairs = _build_allowed_pairs(
        n=cfg.n_elec,
        arrays=list(cfg.arrays),
        min_gap=cfg.min_gap,
        dd_kmax=cfg.dd_kmax,
        schl_a_max=cfg.schl_a_max,
        grad_kmax=cfg.grad_kmax,   # ← これだけ追加
    )

    allowed_pairs = [p for p in allowed_pairs if _canon(p) in y_lut]
    if not allowed_pairs:
        raise RuntimeError("No overlapping designs between allowed_pairs and y_dataset. "
                        "Align arrays/k-limits or regenerate y_dataset.")

    print("[debug] allowed_pairs after y_lut filter:", len(allowed_pairs))
    print("[debug] coverage example (first 3):",
        [ _canon(allowed_pairs[i]) in y_lut for i in range(min(3, len(allowed_pairs))) ])

    # ============ DIAG START: per-array overlap vs y_lut ============
    def _canon(p):
        A,B,M,N = map(int, p)
        if A > B: A,B = B,A
        if M > N: M,N = N,M
        d1, d2 = (A,B), (M,N)
        return (d1[0], d1[1], d2[0], d2[1]) if d1 <= d2 else (d2[0], d2[1], d1[0], d1[1])

    def _count_overlap(tag, pairs_raw, y_lut):
        pairs = { _canon(tuple(map(int,p))) for p in pairs_raw }
        hit   = sum(1 for q in pairs if q in y_lut)
        print(f"[diag] {tag:12s}  made={len(pairs):4d}  overlap={hit:4d}")

    n = cfg.n_elec
    min_gap = cfg.min_gap
    dd_kmax = cfg.dd_kmax
    schl_a_max = cfg.schl_a_max
    grad_kmax = getattr(cfg, "grad_kmax", None)

    pairs_w = _enumerate_wenner_pairs(n, min_gap=min_gap)
    pairs_s = _enumerate_schlumberger_pairs(n, min_gap=min_gap, a_min=1, a_max=schl_a_max)
    pairs_d = _enumerate_dipole_dipole_pairs(n, min_gap=min_gap, k_min=1, k_max=dd_kmax)
    pairs_g = _enumerate_gradient_pairs(n, mn_k_min=1, mn_k_max=grad_kmax) if grad_kmax else []

    print("[diag] n,min_gap,dd_kmax,schl_a_max,grad_kmax =", n, min_gap, dd_kmax, schl_a_max, grad_kmax)
    _count_overlap("wenner",         pairs_w, y_lut)
    _count_overlap("schlumberger",   pairs_s, y_lut)
    _count_overlap("dipole-dipole",  pairs_d, y_lut)
    if pairs_g:
        _count_overlap("gradient",   pairs_g, y_lut)

    import itertools as _it
    all_pairs = list(_it.chain(pairs_w, pairs_s, pairs_d, pairs_g))
    all_pairs_canon = { _canon(tuple(map(int,p))) for p in all_pairs }
    print("[diag] TOTAL made =", len(all_pairs_canon),
        "  TOTAL overlap =", sum(1 for q in all_pairs_canon if q in y_lut))
    # ============ DIAG END ==========================================

    dspace = ERTDesignSpace(
        n_elecs=cfg.n_elec,
        min_gap=cfg.min_gap,
        metric="cdf+whiten",
        allowed_pairs=allowed_pairs,
        debug_whiten=True,
        feature_names=["dAB","dMN","mAB","mMN"],
    )

    def _to_one_based_if_needed(p, n):
        return (p[0]+1, p[1]+1, p[2]+1, p[3]+1) if (min(p)==0 or max(p)==(n-1)) else p

    pairs: list[tuple[int,int,int,int]] = [
        _canon(_to_one_based_if_needed(tuple(map(int,p)), cfg.n_elec)) for p in dspace.pairs
    ]
    assert min(min(p) for p in pairs) >= 1, "pairs still contains 0-based!"

    avail: set[int] = set(range(len(pairs)))
    gi_by_abmn = {tuple(p): i for i, p in enumerate(pairs)}
    print("[debug] pairs(min,max) =", min(min(p) for p in pairs), max(max(p) for p in pairs))

    # --- Warmup set ---
    ABMN_warm = [ _canonical_pair(tuple(map(int, row))) for row in ABMN[:cfg.warmup_steps] ]
    y_warm = yall[:cfg.warmup_steps].astype(float)

    X_warm = []
    for p in ABMN_warm:
        x_dmm = _abmn_unit_to_dmm(_to_unit_box(np.array(p, dtype=float), cfg.n_elec))
        X_warm.append(x_dmm)
    X_warm = np.asarray(X_warm, dtype=np.float64)

    # optional metric feature mapping
    scaler = MetricFeatsScaler(dspace).fit(X_warm) if cfg.use_metric_feats else None
    X_feat = scaler.transform(X_warm) if scaler is not None else X_warm

    # --- Fit GP on warmup ---
    gp = GaussianProcessRegressor(
        kernel=_kernel_separable_rbf_white(cfg.noise_level, compose=cfg.kernel_compose),
        normalize_y=cfg.normalize_y,
        random_state=cfg.seed,
        n_restarts_optimizer=cfg.n_restarts,
    ).fit(X_feat, y_warm)

    
    # --- Logs ---
    params_csv = out_dir / "gpr_params.csv"
    cand_csv   = out_dir / "candidate_stats.csv"
    seq_npz    = out_dir / "seq_log.npz"

    # open once with line buffering to reduce I/O overhead
    f_params = open(params_csv, "w", newline="", encoding="utf-8", buffering=1)
    w_params = csv.writer(f_params)
    w_params.writerow([
        "field","step","phase","kernel","LML",
        "train_rmse","train_mae","train_r2",
        "length_scales","noise_level"
    ])

    f_cand = open(cand_csv, "w", newline="", encoding="utf-8", buffering=1)
    w_cand = csv.writer(f_cand)
    w_cand.writerow([
        "field","step","phase","n_candidates",
        "acq_mean","acq_std","acq_min","acq_max",
        "mu_mean","mu_std","mu_min","mu_max",
        "sd_mean","sd_std","sd_min","sd_max",
        "acq_best","mu_best","sd_best",
        "A","B","M","N",
        "dAB","dMN","mAB","mMN"
    ])
    
    def log_params(step, phase, gp, X, y, writer):
        mu = gp.predict(X, return_std=False)
        rmse = float(np.sqrt(mean_squared_error(y, mu)))
        mae  = float(np.mean(np.abs(y - mu)))
        r2   = float(r2_score(y, mu))

        k = gp.kernel_
        try:
            ls_items, noise = _collect_kernel_hyperparams(k)
        except Exception:
            ls_items, noise = [], np.nan

        if ls_items:
            parts = []
            for tag, arr in ls_items:
                vals = ",".join(f"{v:.6g}" for v in np.ravel(arr))
                parts.append(f"{tag}={vals}")
            ls_str = ";".join(parts)
        else:
            ls_str = ""

        writer.writerow([
            field_index, step, phase, str(gp.kernel_),
            float(getattr(gp, "log_marginal_likelihood_value_", np.nan)),
            rmse, mae, r2, ls_str, (float(noise) if np.isfinite(noise) else "")
        ])

    # warmup log
    if cfg.log_every_step:
        log_params(cfg.warmup_steps-1, "warmup", gp, X_feat, y_warm, w_params)

    # --- Active phase ---

    selected_canon = set(ABMN_warm)
    X_hist = [*X_warm]; y_hist = [*y_warm]; abmn_hist = [*ABMN_warm]

    active_steps = max(0, int(cfg.total_steps) - int(cfg.warmup_steps))
    for t in range(active_steps):
        print(f"[active] step={t+1}/{active_steps}")
        # candidates
        idx_pool = np.fromiter(avail, dtype=np.int32)
        idx_cand, X_cand_raw, pairs_cand = _pick_from_pool(pairs, idx_pool, cfg.random_candidates, rng, cfg.n_elec)

        X_cand = scaler.transform(X_cand_raw) if scaler is not None else X_cand_raw
        mu, sd = gp.predict(X_cand, return_std=True)

        y_best = float(np.max(y_hist))
        sn2 = float(getattr(getattr(gp.kernel_, "k2", None), "noise_level", cfg.noise_level))

        acq = _acquisition_values(cfg, mu, sd, y_best=y_best, observed_noise_var=sn2)

        best_i = int(np.argmax(acq)) if cfg.acquisition.upper() in ("UCB","EI","MAXVAR","MI") else int(np.argmin(acq))
        best_global_idx = int(idx_cand[best_i])
        p_best = _canonical_pair(pairs_cand[best_i])

        avail.discard(best_global_idx)
        selected_canon.add(p_best)

        abmn_hist.append(p_best)
        X_hist.append(X_cand_raw[best_i])

        if p_best not in y_lut:
            raise RuntimeError(f"Chosen design {p_best} not found in y_dataset. Check arrays/k-limits consistency.")

        y_obs = float(y_lut[p_best])
        y_hist.append(y_obs)

        # re-fit GP (incremental fit by re-calling fit for simplicity)
        X_all = np.asarray(X_hist, dtype=np.float64)
        X_all_feat = scaler.transform(X_all) if scaler is not None else X_all
        y_all = np.asarray(y_hist, dtype=float)
        gp.fit(X_all_feat, y_all)

        # logs
        if cfg.log_every_step:
            # candidate stats row
            def _stats(x): 
                x=np.asarray(x); 
                return (x.size,float(x.mean()),float(x.std()),float(x.min()),float(x.max())) if x.size else (0,np.nan,np.nan,np.nan,np.nan)
            n,acq_mean,acq_std,acq_min,acq_max = _stats(acq)
            n2,mu_mean,mu_std,mu_min,mu_max = _stats(mu)
            n3,sd_mean,sd_std,sd_min,sd_max = _stats(sd)
            
            A,B,M,N = p_best
            dmm_best = _abmn_unit_to_dmm(
                _to_unit_box(np.array([A,B,M,N], dtype=float), cfg.n_elec)
            ).astype(float)  # [dAB, dMN, mAB, mMN]

            w_cand.writerow([
                field_index, cfg.warmup_steps+t, "active", n,
                acq_mean, acq_std, acq_min, acq_max,
                mu_mean,  mu_std,  mu_min,  mu_max,
                sd_mean,  sd_std,  sd_min,  sd_max,
                float(acq[best_i]), float(mu[best_i]), float(sd[best_i]),
                int(A), int(B), int(M), int(N),
                float(dmm_best[0]), float(dmm_best[1]), float(dmm_best[2]), float(dmm_best[3])
            ])
            log_params(cfg.warmup_steps+t, "active", gp, X_all_feat, y_all, w_params)

    # Save sequence log
    np.savez_compressed(seq_npz, field=field_index, ABMN=np.asarray(abmn_hist, dtype=int), X=np.asarray(X_hist, dtype=np.float32), y=np.asarray(y_hist, dtype=np.float32))
    # close CSVs
    try:
        f_params.close()
    except Exception:
        pass
    try:
        f_cand.close()
    except Exception:
        pass
    (out_dir / "config_used.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    return out_dir