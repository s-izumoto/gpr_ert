# gpr_sequential_design.py
# -*- coding: utf-8 -*-
"""
GPRを教師に使い、連続空間で獲得関数を最大化 → 離散デザインにスナップ → 測定 → 逐次更新
- 最初の35回は固定デザイン（毎回同じシーケンス）で測定する前提
- 以降は GPR + Acquisition (LCB/EI/UCB) で次デザインを選択

外部出力との接続:
- PCA: pca_ops.py (= build_pca_latent.py) の pca_joint.joblib と Z.npz を使用
- Warmup: ert_physics_forward_wenner.py の rows.npz（例）から ABMN と y を使用
"""
import os
# GUIを使わないバックエンドに固定
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # 念のため

import matplotlib
matplotlib.use("Agg", force=True)
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import numpy as np

# scikit-learn のGPR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.gaussian_process.kernels import Kernel
from design import map_uv_to_embed, ERTDesignSpace
import torch
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import norm
import argparse, sys
import matplotlib.pyplot as plt


# ====== ここだけあなたの環境に合わせてください ======
PCA_JOBLIB     = Path("../data/interim/pca/pca_joint.joblib")    # pca_ops.py が出力
PCA_Z_PATH     = Path("../data/interim/pca/Z.npz")               # 同上（単一ベクトルなら .npy でもOK）
WARMUP_NPZ     = Path("../data/interim/ert_wenner_subset/ert_surrogate_wenner.npz")       # ert_physics_forward_wenner.py の出力（例）
# ==== Reciprocity 一意化済みデザイン空間 ====
N_ELECS = 32
MIN_GAP = 1
DSPACE = ERTDesignSpace(n_elecs=N_ELECS, min_gap=MIN_GAP, metric="cdf+whiten")
# === ログ出力のグローバル変数（mainでセットする） ===
LOG_RUN_DIR     = None   # ラン用フォルダ
LOG_PARAMS_PATH = None   # gpr_params.csv
LOG_CAND_PATH   = None   # candidate_stats.csv
LOG_SEQLOG_PATH = None   # seq_log.npz

# =========================
# 設定
# =========================
@dataclass
class Config:
    bounds: Tuple[Tuple[float,float], Tuple[float,float], Tuple[float,float], Tuple[float,float]] = (
        (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)
    )
    acquisition: str = "LCB"
    kappa: float = 2.0
    xi: float = 0.01
    random_restarts: int = 16
    random_candidates: int = 4096
    noise_level: float = 1e-4
    normalize_y: bool = True
    warmup_steps: int = 35
    seed: int = 42

# === PCAで2D復元して自動保存（pca_ops.pyの出力仕様に準拠） ===
def _reconstruct_field_2d_from_pca__pcaops(field_idx: int, pca_joblib_path: "Path", z_path: "Path") -> np.ndarray:
    from joblib import load as joblib_load
    meta = joblib_load(pca_joblib_path)  # keys: mean, components(k,D), nz, nx, crop_frac, ...
    mean = np.asarray(meta["mean"], dtype=np.float32)            # (D,)
    comps = np.asarray(meta["components"], dtype=np.float32)     # (k, D)
    nz, nx = int(meta["nz"]), int(meta["nx"])
    k = comps.shape[0]

    Z = np.load(z_path, allow_pickle=True)["Z"]                  # (N, k) を想定（kはk_star）
    if not (0 <= field_idx < Z.shape[0]):
        raise IndexError(f"field_index {field_idx} is out of range [0, {Z.shape[0]-1}]")
    z = Z[field_idx, :k].astype(np.float32)                      # (k,)

    x_flat = (z @ comps + mean).astype(np.float32)               # (D,)
    if x_flat.size != nz * nx:
        raise RuntimeError(f"サイズ不一致: flat={x_flat.size} vs nz*nx={nz*nx} (PCA作成時の形状を確認)")
    arr2d = x_flat.reshape(nz, nx)
    return arr2d

def save_field_image_spectral(arr2d_log10, out_path, *, mode="linear", title=""):
    """
    arr2d_log10 : y = log10(rho) の2D配列
    mode:
      - "linear": 10**y を表示し、カラーも linear（ノーマライズなし）
      - "log":    y をそのまま表示（linear ノーマライズ）
    """

    if mode == "linear":
        data = 10.0**arr2d_log10          # 値は線形ρ
        vmin = float(np.nanpercentile(data, 1))
        vmax = float(np.nanpercentile(data, 99))
        cbar_label = "Resistivity (Ωm)"
        im_kwargs = dict(cmap="Spectral_r", vmin=vmin, vmax=vmax)  # ← linear
    elif mode == "log":
        data = arr2d_log10                 # 値は log10(ρ)
        vmin = float(np.nanpercentile(data, 1))
        vmax = float(np.nanpercentile(data, 99))
        cbar_label = "log10 Resistivity (Ωm)"
        im_kwargs = dict(cmap="Spectral_r", vmin=vmin, vmax=vmax)  # ← linear
    else:
        raise ValueError("mode must be 'linear' or 'log'")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 3.6), dpi=150)
    im = plt.imshow(data, origin="upper", **im_kwargs)
    cbar = plt.colorbar(im, shrink=0.9)
    cbar.set_label(cbar_label)
    if title:
        plt.title(title)
    plt.xlabel("x in m")
    plt.ylabel("Depth in m")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[save] {mode} image -> {out_path}")


def _extract_ls_noise_from_kernel(k):
    """
    合成カーネル k から
      - dist（idx=[0,1]）の length_scale 配列
      - pos  （idx=[2,3]）の length_scale 配列
      - noise_level（WhiteKernel）
    を見つけて返す。
    """
    dist_ls = None
    pos_ls  = None
    noise   = None

    def walk(node):
        nonlocal dist_ls, pos_ls, noise
        if isinstance(node, WhiteKernel):
            noise = float(node.noise_level)
            return
        if isinstance(node, SliceKernel):
            base = node.base
            # RBF 以外（Matern 等）にも対応：length_scale があれば拾う
            ls = getattr(base, "length_scale", None)
            if ls is not None:
                ls = np.atleast_1d(ls).astype(float)
                if np.array_equal(node.idx, np.array([0,1])):
                    dist_ls = ls
                elif np.array_equal(node.idx, np.array([2,3])):
                    pos_ls = ls
            return
        # 合成なら子を辿る
        if hasattr(node, "k1") and hasattr(node, "k2"):
            walk(node.k1); walk(node.k2)

    walk(k)
    return dist_ls, pos_ls, noise

class SliceKernel(Kernel):
    """
    base カーネルを X の一部次元（列） idx にだけ適用するラッパ。
    例: SliceKernel(RBF(...), idx=[0,1])  # dAB,dMN にだけ RBF を適用
    """
    requires_vector_input = True

    def __init__(self, base: Kernel, idx):
        self.base = base
        self.idx = np.atleast_1d(np.asarray(idx, dtype=int))

    # --- コア演算 ---
    def __call__(self, X, Y=None, eval_gradient=False):
        Xs = X[:, self.idx]
        Ys = None if Y is None else Y[:, self.idx]
        return self.base(Xs, Ys, eval_gradient=eval_gradient)

    def diag(self, X):
        return self.base.diag(X[:, self.idx])

    def is_stationary(self):
        return self.base.is_stationary()

    # --- ハイパーパラメータ最適化連携（委譲） ---
    @property
    def hyperparameters(self):
        return self.base.hyperparameters

    @property
    def theta(self):
        return self.base.theta

    @theta.setter
    def theta(self, theta):
        self.base.theta = theta

    @property
    def bounds(self):
        return self.base.bounds

    def clone_with_theta(self, theta):
        return SliceKernel(self.base.clone_with_theta(theta), self.idx)

    def get_params(self, deep=True):
        return {"base": self.base, "idx": self.idx}

    def set_params(self, **params):
        if "base" in params:
            self.base = params["base"]
        if "idx" in params:
            self.idx = np.atleast_1d(np.asarray(params["idx"], dtype=int))
        return self

    def __repr__(self):
        return f"SliceKernel(base={self.base!r}, idx={self.idx.tolist()})"

def _load_Z_row(z_npz_path: Path, field_idx: int, *, k_lat_limit: "int | None" = None) -> np.ndarray:
    """Z.npz から行 field_idx を返す（必要なら k_lat で列を切る）"""
    d = np.load(z_npz_path, allow_pickle=False)
    Z = d["Z"].astype(np.float32)
    if field_idx < 0 or field_idx >= Z.shape[0]:
        raise IndexError(f"--field-index {field_idx} is out of range (0..{Z.shape[0]-1})")
    z = Z[field_idx]
    if k_lat_limit is not None:
        z = z[:k_lat_limit]
    return z.ravel()

def _subset_npz_by_Z(npz_path: Path, z_sel: np.ndarray, *, rtol=1e-5, atol=1e-7) -> dict:
    """
    forward生成NPZ（warmup/active）を、選択Zと完全一致（近似一致）する行だけに絞る。
    生成側が各行に Z（またはZrep）を保存している前提。
    返り値は dict（各キー→サブセット配列）。
    """
    d = np.load(npz_path, allow_pickle=True)
    if "Z" not in d.files:
        raise RuntimeError(f"{npz_path} に Z がありません。forward 側で Z(or Zrep) を保存してください。")
    Z_all = d["Z"].astype(np.float32)
    m = np.all(np.isclose(Z_all, z_sel[None, :], rtol=rtol, atol=atol), axis=1)
    if not np.any(m):
        raise RuntimeError(f"{npz_path} に一致する Z の行が見つかりません（field-index / PCA と不整合）")
    return {k: (d[k][m] if hasattr(d[k], "shape") and d[k].shape[0]==Z_all.shape[0] else d[k]) for k in d.files}


# ====== Array enumerators (1-based indices) ======
def enumerate_wenner_pairs(n: int, min_gap: int = 1):
    pairs = []
    # spacing s between adjacent electrodes in the quartet; Wenner-α: (A, M=N-1, N, B) with equal spacing s
    # Pattern: A=i, M=i+s, N=i+2*s, B=i+3*s
    for s in range(min_gap, (n // 3) + 1):
        for A in range(1, n - 3*s + 2):  # inclusive upper bound => +1; python range stop is exclusive => +2
            M = A + s; N = A + 2*s; B = A + 3*s
            pairs.append((A, B, M, N))
    return pairs

def enumerate_schlumberger_pairs(n: int, min_gap: int = 1, a_min: int = 1, a_max: int | None = None):
    pairs = []
    # Schlumberger: current span 2*s centered around potentials; potentials half-span 'a' (1..s-1 by default)
    # Pattern: A=i, B=i+2*s, M=i+s-a, N=i+s+a
    for s in range(max(min_gap, 1), n//2 + 1):
        a_hi = (s-1) if a_max is None else min(a_max, s-1)
        for a in range(max(a_min, 1), a_hi + 1):
            for A in range(1, n - 2*s + 2):
                M = A + s - a; N = A + s + a; B = A + 2*s
                if 1 <= M < N <= n:
                    pairs.append((A, B, M, N))
    return pairs

def enumerate_dipole_dipole_pairs(n: int, min_gap: int = 1, k_min: int = 1, k_max: int | None = None):
    pairs = []
    # Dipole–Dipole: dipole length s for both AB and MN; k is the separation factor between dipoles
    # Pattern: A=i, B=i+s, M=i+(k+1)*s, N=i+(k+2)*s
    max_s = n // 4  # conservative cap so that k>=1 likely fits
    for s in range(max(min_gap,1), max_s + 1):
        k_hi = (n // s) - 3  # ensure N <= n -> i + (k+2)*s <= n with i>=1
        if k_max is not None:
            k_hi = min(k_hi, k_max)
        for k in range(max(k_min,1), k_hi + 1):
            for A in range(1, n - (k+2)*s + 2):
                B = A + s; M = A + (k+1)*s; N = A + (k+2)*s
                pairs.append((A, B, M, N))
    return pairs

def enumerate_gradient_pairs(n: int, mn_k_min: int = 1, mn_k_max: int | None = None):
    """
    Gradient 配列（1-based）:
      - 電流電極固定: A=1, B=n
      - MN は内部でスライドし、N = M + k
      - k は任意に指定（mn_k_min..mn_k_max）
    """
    A_fixed, B_fixed = 1, n
    pairs = []
    k_hi = (n - 2) if (mn_k_max is None) else min(mn_k_max, n - 2)
    for k in range(max(1, mn_k_min), k_hi + 1):
        m_lo = 2
        m_hi = (n - 1) - k
        if m_hi < m_lo:
            continue
        for M in range(m_lo, m_hi + 1):
            N = M + k
            if len({A_fixed, B_fixed, M, N}) != 4:
                continue
            pairs.append((A_fixed, B_fixed, M, N))
    return pairs


def build_allowed_pairs(
    n: int,
    arrays: list[str],
    min_gap: int = 1,
    dd_kmax: int | None = None,
    schl_a_max: int | None = None,
    grad_kmax: int | None = None,               # ← 追加
):
    allowed = []
    arrays_set = {a.lower() for a in arrays}
    if 'wenner' in arrays_set:
        allowed += enumerate_wenner_pairs(n, min_gap=min_gap)
    if 'schlumberger' in arrays_set or 'schl' in arrays_set:
        allowed += enumerate_schlumberger_pairs(n, min_gap=min_gap, a_min=1, a_max=schl_a_max)
    if 'dipole' in arrays_set or 'dipole-dipole' in arrays_set or 'dd' in arrays_set:
        allowed += enumerate_dipole_dipole_pairs(n, min_gap=min_gap, k_min=1, k_max=dd_kmax)
    if 'gradient' in arrays_set:                 # ← 追加
        allowed += enumerate_gradient_pairs(n, mn_k_min=1, mn_k_max=grad_kmax)

    # …以下は同じ（正規化して unique）
    def _canon(p):
        A,B,M,N = p
        dip1 = (A,B); dip2 = (M,N)
        if dip2 < dip1:
            dip1, dip2 = dip2, dip1
        return (dip1[0], dip1[1], dip2[0], dip2[1])
    uniq = sorted({_canon(p) for p in allowed})
    return uniq


# === NEW: metric features scaler & sigma calibration ===
class MetricFeatsScaler:
    """[dAB,dMN,mAB,mMN] -> DSPACE._transform_query -> 標準化(mu, sd)"""
    def __init__(self, dspace):
        self.dspace = dspace
        self.mu = None
        self.sd = None

    def _map(self, X):
        # X: [n,4] in [0,1]^4
        X2 = np.vstack([self.dspace._transform_query(x).reshape(-1) for x in X]).astype(np.float64)
        return X2

    def fit(self, X: np.ndarray):
        X2 = self._map(X)
        self.mu = X2.mean(axis=0, keepdims=True)
        self.sd = X2.std(axis=0, keepdims=True).clip(min=1e-8)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X2 = self._map(X)
        return (X2 - self.mu) / self.sd

def sigma_calibration_scale(y: np.ndarray, mu: np.ndarray, std: np.ndarray) -> float:
    """s = sqrt( E[(y-mu)^2] / E[std^2] )"""
    std_safe = np.maximum(std, 1e-12)
    num = np.mean((y - mu)**2)
    den = np.mean(std_safe**2)
    return 1.0 if den <= 0.0 else float(np.sqrt(num / den))

def apply_sigma_calibration(std: np.ndarray, s: float) -> np.ndarray:
    return std * max(1e-12, s)

def _log_gpr_per_step(
    step_idx: int,
    phase: str,                 # "warmup" or "active"
    X_raw: np.ndarray,          # これまでの [0,1]^4 特徴
    y_obs: np.ndarray,          # これまでの y
    cfg: Config,
    scaler: "MetricFeatsScaler | None",
    csv_path: "Path",
    gp: "GaussianProcessRegressor",   # ← fit 済みの gp
):
    # --- 特徴変換（fit は絶対にしない） ---
    if scaler is not None:
        X_feat = scaler.transform(X_raw)
    else:
        X_feat = X_raw.astype(np.float64)

    # --- 合成カーネルから dist/pos の length_scale と noise を抽出 ---
    k = gp.kernel_
    dist_ls, pos_ls, noise_var = _extract_ls_noise_from_kernel(k)

    # フォールバック（見つからない場合に NaN）
    dist_ls = np.array(dist_ls if dist_ls is not None else [np.nan, np.nan], dtype=float)
    pos_ls  = np.array(pos_ls  if pos_ls  is not None else [np.nan, np.nan], dtype=float)
    noise_var = float(noise_var if noise_var is not None else np.nan)

    # LML（最適化後の現在の theta を渡す）
    lml = float(gp.log_marginal_likelihood(gp.kernel_.theta))

    # --- in-sample 指標 ---
    mu_tr, _std_tr = gp.predict(X_feat, return_std=True)
    rmse = float(np.sqrt(mean_squared_error(y_obs, mu_tr)))
    mae  = float(mean_absolute_error(y_obs, mu_tr))
    r2   = float(r2_score(y_obs, mu_tr))

    # --- print ログ（確認用） ---
    print(f"[gpr] step={step_idx:02d} phase={phase} | kernel={k} | LML={lml:.3f}")
    print(f"[gpr]   dist length_scales={np.array2string(dist_ls, precision=4)}  "
          f"pos length_scales={np.array2string(pos_ls, precision=4)}  "
          f"noise_var={noise_var:.6g}")
    print(f"[gpr]   train RMSE/MAE/R2 = {rmse:.4g}/{mae:.4g}/{r2:.4f}")

    # --- CSV 追記 ---
    import csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "step","phase","n","kernel",
        "dist_ls0","dist_ls1","pos_ls0","pos_ls1",
        "noise_var","LML","train_rmse","train_mae","train_r2"
    ]
    row = [
        int(step_idx), phase, int(len(y_obs)), str(k),
        f"{dist_ls[0]:.8g}", f"{dist_ls[1]:.8g}",
        f"{pos_ls[0]:.8g}",  f"{pos_ls[1]:.8g}",
        f"{noise_var:.8g}", f"{lml:.8g}",
        f"{rmse:.8g}", f"{mae:.8g}", f"{r2:.8g}",
    ]
    write_header = (not csv_path.exists())
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)


def _log_candidate_stats(
    step_idx: int,
    phase: str,                 # "warmup" or "active"
    acq_vals: "np.ndarray | None",
    mu: "np.ndarray | None",
    std: "np.ndarray | None",
    chosen_idx: "int | None",
    csv_path: "Path",
):
    import csv
    def _stats(x):
        if x is None or len(x) == 0:
            return dict(n=0, mean=np.nan, std=np.nan, vmin=np.nan, vmax=np.nan)
        x = np.asarray(x, dtype=float).ravel()
        return dict(n=len(x), mean=float(np.mean(x)), std=float(np.std(x)),
                    vmin=float(np.min(x)), vmax=float(np.max(x)))

    S_acq = _stats(acq_vals)
    S_mu  = _stats(mu)
    S_sd  = _stats(std)
    chosen = dict(
        acq = float(acq_vals[chosen_idx]) if (acq_vals is not None and chosen_idx is not None) else np.nan,
        mu  = float(mu[chosen_idx])        if (mu is not None and chosen_idx is not None) else np.nan,
        sd  = float(std[chosen_idx])       if (std is not None and chosen_idx is not None) else np.nan,
    )

    print(
      f"[cand] step={step_idx:02d} phase={phase} | M={S_acq['n']} "
      f"| acq(mean/std/min/max)={S_acq['mean']:.4g}/{S_acq['std']:.4g}/{S_acq['vmin']:.4g}/{S_acq['vmax']:.4g} "
      f"| mu(mean/std/min/max)={S_mu['mean']:.4g}/{S_mu['std']:.4g}/{S_mu['vmin']:.4g}/{S_mu['vmax']:.4g} "
      f"| sd(mean/std/min/max)={S_sd['mean']:.4g}/{S_sd['std']:.4g}/{S_sd['vmin']:.4g}/{S_sd['vmax']:.4g} "
      f"| chosen(acq,mu,sd)=({chosen['acq']:.4g},{chosen['mu']:.4g},{chosen['sd']:.4g})"
    )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "step","phase","M",
        "acq_mean","acq_std","acq_min","acq_max",
        "mu_mean","mu_std","mu_min","mu_max",
        "sd_mean","sd_std","sd_min","sd_max",
        "chosen_acq","chosen_mu","chosen_sd"
    ]
    write_header = (not csv_path.exists())
    row = [
        int(step_idx), phase, int(S_acq["n"]),
        S_acq["mean"], S_acq["std"], S_acq["vmin"], S_acq["vmax"],
        S_mu["mean"],  S_mu["std"],  S_mu["vmin"],  S_mu["vmax"],
        S_sd["mean"],  S_sd["std"],  S_sd["vmin"],  S_sd["vmax"],
        chosen["acq"], chosen["mu"], chosen["sd"]
    ]
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header: w.writerow(header)
        w.writerow(row)

# ====== ert_physics_forward_wenner の小物を再利用（必要最小限） ======
# （直接importできる構成なら from ert_physics_forward_wenner import ... でもOK）
def _make_sensor_positions(n_elec: int, L_world: float, margin: float):
    dx = (L_world - 2.0*margin) / float(n_elec - 1)
    xs = margin + dx * np.arange(n_elec, dtype=np.float32)
    return xs.astype(np.float32), float(dx), float(L_world - 2.0*margin)



# =========================
# ユーティリティ
# =========================

def _kernel_separable_rbf_white(noise_level: float, compose: str = "sum"):
    """
    位置(mAB,mMN) と 距離(dAB,dMN) を分離したカーネル。
    compose: "sum" なら k = k_dist + k_pos, "prod" なら k = k_dist * k_pos
    """
    # 2次元ずつ別 RBF（各次元でARDさせたいなら length_scale=[1.0,1.0] を維持）
    k_dist = SliceKernel(
        RBF(length_scale=[1.0, 1.0], length_scale_bounds=(1e-4, 1e3)), idx=[0, 1]
    )
    k_pos  = SliceKernel(
        RBF(length_scale=[1.0, 1.0], length_scale_bounds=(1e-4, 1e3)), idx=[2, 3]
    )

    if compose.lower() in ("sum", "add", "+"):
        k_core = k_dist + k_pos       # どちらか片方が近ければ相関↑（和）
    elif compose.lower() in ("prod", "mul", "*"):
        k_core = k_dist * k_pos       # 両方が近いとき相関↑（積）
    else:
        raise ValueError(f"unknown compose mode: {compose}")

    return k_core + WhiteKernel(noise_level=noise_level, noise_level_bounds=(1e-6, 1.0))

def _to_unit_box(designs_int: np.ndarray, ne: int) -> np.ndarray:
    return (designs_int - 1) / float(ne - 1)

def _abmn_unit_to_dmm(x_unit: np.ndarray) -> np.ndarray:
    """
    x_unit: shape (4,) = [A,B,M,N] in [0,1] （_to_unit_boxで正規化済みのインデックス座標）
    return: [dAB, dMN, mAB, mMN] in [0,1]
    """
    A, B, M, N = map(float, x_unit)
    dAB = abs(B - A)
    dMN = abs(N - M)
    mAB = 0.5 * (A + B)
    mMN = 0.5 * (M + N)
    return np.array([dAB, dMN, mAB, mMN], dtype=np.float64)

def _acquisition_values(cfg: Config, mu: np.ndarray, sigma: np.ndarray, y_best: float,
                        observed_noise_var: float | None = None) -> np.ndarray:
    A = cfg.acquisition.upper()
    if A == "LCB":
        return mu - cfg.kappa * sigma
    elif A == "UCB":
        return mu + cfg.kappa * sigma
    elif A == "EI":
        z = (mu - y_best - cfg.xi) / (sigma + 1e-12)
        return (mu - y_best - cfg.xi) * norm.cdf(z) + sigma * norm.pdf(z)
    elif A == "MAXVAR":
        # 予測分散そのものを最大化（= 欲しいのは情報）
        return sigma**2
    elif A == "MI":
        # 相互情報量: 0.5*log(1 + sigma^2 / noise^2)
        # 優先: 学習済み WhiteKernel の noise_level → フォールバック: cfg.noise_level
        sn2 = float(observed_noise_var) if (observed_noise_var is not None) else cfg.noise_level
        sn2 = max(sn2, 1e-12)
        return 0.5 * np.log1p((sigma**2) / sn2)
    else:
        raise ValueError(f"Unknown acquisition: {cfg.acquisition}")

# =========================
# ロード系（ここを実装）
# =========================

def load_pca_field():
    """
    pca_joint.joblib & Z.npz から 1ケース分のZ(=k_lat,) を返す（--field-index に従う）
    """
    from joblib import load as joblib_load
    meta = joblib_load(PCA_JOBLIB)
    k_lat = int(meta["components"].shape[0])
    # --field-index を使用
    z_row = _load_Z_row(PCA_Z_PATH, args.field_index, k_lat_limit=k_lat)
    return z_row

def load_warmup_designs_and_observations() -> Tuple[List[Tuple[int,int,int,int]], Optional[np.ndarray]]:
    # ① PCA から選択Z
    field = load_pca_field()  # (k_lat,)
    # ② warmup NPZ を Z 一致でサブセット
    d = _subset_npz_by_Z(WARMUP_NPZ, np.asarray(field, dtype=np.float32))
    if "ABMN" not in d or "y" not in d:
        raise RuntimeError("Warmup npz に ABMN / y が見つかりません。")
    ABMN = d["ABMN"].astype(np.int32)
    y    = d["y"].astype(np.float32)

    N_WARMUP = 35
    if ABMN.shape[0] < N_WARMUP or y.shape[0] < N_WARMUP:
        raise RuntimeError(f"Warmup不足: ABMN={ABMN.shape[0]}, y={y.shape[0]} (< {N_WARMUP})")

    # 先頭35行採用（生成を固定順にしている前提）
    warm_idx = np.arange(N_WARMUP, dtype=int)
    abmn35 = [_canonical_pair(tuple(map(int, row))) for row in ABMN[warm_idx]]
    y35    = y[warm_idx].astype(np.float32)
    return abmn35, y35

# ============== スナップ ==============
def _canonical_pair(p):
    """(A,B,M,N) と (M,N,A,B) を同一視し，辞書順で小さい方を正規形として返す（1-based想定）"""
    A,B,M,N = map(int, p)
    dip1 = (A,B); dip2 = (M,N)
    if dip2 < dip1:
        dip1, dip2 = dip2, dip1
    return (dip1[0], dip1[1], dip2[0], dip2[1])

# ============== DSPACE サンプリング ==============
def _pick_dspace_candidates(selected_canon: set, n_cand: int, rng: np.random.Generator):
    """
    DSPACE（相反一意化済み）の全離散設計から、未選択の候補を最大 n_cand 抽出して返す。
    戻り値:
      idx_cand:   候補の DSPACE インデックス (np.ndarray[*,])
      X_cand: 候補の [0,1]^4 特徴（ABMN→unit→dmm で作成）
      pairs_cand: 候補の (A,B,M,N) タプルのリスト（1-based）
    """
    pairs = list(DSPACE.pairs)  # [(A,B,M,N), ...] 1-based
    avail_idx = [i for i, p in enumerate(pairs) if _canonical_pair(p) not in selected_canon]
    if len(avail_idx) == 0:
        raise RuntimeError("利用可能な DSPACE 候補が残っていません。")

    if len(avail_idx) <= n_cand:
        idx_cand = np.asarray(avail_idx, dtype=np.int32)
    else:
        idx_cand = rng.choice(avail_idx, size=n_cand, replace=False).astype(np.int32)

    # GPR の特徴量は dAB,dMN,mAB,mMN（ABMN→unit→dmm）を使う
    X_rows = []
    for i in idx_cand:
        x_unit = _to_unit_box(np.array(pairs[i], dtype=float), N_ELECS)     # [A,B,M,N] in [0,1]
        x_dmm  = _abmn_unit_to_dmm(x_unit)                                   # [dAB,dMN,mAB,mMN]
        X_rows.append(x_dmm.astype(np.float32))
    X_cand = np.stack(X_rows, axis=0).astype(np.float32)
    pairs_cand = [pairs[i] for i in idx_cand]
    return idx_cand, X_cand, pairs_cand

# 置換：_build_warmup_dataset_from_file
def _build_warmup_dataset_from_file(n_warmup: int = 35):
    """
    Returns:
      X_warm: [n,4] in [0,1]^4  （dAB/L, dMN/L, mAB/L, mMN/L）
      y_warm: [n]
      ABMN_warm: [n,4] （1-based, canonicalized）
      idx_warm: [n]    （元の行インデックス：今回は 0..n-1）
    """
    abmn_list, y = load_warmup_designs_and_observations()  # [(A,B,M,N)], y
    if len(abmn_list) < n_warmup or y.shape[0] < n_warmup:
        raise ValueError(f"Warmup不足: {len(abmn_list)} / {y.shape[0]} < {n_warmup}")

    ABMN_warm = []
    X_rows = []
    for p in abmn_list[:n_warmup]:
        p_can = _canonical_pair(p)                            # 正規形（1-based）
        ABMN_warm.append(p_can)
        x_unit = _to_unit_box(np.array(p_can, dtype=float), N_ELECS)   # [A,B,M,N] in [0,1]
        x_dmm  = _abmn_unit_to_dmm(x_unit)                            # → [dAB,dMN,mAB,mMN]
        X_rows.append(x_dmm.astype(np.float64))

    X_warm = np.stack(X_rows, axis=0).astype(np.float64)
    y_warm = y[:n_warmup].astype(np.float64)
    ABMN_warm = np.asarray(ABMN_warm, dtype=int)
    idx_warm = np.arange(n_warmup, dtype=int)
    return X_warm, y_warm, ABMN_warm, idx_warm

def _print_warmup_designs_with_y(ABMN_warm: np.ndarray, X_warm: np.ndarray, y_warm: np.ndarray):
    # 表示は ABMN_warm から Dnorm を再計算して行う（X_warm は参照しない）
    ne = N_ELECS
    L_world = 31.0
    margin  = 3.0
    xs, dx, L_inner = _make_sensor_positions(ne, L_world, margin)

    print("\n=== WARMUP designs (used) with measurements ===")
    print("#  k    A  B  M  N   s=M-A    [  dAB    dMN    mAB    mMN ]   y")
    for k, (abmn, yy) in enumerate(zip(ABMN_warm, y_warm), start=1):
        A,B,M,N = map(int, abmn)
        a,b,m,n = A-1,B-1,M-1,N-1
        s = M - A
        dAB = abs(xs[b] - xs[a]); dMN = abs(xs[n] - xs[m])
        mAB = 0.5*(xs[a] + xs[b]); mMN = 0.5*(xs[m] + xs[n])
        Dn  = np.array([dAB/L_inner, dMN/L_inner, (mAB-margin)/L_inner, (mMN-margin)/L_inner], dtype=np.float32)
        print(f"{k:2d}: ({A:2d},{B:2d},{M:2d},{N:2d})  s={s:<2d}    "
              f"[{Dn[0]:6.3f} {Dn[1]:6.3f} {Dn[2]:6.3f} {Dn[3]:6.3f}]   {yy: .6g}")

def _apply_snap_metric_to_feats(X: np.ndarray) -> np.ndarray:
    """
    X: [n,4] の [dAB,dMN,mAB,mMN] を DSPACE の距離変換と同じ写像で座標変換。
    返り値は float64（sklearnの安定性のため）。
    """
    from numpy import vstack
    X2 = vstack([DSPACE._transform_query(x).reshape(-1) for x in X]).astype(np.float64)
    # 推奨: 変換後に各次元を標準化（GPの最適化が安定）
    mu = X2.mean(axis=0, keepdims=True); sd = X2.std(axis=0, keepdims=True).clip(min=1e-8)
    return (X2 - mu) / sd


@dataclass
class GPRWarmupReport:
    kernel_str: str
    length_scales: np.ndarray
    noise_var: float
    log_marginal_likelihood: float
    train_rmse: float
    train_mae: float
    train_r2: float
    loocv_rmse: float
    loocv_mae: float
    loocv_r2: float
    loocv_mlpd: float
    calib_hit_68: float
    calib_hit_95: float
    sigma_scale: float
    dist_ls: np.ndarray
    pos_ls: np.ndarray

def _fit_gpr_warmup(X: np.ndarray, y: np.ndarray, *, noise_level: float, normalize_y: bool, seed: int) -> GaussianProcessRegressor:
    gp = GaussianProcessRegressor(
        kernel=_kernel_separable_rbf_white(noise_level, compose=args.kernel_compose),
        normalize_y=normalize_y,
        random_state=seed,
        n_restarts_optimizer=10,
    )
    gp.fit(X, y)
    return gp

def _interval_hits(y_true: np.ndarray, mu: np.ndarray, std: np.ndarray, z: float) -> float:
    lo = mu - z * std
    hi = mu + z * std
    return float(np.mean((y_true >= lo) & (y_true <= hi)))

def _rmse(y_true, y_pred):
    # 古い sklearn 互換のため squared=False を使わずに RMSE を計算
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def run_warmup_gpr_check(n_warmup: int = 35, *, noise_level: float = 1e-4, normalize_y: bool = True, seed: int = 0) -> GPRWarmupReport:
    X, y, ABMN_warm, idx_warm = _build_warmup_dataset_from_file(n_warmup=n_warmup)
    if args.use_metric_feats:
        scaler = MetricFeatsScaler(DSPACE).fit(X)  # X は dmm を想定
        X = scaler.transform(X)
    _print_warmup_designs_with_y(ABMN_warm, X, y)
    # 学習
    gpr = _fit_gpr_warmup(X, y, noise_level=noise_level, normalize_y=normalize_y, seed=seed)

    # パラメータ抽出
    k = gpr.kernel_

    # 分離カーネル（dist=[dAB,dMN], pos=[mAB,mMN]）とノイズを安全に抽出
    dls, pls, noise_var = _extract_ls_noise_from_kernel(k)

    # 表示やレポート用にまとめる（見つからなければ NaN を入れる）
    dls = np.array(dls if dls is not None else [np.nan, np.nan], dtype=float)
    pls  = np.array(pls  if pls  is not None else [np.nan, np.nan], dtype=float)
    length_scales = np.concatenate([dls, pls])  # [ls_dAB, ls_dMN, ls_mAB, ls_mMN]

    lml = float(gpr.log_marginal_likelihood(gpr.kernel_.theta))

    # in-sample
    mu_tr, std_tr = gpr.predict(X, return_std=True)
    train_rmse = _rmse(y, mu_tr)
    train_mae  = float(mean_absolute_error(y, mu_tr))
    train_r2   = float(r2_score(y, mu_tr))

    # LOOCV
    loo = LeaveOneOut()
    mu_cv = np.zeros_like(y, dtype=float)
    std_cv = np.zeros_like(y, dtype=float)
    for tr, te in loo.split(X):
        gpr_i = _fit_gpr_warmup(X[tr], y[tr], noise_level=noise_level, normalize_y=normalize_y, seed=seed)
        mu_i, std_i = gpr_i.predict(X[te], return_std=True)
        mu_cv[te[0]]  = mu_i[0]
        std_cv[te[0]] = max(1e-12, std_i[0])  # ゼロ除け

    loocv_rmse = _rmse(y, mu_cv)
    loocv_mae  = float(mean_absolute_error(y, mu_cv))
    loocv_r2   = float(r2_score(y, mu_cv))

    # 負の平均対数予測密度（MLPD）
    mlpd = -np.mean(norm.logpdf(y, loc=mu_cv, scale=std_cv))

    # === NEW: sigma post-calibration (LOOCVベース) ===
    s_warm = sigma_calibration_scale(y, mu_cv, std_cv)
    std_cv_cal = apply_sigma_calibration(std_cv, s_warm)

    # キャリブレーション後の命中率
    hit68 = _interval_hits(y, mu_cv, std_cv_cal, z=1.0)
    hit95 = _interval_hits(y, mu_cv, std_cv_cal, z=1.96)

    return GPRWarmupReport(
        kernel_str=str(gpr.kernel_),
        length_scales=length_scales,
        noise_var=noise_var,
        log_marginal_likelihood=lml,
        train_rmse=train_rmse,
        train_mae=train_mae,
        train_r2=train_r2,
        loocv_rmse=loocv_rmse,
        loocv_mae=loocv_mae,
        loocv_r2=loocv_r2,
        loocv_mlpd=float(mlpd),
        calib_hit_68=hit68,
        calib_hit_95=hit95,
        sigma_scale=float(s_warm),
        dist_ls=dls, 
        pos_ls=pls, 
    )

def _print_verify(tag: str, Z_block: np.ndarray, z_seq: np.ndarray):
    """サブセット済みブロック Z_block（形状 [n, k]）が z_seq と一致しているか簡易表示"""
    z0 = Z_block[0].ravel().astype(np.float32)
    z1 = np.asarray(z_seq, dtype=np.float32).ravel()
    k = min(z0.size, z1.size)
    abs_max = float(np.max(np.abs(z0[:k] - z1[:k])))
    rel_max = float(np.max(np.abs(z0[:k] - z1[:k]) / (np.abs(z1[:k]) + 1e-12)))
    ok = np.allclose(z0[:k], z1[:k], rtol=1e-5, atol=1e-7)
    print(f"[verify] {tag}  k={k}  abs_max={abs_max:.3e}  rel_max={rel_max:.3e}  match={ok}")
    if not ok:
        raise RuntimeError(f"{tag}: Z が一致しません。")


# =========================
# メイン
# =========================
def run_sequential_design(cfg: Config, ne: int, total_steps: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)

    field = load_pca_field()  # --field-index で選んだ Z（k_lat,）

    # 1) warmup（Wenner）を Z でサブセットして、直後にZ整合チェック
    warm_view = _subset_npz_by_Z(WARMUP_NPZ, np.asarray(field, dtype=np.float32))
    _print_verify("warmup-vs-seq", warm_view["Z"], field)

    # warmup の ABMN,y を先頭35本に揃える
    assert cfg.warmup_steps == 35
    ABMN_w = warm_view["ABMN"].astype(np.int32)
    y_w    = warm_view["y"].astype(np.float32)
    if ABMN_w.shape[0] < cfg.warmup_steps or y_w.shape[0] < cfg.warmup_steps:
        raise RuntimeError(f"Warmup不足: ABMN={ABMN_w.shape[0]} y={y_w.shape[0]} (< {cfg.warmup_steps})")

    warm_idx = np.arange(cfg.warmup_steps, dtype=int)
    warmup_designs = [_canonical_pair(tuple(map(int, row))) for row in ABMN_w[warm_idx]]
    y_warmup       = y_w[warm_idx]

    # 2) active（dataset=all 等）も Z でサブセットして、直後にZ整合チェック → y辞書
    data_view = _subset_npz_by_Z(Path(args.y_dataset), np.asarray(field, dtype=np.float32))
    _print_verify("dataset-vs-seq", data_view["Z"], field)

    ABMN_all = data_view["ABMN"].astype(np.int32)
    if args.y_dataset_col == "y":
        Y_all = data_view["y"].astype(np.float32)              # log10(rhoa)
    else:
        Y_all = np.log10(data_view["rhoa"].astype(np.float32)) # 必要ならrhoa→log10
    y_lut = { _canonical_pair(tuple(map(int, p))) : float(v) for p, v in zip(ABMN_all, Y_all) }


    assert total_steps >= cfg.warmup_steps

    designs_disc: List[Tuple[int,int,int,int]] = []   # すべて相反正規形で格納
    selected_canon = set()                            # 相反正規形で重複管理
    X_cont_list: List[np.ndarray] = []
    y_list: List[float] = []


    # --- Warm-up（相反正規形に統一してから格納） ---
    if len(warmup_designs) != cfg.warmup_steps:
        raise ValueError("warmup_designs の長さが 35 ではありません。")

    for t in range(cfg.warmup_steps):
        d_raw = warmup_designs[t]
        d_can = _canonical_pair(d_raw)                          # ← ここがポイント
        designs_disc.append(d_can)
        selected_canon.add(d_can)

        # ABMN([0,1]) → dmm([0,1]) に変換してから学習バッファに積む
        x_unit = _to_unit_box(np.array(d_can, dtype=float), ne)   # [A,B,M,N] in [0,1]
        x_dmm  = _abmn_unit_to_dmm(x_unit)                        # [dAB,dMN,mAB,mMN]
        X_cont_list.append(x_dmm.astype(np.float64))

        y = float(y_warmup[t])
        print(f"[warmup] t={t:02d} design={d_can} y={y:.6g}")
        y_list.append(y)

        # === NEW: warmup 各回直後のGPRパラメータ出力 ===
        if args.log_every_step:
            X_sofar = np.stack(X_cont_list, axis=0)
            y_sofar = np.asarray(y_list, dtype=float)

            # 特徴量スケール
            scaler = MetricFeatsScaler(DSPACE).fit(X_sofar) if args.use_metric_feats else None
            X_feat = scaler.transform(X_sofar) if scaler is not None else X_sofar.astype(np.float64)

            # ★ 一度だけfit
            gp = GaussianProcessRegressor(
                kernel=_kernel_separable_rbf_white(cfg.noise_level, compose=args.kernel_compose),
                normalize_y=cfg.normalize_y,
                random_state=cfg.seed,
                n_restarts_optimizer=10,
            )
            gp.fit(X_feat, y_sofar)

            # ★ 再fitしないロガーへ
            _log_gpr_per_step(
                step_idx=len(X_sofar)-1, phase="warmup",
                X_raw=X_sofar, y_obs=y_sofar,
                cfg=cfg, scaler=scaler,
                csv_path=LOG_PARAMS_PATH,
                gp=gp,
            )


    # --- Active（連続提案→スナップ→相反正規形→重複回避） ---
    kernel_prior = None
    while len(designs_disc) < total_steps:
        X = np.stack(X_cont_list, axis=0)
        y = np.asarray(y_list, dtype=float)

        # === NEW: metric features（学習セットでfit→以降は同じ変換を共有） ===
        if args.use_metric_feats:
            scaler = MetricFeatsScaler(DSPACE).fit(X)
            X_feat = scaler.transform(X)
        else:
            scaler = None
            X_feat = X.astype(np.float64)

        kernel_init = kernel_prior if (kernel_prior is not None) else _kernel_separable_rbf_white(cfg.noise_level, compose=args.kernel_compose) 

        gp = GaussianProcessRegressor(
            kernel=kernel_init,
            # kernel=_kernel_ard_rbf_white(cfg.noise_level),
            normalize_y=cfg.normalize_y,
            random_state=cfg.seed,
            n_restarts_optimizer=10,
        )
        gp.fit(X_feat, y)

        dist_ls, pos_ls, noise_var = _extract_ls_noise_from_kernel(gp.kernel_)
        obs_sn2 = float(noise_var) if (noise_var is not None) else None

        kernel_prior = gp.kernel_

        # === NEW: active 各回直後のGPRパラメータ出力（観測を追加する前のモデル状態） ===
        if args.log_every_step:
            X_sofar = X   # 学習に使った生特徴（unit boxの[0,1]^4）
            y_sofar = y

            _log_gpr_per_step(
                step_idx=len(X_cont_list)-1, phase="active",
                X_raw=X_sofar, y_obs=y_sofar,
                cfg=cfg, scaler=scaler,                      # ★ 学習時と同じscaler
                csv_path=LOG_PARAMS_PATH,
                gp=gp,                                       # ★ fit済みを渡す
            )

        # === NEW: in-sample 近似で σ後校正係数を推定 ===
        mu_tr, std_tr = gp.predict(X_feat, return_std=True)
        if args.sigma_calib == "scale":
            s_calib = sigma_calibration_scale(y, mu_tr, std_tr)
        else:
            s_calib = 1.0

        # DSPACE から未選択候補のサンプリング（従来どおり）
        # 予測 mu, sigma の計算が終わった後
        A = cfg.acquisition.upper()
        if A in ("LCB", "EI"):
            y_best = np.min(y)   # 最小化基準
        else:
            y_best = 0.0         # MAXVAR/MI では未使用（ダミー）

        idx_cand, X_cand, pairs_cand = _pick_dspace_candidates(
            selected_canon=selected_canon,
            n_cand=cfg.random_candidates,
            rng=rng
        )
        # Logging: candidate counts per iteration
        total_pairs = len(DSPACE.pairs)
        # ここで実際に残っている候補数を数える（selected_canon に入っていないもの）
        remaining = sum(1 for p in DSPACE.pairs if _canonical_pair(p) not in selected_canon)
        print(f"[active] step={len(designs_disc):02d} candidates: total={total_pairs} remaining={remaining} sampled={len(idx_cand)}")

        # === NEW: 候補にも同じ特徴変換を適用 ===
        if scaler is not None:
            X_cand_feat = scaler.transform(X_cand)
        else:
            X_cand_feat = X_cand.astype(np.float64)

        # 予測 + σ後校正
        mu, sigma = gp.predict(X_cand_feat, return_std=True)
        if args.sigma_calib == "scale":
            sigma = apply_sigma_calibration(sigma, s_calib)

        vals = _acquisition_values(cfg, mu, sigma, y_best, observed_noise_var=obs_sn2)
        best_local = np.argmin(vals) if cfg.acquisition.upper() == "LCB" else np.argmax(vals)

        # === NEW: 候補統計（M と acq/μ/σ の統計＋選択値） ===
        if args.log_every_step:
            _log_candidate_stats(
                step_idx=len(X_cont_list),   # 次に確定するインデックス
                phase="active",
                acq_vals=vals,
                mu=mu,
                std=sigma,
                chosen_idx=int(best_local),
                csv_path=LOG_CAND_PATH,
            )


        d_next = _canonical_pair(pairs_cand[best_local])

        designs_disc.append(d_next)
        selected_canon.add(d_next)

        x_unit_next = _to_unit_box(np.array(d_next, dtype=float), ne)   # [A,B,M,N] in [0,1]
        x_dmm_next  = _abmn_unit_to_dmm(x_unit_next)                    # [dAB,dMN,mAB,mMN]
        X_cont_list.append(x_dmm_next.astype(np.float64))
        if d_next not in y_lut:
            raise RuntimeError(f"指定NPZに選ばれた設計 {d_next} のyが見つかりません。NPZの配列（wenner/Schl/DDの範囲）とk上限を見直してください。")
        y_next = y_lut[d_next]

        print(f"[active] chosen={d_next} y={y_next:.6g}")
        y_list.append(float(y_next))

    return {
        "designs": np.array(designs_disc, dtype=np.int32),               # 1-based (A,B,M,N) の正規形
        "X_cont": np.stack(X_cont_list, axis=0).astype(np.float32),      # 学習に使った連続特徴
        "y": np.array(y_list, dtype=np.float32),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-warmup", action="store_true",
                        help="Use only the first 35 warm-up measurements to fit GPR and print diagnostics, then exit.")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--schl-a-max", type=int, default=None, help="Max half-span a for Schlumberger (defaults to s-1)"),
    parser.add_argument(
        "--dspace-metric",
        choices=["identity", "perdim", "whiten", "cdf", "cdf+whiten"],
        default="cdf+whiten",
        help="ERTDesignSpace の埋め込み距離メトリック"
    )

    # --log-every-step：デフォルトで有効
    parser.add_argument(
        "--log-every-step",
        action="store_true",
        default=True,   # ← 追加
        help="各ステップ（warmup/active）でGPRを再フィットしてパラメータを出力・CSV保存する"
    )

    # --use-metric-feats：デフォルトで有効
    parser.add_argument(
        "--use-metric-feats",
        action="store_true",
        default=True,   # ← 追加
        help="メトリック特徴量（距離等）へ変換してGPRに与える"
    )

    # --arrays：既定を 'wenner,schlumberger,dipole'
    parser.add_argument(
        "--arrays",
        type=str,
        default="wenner,schlumberger,dipole",   # ← 追加/変更
        help="使用する配列をカンマ区切りで指定"
    )

    # --dd-kmax：既定を 6
    parser.add_argument(
        "--dd-kmax",
        type=int,
        default=6,      # ← 追加/変更
        help="Dipole–Dipole の最大オフセット k"
    )

    # --sigma-calib：既定を 'scale'
    parser.add_argument(
        "--sigma-calib",
        type=str,
        choices=["none","scale","isotonic"],
        default="scale",    # ← 追加/変更
        help="予測分散の後校正方式"
    )

    parser.add_argument(
        "--y-dataset",
        type=str,
        default="../data/interim/ert_all_subset/ert_surrogate_all.npz",
        help="ert_physics_forward_all.py で作ったNPZ（ABMN と y を含む）"
    )
    parser.add_argument(
        "--y-dataset-col",
        choices=["y","rhoa"],
        default="y",
        help="NPZ内の列を指定（y=log10(rhoa) が既に入っている場合は y、生のrhoaなら rhoa を指定）"
    )

    parser.add_argument(
        "--kernel-compose",
        choices=["sum", "prod"],
        default="sum",
        help="位置と距離のサブカーネルの合成方法: sum(和) / prod(積)。デフォルト: sum"
    )

    parser.add_argument("--field-index", type=int, default = 0,
        help="PCA Z.npz から使うフィールドの行インデックス")
    
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./gpr_seq_logs",
        help="ログの親ディレクトリ（デフォルト: ./gpr_seq_logs）"
    )

    parser.add_argument(
        "--log-run-tag",
        type=str,
        default=None,
        help="ログのラン名（未指定なら日時+fieldで自動作成）"
    )


    args, _ = parser.parse_known_args()
    from datetime import datetime
    run_tag = args.log_run_tag or f'{datetime.now():%Y%m%d_%H%M%S}-field{int(args.field_index):03d}'
    log_root = Path(args.log_dir)
    log_run_dir = log_root / run_tag
    log_run_dir.mkdir(parents=True, exist_ok=True)

    
    LOG_RUN_DIR     = log_run_dir
    LOG_PARAMS_PATH = log_run_dir / f"gpr_params_field{int(args.field_index):03d}.csv"
    LOG_CAND_PATH   = log_run_dir / f"candidate_stats_field{int(args.field_index):03d}.csv"
    LOG_SEQLOG_PATH = log_run_dir / f"seq_log_field{int(args.field_index):03d}.npz"

    print(f"[log] run dir : {log_run_dir}")
    print(f"[log] params  : {LOG_PARAMS_PATH.name}")
    print(f"[log] cand    : {LOG_CAND_PATH.name}")
    print(f"[log] seqlog  : {LOG_SEQLOG_PATH.name}")

    # Build restricted array pool and override DSPACE with it
    arrays = [a.strip() for a in args.arrays.split(',') if a.strip()]
    allowed_pairs = build_allowed_pairs(N_ELECS, arrays, min_gap=MIN_GAP, dd_kmax=args.dd_kmax, schl_a_max=args.schl_a_max)
    # Recreate DSPACE with the same metric but filtered pairs
    DSPACE = ERTDesignSpace(n_elecs=N_ELECS, min_gap=MIN_GAP, metric=args.dspace_metric, allowed_pairs=allowed_pairs)

    if args.check_warmup:
        rep = run_warmup_gpr_check(
            n_warmup=35,
            noise_level=0.005,
            normalize_y=True,
            seed=args.seed,
        )
        print("\n=== GPR warm-up diagnostics (N=35) ===")
        print(f"* kernel: {rep.kernel_str}")
        print(f"* dist length_scales: {np.array2string(rep.dist_ls, precision=4)}")
        print(f"*  pos length_scales: {np.array2string(rep.pos_ls,  precision=4)}")
        print(f"* noise_var: {rep.noise_var:.6g}   LML: {rep.log_marginal_likelihood:.3f}")
        print(f"* train  RMSE/MAE/R2: {rep.train_rmse:.4g} / {rep.train_mae:.4g} / {rep.train_r2:.4f}")
        print(f"* LOOCV  RMSE/MAE/R2: {rep.loocv_rmse:.4g} / {rep.loocv_mae:.4g} / {rep.loocv_r2:.4f}")
        print(f"* LOOCV  MLPD: {rep.loocv_mlpd:.4f}")
        print(f"* Calibration hit rate: 68%→{100*rep.calib_hit_68:.1f}%   95%→{100*rep.calib_hit_95:.1f}%")
        print(f"* Sigma calibration scale (LOOCV): {rep.sigma_scale:.3f}")

        sys.exit(0)
    
        # --- PCA復元画像の自動保存（オプション不要、pca_ops仕様） ---
    try:
        arr2d = _reconstruct_field_2d_from_pca__pcaops(
            field_idx=int(args.field_index),           # 既存の --field-index をそのまま使用
            pca_joblib_path=PCA_JOBLIB,               # 例: Path("./pca_latent/pca_joint.joblib")
            z_path=PCA_Z_PATH                         # 例: Path("./pca_latent/Z.npz")
        )
        log_png = LOG_RUN_DIR / f"pca_field_logcolor_field{int(args.field_index):03d}.png"
        save_field_image_spectral(arr2d_log10=arr2d, out_path=log_png,
                                mode="log", title="Reconstructed log10 resistivity")
        print("[done] PCA field image saved:", log_png)

        arr2d_linear = np.power(10.0, arr2d, dtype=np.float64)
        lin_png = LOG_RUN_DIR / f"pca_field_linear_field{int(args.field_index):03d}.png"
        save_field_image_spectral(arr2d_log10=arr2d, out_path=lin_png,
                                mode="linear", title="Reconstructed resistivity (Ωm)")
        print("[done] PCA field (linear 10**y) image saved:", lin_png)

    except Exception as e:
        print(f"[warn] PCA field image export skipped: {e}")

    # 従来どおりの逐次デザイン実行
    cfg = Config(acquisition="MI", kappa=2.0, warmup_steps=35, seed=args.seed)
    # cfg = Config(acquisition="UCB", kappa=1.0, warmup_steps=35, seed=args.seed)
    log = run_sequential_design(cfg, ne=32, total_steps=155)
    np.savez_compressed(LOG_SEQLOG_PATH, **log)
    print("[done] saved:", LOG_SEQLOG_PATH)
