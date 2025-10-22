
"""
Toy model for Set Transformer + Cross-Attention Decoder + Evidential Beta–Bernoulli
with time-wise Bayesian pooling. Hyperparameters fixed to production choices.

Run a quick smoke test (single forward):  python toy_evidential_set_transformer.py --smoke
Run a short training:                     python toy_evidential_set_transformer.py --train --epochs 5
"""

import os, math, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from torch.special import digamma, gammaln
import time
import json
from pathlib import Path

# ---------------------------
# Fixed Hyperparameters (あなたの本番想定に合わせています)
# ---------------------------
@dataclass
class HParams:
    input_dim: int = 5                 # 4 design + 1 value
    grid_h: int = 25
    grid_w: int = 100
    n_pixels: int = 25 * 100
    warmup_fixed: int = 35             # first 32 measurements use a fixed pattern
    active_len: int = 120              # following variable-design length
    T: int = 120                       # (overwritten below as warmup_fixed + active_len)                      # history length
    min_meas_per_t: int = 1
    max_meas_per_t: int = 1
    pos_ratio: float = 0.20            # 上位20%が1

    # Set Transformer (Encoder)
    hidden_dim: int = 128

    # Cross-Attention Decoder
    dec_layers: int = 2
    dec_heads: int = 4
    query_dim: int = 35
    attn_dropout: float = 0.1

    # Evidential head
    head_hidden: int = 64
    eps: float = 1e-6

    # Time embedding
    time_embed_dim: int = 16

    # Optimization
    lr: float = 3e-4
    weight_decay: float = 1e-5
    batch_size: int = 8
    epochs: int = 5

    # Loss
    kl_lambda: float = 0.2
    topk_lambda: float = 0.1

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

HP = HParams()
HP.T = HP.warmup_fixed + HP.active_len
HP.device = "cpu"

# どこか HParams 定義の直後あたりに追加（安全チェック）
assert HP.min_meas_per_t == HP.max_meas_per_t, \
    "This code assumes a fixed number of measurements per time step (min == max)."
FIXED_L = HP.min_meas_per_t  # 各時刻の測定個数（固定）

def _load_field_source_idx(npz_path: str):
    """
    優先順: NPZ 埋め込みキー -> meta/shape から連番合成 -> <same_dir>/field_source_idx.npy
    戻り値: shape (n_fields,) の int64（各フィールドの “グローバルID相当”）
    """
    npz_path = str(npz_path)
    # (1) NPZ 内に埋め込みを最優先
    try:
        with np.load(npz_path, allow_pickle=True) as d:
            for k in ("field_source_idx", "field_idx", "field_ids"):
                if k in d.files:
                    return np.asarray(d[k], dtype=np.int64).reshape(-1)

            # (2) 埋め込みが無ければ連番を合成（reduced でも安定）
            meta = {}
            if "meta" in d.files:
                try:
                    meta = json.loads(d["meta"][0])
                except Exception:
                    meta = {}

            # R の推定
            R = None
            if "rows_per_field" in meta:
                try:
                    R = int(meta["rows_per_field"])
                except Exception:
                    R = None
            if R is None and "n_fields" in meta:
                try:
                    nF = int(meta["n_fields"])
                    N_rows = int(d["Dnorm"].shape[0]) if "Dnorm" in d.files else int(d["Z"].shape[0])
                    if nF > 0 and N_rows % nF == 0:
                        R = N_rows // nF
                except Exception:
                    R = None

            # n_fields の推定
            n_fields = None
            try:
                if "n_fields" in meta:
                    n_fields = int(meta["n_fields"])
                if (n_fields is None or n_fields <= 0) and R is not None:
                    N_rows = int(d["Dnorm"].shape[0]) if "Dnorm" in d.files else int(d["Z"].shape[0])
                    n_fields = N_rows // R
            except Exception:
                n_fields = None

            if n_fields is not None and n_fields > 0:
                # reduced では 0..n_fields-1 の連番で十分（決定的）
                return np.arange(n_fields, dtype=np.int64)
    except Exception:
        pass

    # (3) 最後に隣の .npy を試す
    npz_dir = os.path.dirname(os.path.abspath(npz_path))
    cand = os.path.join(npz_dir, "field_source_idx.npy")
    if os.path.isfile(cand):
        try:
            arr = np.load(cand, allow_pickle=True)
            return np.asarray(arr, dtype=np.int64).reshape(-1)
        except Exception:
            pass

    return None


def load_ert_npz(ert_npz_path):
    d = np.load(ert_npz_path, allow_pickle=True)
    Dnorm = np.asarray(d["Dnorm"], dtype=np.float32)  # [N_rows, 4]
    y     = np.asarray(d["y"],     dtype=np.float32)  # [N_rows]
    meta = {}
    if "meta" in d.files:
        try:
            meta = json.loads(d["meta"][0])
        except Exception:
            meta = {}

    # --- R: 1フィールドあたりの行数を決める ---
    R = None
    # (A) 通常版: n_AB * n_MN_per_AB
    if ("n_AB" in meta) and ("n_MN_per_AB" in meta):
        try:
            R = int(meta["n_AB"]) * int(meta["n_MN_per_AB"])
        except Exception:
            R = None
    # (B) Wenner 版: n_fields と総行数から推定
    if (R is None) and ("n_fields" in meta):
        try:
            n_fields = int(meta["n_fields"])
            N_rows   = int(Dnorm.shape[0])
            if n_fields > 0 and N_rows % n_fields == 0:
                R = N_rows // n_fields
        except Exception:
            R = None
    # (C) それでも決まらない場合、ABMN の行数と n_fields から推定
    if (R is None) and (("ABMN" in d.files) or ("ABMN_all" in d.files)) and ("n_fields" in meta):
        key = "ABMN" if "ABMN" in d.files else "ABMN_all"
        ABMN = np.asarray(d[key])
        N_rows = int(ABMN.shape[0])
        n_fields = int(meta.get("n_fields", 0))
        if n_fields > 0 and N_rows % n_fields == 0:
            R = N_rows // n_fields

    if R is None:
        raise RuntimeError(
            "Could not infer per-field row count R. meta keys: "
            + ", ".join(sorted(meta.keys()))
        )

    return Dnorm, y, meta, int(R)

class ERTOracleDataset(Dataset):
    """
    Uses real ERT surrogate (ert_physics_forward) for t >= warmup_fixed
    and oracle masks as labels.
    - Input per step: [Dnorm(4), y(=log10(rhoa))] -> 5D
    - Label: M_oracle (binary), flattened to [N_pixels]
    """
    def __init__(
        self,
        ert_npz_path: str,                 # ← 35回目以降(学習対象)
        oracle_npz_path: str,
        hp: HParams = HP,
        pick_scale_index: int = 0,
        max_fields: int | None = None,
        field_offset: int = 0,
        warmup_npz_path: str | None = None,    # ← 追加: 1〜35回目(履歴のみ)
        warmup_pick: str = "first35",           # "first35" / "stride"
        active_select: str = "random",   # "first" or "random"
        active_seed: int = 42
    ):
        super().__init__()
        self.hp = hp
        self.active_select = str(active_select).lower()
        self.active_seed = int(active_seed)

        # --- 固定測定数（min==max の前提を強制） ---
        assert hp.min_meas_per_t == hp.max_meas_per_t, \
            "ERTOracleDataset assumes a fixed number of measurements per time step (min == max)."
        FIXED_L = int(hp.min_meas_per_t)

        # 35回目以降に使うアクティブ設計の ERT（従来どおり）
        Dn_all_act, y_all_act, meta_act, R_act = load_ert_npz(ert_npz_path)  # :contentReference[oaicite:6]{index=6}

        # Oracle 読み込み（従来どおり）
        d_orc = np.load(oracle_npz_path, allow_pickle=True)
        M_oracle = np.asarray(d_orc["M_oracle"], dtype=np.float32)
        Hc = int(np.array(d_orc["Hc"]).item()); Wc = int(np.array(d_orc["Wc"]).item())
        Nf_total = int(M_oracle.shape[0]); S = int(M_oracle.shape[1])
        s_idx = int(min(max(0, pick_scale_index), S-1))

        # ==== 追加: ROI mask の取得（あれば使う。なければ全1） ====
        roi_mask_c = None
        if "roi_mask" in d_orc.files:
            # 典型: 2D (Hc, Wc) または 3D (S, Hc, Wc)。両方に対応
            raw = d_orc["roi_mask"]
            if raw.ndim == 2:
                roi_mask_c = np.asarray(raw, dtype=np.uint8)         # (Hc, Wc)
            elif raw.ndim == 3:
                roi_mask_c = np.asarray(raw[s_idx], dtype=np.uint8)  # (Hc, Wc)
        # フォールバック（全画素採用）
        if roi_mask_c is None:
            roi_mask_c = np.ones((Hc, Wc), dtype=np.uint8)

        # モデル解像度 (gh, gw) に合わせて ROI を最近傍リピートで拡大
        gh, gw = hp.grid_h, hp.grid_w
        if (Hc, Wc) != (gh, gw):
            rep_h = max(1, gh // Hc); rep_w = max(1, gw // Wc)
            roi_mask = np.kron(roi_mask_c, np.ones((rep_h, rep_w), dtype=np.uint8))[:gh, :gw]
        else:
            roi_mask = roi_mask_c
        # フラット index（True の場所）を保存
        self.roi_mask = roi_mask
        self.roi_idx = np.flatnonzero(roi_mask.reshape(-1) > 0).astype(np.int64)
        self.n_roi = int(self.roi_idx.size)

        # 使用フィールド範囲
        start_f = int(field_offset)
        end_f   = Nf_total if max_fields is None else min(Nf_total, start_f + int(max_fields))
        assert 0 <= start_f < end_f
        use_Nf  = end_f - start_f

        # === ここから 追加：field_source_idx の一致検証（寛容版） ===
        # アクティブ側
        fidx_act_full = _load_field_source_idx(ert_npz_path)
        # アクティブの n_fields を shape から計算（安全）
        n_fields_act = (Dn_all_act.shape[0] // R_act)
        if fidx_act_full is None:
            print("[warn] field_source_idx not found for active NPZ; synthesizing 0..n_fields-1")
            fidx_act = np.arange(n_fields_act, dtype=np.int64)
        else:
            fidx_act = np.asarray(fidx_act_full, dtype=np.int64).reshape(-1)
            if fidx_act.shape[0] != n_fields_act:
                print(f"[warn] active field_source_idx length={fidx_act.shape[0]} != n_fields={n_fields_act}; ignoring and synthesizing")
                fidx_act = np.arange(n_fields_act, dtype=np.int64)

        # ウォームアップ側（ある場合のみ）
        fidx_warm = None
        if warmup_npz_path is not None:
            fidx_warm_full = _load_field_source_idx(warmup_npz_path)
            # ウォームアップの n_fields を shape から計算（安全）
            try:
                Dn_all_warm_tmp, _, _, R_warm_tmp = load_ert_npz(warmup_npz_path)
                n_fields_warm = (Dn_all_warm_tmp.shape[0] // R_warm_tmp)
            except Exception:
                n_fields_warm = None

            if fidx_warm_full is None or (n_fields_warm is not None and len(fidx_warm_full) != n_fields_warm):
                print("[warn] field_source_idx not found or length mismatch for warmup NPZ; synthesizing 0..n_fields-1")
                if n_fields_warm is None:
                    # warmup の shape がまだ無い場合は active に合わせておく（後でスライスするので大丈夫）
                    n_fields_warm = n_fields_act
                fidx_warm = np.arange(n_fields_warm, dtype=np.int64)
            else:
                fidx_warm = np.asarray(fidx_warm_full, dtype=np.int64).reshape(-1)

        # 使用フィールド範囲のスライス
        act_slice  = fidx_act[start_f:end_f]
        if fidx_warm is not None:
            warm_slice = fidx_warm[start_f:end_f]
            if act_slice.shape != warm_slice.shape or not np.array_equal(act_slice, warm_slice):
                # ここは “警告だけ” 出して続行（決定的シード用の env_id は act を使う）
                msg = "(shape mismatch)" if act_slice.shape != warm_slice.shape else "(values differ)"
                print(f"[warn] field_source_idx mismatch between warmup and active for range [{start_f}:{end_f}) {msg}. Proceeding.")
        # === 一致検証ここまで（例外を投げない） ===


        # 行スライス：アクティブ分
        row_start = start_f * R_act
        row_end   = end_f   * R_act
        Dn_all_act = Dn_all_act[row_start:row_end]  # [use_Nf*R_act, 4]
        y_all_act  = y_all_act[row_start:row_end]   # [use_Nf*R_act]

        # 追加：ウォームアップ分（Wenner）
        if warmup_npz_path is not None:
            Dn_all_warm, y_all_warm, meta_w, R_warm = load_ert_npz(warmup_npz_path)
            # 同じフィールド順であることを仮定（必要なら field_source_idx.npy の一致確認を入れる）
            row_start_w = start_f * R_warm
            row_end_w   = end_f   * R_warm
            Dn_all_warm = Dn_all_warm[row_start_w:row_end_w]  # [use_Nf*R_warm, 4]
            y_all_warm  = y_all_warm[row_start_w:row_end_w]   # [use_Nf*R_warm]
        else:
            Dn_all_warm = None
            y_all_warm = None
            R_warm = 0

        # 出力解像度
        gh, gw = hp.grid_h, hp.grid_w
        self.n_pixels = gh * gw
        self.samples = []

        # ===== フィールドごとに 1 サンプル =====
        for i in range(use_Nf):
            # 教師（バイナリ）
            mo = M_oracle[start_f + i, s_idx]
            if (Hc, Wc) != (gh, gw):
                rep_h = max(1, gh // Hc); rep_w = max(1, gw // Wc)
                mo_resized = np.kron(mo, np.ones((rep_h, rep_w), dtype=np.float32))[:gh, :gw]
            else:
                mo_resized = mo
            y_full = mo_resized.astype(np.float32).reshape(-1)      # (gh*gw,)
            y_bin  = y_full[self.roi_idx]                           # (n_roi,)


            # === ここから置換: 固定L・マスクなし版 ===
            sets_T = []  # 各時刻の [L,5] を積み上げ → 最終的に [T,L,5] へ

            # ----- (1) ウォームアップ hp.warmup_fixed ステップ：履歴としてのみ使用 -----
            if Dn_all_warm is not None:
                row0_w = i * R_warm
                Dn_w = Dn_all_warm[row0_w: row0_w + R_warm]  # [R_warm, 4]
                y_w  = y_all_warm[row0_w: row0_w + R_warm]   # [R_warm]

                # 取り出し方：first / stride
                if hp.warmup_fixed <= Dn_w.shape[0]:
                    if warmup_pick == "stride":
                        idx = np.linspace(0, Dn_w.shape[0] - 1, hp.warmup_fixed, dtype=int)
                    else:
                        idx = np.arange(hp.warmup_fixed, dtype=int)
                else:
                    base = np.arange(Dn_w.shape[0], dtype=int)
                    pad  = np.full(hp.warmup_fixed - Dn_w.shape[0], Dn_w.shape[0] - 1, dtype=int)
                    idx  = np.concatenate([base, pad], axis=0)

                for t in idx:
                    v5 = np.concatenate([Dn_w[t], np.array([y_w[t]], dtype=np.float32)], axis=0)  # [5]
                    # 固定Lにするため v5 を L 回複製 → [L,5]
                    sets_T.append(np.repeat(v5[None, :], FIXED_L, axis=0).astype(np.float32))
            else:
                # ウォームアップデータが無い場合のダミー
                v5 = np.array([0.5, 0.02, 0.06, 0.0, 0.0], dtype=np.float32)
                dummy_L5 = np.repeat(v5[None, :], FIXED_L, axis=0)  # [L,5]
                for _ in range(hp.warmup_fixed):
                    sets_T.append(dummy_L5.copy())

            # ----- (2) アクティブ hp.active_len ステップ：学習対象 -----
            if 'R_act' not in locals() or R_act <= 0:
                raise ValueError(f"[ERTOracleDataset] Invalid R_act={R_act}. Active split is empty or undefined.")

            row0 = i * R_act  # このフィールドの先頭行（アクティブ部）
            all_idx = np.arange(R_act, dtype=int)

            # 環境Zごとに決定的な RNG（field_source_idx を使う）
            env_id = int(act_slice[i]) if 'act_slice' in locals() else i
            rng = np.random.RandomState(self.active_seed ^ env_id)

            if self.active_select == "random":
                if R_act >= hp.active_len:
                    pick_idx = rng.choice(all_idx, size=hp.active_len, replace=False)
                else:
                    reps = (hp.active_len + R_act - 1) // R_act  # 切り上げ
                    seq = np.concatenate([rng.permutation(all_idx) for _ in range(reps)], axis=0)
                    pick_idx = seq[:hp.active_len]
            else:
                take = min(hp.active_len, R_act)
                base = np.arange(take, dtype=int)
                if take < hp.active_len:
                    pad = np.full(hp.active_len - take, take - 1, dtype=int)
                    pick_idx = np.concatenate([base, pad], axis=0)
                else:
                    pick_idx = base

            Dn_i = Dn_all_act[row0 + pick_idx]  # [active_len, 4]
            yi   = y_all_act [row0 + pick_idx]  # [active_len]

            for t in range(hp.active_len):
                v5 = np.concatenate([Dn_i[t], np.array([yi[t]], dtype=np.float32)], axis=0)  # [5]
                sets_T.append(np.repeat(v5[None, :], FIXED_L, axis=0).astype(np.float32))     # [L,5]

            # --- 最終的に [T,L,5] に積む ---
            sets_arr = np.stack(sets_T, axis=0)  # [T, L, 5]

            # ここでは numpy のまま保持して __getitem__ で Tensor 化してもOKだし、
            # 直ちに Tensor 化してもOK（好みで）。混在しないよう統一してください。
            self.samples.append((
                torch.from_numpy(sets_arr),                       # [T,L,5]
                torch.from_numpy(y_bin.astype(np.float32))        # [n_roi]
            ))
            # === 置換ここまで ===

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 旧: return (sets_list, masks_list, y_bin)
        return self.samples[idx]  # (sets:[T,L,5], y:[n_roi])


def collate_fn(batch):
    """
    batch: list of (sets, y) where
      sets: [T, L, 5], y: [N] (ROIピクセル数)
    return:
      sets: [B, T, L, 5], y: [B, N]
    """
    sets = torch.stack([b[0] for b in batch], dim=0)  # [B,T,L,5]
    y    = torch.stack([b[1] for b in batch], dim=0)  # [B,N]
    return sets, y

class DeepSetsEncoder(nn.Module):
    """
    Deep Sets (Zaheer+ 2017) の最小実装:
      h_i = phi(x_i)
      H   = AGG_i h_i   (# 集合プーリング: mean/sum)
      z   = rho(H)
    返り値は [B, D]（= コンテキストベクトル）で、元の SetEncoder と同じ形に揃えています。
    """
    def __init__(self, in_dim, hidden_dim=128, agg="mean"):
        super().__init__()
        self.agg = str(agg).lower()
        assert self.agg in ("mean", "sum")

        # φ: 各要素に同じMLPを適用（要素ごと）
        self.phi = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # ρ: 集合プール後のまとめMLP（出力次元は hidden_dim のままにして decoder と整合）
        self.rho = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, X, mask=None):
        """
        X: [B, L, in_dim]
        mask: [B, L] (True=有効) 省略可
        return: [B, hidden_dim]
        """
        H = self.phi(X)  # [B, L, D]

        if mask is None: # if there is no missing data
            if self.agg == "mean":
                pooled = H.mean(dim=1)
            else:
                pooled = H.sum(dim=1)
        else:
            M = mask.to(H.dtype).unsqueeze(-1)  # [B,L,1]
            if self.agg == "mean":
                denom = M.sum(dim=1).clamp(min=1.0)
                pooled = (H * M).sum(dim=1) / denom
            else:
                pooled = (H * M).sum(dim=1)

        z = self.rho(pooled)  # [B, D]
        return z


class FiLMDecoder(nn.Module):
    """
    座標を特徴へ写像し、文脈 z で FiLM(Feature-wise Linear Modulation) 条件付け。
    return: [B, N, hidden_dim]
    """
    def __init__(self, query_dim, ctx_dim, hidden_dim=128, dropout=0.0):
        super().__init__()
        self.query_proj = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.film = nn.Linear(ctx_dim, 2*hidden_dim)  # -> (gamma, beta)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ff  = nn.Sequential(
            nn.Linear(hidden_dim, 4*hidden_dim), nn.GELU(),
            nn.Linear(4*hidden_dim, hidden_dim)
        )
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, pixel_queries, ctx_vec):
        """
        pixel_queries: [B, N, Dq],  ctx_vec: [B, Dc]
        returns: [B, N, H]
        """
        B, N, _ = pixel_queries.shape
        q = self.query_proj(pixel_queries)          # [B,N,H]
        gamma, beta = self.film(ctx_vec).chunk(2, dim=-1)  # [B,H], [B,H]
        gamma = gamma.unsqueeze(1).expand(B, N, -1)         # [B,N,H]
        beta  = beta.unsqueeze(1).expand(B, N, -1)          # [B,N,H]

        h = gamma * q + beta                        # FiLM
        h = self.drop(h)
        h = self.ln1(h)
        h = self.ln2(h + self.ff(h))                # 残差 + FFN
        return h

# ---------------------------
# Evidential Head（α, β）
# ---------------------------
class EvidentialHead(nn.Module):
    def __init__(self, in_dim, eps=1e-6):
        super().__init__()
        self.a = nn.Linear(in_dim, 1)
        self.b = nn.Linear(in_dim, 1)
        self.eps = eps
    def forward(self, U):
        alpha = F.softplus(self.a(U)) + 1.0 + self.eps
        beta  = F.softplus(self.b(U)) + 1.0 + self.eps
        return alpha.squeeze(-1), beta.squeeze(-1)

# ---------------------------
# Full Model（時刻ごとforward → 後段で時間方向sum）
# ---------------------------
class ToyModel(nn.Module):
    def __init__(self, hp: HParams, roi_idx: np.ndarray | None = None):
        super().__init__()
        self.hp = hp
        self.encoder = DeepSetsEncoder(
            in_dim=hp.input_dim, hidden_dim=hp.hidden_dim, 
            agg="mean"
        )

        self.decoder = FiLMDecoder(
            query_dim=hp.query_dim, ctx_dim=hp.hidden_dim,
            hidden_dim=hp.hidden_dim, dropout=hp.attn_dropout
        )

        self.head = EvidentialHead(in_dim=hp.hidden_dim, eps=hp.eps)

        # クエリ（ROI 限定可）
        gh, gw = hp.grid_h, hp.grid_w
        xs = torch.linspace(0, 1, gw).repeat(gh)
        zs = torch.linspace(0, 1, gh).unsqueeze(1).repeat(1, gw).reshape(-1)
        full_queries = torch.stack([xs, zs], dim=-1)  # [gh*gw, 2]
        if roi_idx is not None:
            roi_idx_t = torch.from_numpy(roi_idx.astype(np.int64))
            full_queries = full_queries.index_select(0, roi_idx_t)
        self.register_buffer("pixel_queries", full_queries)   # [n_roi, 2]
        self.query_proj = nn.Linear(2, hp.query_dim)

        # （任意）固定Lの前提チェック
        # assert hp.min_meas_per_t == hp.max_meas_per_t

    def forward_time(self, S_t):
        """
        S_t: [B, L, 5]
        """
        try:
            H_t = self.encoder(S_t, None)
        except TypeError:
            H_t = self.encoder(S_t)

        ctx = H_t
        B = S_t.size(0)
        Q = self.query_proj(self.pixel_queries).unsqueeze(0).expand(B, -1, -1)  # [B,N_roi,Dq]
        U_t = self.decoder(Q, ctx)               # [B,N_roi,H]
        alpha_t, beta_t = self.head(U_t)         # [B,N_roi]
        return alpha_t, beta_t

    def forward_sequence(self, sets):
        """
        sets: [B, T, L, 5]
        """
        B, T, L, D = sets.shape
        alphas, betas = [], []
        for t in range(T):
            a_t, b_t = self.forward_time(sets[:, t, :, :])
            alphas.append(a_t.unsqueeze(1))
            betas.append(b_t.unsqueeze(1))
        return torch.cat(alphas, dim=1), torch.cat(betas, dim=1)


# ---------------------------
# Evidential Beta–Bernoulli Loss（Bernoulli仮定 + Betaの事前）
# ---------------------------
def evidential_beta_bernoulli_loss(y, alpha, beta, kl_lambda=0.1, eps=1e-12):
    """
    y:     [B, N]  （0/1 ターゲット）
    alpha: [B, N]  （>1）
    beta:  [B, N]  （>1）
    """
    # 数値安定のため軽くクランプ
    a = torch.clamp(alpha, min=1.0 + 1e-6)
    b = torch.clamp(beta,  min=1.0 + 1e-6)

    # 期待確率 E[p] = a / (a + b)
    p1 = a / (a + b + eps)

    # Bernoulli の NLL
    nll = - (y * torch.log(p1 + eps) + (1.0 - y) * torch.log(1.0 - p1 + eps)).mean()

    # KL( Beta(a,b) || Beta(1,1) )
    # 正しい閉形式：KL = -log B(a,b) + (a-1)ψ(a) + (b-1)ψ(b) - (a+b-2)ψ(a+b)
    logB = (gammaln(a) + gammaln(b) - gammaln(a + b))
    kl   = (- logB
            + (a - 1.0) * digamma(a)
            + (b - 1.0) * digamma(b)
            - (a + b - 2.0) * digamma(a + b)).mean()

    # 数値誤差でごく小さく負になるのを防ぐ（任意）
    kl = torch.clamp(kl, min=0.0)

    loss = nll + kl_lambda * kl
    return loss, float(nll.item()), float(kl.item())


# ---------------------------
# Train / Smoke
# ---------------------------
def run_train(epochs=5, n_train=16, n_val=4, save_dir="./outputs"):
    device = HP.device
    os.makedirs(save_dir, exist_ok=True)

    # 使うファイル
    ERT_NPZ   = "../data/interim/surrogate_ds/ert_surrogate_reduced.npz"
    ORACLE_NPZ= "../data/processed/oracle_pairs_reduced/oracle_pairs_reduced_diverse.npz"
    WARMUP_NPZ= "../data/interim/ert_wenner_subset/ert_surrogate_wenner_reduced.npz"

    # --- DataSets: 学習/検証ともに ERTOracleDataset を使う ---
    train_ds = ERTOracleDataset(
        ert_npz_path=ERT_NPZ,
        oracle_npz_path=ORACLE_NPZ,
        hp=HP, pick_scale_index=0,
        max_fields=n_train, field_offset=0,
        warmup_npz_path=WARMUP_NPZ,
        warmup_pick="first35",
        active_select="random",
        active_seed=20251021
    )
    val_ds = ERTOracleDataset(
        ert_npz_path=ERT_NPZ,
        oracle_npz_path=ORACLE_NPZ,
        hp=HP, pick_scale_index=0,
        max_fields=n_val, field_offset=n_train,
        warmup_npz_path=WARMUP_NPZ,
        warmup_pick="first35",
        active_select="random",
        active_seed=20251021
    )

    # --- 追加: ROI をモデルへ（train の ROI を採用。val も同形前提） ---
    roi_idx = getattr(train_ds, "roi_idx", None)

    # --- collate はタプル専用（辞書対応をやめる） ---
    train_dl = DataLoader(train_ds, batch_size=HP.batch_size, shuffle=True,
                          collate_fn=collate_fn, drop_last=False)
    val_dl   = DataLoader(val_ds,   batch_size=HP.batch_size, shuffle=False,
                          collate_fn=collate_fn, drop_last=False)

    # モデル&最適化（roi_idx を渡す）
    model = ToyModel(HP, roi_idx=roi_idx).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=HP.lr, weight_decay=HP.weight_decay)  
    
    def run_epoch(loader, train=True):
        if train: model.train()
        else:     model.eval()
        total_loss = total_nll = total_kl = 0.0; nb = 0

        with torch.set_grad_enabled(train):
            for sets, y in loader:                # ← (sets_list, masks_list, y) ではなく (sets, y)
                sets = sets.to(device)            # [B,T,L,5]
                y    = y.to(device)               # [B,N]
                # warm-up を飛ばしてアクティブ部分のみを損失に使う
                alpha_T, beta_T = model.forward_sequence(sets[:, HP.warmup_fixed:, :, :])  # [B, Ta, N]
                alpha = alpha_T.sum(dim=1)
                beta  = beta_T.sum(dim=1)
                loss, nll_val, kl_val = evidential_beta_bernoulli_loss(y, alpha, beta, kl_lambda=HP.kl_lambda)
                p = alpha / (alpha + beta + 1e-12)
                topk_reg = (p.mean(dim=1) - 0.2).pow(2).mean()
                loss = loss + HP.topk_lambda * topk_reg

                if train:
                    opt.zero_grad(set_to_none=True); loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                    opt.step()

                total_loss += loss.item(); total_nll += nll_val; total_kl += kl_val; nb += 1
        return total_loss/nb, total_nll/nb, total_kl/nb


    log_lines = []
    t0_total = time.perf_counter()  # ← 総計測スタート

    for ep in range(1, epochs+1):
        ep_start = time.perf_counter()  # ← エポック計測スタート
        tr_loss, tr_nll, tr_kl = run_epoch(train_dl, train=True)
        va_loss, va_nll, va_kl = run_epoch(val_dl, train=False)

        with torch.no_grad():
            sets, y = next(iter(val_dl))
            sets = sets.to(device)
            alpha_T, beta_T = model.forward_sequence(sets[:, HP.warmup_fixed:, :, :])
            alpha = alpha_T.sum(dim=1); beta = beta_T.sum(dim=1)
            p_mean = (alpha / (alpha + beta + 1e-12)).mean().item()
            evidence_mean = (alpha + beta).mean().item()


        ep_sec = time.perf_counter() - ep_start  # ← このエポックの所要秒
        msg = (f"[Epoch {ep:02d}] train={tr_loss:.4f} (nll={tr_nll:.4f}, kl={tr_kl:.4f}) "
               f"| val={va_loss:.4f} | mean(p)={p_mean:.3f} | mean(evidence)={evidence_mean:.1f} "
               f"| time/epoch={ep_sec:.2f}s")
        print(msg, flush=True)
        log_lines.append(msg)

    total_sec = time.perf_counter() - t0_total  # ← 総所要秒
    total_msg = f"[TOTAL] time_all_epochs={total_sec:.2f}s (avg/epoch={total_sec/epochs:.2f}s)"
    print(total_msg, flush=True)
    log_lines.append(total_msg)

    torch.save({"model_state_dict": model.state_dict(), "hparams": HP.__dict__},
               os.path.join(save_dir, "toy_model.pt"))
    with open(os.path.join(save_dir, "toy_training_log.txt"), "w") as f:
        f.write("\n".join(log_lines))


# =========================
# Test (single field) + PNG
# =========================
@torch.no_grad()
def run_test_single_field(
    ckpt_path: str,
    out_dir: str | None = None,
    field_offset: int = 0,                 # 先頭から何番目のフィールドか
    first_output_step: int = 35,
    last_output_step: int | None = 155,
    save_alpha_beta: bool = False,
    stamp: bool = False,                   # ★ 追加: 常にタイムスタンプを付けたいとき True
):
    import matplotlib.pyplot as plt

    # -----------------------------
    # 1) 出力ディレクトリの決定と正規化
    # -----------------------------
    # デフォルトは repo 直下の "outputs_test_single"
    if out_dir is None:
        out_dir = "outputs_test_single"

    # {TIMESTAMP} テンプレートをサポート
    ts = time.strftime('%Y%m%d_%H%M%S')
    out_dir = out_dir.replace("{TIMESTAMP}", ts)

    # --stamp 指定なら "run_YYYYMMDD_HHMMSS" を末尾に付与
    if stamp:
        out_dir = str(Path(out_dir) / f"run_{ts}")

    # ~ 展開 & 絶対パス化（JupyterでもPowerShellでも安定）
    out_dir_path = Path(out_dir).expanduser().resolve()
    out_dir_path.mkdir(parents=True, exist_ok=True)

    device = HP.device

    # -----------------------------
    # 2) データパス（必要に応じて調整）
    # -----------------------------
    ERT_NPZ    = "../data/interim/surrogate_ds/ert_surrogate_reduced.npz"
    ORACLE_NPZ = "../data/processed/oracle_pairs_reduced/oracle_pairs_reduced_diverse.npz"
    WARMUP_NPZ = "../data/interim/ert_wenner_subset/ert_surrogate_wenner_reduced.npz"

    # ===== 1フィールドだけロード =====
    test_ds = ERTOracleDataset(
        ert_npz_path=ERT_NPZ,
        oracle_npz_path=ORACLE_NPZ,
        hp=HP, pick_scale_index=0,
        max_fields=1, field_offset=field_offset,     # ← 1フィールド指定
        warmup_npz_path=WARMUP_NPZ,
        warmup_pick="first35",
        active_select="random",
        active_seed=20251021
    )
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # ===== モデル復元 =====
    roi_idx = getattr(test_ds, "roi_idx", None)
    model = ToyModel(HP, roi_idx=roi_idx).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    gh, gw = HP.grid_h, HP.grid_w
    T_total = HP.warmup_fixed + HP.active_len  # 例: 35 + 120 = 155
    t_first = max(0, first_output_step - 1)
    t_last  = (T_total if last_output_step is None else min(last_output_step, T_total)) - 1
    assert 0 <= t_first <= t_last < T_total, f"Invalid range: {first_output_step}..{last_output_step}"

    # 保存先（フィールド別サブフォルダ。絶対パス）
    sample_dir = (out_dir_path / f"field_{field_offset:04d}")
    sample_dir.mkdir(parents=True, exist_ok=True)

    # ===== 推論 =====
    # スタック用（必要なら使う）
    stacked_p = []
    stacked_alpha = []
    stacked_beta  = []

    for (sets, _y_bin) in test_dl:   # ← (sets, y) のみ
        sets = sets.to(device)       # [1,T,L,5]
        alpha_T, beta_T = model.forward_sequence(sets)  # [1,T,N_roi]

        alpha_cum = torch.zeros_like(alpha_T[:, 0, :])
        beta_cum  = torch.zeros_like(beta_T[:, 0, :])

        for t in range(t_first, t_last + 1):
            a = alpha_T[:, t, :].squeeze(0)  # [N_roi]
            b = beta_T[:, t, :].squeeze(0)   # [N_roi]
            alpha_cum += a
            beta_cum  += b

            p = (alpha_cum / (alpha_cum + beta_cum + 1e-12)).squeeze(0).cpu().numpy()

            # p: shape [N_roi], 各フレームのROI内確率
            vals = np.asarray(p, dtype=np.float32)
            vals = vals[np.isfinite(vals)]                 # NaN/Inf除去
            if vals.size == 0:
                vmin, vmax = 0.0, 1.0                      # フォールバック
            else:
                # ロバスト（外れ値除去）：2〜98% のパーセンタイル
                vmin = float(np.percentile(vals, 2))
                vmax = float(np.percentile(vals, 98))
                vmin = max(0.0, min(1.0, vmin))
                vmax = max(0.0, min(1.0, vmax))
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
                    m = float(np.nanmean(vals)) if vals.size else 0.0
                    eps = 1e-6
                    vmin, vmax = m - eps, m + eps

            # フル格子に復元
            full_p_vec = np.zeros((gh * gw,), dtype=np.float32)
            if roi_idx is None:
                full_p_vec[:] = p
            else:
                full_p_vec[roi_idx] = p

            # 2D 化
            full_p_img = np.ascontiguousarray(full_p_vec.reshape(gh, gw), dtype=np.float32)

            # --- PNG 保存 ---
            fig = plt.figure()
            plt.imshow(full_p_img, origin="upper", vmin=vmin, vmax=vmax)
            plt.axis("off")
            plt.tight_layout(pad=0)
            fig.savefig(sample_dir / f"p_step_{t+1:03d}.png", dpi=150, bbox_inches="tight", pad_inches=0)
            plt.close(fig)

            # --- NPY 保存 ---
            np.save(sample_dir / f"p_step_{t+1:03d}.npy", full_p_img)

    print(f"[TEST single] saved to: {sample_dir.resolve()}")

# ---- CLI 追加 ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test",  action="store_true")
    parser.add_argument("--test1", action="store_true", help="Single-field test with PNG export")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--out", type=str, default="./outputs")
    parser.add_argument("--ckpt", type=str, default="./outputs/toy_model.pt")
    parser.add_argument("--field_offset", type=int, default=0)
    parser.add_argument("--first_step", type=int, default=33)
    parser.add_argument("--last_step",  type=int, default=155)
    parser.add_argument("--save_ab", action="store_true")
    parser.add_argument(
        "--stamp", action="store_true",
        help="OUT の末尾に run_YYYYMMDD_HHMMSS を付けて保存する"
    )
    args = parser.parse_args()

    if args.train:
        run_train(epochs=args.epochs, save_dir=args.out)
    if args.test1:
        # 新規: 1フィールドだけPNG込みで保存
        run_test_single_field(
            ckpt_path=args.ckpt,
            out_dir=args.out,
            field_offset=args.field_offset,
            first_output_step=args.first_step,
            last_output_step=args.last_step,
            save_alpha_beta=args.save_ab,
            stamp=args.stamp,
        )