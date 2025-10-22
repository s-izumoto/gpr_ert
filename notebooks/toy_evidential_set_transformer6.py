
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
    enc_heads: int = 4
    enc_layers: int = 2
    pma_seeds: int = 1
    enc_dropout: float = 0.1

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
    batch_size: int = 2
    epochs: int = 5

    # Loss
    kl_lambda: float = 0.1
    topk_lambda: float = 0.1

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

HP = HParams()
HP.T = HP.warmup_fixed + HP.active_len
HP.device = "cpu"

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

# ---------------------------
# Synthetic Dataset（静的な二値GT、時刻ごとの可変個数の測定を合成）
# ---------------------------
class ToyERTDataset(Dataset):
    def __init__(self, n_samples: int = 20, hp: HParams = HP, noise_std: float = 0.05, seed: int = 0):
        super().__init__()
        self.hp = hp
        rng = np.random.RandomState(seed)
        self.samples = []
        # pixel grid in [0,1]^2
        gh, gw = hp.grid_h, hp.grid_w
        x = np.linspace(0, 1, gw, dtype=np.float32)
        z = np.linspace(0, 1, gh, dtype=np.float32)
        X, Z = np.meshgrid(x, z)
        self.pixel_coords = np.stack([X, Z], axis=-1).reshape(-1, 2)

        for _ in range(n_samples):
            # 静的GT（二値）。楕円スコアの上位20%を1に。
            cx, cz = rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7)
            ax, az = rng.uniform(0.1, 0.25), rng.uniform(0.1, 0.25)
            angle = rng.uniform(0, np.pi)
            R = np.array([[np.cos(angle), -np.sin(angle)],
                          [np.sin(angle),  np.cos(angle)]], dtype=np.float32)
            rel = (self.pixel_coords - np.array([cx, cz], dtype=np.float32)).dot(R.T)
            score = -((rel[:,0]/ax)**2 + (rel[:,1]/az)**2)
            k = int(hp.n_pixels * hp.pos_ratio)
            idx = np.argpartition(score, -k)[-k:]
            y = np.zeros(hp.n_pixels, dtype=np.float32); y[idx] = 1.0
            # 合成のため連続スコアを[0,1]正規化
            cont = (score - score.min()) / (score.max() - score.min() + 1e-9)

            # 各時刻の測定セットを生成（design4 + value1）
            S_list = []
            for t in range(hp.warmup_fixed + hp.active_len):
                n_t = 1  # 1 measurement per time step (adjust if needed)
                meas = []
                for _m in range(n_t):
                    if t < hp.warmup_fixed:
                        # Fixed pattern during warm-up
                        frac = (t + 0.5) / hp.warmup_fixed
                        src_x = float(frac)
                        src_z = 0.02
                        spacing = 0.06
                        orient = 0.0
                    else:
                        # Variable/random designs after warm-up
                        src_x = rng.uniform(0, 1)
                        src_z = rng.uniform(0, 1)
                        spacing = rng.uniform(0.02, 0.2)
                        orient = rng.uniform(0, np.pi) / np.pi
                    dx = self.pixel_coords[:, 0] - src_x
                    dz = self.pixel_coords[:, 1] - src_z
                    r2 = dx*dx + dz*dz
                    sigma2 = (spacing*2.0)**2 + 1e-6
                    w = np.exp(-r2 / (2*sigma2)).astype(np.float32)
                    val = (w * cont).mean() + rng.normal(0, noise_std)
                    val = float(np.tanh(val * 3.0))
                    meas.append([src_x, src_z, spacing, orient, val])
                S_list.append(np.array(meas, dtype=np.float32))

            self.samples.append({"sets": S_list, "label": y.astype(np.float32)})

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

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
                print("roi mask found in oracle NPZ (2D).")
            elif raw.ndim == 3:
                roi_mask_c = np.asarray(raw[s_idx], dtype=np.uint8)  # (Hc, Wc)
                print("roi mask found in oracle NPZ (3D).")
        # フォールバック（全画素採用）
        if roi_mask_c is None:
            roi_mask_c = np.ones((Hc, Wc), dtype=np.uint8)
            print("roi mask not found in oracle NPZ. Using all-ones mask.")

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

            sets_list, masks_list = [], []

            # ----- (1) ウォームアップ 35 ステップ：Wenner の 35 件を履歴として使用 -----
            if Dn_all_warm is not None:
                row0_w = i * R_warm
                Dn_w = Dn_all_warm[row0_w: row0_w + R_warm]   # [R_warm, 4]
                y_w  = y_all_warm[row0_w: row0_w + R_warm]    # [R_warm]

                # 取り出し方：first32（そのまま先頭から） or stride（間引き均等サンプリング）
                if hp.warmup_fixed <= Dn_w.shape[0]:
                    if warmup_pick == "stride":
                        idx = np.linspace(0, Dn_w.shape[0]-1, hp.warmup_fixed, dtype=int)
                    else:
                        idx = np.arange(hp.warmup_fixed, dtype=int)
                else:
                    # 足りないときは末尾を複製して 35 に揃える
                    base = np.arange(Dn_w.shape[0], dtype=int)
                    pad  = np.full(hp.warmup_fixed - Dn_w.shape[0], Dn_w.shape[0]-1, dtype=int)
                    idx  = np.concatenate([base, pad], axis=0)

                for t in idx:
                    v5 = np.concatenate([Dn_w[t], np.array([y_w[t]], dtype=np.float32)], axis=0)  # [5]
                    sets_list.append(torch.from_numpy(v5[None, :]))
                    masks_list.append(torch.ones((1, 5), dtype=torch.bool))  # 使うけど損失には寄与しない（下で start=35）
            else:
                # 従来のダミー（互換）
                for _ in range(hp.warmup_fixed):
                    sets_list.append(torch.from_numpy(np.array([[0.5, 0.02, 0.06, 0.0, 0.0]], dtype=np.float32)))
                    masks_list.append(torch.ones((1, 5), dtype=torch.bool))

            # ----- (2) アクティブ 120 ステップ：学習対象（ランダム or 先頭） -----
            # ここまでに必ず R_act, Dn_all_act, y_all_act が定義済み
            if 'R_act' not in locals() or R_act <= 0:
                raise ValueError(f"[ERTOracleDataset] Invalid R_act={R_act}. Active split is empty or undefined.")

            row0 = i * R_act  # このフィールドの先頭行（アクティブ部）
            all_idx = np.arange(R_act, dtype=int)

            # 環境Zごとに決定的な RNG： field_source_idx を使う
            env_id = int(act_slice[i]) if 'act_slice' in locals() else i
            rng = np.random.RandomState(self.active_seed ^ env_id)

            if self.active_select == "random":
                if R_act >= hp.active_len:
                    # 重複なしで 120 抜き出し
                    pick_idx = rng.choice(all_idx, size=hp.active_len, replace=False)
                else:
                    # R_act が 120 未満でも、シャッフルを繰り返して 120 個に満たす
                    reps = (hp.active_len + R_act - 1) // R_act  # 切り上げ
                    seq = np.concatenate([rng.permutation(all_idx) for _ in range(reps)], axis=0)
                    pick_idx = seq[:hp.active_len]
            else:
                # 互換：先頭から（足りなければ末尾複製で埋める）
                take = min(hp.active_len, R_act)
                base = np.arange(take, dtype=int)
                if take < hp.active_len:
                    pad = np.full(hp.active_len - take, take - 1, dtype=int)
                    pick_idx = np.concatenate([base, pad], axis=0)
                else:
                    pick_idx = base

            # 実データ抽出（row0 オフセットを忘れずに）
            Dn_i = Dn_all_act[row0 + pick_idx]      # [active_len, 4]
            yi   = y_all_act [row0 + pick_idx]      # [active_len]

            # 35回目以降の 120 ステップを push
            for t in range(hp.active_len):
                v5 = np.concatenate([Dn_i[t], np.array([yi[t]], dtype=np.float32)], axis=0)
                sets_list.append(torch.from_numpy(v5[None, :]))
                masks_list.append(torch.ones((1, 5), dtype=torch.bool))
            
            self.samples.append((
                sets_list,
                masks_list,
                torch.from_numpy(y_bin.astype(np.float32))   # [n_roi]
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 常に (sets_list, masks_list, y_bin) のタプルを返す（collateを安定させる）
        return self.samples[idx]

def collate_fn(batch):
    """
    batch: list of (sets_list, masks_list, y) だけを想定
      - sets_list: 長さ T のリスト。各要素は [n_i(t), D] の Tensor（D=5）
      - masks_list: 同じ長さ T のリスト（使わなくてもOKだが True=有効を前提）
      - y: [N_pixels] の Tensor
    返り値:
      sets_padded: 長さ T のリスト。各要素は [B, L_t, D]
      masks:       長さ T のリスト。各要素は [B, L_t] (True=有効)
      y:           [B, N_pixels]
    """

    assert isinstance(batch[0], (tuple, list)) and len(batch[0]) == 3, \
        f"Dataset must return (sets_list, masks_list, y). Got: {type(batch[0])}, len={len(batch[0])}"

    sets0, _, y0 = batch[0]
    T = len(sets0)
    D = sets0[0].shape[1]  # (=5)

    # 各時刻の最大要素数 L_t
    max_n_per_t = [0]*T
    for t in range(T):
        mx = 0
        for (sets_list, _, _) in batch:
            mx = max(mx, sets_list[t].shape[0])
        max_n_per_t[t] = mx

    B = len(batch)
    sets_padded, masks = [], []
    for t in range(T):
        L_t = max_n_per_t[t]
        S_pad = torch.zeros(B, L_t, D, dtype=torch.float32)
        M     = torch.zeros(B, L_t, dtype=torch.bool)
        for b, (sets_list, masks_list, _) in enumerate(batch):
            S = sets_list[t]              # [n_b(t), D]
            n = S.shape[0]
            S_pad[b, :n, :] = S
            M[b, :n] = True               # True=有効トークン
        sets_padded.append(S_pad)
        masks.append(M)

    y = torch.stack([yb for (_, _, yb) in batch], dim=0)  # [B, N_pixels]
    return sets_padded, masks, y

# ---------------------------
# Set Transformer (SAB + PMA)
# ---------------------------
class MultiheadSelfAttention(nn.Module):
    """
    Safety-first self-attention (multi-head) that avoids nn.MultiheadAttention crashes
    by doing attention with explicit matmuls & masking on CPU. batch_first=True 相当。
    """
    def __init__(self, d_model, heads=4, dropout=0.1):
        super().__init__()
        assert d_model % heads == 0, "d_model must be divisible by heads"
        self.d_model = d_model
        self.h = heads
        self.dh = d_model // heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def _ensure_key_mask(self, x, key_padding_mask):
        """
        入力側の M は True=有効（実データ）という前提が多いので、
        ここで PyTorch 的な pad=True に変換する。
        返り値: pad_mask [B, 1, 1, L] （True=無視=pad）
        """
        B, L, _ = x.shape
        if key_padding_mask is None:
            return None
        M = key_padding_mask.to(torch.bool)
        if M.size(1) != L:
            M = M[:, :L]  # 安全策
        pad = (~M).contiguous()         # True=pad
        # 全部 pad だと softmax が NaN になるので最小限 1点だけ通す
        all_pad = pad.all(dim=1)
        if all_pad.any():
            idx = torch.nonzero(all_pad, as_tuple=False).flatten()
            pad[idx, 0] = False
        return pad[:, None, None, :]    # [B,1,1,L]

    def forward(self, x, key_padding_mask=None):
        """
        x: [B, L, D], key_padding_mask: [B, L] with True=有効（想定）
        """
        B, L, D = x.shape
        pad_mask = self._ensure_key_mask(x, key_padding_mask)

        # Projections & head-split
        q = self.q_proj(x).view(B, L, self.h, self.dh).transpose(1, 2)  # [B,h,L,dh]
        k = self.k_proj(x).view(B, L, self.h, self.dh).transpose(1, 2)  # [B,h,L,dh]
        v = self.v_proj(x).view(B, L, self.h, self.dh).transpose(1, 2)  # [B,h,L,dh]

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.dh ** 0.5)  # [B,h,L,L]
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask, -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.drop(attn)
        out = torch.matmul(attn, v)  # [B,h,L,dh]

        # Merge heads
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.o_proj(out)

        # Residual + FFN
        x = self.ln1(x + out)
        x = self.ln2(x + self.ff(x))
        return x

class PMA(nn.Module):
    """
    Pooling by Multihead Attention: uses learnable seeds S as queries,
    and attends over set X (keys/values). Safety-first implementation.
    """
    def __init__(self, d_model, k=1, heads=4, dropout=0.1):
        super().__init__()
        assert d_model % heads == 0
        self.k = k
        self.d_model = d_model
        self.h = heads
        self.dh = d_model // heads
        self.S = nn.Parameter(torch.randn(k, d_model))

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model)

    def _ensure_key_mask(self, X, key_padding_mask):
        # True=有効 → True=pad に変換
        if key_padding_mask is None:
            return None
        B, L, _ = X.shape
        M = key_padding_mask.to(torch.bool)
        if M.size(1) != L:
            M = M[:, :L]
        pad = (~M).contiguous()                 # True=pad
        all_pad = pad.all(dim=1)
        if all_pad.any():
            idx = torch.nonzero(all_pad, as_tuple=False).flatten()
            pad[idx, 0] = False
        return pad[:, None, None, :]            # [B,1,1,L]

    def forward(self, X, key_padding_mask=None):
        """
        X: [B, L, D], key_padding_mask: [B, L] with True=有効（想定）
        returns: [B, k, D]
        """
        B, L, D = X.shape
        pad_mask = self._ensure_key_mask(X, key_padding_mask)

        # Seeds as queries
        S = self.S.unsqueeze(0).expand(B, -1, -1)  # [B,k,D]

        # Projections & head split
        q = self.q_proj(S).view(B, self.k, self.h, self.dh).transpose(1, 2)  # [B,h,k,dh]
        k = self.k_proj(X).view(B, L, self.h, self.dh).transpose(1, 2)       # [B,h,L,dh]
        v = self.v_proj(X).view(B, L, self.h, self.dh).transpose(1, 2)       # [B,h,L,dh]

        # Cross-attention: seeds -> set
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.dh ** 0.5)      # [B,h,k,L]
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask, -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.drop(attn)
        out = torch.matmul(attn, v)  # [B,h,k,dh]

        out = out.transpose(1, 2).contiguous().view(B, self.k, D)
        return self.ln(self.o_proj(out))  # [B,k,D]

class SetEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, heads=4, layers=2, pma_seeds=1, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.sabs = nn.ModuleList([MultiheadSelfAttention(hidden_dim, heads, dropout=dropout) for _ in range(layers)])
        self.pma = PMA(d_model=hidden_dim, k=pma_seeds, heads=heads, dropout=dropout)
    def forward(self, X, mask):
        h = self.in_proj(X)
        for sab in self.sabs: h = sab(h, key_padding_mask=mask)
        pooled = self.pma(h, key_padding_mask=mask)  # [B, seeds, D]
        return pooled.squeeze(1)

class CrossAttentionDecoder(nn.Module):
    """
    各ピクセル（Query）が、Encoder文脈1トークン（Key/Value）にクロスアテンションする軽量Decoder。
    O(N) で動作し、SetDecoderSAB の O(N^2) を回避します。
    """
    def __init__(self, query_dim, ctx_dim, hidden_dim=128, heads=4, dropout=0.0):
        super().__init__()
        assert hidden_dim % heads == 0
        self.h = heads
        self.dh = hidden_dim // heads
        # Q: pixel queries, K/V: context(=1 token)
        self.q_proj = nn.Linear(query_dim, hidden_dim)
        self.k_proj = nn.Linear(ctx_dim,   hidden_dim)
        self.v_proj = nn.Linear(ctx_dim,   hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ff  = nn.Sequential(
            nn.Linear(hidden_dim, 4*hidden_dim), nn.GELU(),
            nn.Linear(4*hidden_dim, hidden_dim)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, pixel_queries, ctx_vec):
        """
        pixel_queries: [B, N, Dq]; ctx_vec: [B, Dc]
        return: [B, N, H]
        """
        B, N, _ = pixel_queries.shape
        # proj & head-split
        q = self.q_proj(pixel_queries).view(B, N, self.h, self.dh).transpose(1, 2)  # [B,h,N,dh]
        k = self.k_proj(ctx_vec).view(B, 1, self.h, self.dh).transpose(1, 2)        # [B,h,1,dh]
        v = self.v_proj(ctx_vec).view(B, 1, self.h, self.dh).transpose(1, 2)        # [B,h,1,dh]
        # scaled dot-product attention（長さ1に対するsoftmaxは自明だが書式を統一）
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.dh ** 0.5)            # [B,h,N,1]
        attn   = torch.softmax(scores, dim=-1)
        attn   = self.drop(attn)
        out    = torch.matmul(attn, v)                                              # [B,h,N,dh]
        out    = out.transpose(1, 2).contiguous().view(B, N, -1)                    # [B,N,H]
        # 残差 + FFN
        x = self.ln1(out)
        x = self.ln2(x + self.ff(x))
        return x

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
        self.encoder = SetEncoder(in_dim=hp.input_dim, hidden_dim=hp.hidden_dim,
                                  heads=hp.enc_heads, layers=hp.enc_layers, pma_seeds=hp.pma_seeds, dropout=hp.enc_dropout)
        self.decoder = CrossAttentionDecoder(query_dim=hp.query_dim, ctx_dim=hp.hidden_dim,
                                  hidden_dim=hp.hidden_dim, heads=hp.dec_heads, dropout=hp.attn_dropout)
        self.head = EvidentialHead(in_dim=hp.hidden_dim, eps=hp.eps)

        # === 既存のクエリ生成（全画素） ===
        gh, gw = hp.grid_h, hp.grid_w
        xs = torch.linspace(0, 1, gw).repeat(gh)
        zs = torch.linspace(0, 1, gh).unsqueeze(1).repeat(1, gw).reshape(-1)
        full_queries = torch.stack([xs, zs], dim=-1)  # [gh*gw, 2]

        # === 追加: ROI 限定 ===
        if roi_idx is not None:
            roi_idx_t = torch.from_numpy(roi_idx.astype(np.int64))
            full_queries = full_queries.index_select(0, roi_idx_t)

        self.register_buffer("pixel_queries", full_queries)   # [n_roi, 2] or [gh*gw, 2]
        self.query_proj = nn.Linear(2, hp.query_dim)

    def forward_time(self, S_t, M_t):
        B = S_t.size(0)
        X_in = S_t
        H_t = self.encoder(X_in, M_t)  # [B, D]
        KV = H_t.unsqueeze(1)                        # [B, 1, D]（各時刻1トークン）
        Q = self.query_proj(self.pixel_queries).unsqueeze(0).expand(B, -1, -1)  # [B, N, Dq]
        U_t = self.decoder(Q, H_t)  # ← SAB デコーダ
        alpha_t, beta_t = self.head(U_t)  # [B, N]

        return alpha_t, beta_t

    def forward_sequence(self, sets_list, masks_list):
        alphas, betas = [], []
        T = len(sets_list)
        for t in range(T):
            S_t = sets_list[t].to(self.pixel_queries.device)
            M_t = masks_list[t].to(self.pixel_queries.device)
            a_t, b_t = self.forward_time(S_t, M_t)
            alphas.append(a_t.unsqueeze(1)); betas.append(b_t.unsqueeze(1))
        return torch.cat(alphas, dim=1), torch.cat(betas, dim=1)


# ---------------------------
# Evidential Beta–Bernoulli Loss（Bernoulli仮定 + Betaの事前）
# ---------------------------
def evidential_beta_bernoulli_loss(y, alpha, beta, kl_lambda=0.1, eps=1e-12):
    """
    y:     [B, N]  （0/1 のターゲット）
    alpha: [B, N]  （>1）
    beta:  [B, N]  （>1）
    """
    # 期待確率 E[p] = alpha / (alpha + beta)
    p1 = alpha / (alpha + beta + eps)

    # 負の対数尤度（Bernoulli）
    nll = - (y * torch.log(p1 + eps) + (1.0 - y) * torch.log(1.0 - p1 + eps)).mean()

    # KL( Beta(alpha,beta) || Beta(1,1) )
    a, b = alpha, beta
    B_ab = torch.exp(gammaln(a) + gammaln(b) - gammaln(a + b))
    # Beta(1,1) は一様分布。KL の閉形式（定数項は 0 扱いで OK）
    kl = (torch.log(B_ab + eps)
          + (a - 1.0) * (digamma(a) - digamma(a + b))
          + (b - 1.0) * (digamma(b) - digamma(a + b))).mean()

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
        else: model.eval()
        total_loss = total_nll = total_kl = 0.0; nb = 0
        with torch.set_grad_enabled(train):
            for sets_list, masks_list, y in loader:
                sets_list = [s.to(device) for s in sets_list]
                masks_list = [m.to(device) for m in masks_list]
                y = y.to(device)
                start = HP.warmup_fixed
                alpha_T, beta_T = model.forward_sequence(sets_list[start:], masks_list[start:])
                alpha = alpha_T.sum(dim=1)
                beta  = beta_T.sum(dim=1)
                loss, nll_val, kl_val = evidential_beta_bernoulli_loss(y, alpha, beta, kl_lambda=HP.kl_lambda)
                # optional: top-20% constraint
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
            sets_list, masks_list, y = next(iter(val_dl))
            sets_list = [s.to(device) for s in sets_list]; masks_list = [m.to(device) for m in masks_list]
            start = HP.warmup_fixed
            alpha_T, beta_T = model.forward_sequence(sets_list[start:], masks_list[start:])
            alpha = alpha_T.sum(dim=1)
            beta  = beta_T.sum(dim=1)
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


def run_smoke(n_val=1):
    device = HP.device
    val_ds = ToyERTDataset(n_samples=n_val, hp=HP, noise_std=0.05, seed=7)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
    model = ToyModel(HP).to(device)
    sets_list, masks_list, y = next(iter(val_dl))
    sets_list = [s.to(device) for s in sets_list]; masks_list = [m.to(device) for m in masks_list]
    start = HP.warmup_fixed
    alpha_T, beta_T = model.forward_sequence(sets_list[start:], masks_list[start:])
    alpha = alpha_T.sum(dim=1)
    beta  = beta_T.sum(dim=1)
    p = alpha / (alpha + beta + 1e-12)
    print("SMOKE TEST OK")
    print("alpha.shape, beta.shape, p.shape:", alpha.shape, beta.shape, p.shape)
    print("p.mean():", p.mean().detach().item(), " evidence.mean():", (alpha+beta).mean().detach().item())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run a quick smoke test (single forward)")
    parser.add_argument("--train", action="store_true", help="Run a short training loop")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs for training")
    parser.add_argument("--out", type=str, default="./outputs", help="Output directory")
    args = parser.parse_args()
    if args.smoke:
        run_smoke()
    if args.train:
        run_train(epochs=args.epochs, save_dir=args.out)
