
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
# ---------------------------
# Fixed Hyperparameters (あなたの本番想定に合わせています)
# ---------------------------
@dataclass
class HParams:
    input_dim: int = 5                 # 4 design + 1 value
    grid_h: int = 25
    grid_w: int = 100
    n_pixels: int = 25 * 100
    T: int = 120                       # history length
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
    query_dim: int = 32
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
HP.device = "cpu"

def sinusoidal_position_embedding(x: torch.Tensor, dim: int) -> torch.Tensor:
    if x.dim() == 1:
        x = x.unsqueeze(-1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=x.device) * (-math.log(10000.0) / dim))
    sin = torch.sin(x * div_term)
    cos = torch.cos(x * div_term)
    pe = torch.zeros(*x.shape[:-1], dim, device=x.device)
    pe[..., 0::2] = sin
    pe[..., 1::2] = cos
    return pe

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
            for t in range(hp.T):
                n_t = rng.randint(hp.min_meas_per_t, hp.max_meas_per_t + 1)
                meas = []
                for _m in range(n_t):
                    src_x = rng.uniform(0, 1)
                    src_z = rng.uniform(0, 1)
                    spacing = rng.uniform(0.02, 0.2)
                    orient = rng.uniform(0, np.pi) / np.pi
                    dx = self.pixel_coords[:, 0] - src_x
                    dz = self.pixel_coords[:, 1] - src_z
                    r2 = dx*dx + dz*dz
                    sigma2 = (spacing*2.0)**2 + 1e-6
                    w = np.exp(-r2 / (2*sigma2)).astype(np.float32)
                    # positive領域に相関する値 + ノイズ
                    val = (w * cont).mean() + rng.normal(0, noise_std)
                    val = float(np.tanh(val * 3.0))  # 概ね[-1,1]
                    meas.append([src_x, src_z, spacing, orient, val])
                S_list.append(np.array(meas, dtype=np.float32))
            self.samples.append({"sets": S_list, "label": y.astype(np.float32)})

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def collate_fn(batch):
    B, T = len(batch), HP.T
    max_n_per_t = []
    for t in range(T):
        mx = 0
        for b in range(B):
            mx = max(mx, batch[b]["sets"][t].shape[0])
        max_n_per_t.append(mx)
    sets_padded, masks = [], []
    for t in range(T):
        max_n = max_n_per_t[t]
        S_pad = torch.zeros(B, max_n, HP.input_dim, dtype=torch.float32)
        M = torch.zeros(B, max_n, dtype=torch.bool)
        for b in range(B):
            S = torch.from_numpy(batch[b]["sets"][t]); n = S.shape[0]
            S_pad[b, :n, :] = S; M[b, :n] = 1
        sets_padded.append(S_pad); masks.append(M)
    labels = torch.tensor(np.stack([b["label"] for b in batch], axis=0), dtype=torch.float32)
    return sets_padded, masks, labels

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


class SetDecoderSAB(nn.Module):
    """
    コンテキスト（各時刻の集合埋め込み H_t を1トークン化）と
    全ピクセル query 埋め込みを連結した “セット” に SAB を通す Decoder。
    出力はピクセルごとの埋め込み列（先頭のコンテキストは切り落とす）。
    """
    def __init__(self, query_dim, ctx_dim, hidden_dim=128, heads=4, layers=2):
        super().__init__()
        self.q_proj   = nn.Linear(query_dim, hidden_dim)
        self.ctx_proj = nn.Linear(ctx_dim,   hidden_dim)
        self.sabs = nn.ModuleList([MultiheadSelfAttention(hidden_dim, heads, dropout=0.0)
                                   for _ in range(layers)])
        self.ln_out = nn.LayerNorm(hidden_dim)

    def forward(self, pixel_queries, ctx_vec):
        """
        pixel_queries: [B, N, query_dim]
        ctx_vec:       [B, ctx_dim]  （EncoderのPMA出力）
        return:        [B, N, hidden_dim]
        """
        B, N, _ = pixel_queries.size()
        Q  = self.q_proj(pixel_queries)        # [B, N, H]
        C  = self.ctx_proj(ctx_vec).unsqueeze(1)  # [B, 1, H]
        X  = torch.cat([C, Q], dim=1)          # [B, 1+N, H]
        # SAB × L
        mask = None
        for sab in self.sabs:
            X = sab(X, key_padding_mask=mask)
        X = self.ln_out(X)
        return X[:, 1:, :]                     # 先頭(コンテキスト)を除いて [B, N, H]

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
    def __init__(self, hp: HParams):
        super().__init__()
        self.hp = hp
        self.encoder = SetEncoder(in_dim=hp.input_dim, hidden_dim=hp.hidden_dim,
                                  heads=hp.enc_heads, layers=hp.enc_layers, pma_seeds=hp.pma_seeds, dropout=hp.enc_dropout)
        self.decoder = CrossAttentionDecoder(query_dim=hp.query_dim, ctx_dim=hp.hidden_dim,
                                  hidden_dim=hp.hidden_dim, heads=hp.dec_heads, dropout=hp.attn_dropout)
        self.head = EvidentialHead(in_dim=hp.hidden_dim, eps=hp.eps)
        # Pixel queries from coordinates
        gh, gw = hp.grid_h, hp.grid_w
        xs = torch.linspace(0, 1, gw).repeat(gh)
        zs = torch.linspace(0, 1, gh).unsqueeze(1).repeat(1, gw).reshape(-1)
        self.register_buffer("pixel_queries", torch.stack([xs, zs], dim=-1))
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
        for t in range(self.hp.T):
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
    ds = ToyERTDataset(n_samples=n_train, hp=HP, noise_std=0.05, seed=42)
    dl = DataLoader(ds, batch_size=HP.batch_size, shuffle=True, collate_fn=collate_fn)
    val_ds = ToyERTDataset(n_samples=n_val, hp=HP, noise_std=0.05, seed=7)
    val_dl = DataLoader(val_ds, batch_size=HP.batch_size, shuffle=False, collate_fn=collate_fn)

    model = ToyModel(HP).to(device)
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
                alpha_T, beta_T = model.forward_sequence(sets_list, masks_list)
                alpha = alpha_T.sum(dim=1); beta = beta_T.sum(dim=1)  # Bayesian pooling over time
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
        tr_loss, tr_nll, tr_kl = run_epoch(dl, train=True)
        va_loss, va_nll, va_kl = run_epoch(val_dl, train=False)

        with torch.no_grad():
            sets_list, masks_list, y = next(iter(val_dl))
            sets_list = [s.to(device) for s in sets_list]; masks_list = [m.to(device) for m in masks_list]
            alpha_T, beta_T = model.forward_sequence(sets_list, masks_list)
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


def run_smoke(n_val=1):
    device = HP.device
    val_ds = ToyERTDataset(n_samples=n_val, hp=HP, noise_std=0.05, seed=7)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
    model = ToyModel(HP).to(device)
    sets_list, masks_list, y = next(iter(val_dl))
    sets_list = [s.to(device) for s in sets_list]; masks_list = [m.to(device) for m in masks_list]
    alpha_T, beta_T = model.forward_sequence(sets_list, masks_list)
    alpha = alpha_T.sum(dim=1); beta = beta_T.sum(dim=1)
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
