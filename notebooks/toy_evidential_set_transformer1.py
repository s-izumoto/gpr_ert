
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
    max_meas_per_t: int = 4
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
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))
        self.ln2 = nn.LayerNorm(dim)
    def forward(self, x, key_padding_mask=None):
        h, _ = self.attn(x, x, x, key_padding_mask=~key_padding_mask if key_padding_mask is not None else None, need_weights=False)
        x = self.ln(x + h); x = self.ln2(x + self.ff(x)); return x

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds=1):
        super().__init__()
        self.S = nn.Parameter(torch.randn(num_seeds, dim))
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.ln = nn.LayerNorm(dim)
    def forward(self, X, key_padding_mask=None):
        B = X.size(0); S = self.S.unsqueeze(0).expand(B, -1, -1)
        out, _ = self.attn(S, X, X, key_padding_mask=~key_padding_mask if key_padding_mask is not None else None, need_weights=False)
        return self.ln(out)

class SetEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, heads=4, layers=2, pma_seeds=1, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.sabs = nn.ModuleList([MultiheadSelfAttention(hidden_dim, heads, dropout=dropout) for _ in range(layers)])
        self.pma = PMA(hidden_dim, heads, num_seeds=pma_seeds)
    def forward(self, X, mask):
        h = self.in_proj(X)
        for sab in self.sabs: h = sab(h, key_padding_mask=mask)
        pooled = self.pma(h, key_padding_mask=mask)  # [B, seeds, D]
        return pooled.squeeze(1)

# ---------------------------
# Cross-Attention Decoder（ピクセルをQuery、時刻ごとの集合表現をK/V）
# ---------------------------
class CrossAttentionDecoder(nn.Module):
    def __init__(self, query_dim, kv_dim, hidden_dim=128, heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.q_proj = nn.Linear(query_dim, hidden_dim)
        self.kv_proj = nn.Linear(kv_dim, hidden_dim)
        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(nn.ModuleList([
                nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=heads, dropout=dropout, batch_first=True),
                nn.LayerNorm(hidden_dim),
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim*4), nn.GELU(), nn.Linear(hidden_dim*4, hidden_dim)),
                nn.LayerNorm(hidden_dim)
            ]))
    def forward(self, queries, keys_values):
        Q = self.q_proj(queries); KV = self.kv_proj(keys_values); x = Q
        for attn, ln1, ff, ln2 in self.layers:
            h, _ = attn(x, KV, KV, need_weights=False); x = ln1(x + h); x = ln2(x + ff(x))
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
    def __init__(self, hp: HParams):
        super().__init__()
        self.hp = hp
        self.encoder = SetEncoder(in_dim=hp.input_dim + hp.time_embed_dim, hidden_dim=hp.hidden_dim,
                                  heads=hp.enc_heads, layers=hp.enc_layers, pma_seeds=hp.pma_seeds, dropout=hp.enc_dropout)
        self.decoder = CrossAttentionDecoder(query_dim=hp.query_dim, kv_dim=hp.hidden_dim,
                                             hidden_dim=hp.hidden_dim, heads=hp.dec_heads, layers=hp.dec_layers, dropout=hp.attn_dropout)
        self.head = EvidentialHead(in_dim=hp.hidden_dim, eps=hp.eps)
        # Pixel queries from coordinates
        gh, gw = hp.grid_h, hp.grid_w
        xs = torch.linspace(0, 1, gw).repeat(gh)
        zs = torch.linspace(0, 1, gh).unsqueeze(1).repeat(1, gw).reshape(-1)
        self.register_buffer("pixel_queries", torch.stack([xs, zs], dim=-1))
        self.query_proj = nn.Linear(2, hp.query_dim)

    def forward_time(self, S_t, M_t, t_norm):
        B = S_t.size(0)
        t_embed = sinusoidal_position_embedding(torch.full((B, 1), t_norm, device=S_t.device), self.hp.time_embed_dim)
        t_embed = t_embed.unsqueeze(1).expand(-1, S_t.size(1), -1)
        X_in = torch.cat([S_t, t_embed], dim=-1)
        H_t = self.encoder(X_in, M_t)               # [B, D]
        KV = H_t.unsqueeze(1)                        # [B, 1, D]（各時刻1トークン）
        Q = self.query_proj(self.pixel_queries).unsqueeze(0).expand(B, -1, -1)  # [B, N, Dq]
        U_t = self.decoder(Q, KV)
        alpha_t, beta_t = self.head(U_t)            # [B, N]
        return alpha_t, beta_t

    def forward_sequence(self, sets_list, masks_list):
        alphas, betas = [], []
        for t in range(self.hp.T):
            S_t = sets_list[t].to(self.pixel_queries.device)
            M_t = masks_list[t].to(self.pixel_queries.device)
            t_norm = (t + 1) / self.hp.T
            a_t, b_t = self.forward_time(S_t, M_t, t_norm)
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
    print("p.mean():", float(p.mean()), " evidence.mean():", float((alpha+beta).mean()))

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
