"""Kernel point attention (KPA) network, shared by train_kpa.py and the CLIP-PCQA "kpa" branch.

Fixed-radius neighbourhoods, Gaussian-kernel-weighted gated SUM of neighbour messages rescaled by the
true neighbour count (sampling-sensitive), then a transformer over query tokens and mean/max/attention
pooling. See train_kpa.py for the full description and the fixed-k ablation.
"""
import torch, torch.nn as nn

H = torch.tensor([0.003, 0.006, 0.012])
D_TOKEN = 64 * (len(H) + 1)
POOLED_DIM = 3 * D_TOKEN


class KernelLayer(nn.Module):
    """One bandwidth: kernel-weighted, gated SUM of neighbour messages (sampling-sensitive)."""
    def __init__(self, h, d=64, fixed_k=False):
        super().__init__()
        self.h, self.fixed_k = h, fixed_k
        self.msg = nn.Sequential(nn.Linear(3 + 3 + 1, d), nn.GELU(), nn.Linear(d, d))
        self.gate = nn.Linear(d, 1)

    def forward(self, q, nb, cnt):  # q [B,Q,6], nb [B,Q,M,6], cnt [B,Q]
        off = nb[..., :3] / self.h
        valid = (nb[..., :3].abs().sum(-1) > 0).float()               # zero padding
        d2 = (off ** 2).sum(-1, keepdim=True)
        x = torch.cat([off, nb[..., 3:] - q[:, :, None, 3:], d2], -1)
        m = self.msg(x)
        if self.fixed_k:  # PTv2-like: softmax over whatever neighbours exist, no count information
            logit = self.gate(m).squeeze(-1).masked_fill(valid == 0, -1e4)
            return (torch.softmax(logit, -1)[..., None] * m).sum(2)
        w = torch.exp(-d2 / 2) * torch.sigmoid(self.gate(m)) * valid[..., None]
        scale = (cnt / valid.sum(-1).clamp_min(1))[..., None]            # unbiased for the subsampling
        o = (w * m).sum(2) * scale / 32                                  # sum over the full 3h ball
        return torch.sign(o) * torch.log1p(o.abs())                       # compress the dynamic range, keep order


class KPANet(nn.Module):
    def __init__(self, d=64, fixed_k=False):
        super().__init__()
        self.layers = nn.ModuleList([KernelLayer(float(h), d, fixed_k) for h in H])
        self.q_embed = nn.Linear(6, d)
        self.norm = nn.LayerNorm(d * (len(H) + 1))
        enc = nn.TransformerEncoderLayer(d * (len(H) + 1), 4, d * 4, 0.1, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(enc, 2)
        D = d * (len(H) + 1)
        self.att = nn.Linear(D, 1)
        self.head = nn.Sequential(nn.Linear(3 * D, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 1))

    def forward(self, q, nb, cnt):
        f = [self.q_embed(q)] + [l(q, nb[:, s], cnt[:, s]) for s, l in enumerate(self.layers)]
        t = self.tr(self.norm(torch.cat(f, -1)))
        self.last_tokens = t  # per-query tokens [B,Q,D_TOKEN], used by the FCA alignment loss in CLIPPCQA_Net
        w = torch.softmax(self.att(t), 1)
        return self.head(torch.cat([t.mean(1), t.max(1).values, (w * t).sum(1)], -1)).squeeze(-1)


class SerialAttnBlock(nn.Module):
    """PTv3-style window attention along a space-filling-curve order of the query tokens (pre-LN, + MLP)."""
    def __init__(self, c, heads=4, window=512):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(c), nn.LayerNorm(c)
        self.attn = nn.MultiheadAttention(c, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(c, 4 * c), nn.GELU(), nn.Linear(4 * c, c))
        self.window = window

    def forward(self, t, order):  # t [B,Q,C], order [B,Q]
        B, Q, C = t.shape
        x = torch.gather(self.n1(t), 1, order[..., None].expand(-1, -1, C))        # serialise
        W = min(self.window, Q)
        pad = (-Q) % W
        x = torch.nn.functional.pad(x, (0, 0, 0, pad)).reshape(-1, W, C)
        mask = torch.zeros(B, Q + pad, dtype=torch.bool, device=t.device)
        mask[:, Q:] = True
        y = self.attn(x, x, x, key_padding_mask=mask.reshape(-1, W), need_weights=False)[0].reshape(B, -1, C)[:, :Q]
        t = t.scatter_add(1, order[..., None].expand(-1, -1, C), y)                # de-serialise + residual
        return t + self.mlp(self.n2(t))


class SKPANet(nn.Module):
    """S-KPA: FPS queries (kpa_prep.py, QUERY=fps, KPA_Q=16384) x kernel-KNN local operator (KernelLayer, fixed-radius,
    density-sensitive) x PTv3-style serialised window attention (alternating Z-order / Hilbert per block), then
    mean / max / attention pooling. Same POOLED_DIM as KPANet, so it is a drop-in for the CLIP-PCQA branch."""
    def __init__(self, d=64, blocks=2, window=512, global_last=False):
        super().__init__()
        self.layers = nn.ModuleList([KernelLayer(float(h), d) for h in H])
        # global_last: one full (global) attention layer after the window blocks (affordable for <= ~4k queries)
        self.glob = nn.TransformerEncoderLayer(D_TOKEN, 4, 4 * D_TOKEN, 0.1, batch_first=True, norm_first=True) if global_last else None
        self.q_embed = nn.Linear(6, d)
        self.norm = nn.LayerNorm(D_TOKEN)
        self.blocks = nn.ModuleList([SerialAttnBlock(D_TOKEN, window=window) for _ in range(blocks)])
        self.att = nn.Linear(D_TOKEN, 1)
        self.head = nn.Sequential(nn.Linear(POOLED_DIM, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 1))

    def forward(self, q, nb, cnt, order):  # order [B,2,Q] long
        f = [self.q_embed(q)] + [l(q, nb[:, s], cnt[:, s]) for s, l in enumerate(self.layers)]
        t = self.norm(torch.cat(f, -1))
        for i, b in enumerate(self.blocks):
            t = b(t, order[:, i % order.shape[1]])
        if self.glob is not None:
            t = self.glob(t)
        self.last_tokens = t
        w = torch.softmax(self.att(t), 1)
        return self.head(torch.cat([t.mean(1), t.max(1).values, (w * t).sum(1)], -1)).squeeze(-1)
