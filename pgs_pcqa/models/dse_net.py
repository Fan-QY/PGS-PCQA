"""DSE: density-preserving serialised encoder (PTv3-style) for point-cloud quality, used as the CLIP-PCQA "dse" branch.

Input per sample = dse_prep.py voxels of the FULL cloud (grid G): integer coords, features
[log point count, mean in-voxel offset (3), mean rgb (3)] and two serialisation orders (Z-order, Hilbert).
Three changes w.r.t. PTv3, all aimed at sampling / defect distortions:
  * grid pooling keeps the point count (density) and in-voxel offset instead of discarding them;
  * xCPE is a submanifold 3x3x3 conv in which every EMPTY neighbour direction adds its own learned
    embedding, so holes / missing surface ("there should be a point here") are explicit;
  * attention runs in windows of W voxels along the space-filling curve (alternating curves per block),
    covering the whole cloud instead of a few thousand query points.
Pure PyTorch (neighbour lookup by sorted keys + searchsorted), no spconv needed.
"""
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

OFFS = torch.tensor([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])  # 27, incl. centre
FEAT_DIM = 7


def neighbours(coord):
    """coord [N,3] long (< 1024) -> nbr [N,27] long, -1 where the neighbour voxel is empty."""
    key = (coord[:, 0] << 20) | (coord[:, 1] << 10) | coord[:, 2]
    sk, perm = torch.sort(key)
    q = coord[:, None, :] + OFFS.to(coord.device)[None]                       # [N,27,3]
    ok = ((q >= 0) & (q < 1024)).all(-1)
    qk = (q[..., 0] << 20) | (q[..., 1] << 10) | q[..., 2]
    pos = torch.searchsorted(sk, qk.clamp_min(0)).clamp_max(len(sk) - 1)
    hit = ok & (sk[pos] == qk)
    return torch.where(hit, perm[pos], torch.full_like(pos, -1))


class SubmConvEmpty(nn.Module):
    """3x3x3 submanifold conv; empty neighbour k contributes the learned vector e_k instead of nothing."""
    def __init__(self, c, empty=True):
        super().__init__()
        self.w = nn.Parameter(torch.randn(27, c, c) * (1.0 / (27 * c) ** 0.5))
        # ablation empty=False: plain submanifold conv (an empty neighbour contributes nothing, as in PTv3's xCPE)
        self.empty = nn.Parameter(torch.zeros(27, c), requires_grad=empty)
        self.norm = nn.LayerNorm(c)

    def forward(self, f, nbr):
        m = (nbr >= 0)
        # index_select (backward = atomic index_add) instead of f[nbr]: advanced-index backward sorts N*27 indices
        g = f.index_select(0, nbr.clamp_min(0).reshape(-1)).view(*nbr.shape, -1) * m[..., None].to(f.dtype)  # [N,27,C]
        out = torch.einsum('nkc,kco->no', g, self.w) + (~m).to(f.dtype) @ self.empty
        return self.norm(out)


class SerialBlock(nn.Module):
    def __init__(self, c, heads=4, window=512, empty=True, attn=True):
        super().__init__()
        self.cpe = SubmConvEmpty(c, empty)
        self.use_attn = attn  # ablation attn=False: conv + MLP only, no serialised window attention
        self.n1, self.n2 = nn.LayerNorm(c), nn.LayerNorm(c)
        self.attn = nn.MultiheadAttention(c, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(c, 4 * c), nn.GELU(), nn.Linear(4 * c, c))
        self.window = window

    def forward(self, f, nbr, order):
        f = f + self.cpe(f, nbr)
        if not self.use_attn:
            return f + self.mlp(self.n2(f))
        x = self.n1(f).index_select(0, order)                                  # serialise
        N, C, W = x.shape[0], x.shape[1], self.window
        pad = (-N) % W
        xw = F.pad(x, (0, 0, 0, pad)).view(-1, W, C)
        mask = torch.zeros(xw.shape[:2], dtype=torch.bool, device=f.device)
        if pad:
            mask.view(-1)[N:] = True
        # windows in chunks: one call over ~2k windows of a 1M-voxel cloud would need >7 GB for the attention alone
        y = torch.cat([self.attn(xw[i:i + 256], xw[i:i + 256], xw[i:i + 256], key_padding_mask=mask[i:i + 256],
                                 need_weights=False)[0] for i in range(0, len(xw), 256)]).reshape(-1, C)[:N]
        f = f.index_add(0, order, y)                                           # de-serialise + residual
        return f + self.mlp(self.n2(f))


class DSENet(nn.Module):
    def __init__(self, c=64, blocks=2, window=512, empty=True, count=True, attn=True, ckpt=False):
        super().__init__()
        self.ckpt = ckpt  # gradient checkpointing per block: needed when training on ALL voxels (up to ~1M per cloud)
        self.count = count  # ablation count=False: the log point count channel is zeroed (density-blind voxels)
        self.embed = nn.Sequential(nn.Linear(FEAT_DIM, c), nn.GELU(), nn.Linear(c, c))
        self.blocks = nn.ModuleList([SerialBlock(c, window=window, empty=empty, attn=attn) for _ in range(blocks)])
        self.att = nn.Linear(c, 1)
        self.out_dim = 3 * c

    def forward_one(self, coord, feat, order):
        nbr = neighbours(coord)
        if not self.count:
            feat = torch.cat([torch.zeros_like(feat[:, :1]), feat[:, 1:]], 1)
        f = self.embed(feat)
        for i, b in enumerate(self.blocks):
            o = order[i % order.shape[0]]                                      # alternate Z-order / Hilbert
            f = checkpoint(b, f, nbr, o, use_reentrant=False) if self.ckpt and self.training else b(f, nbr, o)
        w = torch.softmax(self.att(f), 0)
        return torch.cat([f.mean(0), f.max(0).values, (w * f).sum(0)])

    def forward(self, items):
        """items: list of dicts with coord [N,3] long, feat [N,7] float, order [2,N] long (one per sample)."""
        return torch.stack([self.forward_one(d['coord'], d['feat'], d['order']) for d in items])
