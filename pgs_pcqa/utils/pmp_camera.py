"""Camera utilities for PMP-PCQA (point <-> image-token mapping).

Exactly mirrors ../render_projections.py:
  * cloud normalised by normalize_verts (mean-centred, divided by the max radius),
  * look_at_view_transform(1, elev, azim) + FoVOrthographicCameras(znear=0.01) with the default
    unit NDC box, so NDC xy == view-space xy and view-space z is the depth,
  * 1024x1024 points rasterisation, splat radius 0.003 (NDC units),
  * the saved PNG is img[y0:y1, x0:x1] where (y0, y1, x0, x1) = (min, max) of the mask indices.

pytorch3d conventions: X_view = X_world @ R + T (row vectors); NDC +x points left and +y up, the
rasteriser's pixel (row i, col j) has its centre at NDC x = 1 - (2j+1)/W, y = 1 - (2i+1)/H, i.e. the
continuous screen coordinate is U = (1 - x_ndc) * W/2, V = (1 - y_ndc) * H/2 (pixel centre at j+0.5).

Image-token frame: the dataset crops a window (top, left, side) from the PNG and resizes it to 224
(scale s = 224/side, s = 1 when no resize).  A 224-crop pixel (u, v) therefore sits at render screen
coordinates U = ox + u/s, V = oy + v/s with ox = x0 + left, oy = y0 + top.  crop_meta = (ox, oy, s).
"""
import math
import numpy as np
import torch

VIEWS = [(0, 0), (0, 90), (0, 180), (0, -90), (90, 0), (-90, 0)]
RENDER_SIZE = 1024
SPLAT_RADIUS = 0.003


def view_RT(views=VIEWS):
    """R [M,3,3], T [M,3] of the render cameras (float32, CPU)."""
    from pytorch3d.renderer import look_at_view_transform
    Rs, Ts = [], []
    for elev, azim in views:
        R, T = look_at_view_transform(1, elev, azim)
        Rs.append(R[0]); Ts.append(T[0])
    return torch.stack(Rs).float(), torch.stack(Ts).float()


def normalize_verts(verts):
    verts = verts - verts.mean(axis=0)
    return verts / np.max(np.sqrt((verts ** 2).sum(axis=1)))


def world_to_screen(xyz, R, T, size=RENDER_SIZE):
    """xyz [..., 3] world -> (U, V, depth) continuous render-screen coords, for one camera (R [3,3], T [3])."""
    xv = xyz @ R + T
    U = (1 - xv[..., 0]) * size / 2
    V = (1 - xv[..., 1]) * size / 2
    return U, V, xv[..., 2]


def mask_bbox(xyz_np, R, T, size=RENDER_SIZE, radius=SPLAT_RADIUS):
    """Analytic mask bounding box of the splat rasterisation: (y0, y1, x0, x1) with the same min/max
    convention as render_projections.crop_box. A pixel is covered when its centre is within `radius`
    (NDC) of a projected point; in pixels the radius is radius*size/2."""
    xv = xyz_np @ R.numpy() + T.numpy()
    xv = xv[xv[:, 2] >= 0]  # the rasteriser skips points behind the camera
    U = (1 - xv[:, 0]) * size / 2
    V = (1 - xv[:, 1]) * size / 2
    rp = radius * size / 2

    def lo_hi(a, b):
        # covered index range along axis a given the offset to the nearest pixel centre along b
        db = np.abs(b - (np.floor(b) + 0.5))           # distance to the nearest pixel centre on the other axis
        h = np.sqrt(np.clip(rp ** 2 - db ** 2, 0, None))
        lo = np.ceil(a - 0.5 - h + 1e-9).astype(np.int64)    # smallest j with (j+.5-a)^2 < h^2
        hi = np.floor(a - 0.5 + h - 1e-9).astype(np.int64)   # largest j
        ok = (hi >= lo) & (db < rp)
        return lo[ok], hi[ok]

    xl, xh = lo_hi(U, V)
    yl, yh = lo_hi(V, U)
    x0, x1 = max(int(xl.min()), 0), min(int(xh.max()), size - 1)
    y0, y1 = max(int(yl.min()), 0), min(int(yh.max()), size - 1)
    return y0, y1, x0, x1


# ----------------------------------------------------------------------------------------------- torch, batched
def token_ray_points(crop_meta, R, T, n_depth=64, d_min=0.0, d_max=2.0, img=224, grid=14, size=RENDER_SIZE):
    """World coordinates of the n_depth samples on the ray of every image-token centre (Eq. 3-4).
    crop_meta [B,M,3] (ox, oy, s); R [M,3,3]; T [M,3] -> [B, M, grid*grid, n_depth, 3]."""
    dev = crop_meta.device
    step = img / grid
    n = torch.arange(grid * grid, device=dev)
    u = step * ((n % grid).float() + 0.5)                     # Eq. 3
    v = step * (torch.div(n, grid, rounding_mode='floor').float() + 0.5)
    ox, oy, s = crop_meta[..., 0:1], crop_meta[..., 1:2], crop_meta[..., 2:3]   # [B,M,1]
    U = ox + u / s                                             # [B,M,N]
    V = oy + v / s
    x = 1 - 2 * U / size
    y = 1 - 2 * V / size
    d = d_min + torch.arange(n_depth, device=dev).float() * (d_max - d_min) / (n_depth - 1)   # Eq. 4
    B, M, N = x.shape
    xv = torch.stack([x.unsqueeze(-1).expand(B, M, N, n_depth), y.unsqueeze(-1).expand(B, M, N, n_depth),
                      d.view(1, 1, 1, -1).expand(B, M, N, n_depth)], -1)           # view coords
    # X_world = (X_view - T) @ R^T
    Rm, Tm = R.to(dev), T.to(dev)
    return torch.einsum('bmndj,mij->bmndi', xv - Tm.view(1, M, 1, 1, 3), Rm)


def points_to_tokens(xyz, crop_meta, R, T, img=224, grid=14, size=RENDER_SIZE):
    """Project world points into every view and return the token index they fall in (Eq. 7-8).
    xyz [B,P,3], crop_meta [B,M,3] -> idx [B,P,M] (long), valid [B,P,M] (bool: inside the 224 crop)."""
    dev = xyz.device
    Rm, Tm = R.to(dev), T.to(dev)
    xv = torch.einsum('bpj,mjk->bmpk', xyz, Rm) + Tm.view(1, -1, 1, 3)       # [B,M,P,3]
    U = (1 - xv[..., 0]) * size / 2
    V = (1 - xv[..., 1]) * size / 2
    ox, oy, s = crop_meta[..., 0:1], crop_meta[..., 1:2], crop_meta[..., 2:3]
    u = (U - ox) * s
    v = (V - oy) * s
    valid = (u >= 0) & (u < img) & (v >= 0) & (v < img)
    step = img / grid
    col = torch.clamp(torch.floor(u / step), 0, grid - 1).long()
    row = torch.clamp(torch.floor(v / step), 0, grid - 1).long()
    idx = row * grid + col
    return idx.permute(0, 2, 1), valid.permute(0, 2, 1)
