"""Neighbourhoods for the kernel point attention (KPA) network (CPU).

Per cloud (normalised to the unit sphere, same units as the 0.003 splat radius): Q random query
points (not FPS) and, for each bandwidth h in H, up to M neighbours from the FULL cloud within 3h,
plus the true neighbour count (so kernel sums can be corrected for the subsampling). Nothing is
normalised by local spacing: offsets are stored in absolute units.

Output <out_dir>/<name>.npz:
  q    [Q, 6]        query xyz + rgb
  nb   [S, Q, M, 6]  neighbour offset (xyz - query) + neighbour rgb, zero-padded
  cnt  [S, Q]        true number of neighbours within 3h (excluding the query)
usage: python kpa_prep.py <ply_root> <out_dir> [n_proc]
"""
import os, sys, glob
import numpy as np
from multiprocessing import Pool

Q, M = int(os.environ.get('KPA_Q', 2048)), 32   # KPA_Q: number of query points (S-KPA uses 16384)
SERIAL = os.environ.get('KPA_SERIAL') == '1'      # also store Z-order / Hilbert orders of the queries (S-KPA)
SURF = os.environ.get('KPA_SURF') == '1'  # add hole-covering surface queries
QUERY = os.environ.get('QUERY', 'random')  # 'fps': farthest-point queries (even coverage incl. sparse / damaged areas); 'mix': half FPS, half random


def query_indices(xyz, n, rng):
    """Query point indices for the chosen QUERY mode: random (density-proportional), fps (even coverage) or mix
    (half each: fps covers sparse / damaged regions, random keeps the dense, colour-rich ones)."""
    if QUERY == 'fps':
        return fps_indices(xyz, n, rng)
    if QUERY == 'mix':
        return np.r_[fps_indices(xyz, n // 2, rng), rng.choice(len(xyz), n - n // 2, replace=len(xyz) < n - n // 2)]
    return rng.choice(len(xyz), min(n, len(xyz)), replace=len(xyz) < n)


def fps_indices(xyz, n, rng, pool=50000):
    """Approximate farthest-point sampling: greedy FPS over a random pool of `pool` points (exact FPS on
    1M-point clouds is too slow); returns indices into xyz."""
    cand = rng.choice(len(xyz), min(pool, len(xyz)), replace=False)
    p = xyz[cand]
    sel = [int(rng.integers(len(p)))]
    d = np.linalg.norm(p - p[sel[0]], axis=1)
    for _ in range(min(n, len(p)) - 1):
        sel.append(int(d.argmax()))
        d = np.minimum(d, np.linalg.norm(p - p[sel[-1]], axis=1))
    return cand[np.array(sel)]
H = [0.003, 0.006, 0.012]


def read(path):
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(path)
        return np.asarray(pcd.points, np.float64), np.asarray(pcd.colors, np.float32)
    except (ImportError, OSError):
        from pytorch3d.io import IO
        pc = IO().load_pointcloud(path)
        return pc.points_packed().numpy().astype(np.float64), pc.features_packed().numpy().astype(np.float32)


def surface_queries(xyz, n, rng, img=128, splat=0.012):
    """Query positions on the EXPECTED surface, including where points are missing: render the cloud
    with a large splat (which closes gaps) from the 6 standard views at low resolution and back-project
    random covered pixels. Works on CPU."""
    import torch
    from pytorch3d.structures import Pointclouds
    from pytorch3d.renderer import look_at_view_transform, FoVOrthographicCameras, PointsRasterizationSettings, PointsRasterizer
    pc = Pointclouds(points=[torch.tensor(xyz, dtype=torch.float32)])
    out = []
    for elev, azim in [(0, 0), (0, 90), (0, 180), (0, -90), (90, 0), (-90, 0)]:
        R, T = look_at_view_transform(1, elev, azim)
        cam = FoVOrthographicCameras(R=R, T=T, znear=0.01)
        frag = PointsRasterizer(cameras=cam, raster_settings=PointsRasterizationSettings(image_size=img, radius=splat, points_per_pixel=1))(pc)
        z = frag.zbuf[0, ..., 0]
        ys, xs = torch.nonzero(z > 0, as_tuple=True)
        # pixel centre -> NDC (PyTorch3D: +x left, +y up, pixel (0,0) at top-left)
        ndc_x = 1 - (2 * xs.float() + 1) / img
        ndc_y = 1 - (2 * ys.float() + 1) / img
        pts = cam.unproject_points(torch.stack([ndc_x, ndc_y, z[ys, xs]], 1)[None], world_coordinates=True)[0]
        out.append(pts.numpy())
    out = np.concatenate(out)
    return out[rng.choice(len(out), n, replace=len(out) < n)]


def prep(path, out_dir):
    dst = os.path.join(out_dir, os.path.basename(path)[:-4] + '.npz')
    if os.path.exists(dst):
        return
    from scipy.spatial import cKDTree
    xyz, rgb = read(path)
    if len(rgb) != len(xyz):
        rgb = np.zeros((len(xyz), 3), np.float32)
    xyz = xyz - xyz.mean(0)
    xyz /= np.linalg.norm(xyz, axis=1).max()
    tree = cKDTree(xyz)
    rng = np.random.default_rng(0)
    if SURF:  # half random points, half back-projected surface positions (cover holes)
        qi = rng.choice(len(xyz), Q // 2, replace=len(xyz) < Q // 2)
        qpos = np.r_[xyz[qi], surface_queries(xyz, Q - Q // 2, rng)]
        qcol = np.r_[rgb[qi], rgb[tree.query(qpos[Q // 2:])[1]]]   # surface queries take the nearest colour
        src = np.r_[np.zeros(Q // 2), np.ones(Q - Q // 2)].astype(np.int8)
    else:
        qi = query_indices(xyz, Q, rng)
        qpos, qcol, src = xyz[qi], rgb[qi], np.zeros(len(qi), np.int8)
    nb = np.zeros((len(H), len(qpos), M, 6), np.float16)
    cnt = np.zeros((len(H), len(qpos)), np.int32)
    for s, h in enumerate(H):
        for i, n in enumerate(tree.query_ball_point(qpos, 3 * h, workers=1)):
            d = np.linalg.norm(xyz[n] - qpos[i], axis=1) if n else np.zeros(0)
            n = [j for j, dj in zip(n, d) if dj > 0]                 # drop the query point itself
            cnt[s, i] = len(n)
            if len(n) > M:
                n = rng.choice(n, M, replace=False)
            if len(n):
                nb[s, i, :len(n), :3] = xyz[n] - qpos[i]
                nb[s, i, :len(n), 3:] = rgb[n]
    extra = {}
    if SERIAL:  # serialisation of the query points for S-KPA's window attention (curves from dse_prep.py)
        from dse_prep import morton, hilbert
        g = np.clip(np.floor((qpos + 1) / 0.002), 0, 1023).astype(np.int64)
        extra['order'] = np.stack([np.argsort(morton(g), kind='stable'), np.argsort(hilbert(g), kind='stable')]).astype(np.int32)
    np.savez(dst, q=np.c_[qpos, qcol].astype(np.float16), nb=nb, cnt=cnt, src=src, **extra)


if __name__ == '__main__':
    root, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(root, '**', '*.ply'), recursive=True))
    files = files[int(os.environ.get('SHARD', 0))::int(os.environ.get('NSHARD', 1))]  # split across jobs
    with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 16) as p:
        p.starmap(prep, [(f, out_dir) for f in files], chunksize=1)
    print('KPA_PREP_DONE', len(os.listdir(out_dir)))
