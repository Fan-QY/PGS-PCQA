"""Render 6-view color + depth projections, following CLIP-PCQA/get_projections.py.

Same camera views, rasterization settings, white background and mask crop as
the original script; only I/O differs (reads .ply via open3d, takes paths as args,
shards the file list across processes).

usage: python render_projections.py <ply_root> <out_root> <shard_id> <num_shards>
"""
import os, sys, glob
import numpy as np
import torch
import cv2
try:  # open3d needs host OpenGL libs; on some headless hosts that import fails
    import open3d as o3d
except OSError:
    o3d = None
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (look_at_view_transform, FoVOrthographicCameras,
                                PointsRasterizationSettings, PointsRenderer,
                                PointsRasterizer, AlphaCompositor)

VIEWS = [(0, 0), (0, 90), (0, 180), (0, -90), (90, 0), (-90, 0)]
if os.environ.get('N_VIEWS', '6') == '20':
    # 20 views = face centres of a regular icosahedron (as in R3-PCQA), as (elevation, azimuth) in degrees
    import itertools
    phi = (1 + 5 ** 0.5) / 2
    verts = np.array([v for a, b in itertools.product((-1, 1), (-phi, phi))
                      for v in ((0, a, b), (a, b, 0), (b, 0, a))], dtype=float)
    faces = [f for f in itertools.combinations(range(12), 3)
             if all(abs(np.linalg.norm(verts[i] - verts[j]) - 2) < 1e-6 for i, j in itertools.combinations(f, 2))]
    cent = np.array([verts[list(f)].mean(0) for f in faces])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    VIEWS = [(float(np.degrees(np.arcsin(c[1]))), float(np.degrees(np.arctan2(c[0], c[2])))) for c in cent]


def normalize_verts(verts):
    verts = verts - verts.mean(axis=0)
    return verts / np.max(np.sqrt((verts ** 2).sum(axis=1)))


def crop_box(mask):
    ys, xs = np.nonzero(mask)
    return ys.min(), ys.max(), xs.min(), xs.max()


def render_one(path, out_color, out_depth, device):
    stem = os.path.splitext(os.path.basename(path))[0]
    if all(os.path.exists(os.path.join(out_color, f'{stem}_view_{k}.png')) for k in range(len(VIEWS))):
        return
    if o3d is not None:
        pcd = o3d.io.read_point_cloud(path)
        xyz = np.asarray(pcd.points, dtype=np.float32)
        rgb = np.asarray(pcd.colors, dtype=np.float32)  # open3d gives [0,1]
    else:
        from pytorch3d.io import IO
        pc = IO().load_pointcloud(path)
        xyz = pc.points_packed().numpy().astype(np.float32)
        rgb = pc.features_packed().numpy().astype(np.float32)  # pytorch3d also gives [0,1]
    keep = float(os.environ.get("DOWNSAMPLE", "1"))  # resampling ladder for self-anchoring
    if keep < 1:
        idx = np.random.default_rng(0).choice(len(xyz), int(len(xyz) * keep), replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
    verts = torch.tensor(normalize_verts(xyz), device=device)
    pc = Pointclouds(points=[verts], features=[torch.tensor(rgb, device=device)])
    for k, (elev, azim) in enumerate(VIEWS):
        R, T = look_at_view_transform(1, elev, azim)
        cameras = FoVOrthographicCameras(device=device, R=R, T=T, znear=0.01)
        rs = PointsRasterizationSettings(image_size=int(os.environ.get("IMAGE_SIZE", "1024")), radius=float(os.environ.get("SPLAT_RADIUS", "0.003")), points_per_pixel=10)
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=rs)
        renderer = PointsRenderer(rasterizer=rasterizer, compositor=AlphaCompositor())
        depth = rasterizer(pc)[1][0, :, :, 0].cpu()
        mask = (depth != -1).numpy()
        depth = torch.where(depth == -1, torch.tensor(0.0), depth).numpy()
        depth_img = (depth / depth[depth != 0].max() * 255).astype(np.uint8)
        img = renderer(pc)[0, ..., :3].cpu().numpy()
        img[~mask] = 1
        img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        # original crops with [min:max) of the mask bounding box
        y0, y1, x0, x1 = crop_box(mask)
        name = f'{stem}_view_{k}.png'
        cv2.imwrite(os.path.join(out_color, name), img[y0:y1, x0:x1])
        cv2.imwrite(os.path.join(out_depth, name), depth_img[y0:y1, x0:x1])


if __name__ == '__main__':
    ply_root, out_root, shard, nshard = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    out_color, out_depth = os.path.join(out_root, '6view'), os.path.join(out_root, '6view_depth')
    os.makedirs(out_color, exist_ok=True); os.makedirs(out_depth, exist_ok=True)
    files = sorted(glob.glob(os.path.join(ply_root, '**', '*.ply'), recursive=True))[shard::nshard]
    device = torch.device('cuda')
    for i, f in enumerate(files):
        render_one(f, out_color, out_depth, device)
        print(f'[{shard}] {i + 1}/{len(files)} {os.path.basename(f)}', flush=True)
    print('SHARD_DONE', shard)
