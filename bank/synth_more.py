"""Generic (non-codec) synthetic distortions of a SOURCE reference, written as PLY, for the within-object severity ranking
loss. One distortion bank for every source dataset: colour noise (Gaussian, salt-and-pepper, luma), global colour shifts
(contrast, saturation, brightness), colour bit-depth quantisation, local geometric deformation (missing / offset / rotated
patches), geometric Gaussian displacement and random downsampling; 5 levels each (L1 mildest). Geometric magnitudes are relative to the
cloud's median nearest-neighbour spacing s, patch sizes to its bounding-sphere radius. Deterministic per (cloud, type).
usage: python synth_more.py <vox_ref.ply> <out_dir>
"""
import os, sys, zlib
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep_refs import read_ply, write_ply

LV = {'cgn': [6, 12, 20, 30, 45], 'csp': [0.01, 0.025, 0.05, 0.1, 0.2], 'clm': [6, 12, 20, 30, 45],
      'cct': [0.8, 0.65, 0.5, 0.35, 0.2], 'csat': [0.75, 0.55, 0.35, 0.18, 0.0], 'cms': [15, 30, 45, 65, 90],
      'cq': [5, 4, 3, 2, 1], 'lmiss': [0.02, 0.04, 0.07, 0.11, 0.16], 'loff': [2, 4, 7, 11, 16],
      'lrot': [6, 12, 20, 30, 45], 'gshift': [0.5, 1, 2, 3, 4.5],
      'ds': [0.7, 0.5, 0.35, 0.2, 0.1]}   # ds: random downsampling, keep ratio
Y = np.array([0.299, 0.587, 0.114])


def patches(xyz, rng, frac, n=6):
    """n spherical patches around random points covering ~frac of the points in total -> list of index arrays"""
    tree = cKDTree(xyz); k = max(8, int(frac * len(xyz) / n))
    return [tree.query(xyz[rng.integers(len(xyz))], k=k)[1] for _ in range(n)]


def rot(axis, deg):
    a = np.deg2rad(deg); axis = axis / np.linalg.norm(axis); K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def degrade(xyz, rgb, t, v, s, rng):
    xyz, rgb = xyz.copy(), rgb.astype(np.float64).copy()
    if t == 'cgn':
        rgb += rng.normal(0, v, rgb.shape)
    elif t == 'csp':
        m = rng.random(len(rgb)) < v; rgb[m] = np.where(rng.random((m.sum(), 1)) < 0.5, 0, 255)
    elif t == 'clm':
        rgb += rng.normal(0, v, (len(rgb), 1))
    elif t == 'cct':
        rgb = rgb.mean(0) + (rgb - rgb.mean(0)) * v
    elif t == 'csat':
        g = (rgb @ Y)[:, None]; rgb = g + (rgb - g) * v
    elif t == 'cms':
        rgb += v * (1 if rng.random() < 0.5 else -1)
    elif t == 'cq':
        q = 256 // (2 ** v); rgb = np.floor(rgb / q) * q + q / 2
    elif t == 'lmiss':
        keep = np.ones(len(xyz), bool)
        for p in patches(xyz, rng, v): keep[p] = False
        xyz, rgb = xyz[keep], rgb[keep]
    elif t == 'loff':
        for p in patches(xyz, rng, 0.06):
            d = rng.normal(size=3); xyz[p] += d / np.linalg.norm(d) * v * s
    elif t == 'lrot':
        for p in patches(xyz, rng, 0.06):
            c = xyz[p].mean(0); xyz[p] = (xyz[p] - c) @ rot(rng.normal(size=3), v).T + c
    elif t == 'gshift':
        xyz += rng.normal(0, v * s, xyz.shape)
    elif t == 'ds':
        keep = rng.random(len(xyz)) < v; xyz, rgb = xyz[keep], rgb[keep]
    return xyz, np.clip(rgb, 0, 255)


if __name__ == '__main__':
    src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
    b = os.path.basename(src)[:-4]; xyz, rgb = read_ply(src)
    s = float(np.median(cKDTree(xyz).query(xyz[np.random.default_rng(0).choice(len(xyz), min(20000, len(xyz)), replace=False)], k=2)[0][:, 1]))
    only = os.environ.get('TYPES', '')   # e.g. TYPES=ds: generate only these types (bank extension)
    for t, levels in LV.items():
        if only and t not in only.split(','): continue
        for i, v in enumerate(levels):
            dst = os.path.join(out, f'{b}_{t}_L{i + 1}.ply')
            if os.path.exists(dst): continue
            rng = np.random.default_rng(zlib.crc32(f'{b}_{t}'.encode()))   # same patches / noise field across levels of a type
            write_ply(dst, *degrade(xyz, rgb, t, v, s, rng))
    print(f'{b}: s={s:.3f}, {len(LV) * 5} samples')
