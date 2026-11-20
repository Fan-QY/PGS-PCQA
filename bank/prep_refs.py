"""Reference clouds -> G-PCC input: coordinates scaled so the longest bbox side spans 0..1023 (10-bit), rounded,
duplicates merged (mean colour), written as binary little-endian PLY with float x,y,z and uchar red,green,blue.
Reads ascii or binary PLY with any property order (only x,y,z,red,green,blue are used; faces are ignored).
usage: python prep_refs.py <in.ply> <out.ply>
"""
import sys
import numpy as np

TYPES = {'float': 'f4', 'float32': 'f4', 'double': 'f8', 'float64': 'f8', 'int8': 'i1', 'uint16': 'u2', 'int16': 'i2', 'uint32': 'u4', 'uchar': 'u1', 'uint8': 'u1', 'int': 'i4', 'int32': 'i4',
         'uint': 'u4', 'short': 'i2', 'ushort': 'u2', 'char': 'i1'}


def read_ply(path):
    with open(path, 'rb') as f:
        fmt, n, props, in_vertex = None, 0, [], False
        while True:
            line = f.readline().decode('ascii', 'ignore').strip()
            if line.startswith('format'):
                fmt = line.split()[1]
            elif line.startswith('element'):
                in_vertex = line.split()[1] == 'vertex'
                if in_vertex:
                    n = int(line.split()[2])
            elif line.startswith('property') and in_vertex and line.split()[1] != 'list':
                props.append((line.split()[2], TYPES[line.split()[1]]))
            elif line == 'end_header':
                break
        if fmt == 'ascii':
            a = np.loadtxt(f, max_rows=n, usecols=range(len(props)), dtype=np.float64)
            cols = {p: a[:, i] for i, (p, _) in enumerate(props)}
        else:
            dt = np.dtype([(p, ('<' if fmt == 'binary_little_endian' else '>') + t) for p, t in props])
            a = np.frombuffer(f.read(n * dt.itemsize), dtype=dt, count=n)
            cols = {p: a[p].astype(np.float64) for p, _ in props}
    xyz = np.stack([cols['x'], cols['y'], cols['z']], 1)
    rgb = np.stack([cols['red'], cols['green'], cols['blue']], 1)
    return xyz, rgb


def write_ply(path, xyz, rgb):
    dt = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    v = np.empty(len(xyz), dtype=dt)
    v['x'], v['y'], v['z'] = xyz.T
    v['red'], v['green'], v['blue'] = np.clip(np.round(rgb), 0, 255).astype(np.uint8).T
    with open(path, 'wb') as f:
        f.write((f'ply\nformat binary_little_endian 1.0\nelement vertex {len(v)}\nproperty float x\nproperty float y\n'
                 'property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n').encode())
        f.write(v.tobytes())


if __name__ == '__main__':
    xyz, rgb = read_ply(sys.argv[1])
    q = np.round((xyz - xyz.min(0)) * (1023.0 / (xyz.max(0) - xyz.min(0)).max())).astype(np.int64)
    key = (q[:, 0] << 22) | (q[:, 1] << 11) | q[:, 2]
    u, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    col = np.zeros((len(u), 3)); np.add.at(col, inv, rgb); col /= cnt[:, None]
    pos = np.stack([u >> 22, (u >> 11) & 2047, u & 2047], 1).astype(np.float64)
    write_ply(sys.argv[2], pos, col)
    print(f'{sys.argv[1]}: {len(xyz)} -> {len(u)} points (10-bit voxels)')
