"""TMC13 reconstructions (float64 xyz, colour stored g,b,r, empty face list) -> plain binary PLY with float32 x,y,z and
uchar red,green,blue, which open3d reads reliably. usage: python clean_ply.py <in_dir> <out_dir>"""
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep_refs import read_ply, write_ply
os.makedirs(sys.argv[2], exist_ok=True)
for f in sorted(glob.glob(os.path.join(sys.argv[1], '*.ply'))):
    xyz, rgb = read_ply(f); write_ply(os.path.join(sys.argv[2], os.path.basename(f)), xyz, rgb)
print('cleaned', len(glob.glob(os.path.join(sys.argv[2], '*.ply'))))
