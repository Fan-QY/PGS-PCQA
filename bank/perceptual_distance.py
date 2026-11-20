"""Perceptual distance d(.) of the synthetic bank (the "teacher" of the perceptual gate): full-reference distance between
each synthetic sample's 6 rendered colour views and the 6 views of its (10-bit voxelised) reference.
  deep : LPIPS-style deep-feature distance with ImageNet VGG19 (torchvision weights, not trained): relu1_2, relu2_2,
         relu3_4, relu4_4, relu5_4 features, unit-normalised along channels, squared difference averaged over space,
         layers averaged uniformly (i.e. LPIPS without its learned linear layer)
  ssim : grey-level SSIM (skimage, optional) as a classical reference; not used in training
Views are aligned by resizing the distorted projection to the reference projection's size (same camera and per-cloud
normalisation), then both to <=512 px on the long side; distances are averaged over the 6 views.
usage: python bank/perceptual_distance.py <bank 6view dir> <reference 6view dir> <csv name,mos> <ref-of-name regex> <out.csv>
       e.g. regex '(.+)_[a-z]+_L\d+\.ply$' maps '<ref>_<type>_L<level>.ply' to '<ref>'.
The training code reads the columns name,deep of <out.csv> (--synth_teacher). VGG19 weights: env VGG19_PTH or the
torchvision cache.
"""
import sys, re, os
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from PIL import Image
from torchvision.models import vgg19
try:
    from skimage.metrics import structural_similarity   # optional classical reference
except ImportError:
    structural_similarity = None

dist_dir, ref_dir, csv, pat, out = sys.argv[1:6]
dev = torch.device('cuda')
vgg = vgg19(); vgg.load_state_dict(torch.load(os.environ.get('VGG19_PTH', os.path.expanduser('~/.cache/torch/hub/checkpoints/vgg19-dcbb9e9d.pth'))))
feat = vgg.features.eval().to(dev)
TAPS = {3, 8, 17, 26, 35}   # relu1_2, relu2_2, relu3_4, relu4_4, relu5_4
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev)[:, None, None]; STD = torch.tensor([0.229, 0.224, 0.225], device=dev)[:, None, None]


def load(p, size=None):
    im = Image.open(p).convert('RGB')
    if size is not None and im.size != size:
        im = im.resize(size, Image.BILINEAR)
    s = 512 / max(im.size)
    if s < 1:
        im = im.resize((max(1, round(im.size[0] * s)), max(1, round(im.size[1] * s))), Image.BILINEAR)
    if min(im.size) < 64:   # very thin views (some LS objects): VGG's five poolings need >= 32 px on each side
        t = 64 / min(im.size); im = im.resize((round(im.size[0] * t), round(im.size[1] * t)), Image.BILINEAR)
    return np.asarray(im)


@torch.no_grad()
def deep(a, b):
    x = torch.stack([torch.from_numpy(a), torch.from_numpy(b)]).to(dev).permute(0, 3, 1, 2).float() / 255
    x = (x - MEAN) / STD; d = []
    for i, layer in enumerate(feat):
        x = layer(x)
        if i in TAPS:
            u = x / (x.norm(dim=1, keepdim=True) + 1e-10); d.append(((u[0] - u[1]) ** 2).sum(0).mean().item())
        if i >= max(TAPS): break
    return float(np.mean(d))


df = pd.read_csv(csv, encoding='utf-8-sig'); rows = []
for n, mos in zip(df.name, df.mos):
    ref = re.match(pat, n).group(1); st = n[:-4]; dd, ss = [], []
    for v in range(6):
        r = Image.open(os.path.join(ref_dir, f'{ref}_view_{v}.png'))
        a = load(os.path.join(ref_dir, f'{ref}_view_{v}.png')); b = load(os.path.join(dist_dir, f'{st}_view_{v}.png'), r.size)
        if b.shape != a.shape:
            b = np.asarray(Image.fromarray(b).resize((a.shape[1], a.shape[0]), Image.BILINEAR))
        dd.append(deep(a, b)); ss.append(structural_similarity(a.mean(-1), b.mean(-1), data_range=255) if structural_similarity else np.nan)
    rows.append({'name': n, 'mos': mos, 'ref': ref, 'deep': np.mean(dd), 'ssim': np.mean(ss)})
pd.DataFrame(rows).to_csv(out, index=False); print('wrote', len(rows))
