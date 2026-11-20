"""Per-sample predictions of CLIP-PCQA checkpoints on each fold's test split.

usage: python eval_folds.py --code <CLIP-PCQA dir> --ckpt_pattern '<dir>/SJTU_{k}_best_model.pth'
         --folds 1-9 --database SJTU --color <6view dir> --depth <6view_depth dir>
         [--use_geometry --pc <point cloud dir>] --out preds.csv
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import torch
from torchvision import transforms

ap = argparse.ArgumentParser()
ap.add_argument('--code', required=True)
ap.add_argument('--ckpt_pattern', required=True)
ap.add_argument('--folds', default='1-9')
ap.add_argument('--database', default='SJTU')
ap.add_argument('--color', required=True)
ap.add_argument('--depth', required=True)
ap.add_argument('--use_geometry', action='store_true')
ap.add_argument('--pc', default='')
ap.add_argument('--color2', default='')
ap.add_argument('--color2_mode', default='delta')
ap.add_argument('--tam', action='store_true', help='model was trained with text-guided token pooling (--tam)')
ap.add_argument('--kpa_fuse', action='store_true', help='model was trained with --kpa_fuse')
ap.add_argument('--kpa_mix', action='store_true', help='model was trained with --kpa_mix (needs the aux head at test time)')
ap.add_argument('--kpa_gate', action='store_true', help='model was trained with --kpa_gate (the aux head is then needed at test time)')
ap.add_argument('--out', required=True)
ap.add_argument('--view_mode', default='crop', choices=['crop', 'resize', 'fragment'])
ap.add_argument('--view_size', type=int, default=224)
ap.add_argument('--csv_suffix', default='', help='evaluate on test_<k><suffix>.csv (inner split of a nested run)')
ap.add_argument('--views', type=int, default=6)
ap.add_argument('--tta', type=int, default=1, help='1 = centre crop; 5 = centre + 4 corner crops; 9 / 16 = 3x3 / 4x4 grid of crops (as main.py --test_tta)')
ap.add_argument('--tta_fg_weight', action='store_true', help='weight the crops by their object-pixel share (as main.py --tta_fg_weight)')
ap.add_argument('--score_db', default='', help='cross-dataset: build the model with THIS database\'s score bins (the training '
                'database of the checkpoint) while --database selects the test data / csv files (default: --database)')
ap.add_argument('--wise', type=float, default=1.0, help='WiSE-FT: CLIP encoder weights = wise * fine-tuned + (1 - wise) * zero-shot (OpenAI); 1 = checkpoint as is')
ap.add_argument('--tta_batched', action='store_true', help='all crops of a sample from ONE read of its views, one forward batch (as main.py --tta_batched); same predictions, faster')
a = ap.parse_args()

sys.path.insert(0, a.code)
from models.CLIPPCQA_Net import CLIPPCQA_Net
from utils.CLIPPCQA_Dataset import CLIPPCQA_Dataset, pcqa_collate, to_device

INFO = {'SJTU': ('sjtu_data_info', [2.0, 4.0, 6.0, 8.0, 10.0]),
        'LS_PCQA_part': ('ls_pcqa_data_info', [1.0, 2.0, 3.0, 4.0, 5.0]),
        'BASICS': ('basics_data_info', [0.0, 1.0, 2.0, 3.0, 4.0]),
        'WPC': ('wpc_data_info', [20.0, 40.0, 60.0, 80.0, 100.0])}
info_dir, score_list = INFO[a.database]
if a.score_db:
    score_list = INFO[a.score_db][1]

# same model config as main.py (parse_args defaults + extend_args)
args = argparse.Namespace(n_ctx=16, csc=True, ctx_init='', prec='fp32', class_token_position='middle',
                          subsample_classes='all', use_geometry=a.use_geometry, ablation_mask=None,
                          data_dir_color2=a.color2, color2_mode=a.color2_mode, tam=a.tam, kpa_fuse=a.kpa_fuse, kpa_gate=a.kpa_gate, kpa_mix=a.kpa_mix, view_size=a.view_size)
quality_classes = ['bad', 'poor', 'fair', 'good', 'excellent']
if a.tta in (9, 16):  # same positions and order as main.py
    g = np.linspace(0, 1, int(round(a.tta ** 0.5)))
    POS = [(float(y), float(x)) for y in g for x in g]
else:
    POS = [(0.5, 0.5), (0, 0), (0, 1), (1, 0), (1, 1)][:a.tta]


def crop_weights(names):
    """[n, len(POS)] crop weights, identical to main.py: object (non-white) pixel share of each colour crop averaged
    over the views (narrow views: the centre square resized to 224, as the dataset feeds it), normalised per sample."""
    from PIL import Image
    w = np.ones((len(names), len(POS)))
    if a.tta_fg_weight:
        for i_n, nm in enumerate(names):
            for i_v in range(a.views):
                im = Image.open(os.path.join(a.color, nm.split('.')[0] + '_view_' + str(i_v) + '.png')).convert('RGB')
                if min(im.size) < 224:
                    im = transforms.Resize((224, 224))(transforms.CenterCrop(min(im.size))(im))
                im = np.asarray(im); hh, ww = im.shape[:2]
                for c_i, (py, px) in enumerate(POS):
                    t0, l0 = int(round((hh - 224) * py)), int(round((ww - 224) * px))
                    w[i_n, c_i] += (im[t0:t0 + 224, l0:l0 + 224].min(-1) < 240).mean() / a.views - (1.0 / a.views)
        w = w + 1e-6
    return w / w.sum(1, keepdims=True)


def crop_at(py, px):
    # same window for colour and depth (both go through this transform); images narrower than 224
    # were already squared and resized by the dataset, so the crop is then the whole image
    def f(im):
        w, h = im.size
        return transforms.functional.crop(im, int(round((h - 224) * py)), int(round((w - 224) * px)), 224, 224)
    return transforms.Compose([transforms.Lambda(f), transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
device = torch.device('cuda')
lo, hi = map(int, a.folds.split('-'))
rows = []
for k in range(lo, hi + 1):
    test_csv = os.path.join(a.code, 'csvfiles', info_dir, f'test_{k}{a.csv_suffix}.csv')
    raw = os.path.join(a.code, 'csvfiles', info_dir, 'total_raw_score.csv')
    dss = [CLIPPCQA_Dataset(data_dir_color=a.color, data_dir_depth=a.depth, datainfo_path=test_csv,
                            raw_score_path=raw, transform=crop_at(*pos), img_length_read=a.views, is_train=False,
                            data_dir_pc=a.pc, use_geometry=a.use_geometry, geom_num_points=2048,
                            data_dir_color2=a.color2) for pos in POS]
    if a.view_mode != 'crop':  # the dataset produces the final view (see main.py --view_mode); no TTA
        assert a.tta == 1
        plain = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        for d_ in dss:
            d_.transform, d_.view_mode = plain, a.view_mode
            if a.view_mode == 'resize':
                d_.crop_size = a.view_size
    ds = dss[0]
    names = list(pd.read_csv(test_csv)['name'])
    pc_paths = [ds._resolve_point_cloud_path(i, n) for i, n in enumerate(names)] if a.use_geometry else [''] * len(names)
    sd = torch.load(a.ckpt_pattern.format(k=k), map_location='cpu')
    if a.kpa_gate or a.kpa_mix:  # the gate reads the aux head: rebuild it with the checkpoint's number of distortion types
        from utils.deg_labels import P_MAX
        w = [v for kk, v in sd.items() if kk.endswith('geom_aux_head.3.weight')][0]
        args.aux_ntypes = w.shape[0] // (1 + P_MAX)
    model = CLIPPCQA_Net(device, args, score_list, quality_classes).to(device)
    sd = {kk[7:] if kk.startswith('module.') else kk: v for kk, v in sd.items()}
    if a.wise != 1.0:  # zero-shot CLIP weights as built by the model, before the checkpoint overwrites them
        zs = [{n: t.detach().clone() for n, t in enc.state_dict().items()} for enc in (model.clip_texture, model.clip_depth)]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if a.wise != 1.0:  # in place, so modules sharing these tensors (text encoder) stay consistent
        with torch.no_grad():
            for enc, z in zip((model.clip_texture, model.clip_depth), zs):
                for n, t in enc.state_dict().items():
                    if t.is_floating_point():
                        t.copy_(a.wise * t + (1 - a.wise) * z[n].to(t.device))
    # training-only heads (aux degradation head, FCA projection) may be absent from the eval model; anything else must match
    assert not [k for k in unexpected if not k.startswith(('geom_aux_head', 'geom_fca_', 'geom_kpa_auxhead', 'geom_kpa_drophead'))] and not missing, \
        f'checkpoint/model mismatch (missing={missing}, unexpected={unexpected}): pass the matching model flags'
    if missing or unexpected:
        print(f'fold {k}: missing={missing[:5]} unexpected={unexpected[:5]}', flush=True)
    model.eval()
    scores = np.zeros(len(names)); cw = crop_weights(names)
    batched = a.tta_batched and len(POS) > 1
    if batched:  # same windows / narrow-view handling as main.py _TTAViews
        assert not a.use_geometry and a.view_mode == 'crop'
        norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        from PIL import Image

        class _TTAViews(torch.utils.data.Dataset):
            def __len__(self):
                return len(names)
            def __getitem__(self, i):
                item = dss[0][i]
                nm = names[i].split('.')[0]; cols, deps = [], []
                for v in range(a.views):
                    c = Image.open(os.path.join(a.color, nm + '_view_' + str(v) + '.png')).convert('RGB')
                    d = Image.open(os.path.join(a.depth, nm + '_view_' + str(v) + '.png')).convert('RGB')
                    w, h = c.size
                    assert d.size == c.size
                    if min(w, h) < 224:
                        sq = transforms.Compose([transforms.CenterCrop(min(w, h)), transforms.Resize((224, 224))])
                        c, d = sq(c), sq(d); w, h = c.size
                    cc, dd = [], []
                    for py, px in POS:
                        t0, l0 = int(round((h - 224) * py)), int(round((w - 224) * px))
                        cc.append(norm(transforms.functional.crop(c, t0, l0, 224, 224))); dd.append(norm(transforms.functional.crop(d, t0, l0, 224, 224)))
                    cols.append(torch.stack(cc)); deps.append(torch.stack(dd))
                return torch.stack(cols, 1), torch.stack(deps, 1), (item[2] if a.color2 else None), float(item[-2])

        with torch.no_grad():
            for i, (cols, deps, c2s, mos) in enumerate(torch.utils.data.DataLoader(_TTAViews(), batch_size=None, shuffle=False, num_workers=8)):
                sc = []
                for c0 in range(0, cols.shape[0], 16):
                    cb, db = cols[c0:c0 + 16].to(device), deps[c0:c0 + 16].to(device)
                    c2b = None if c2s is None else [t[None].repeat(cb.shape[0], *([1] * t.dim())).to(device) for t in c2s]
                    sc.append(model(cb, db, point_patches=None, color2_imgs=c2b)[2].squeeze(1).float().cpu())
                scores[i] = float((torch.cat(sc).numpy() * cw[i]).sum())
                rows.append({'fold': k, 'name': names[i], 'mos': mos, 'pred': scores[i], 'pc_path': pc_paths[i]})
    for j_ds, ds in enumerate([] if batched else dss):
      loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0 if a.use_geometry else 8, collate_fn=pcqa_collate)
      with torch.no_grad():
        for i, batch in enumerate(loader):
            if a.use_geometry:
                imgs, depth, pts, mos, _ = batch
                pts = pts.to(device)
            elif a.color2:
                imgs, depth, c2, mos, _ = batch
                pts = None
            else:
                imgs, depth, mos, _ = batch
                pts = None
            c2 = to_device(c2, device) if a.color2 else None
            _, _, score, _, _ = model(imgs.to(device), depth.to(device), point_patches=pts, color2_imgs=c2)
            scores[i] += cw[i, j_ds] * score.item()
            if ds is dss[-1]:
                rows.append({'fold': k, 'name': names[i], 'mos': mos.item(), 'pred': scores[i],
                             'pc_path': pc_paths[i]})
    print(f'fold {k} done: {len(names)} samples', flush=True)
    del model; torch.cuda.empty_cache()
pd.DataFrame(rows).to_csv(a.out, index=False)
print('WROTE', a.out)
