import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
import random
from torchvision import transforms
from torch.utils import data
from PIL import Image

class CLIPPCQA_Dataset(data.Dataset):
    """Read data from the original dataset for feature extraction"""
    def __init__(
        self,
        data_dir_color,
        data_dir_depth,
        datainfo_path,
        raw_score_path,
        transform,
        crop_size = 224,
        img_length_read = 6,
        is_train = True,
        data_dir_pc = "",
        use_geometry = False,
        geom_num_points = 2048,
        data_dir_color2 = "",
    ):
        super(CLIPPCQA_Dataset, self).__init__()
        self.aux = None  # name -> (type_id, params); set by main.py for --aux_deg, appended to each item
        self.shared_crop = False  # --shared_crop: colour and depth of a view get the SAME random crop window (training)
        self.view_pool = None  # --view_pool: list of (colour dir, depth dir, view id) to draw the training views from
        self.hflip = False  # --hflip: random horizontal mirror of each training view (colour and depth together)
        self.fca_bbox = None  # --fca: name -> [6,4] render mask bbox (y0,y1,x0,x1); crops then also return (ox,oy,s)
        # --view_mode (set by main.py / eval_folds.py): 'crop' = CLIP-PCQA default (224 crop of the view);
        # 'resize' = whole view padded to a square and resized to crop_size; 'fragment' = 7x7 grid over the whole view,
        # one native-resolution 32x32 patch per cell (random in training, cell centre at test), stitched to 224
        self.view_mode = 'crop'
        dataInfo = pd.read_csv(datainfo_path, header = 0, sep=',', index_col=False, encoding="utf-8-sig")
        self.dataInfo = dataInfo
        self.raw_score = pd.read_csv(raw_score_path, header = 0, sep=',', index_col=False, encoding="utf-8-sig")
        self.ply_name = dataInfo[['name']]
        self.ply_mos = dataInfo['mos']
        self.crop_size = crop_size
        self.data_dir_color = data_dir_color
        self.data_dir_depth = data_dir_depth
        self.transform = transform
        self.img_length_read = img_length_read
        self.length = len(self.ply_name)
        self.is_train = is_train
        self.data_dir_pc = data_dir_pc
        self.use_geometry = use_geometry
        self.geom_num_points = geom_num_points
        # second color rendering of the same views (e.g. smaller splat radius); cropped identically to color
        # a .npz instead of image dirs = precomputed frozen-CLIP self-anchor vectors per sample
        self.anchor = None
        self.point_feat_dir = None
        self.kpa_dir = None
        self.dse_dir = None
        npz = [f for f in os.listdir(data_dir_color2)[:5] if f.endswith('.npz')] if os.path.isdir(data_dir_color2) else []
        if npz and 'coord' in np.load(os.path.join(data_dir_color2, npz[0])).files:
            # full-cloud voxels (dse_prep.py) for the density-preserving serialised encoder (DSE) branch
            self.dse_dir, data_dir_color2 = data_dir_color2, ''
        elif npz:
            # KPA neighbourhoods (kpa_prep.py) for the kernel point attention branch
            self.kpa_dir, data_dir_color2 = data_dir_color2, ''
        elif os.path.isdir(data_dir_color2) and any(f.endswith('.npy') for f in os.listdir(data_dir_color2)[:5]):
            # per-point geometric features (geo_feats.py) for the trainable geometry branch
            self.point_feat_dir, data_dir_color2 = data_dir_color2, ''
        if data_dir_color2.endswith('.npz'):
            z = np.load(data_dir_color2)
            self.anchor = {k: torch.from_numpy((z[k][1:] - z[k][0]).reshape(-1)).float() for k in z.files}
            data_dir_color2 = ''
        self.data_dir_color2 = data_dir_color2
        self.post = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.geom_path_columns = ["pointcloud", "point_cloud", "pc_path", "path", "name"]
        self.pc_file_index = self._build_point_cloud_index() if self.use_geometry and self.data_dir_pc else {}

    def _build_point_cloud_index(self):
        index = {}
        supported_exts = (".ply", ".xyz", ".txt", ".npy", ".pt")
        for root, _, files in os.walk(self.data_dir_pc):
            for filename in files:
                if filename.lower().endswith(supported_exts):
                    stem = os.path.splitext(filename)[0]
                    path = os.path.join(root, filename)
                    index.setdefault(stem, path)
                    index.setdefault(stem.lower(), path)
        return index

    def _candidate_point_cloud_stems(self, key):
        stem = os.path.splitext(os.path.basename(str(key)))[0]
        candidates = [stem]

        # SJTU distorted samples can be named statue_16.ply while the
        # corresponding reference point cloud is statue.ply.
        candidates.append(re.sub(r"_\d+$", "", stem))

        # WPC-style samples keep the reference object before these distortion tags.
        candidates.append(re.sub(r"_(gQP|gsigma|level|pqs|tsl)_.*$", "", stem))

        # LS-PCQA/BASICS-style samples keep the object before the distortion token.
        for token in [
            "Octree",
            "GammaNoise",
            "MultiplicativeGaussianNoise",
            "reconstruction",
            "limitlossyG-lossyA",
            "LumaNoise",
            "lossless-geom-lossy-attrs",
        ]:
            marker = "_" + token
            if marker in stem:
                candidates.append(stem.split(marker)[0])

        prefix_matches = [
            indexed_stem for indexed_stem in self.pc_file_index
            if stem.startswith(indexed_stem + "_")
        ]
        candidates.extend(sorted(prefix_matches, key=len, reverse=True))

        unique_candidates = []
        for candidate in candidates:
            if candidate and candidate not in unique_candidates:
                unique_candidates.append(candidate)
        return unique_candidates

    def _read_ply_xyz(self, pc_path):
        try:
            import open3d as o3d
            return np.asarray(o3d.io.read_point_cloud(pc_path).points, dtype=np.float32)
        except Exception:
            pass

        with open(pc_path, "rb") as f:
            header_lines = []
            while True:
                line = f.readline()
                if not line:
                    raise ValueError(f"Invalid PLY header: {pc_path}")
                decoded = line.decode("ascii", errors="ignore").strip()
                header_lines.append(decoded)
                if decoded == "end_header":
                    break

            fmt = next(line for line in header_lines if line.startswith("format ")).split()[1]
            vertex_count = 0
            vertex_props = []
            in_vertex = False
            for line in header_lines:
                parts = line.split()
                if len(parts) >= 3 and parts[0] == "element":
                    in_vertex = parts[1] == "vertex"
                    if in_vertex:
                        vertex_count = int(parts[2])
                elif in_vertex and len(parts) >= 3 and parts[0] == "property" and parts[1] != "list":
                    vertex_props.append((parts[2], parts[1]))

            prop_names = [name for name, _ in vertex_props]
            if not all(axis in prop_names for axis in ["x", "y", "z"]):
                raise ValueError(f"PLY file does not contain x/y/z properties: {pc_path}")

            if fmt == "ascii":
                data = np.loadtxt(f, max_rows=vertex_count)
                cols = [prop_names.index(axis) for axis in ["x", "y", "z"]]
                return data[:, cols].astype(np.float32)

            if fmt != "binary_little_endian":
                raise ValueError(f"Unsupported PLY format '{fmt}' in {pc_path}")

            dtype_map = {
                "char": "i1", "uchar": "u1", "int8": "i1", "uint8": "u1",
                "short": "<i2", "ushort": "<u2", "int16": "<i2", "uint16": "<u2",
                "int": "<i4", "uint": "<u4", "int32": "<i4", "uint32": "<u4",
                "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
            }
            dtype = np.dtype([(name, dtype_map[prop_type]) for name, prop_type in vertex_props])
            data = np.frombuffer(f.read(vertex_count * dtype.itemsize), dtype=dtype, count=vertex_count)
            return np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)

    def _resolve_point_cloud_path(self, idx, img_name):
        row = self.dataInfo.iloc[idx]

        keys = []
        for column in self.geom_path_columns:
            if column in row and pd.notna(row[column]):
                keys.append(str(row[column]))

        keys.append(img_name)

        for key in keys:
            if key in self.pc_file_index:
                return self.pc_file_index[key]
            if str(key).lower() in self.pc_file_index:
                return self.pc_file_index[str(key).lower()]

            # try absolute or relative paths
            key_path = key if os.path.isabs(key) else os.path.join(self.data_dir_pc, key)
            if os.path.exists(key_path):
                return key_path

            for stem in self._candidate_point_cloud_stems(key):
                if stem in self.pc_file_index:
                    return self.pc_file_index[stem]
                if stem.lower() in self.pc_file_index:
                    return self.pc_file_index[stem.lower()]

                for ext in [".ply", ".xyz", ".txt", ".npy", ".pt"]:
                    candidate = os.path.join(self.data_dir_pc, stem + ext)
                    if os.path.exists(candidate):
                        return candidate

        examples = list(self.pc_file_index.keys())[:10]
        raise FileNotFoundError(
            f"Point cloud file not found for {img_name} in {self.data_dir_pc}. "
            f"Tried exact sample names and reference-name fallbacks. Indexed examples: {examples}"
        )


    def _load_point_cloud(self, idx, img_name):
        pc_path = self._resolve_point_cloud_path(idx, img_name)
        if pc_path.endswith(".npy"):
            points = np.load(pc_path)
        elif pc_path.endswith(".pt"):
            points = torch.load(pc_path)
            if torch.is_tensor(points):
                points = points.detach().cpu().numpy()
        elif pc_path.endswith(".ply"):
            points = self._read_ply_xyz(pc_path)
        else:
            points = np.loadtxt(pc_path)

        points = np.asarray(points, dtype=np.float32)[:, :3]
        if points.shape[0] >= self.geom_num_points:
            indices = np.linspace(0, points.shape[0] - 1, self.geom_num_points).astype(np.int64)
            points = points[indices]
        else:
            pad_count = self.geom_num_points - points.shape[0]
            pad = np.repeat(points[-1:, :], pad_count, axis=0)
            points = np.concatenate([points, pad], axis=0)
        return torch.from_numpy(points)

    def _load_color2(self, img_name, i, raw, cur, d=None, reuse_box=False):
        """Load view i of the second rendering and crop it exactly like the color frame.

        `cur` is the color frame after the small-side handling; this method applies the final
        224 crop to both and stashes the color result in self._color_cropped."""
        path = os.path.join(d or self.data_dir_color2, img_name.split('.')[0] + '_view_' + str(i) + '.png')
        img2 = Image.open(path).convert('RGB').resize(raw.size, Image.BICUBIC)
        w, h = raw.size
        cw, ch = cur.size
        if (cw, ch) != (w, h):
            # small-side branch ran on color: redo the same square crop deterministically for img2
            side = min(w, h)
            img2 = transforms.Resize((self.crop_size, self.crop_size))(transforms.CenterCrop(side)(img2))
        if reuse_box:  # further rungs of a resampling ladder share the first rung's crop
            top, left, hh, ww = self._crop_box
        elif self.is_train:
            top, left, hh, ww = transforms.RandomCrop.get_params(cur, (self.crop_size, self.crop_size))
        else:
            top, left = (ch - self.crop_size) // 2, (cw - self.crop_size) // 2
            hh = ww = self.crop_size
        self._crop_box = (top, left, hh, ww)
        return self.post(transforms.functional.crop(img2, top, left, hh, ww))

    def __len__(self):
        return self.length

    def _crop_with_meta(self, a, b, bbox):
        """Same rules as the baseline (short side < crop: square short-side crop resized to the crop size; else a
        crop-size window), random in training and centred at test, applied to colour AND depth. Returns the crop's
        render-frame offset (ox, oy) and scale s, so points can be projected onto the 14x14 tokens (pmp_camera)."""
        y0, _, x0, _ = (int(v) for v in bbox)
        w, h = a.size
        S = self.crop_size
        side = min(w, h, S) if min(w, h) >= S else min(w, h)
        if self.is_train:
            top, left = np.random.randint(0, h - side + 1), np.random.randint(0, w - side + 1)
        else:
            top, left = int(round((h - side) / 2.0)), int(round((w - side) / 2.0))
        a = transforms.functional.crop(a, top, left, side, side)
        b = transforms.functional.crop(b, top, left, side, side)
        if side != S:
            a, b = transforms.Resize((S, S))(a), transforms.Resize((S, S))(b)
        return a, b, torch.tensor([x0 + left, y0 + top, S / side], dtype=torch.float32)

    DSE_MAX_TRAIN = 65536

    def _load_dse(self, img_name):
        """DSE voxels; in training a random contiguous run of DSE_MAX_TRAIN voxels along the Hilbert curve (a compact
        region, like PTv3's sphere crop), with both serialisation orders restricted and re-indexed."""
        z = np.load(os.path.join(self.dse_dir, img_name.split('.')[0] + '.npz'))
        coord, feat, order = z['coord'].astype(np.int64), z['feat'].astype(np.float32), z['order'].astype(np.int64)
        n = len(coord)
        if self.is_train and n > self.DSE_MAX_TRAIN:
            st = np.random.randint(0, n - self.DSE_MAX_TRAIN + 1)
            keep = order[1][st:st + self.DSE_MAX_TRAIN]
            remap = np.full(n, -1); remap[keep] = np.arange(len(keep))
            order = np.stack([remap[o][remap[o] >= 0] for o in order])
            coord, feat = coord[keep], feat[keep]
        return {'coord': torch.from_numpy(coord), 'feat': torch.from_numpy(feat), 'order': torch.from_numpy(order)}

    def _fragments(self, a, b, grid=7, patch=32):
        """FAST-VQA-style grid mini-patches: the view is split into grid x grid cells, one patch x patch window at native
        resolution is taken from every cell (uniformly inside the cell in training, the cell centre at test) and the
        windows are stitched into a (grid*patch)^2 image; the same windows are used for colour and depth."""
        w, h = a.size
        out_a, out_b = Image.new('RGB', (grid * patch, grid * patch)), Image.new('RGB', (grid * patch, grid * patch))
        for gy in range(grid):
            for gx in range(grid):
                y0, y1 = h * gy // grid, h * (gy + 1) // grid
                x0, x1 = w * gx // grid, w * (gx + 1) // grid
                ry, rx = max(y1 - y0 - patch, 0), max(x1 - x0 - patch, 0)
                if self.is_train:
                    ty, tx = y0 + random.randint(0, ry), x0 + random.randint(0, rx)
                else:
                    ty, tx = y0 + ry // 2, x0 + rx // 2
                ty, tx = min(ty, max(h - patch, 0)), min(tx, max(w - patch, 0))
                box = (tx, ty, tx + patch, ty + patch)
                out_a.paste(a.crop(box), (gx * patch, gy * patch)); out_b.paste(b.crop(box), (gx * patch, gy * patch))
        return out_a, out_b

    @staticmethod
    def _same_crop(a, b, size):
        """One random size x size window, applied to both images (colour and its depth map)."""
        i, j, h, w = transforms.RandomCrop.get_params(a, (size, size))
        return transforms.functional.crop(a, i, j, h, w), transforms.functional.crop(b, i, j, h, w)

    def __getitem__(self, idx):
        item = self._getitem(idx)
        if self.aux is None:
            return item
        t, p = self.aux[self.ply_name.iloc[idx, 0]]
        return (*item, torch.tensor(t), torch.from_numpy(p))

    def _getitem(self, idx):

        img_name = self.ply_name.iloc[idx,0] 
        img_channel = 3
        img_height_crop = self.crop_size
        img_width_crop = self.crop_size
       
        img_length_read = self.img_length_read       
        projected_img = torch.zeros([img_length_read, img_channel, img_height_crop, img_width_crop])
        depth_img = torch.zeros([img_length_read, img_channel, img_height_crop, img_width_crop])
        # several comma-separated dirs = resampling ladder -> [views, rungs, C, H, W]
        c2_dirs = self.data_dir_color2.split(',') if self.data_dir_color2 else []
        c2_shape = [img_length_read] + ([len(c2_dirs)] if len(c2_dirs) > 1 else []) + [img_channel, img_height_crop, img_width_crop]
        color2_img = torch.zeros(c2_shape) if self.data_dir_color2 else None

        crop_meta = torch.zeros([img_length_read, 3])  # per view (ox, oy, s): 224-crop pixel (u,v) -> render (ox+u/s, oy+v/s)
        # read images
        img_read_index = 0
        views = [(self.data_dir_color, self.data_dir_depth, i) for i in range(img_length_read)]
        if self.is_train and self.view_pool:  # viewpoint augmentation: a random subset of the rendered views
            views = random.sample(self.view_pool, img_length_read)
        for i in range(img_length_read):
            # load images
            dc, dd, vi = views[i]
            image_name = os.path.join(dc, img_name.split('.')[0] + '_view_' + str(vi) + '.png')
            depth_name = os.path.join(dd, img_name.split('.')[0] + '_view_' + str(vi) + '.png')
            if os.path.exists(image_name) and os.path.exists(depth_name):
                read_frame = Image.open(image_name)
                read_frame = read_frame.convert('RGB')
                depth_frame = Image.open(depth_name)
                depth_frame = depth_frame.convert('RGB')
                if self.is_train and self.hflip and random.random() < 0.5:
                    read_frame = read_frame.transpose(Image.FLIP_LEFT_RIGHT)
                    depth_frame = depth_frame.transpose(Image.FLIP_LEFT_RIGHT)

                if self.fca_bbox is not None:  # one shared crop for colour and depth, with its render-frame offsets
                    read_frame, depth_frame, crop_meta[i] = self._crop_with_meta(read_frame, depth_frame, self.fca_bbox[img_name.split('.')[0]][i])
                    projected_img[i], depth_img[i] = self.post(read_frame), self.post(depth_frame)
                    img_read_index += 1
                    continue
                read_frame_raw = read_frame
                width, height = read_frame.size
                self._crop_box = None
                if self.view_mode == 'resize':
                    side = max(width, height)
                    pad = ((side - width) // 2, (side - height) // 2, side - width - (side - width) // 2, side - height - (side - height) // 2)
                    read_frame = transforms.Resize((self.crop_size, self.crop_size))(transforms.Pad(pad, fill=(255, 255, 255))(read_frame))
                    depth_frame = transforms.Resize((self.crop_size, self.crop_size))(transforms.Pad(pad, fill=(0, 0, 0))(depth_frame))
                elif self.view_mode == 'fragment':
                    read_frame, depth_frame = self._fragments(read_frame, depth_frame)
                elif os.environ.get('PCQA_FULL_VIEW') == '1' and not self.data_dir_color2:
                    # whole view: pad to a square (white colour / black depth background, as rendered), resize
                    side = max(width, height)
                    pad = ((side - width) // 2, (side - height) // 2, side - width - (side - width) // 2, side - height - (side - height) // 2)
                    read_frame = transforms.Resize((self.crop_size, self.crop_size))(transforms.Pad(pad, fill=(255, 255, 255))(read_frame))
                    depth_frame = transforms.Resize((self.crop_size, self.crop_size))(transforms.Pad(pad, fill=(0, 0, 0))(depth_frame))
                elif (not self.is_train or self.data_dir_color2) and min(width, height) < self.crop_size:
                    # deterministic test-time crop, same window for color and depth
                    side = min(width, height)
                    read_frame = transforms.Resize((self.crop_size, self.crop_size))(transforms.CenterCrop(side)(read_frame))
                    depth_frame = transforms.Resize((self.crop_size, self.crop_size))(transforms.CenterCrop(side)(depth_frame))
                elif self.shared_crop and self.is_train and min(width, height) < self.crop_size:
                    read_frame, depth_frame = self._same_crop(read_frame, depth_frame, min(width, height))
                    read_frame = transforms.Resize((self.crop_size, self.crop_size))(read_frame)
                    depth_frame = transforms.Resize((self.crop_size, self.crop_size))(depth_frame)
                elif width < self.crop_size and width < height:
                    read_frame = transforms.RandomCrop(width)(read_frame)
                    read_frame = transforms.Resize((self.crop_size, self.crop_size))(read_frame)
                    depth_frame = transforms.RandomCrop(width)(depth_frame)
                    depth_frame = transforms.Resize((self.crop_size, self.crop_size))(depth_frame)
                elif height < self.crop_size:
                    read_frame = transforms.RandomCrop(height)(read_frame)
                    read_frame = transforms.Resize((self.crop_size, self.crop_size))(read_frame)
                    depth_frame = transforms.RandomCrop(height)(depth_frame)
                    depth_frame = transforms.Resize((self.crop_size, self.crop_size))(depth_frame)

                if self.data_dir_color2:
                    if len(c2_dirs) > 1:
                        for j, d in enumerate(c2_dirs):
                            color2_img[i, j] = self._load_color2(img_name, i, read_frame_raw, read_frame, d, reuse_box=j > 0)
                    else:
                        color2_img[i] = self._load_color2(img_name, i, read_frame_raw, read_frame)
                if self.data_dir_color2:
                    read_frame = self.post(transforms.functional.crop(read_frame, *self._crop_box))
                    depth_frame = self.transform(depth_frame)
                elif self.shared_crop and self.is_train:
                    read_frame, depth_frame = self._same_crop(read_frame, depth_frame, self.crop_size)
                    read_frame, depth_frame = self.post(read_frame), self.post(depth_frame)
                else:
                    read_frame = self.transform(read_frame)
                    depth_frame = self.transform(depth_frame)
                projected_img[i] = read_frame
                depth_img[i] = depth_frame

                img_read_index += 1

            elif os.path.exists(image_name):
                print(depth_name)
                print('Depth_image do not exist!')
            else:
                print(image_name)
                print('Image do not exist!')

        if img_read_index < img_length_read:
            for j in range(img_read_index, img_length_read):
                projected_img[j] = projected_img[img_read_index-1]
                depth_img[j] = depth_img[img_read_index-1]

        y_mos = self.ply_mos.iloc[idx]
        y_label = torch.FloatTensor(np.array(y_mos))

        raw_score_rows = self.raw_score[self.raw_score['name'] == img_name]
        if len(raw_score_rows) == 0:  # WPC: 80 G-PCC predlift names carry '_rounded' in the split csv but not in the raw-score file
            raw_score_rows = self.raw_score[self.raw_score['name'] == img_name.replace('_rounded', '')]
        if len(raw_score_rows) == 0:
            num_raters = self.raw_score.shape[1] - 1
            raw_score_tensor_data = np.full(num_raters, float(y_mos), dtype=np.float32)
        else:
            raw_score_tensor_data = raw_score_rows.iloc[:, 1:].values.flatten().astype(np.float32)

        if self.use_geometry:
            point_patch = self._load_point_cloud(idx, img_name)
            return projected_img, depth_img, point_patch, y_label, raw_score_tensor_data
        if self.data_dir_color2:
            return projected_img, depth_img, color2_img, y_label, raw_score_tensor_data
        if self.anchor is not None:
            return projected_img, depth_img, self.anchor[img_name.split('.')[0]], y_label, raw_score_tensor_data
        if self.dse_dir is not None:
            return projected_img, depth_img, self._load_dse(img_name), y_label, raw_score_tensor_data
        if self.kpa_dir is not None:
            z = np.load(os.path.join(self.kpa_dir, img_name.split('.')[0] + '.npz'))
            q, nb, cnt = (torch.from_numpy(z[k].astype(np.float32)) for k in ('q', 'nb', 'cnt'))
            if self.is_train:  # random half of the queries, as in train_kpa.py
                sub = torch.randperm(q.shape[0])[:q.shape[0] // 2]
                q, nb, cnt = q[sub], nb[:, sub], cnt[:, sub]
            if self.fca_bbox is not None:
                return projected_img, depth_img, (q, nb, cnt, crop_meta), y_label, raw_score_tensor_data
            return projected_img, depth_img, (q, nb, cnt), y_label, raw_score_tensor_data
        if self.point_feat_dir is not None:
            pf = torch.from_numpy(np.load(os.path.join(self.point_feat_dir, img_name.split('.')[0] + '.npy')).astype(np.float32))
            if self.is_train:  # random half of the query points, as in train_geonet.py
                pf = pf[torch.randint(0, len(pf), (len(pf) // 2,))]
            return projected_img, depth_img, pf, y_label, raw_score_tensor_data

        return projected_img, depth_img, y_label, raw_score_tensor_data


def pcqa_collate(batch):
    """default_collate, except that per-sample dicts (variable-size DSE voxels) are kept as a list."""
    from torch.utils.data.dataloader import default_collate
    return [list(col) if isinstance(col[0], dict) else default_collate(list(col)) for col in zip(*batch)]


def to_device(x, device):
    """Tensor, list/tuple of tensors, or dict of tensors (DSE voxels), possibly nested -> device."""
    if isinstance(x, (list, tuple)):
        return [to_device(t, device) for t in x]
    if isinstance(x, dict):
        return {k: v.to(device) for k, v in x.items()}
    return x.to(device)
