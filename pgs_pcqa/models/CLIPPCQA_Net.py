import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import numpy as np
import copy
import math
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from utils.geometry_features import compute_geom_features

_tokenizer = _Tokenizer()
clip_vis = os.environ.get('PCQA_CLIP', 'ViT-B/16')  # e.g. ViT-L/14 for the stronger backbone
GEOM_FEATURE_DIM = 22

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]).to(x.device), tokenized_prompts.argmax(dim=-1).to(x.device)] @ self.text_projection.to(x.device)
        
        return x


class PromptLearner(nn.Module):
    def __init__(self, device, args, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = args.n_ctx
        ctx_init = args.ctx_init
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.device = device

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if args.csc:
                # print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                # print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype).to(self.device)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = args.class_token_position

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class LoRA(nn.Module):
    """weight -> weight + (alpha / r) * B @ A (torch parametrization); B starts at zero, so the model starts as the plain one"""
    def __init__(self, w, r, alpha):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(r, w.shape[1], device=w.device) / math.sqrt(w.shape[1]))
        self.lora_B = nn.Parameter(torch.zeros(w.shape[0], r, device=w.device)); self.scale = alpha / r

    def forward(self, w):
        return w + self.scale * (self.lora_B @ self.lora_A)


def add_lora(visual, r, alpha):
    """low-rank adapters on every attention in-/out-projection and MLP matrix of a CLIP ViT"""
    from torch.nn.utils import parametrize
    for blk in visual.transformer.resblocks:
        for mod, name in ((blk.attn, 'in_proj_weight'), (blk.attn.out_proj, 'weight'), (blk.mlp.c_fc, 'weight'), (blk.mlp.c_proj, 'weight')):
            parametrize.register_parametrization(mod, name, LoRA(getattr(mod, name).detach(), r, alpha))


def plain_state_dict(model):
    """state dict with every LoRA-parametrized weight merged back under its original key (loads into a model without LoRA)"""
    sd, out = model.state_dict(), {}
    for k, v in sd.items():
        if '.parametrizations.' not in k:
            out[k] = v
        elif k.endswith('.original'):
            mod_path, name = k.split('.parametrizations.')[0], k.split('.parametrizations.')[1][:-len('.original')]
            mod = model
            for part in mod_path.split('.'):
                mod = getattr(mod, part)
            out[mod_path + '.' + name] = getattr(mod, name).detach()
    return out


class CLIPPCQA_Net(nn.Module):
    def __init__(self, device, args, score_list, quality_classes):
        super(CLIPPCQA_Net, self).__init__()
        self.device = device
        self.score_list = score_list
        self.num_quality_classes = len(quality_classes)
        self.use_geometry = getattr(args, "use_geometry", False)
        self.ablation_mask = getattr(args, "ablation_mask", None)

        self.clip_texture, _ = clip.load(clip_vis)
        self.clip_texture = self.clip_texture.to(torch.float32)

        self.clip_depth, _ = clip.load(clip_vis)
        self.clip_depth = self.clip_depth.to(torch.float32)
        init = getattr(args, "clip_init", "")
        if init:  # quality-pretrained CLIP (e.g. LIQE) in place of the OpenAI weights, for both encoders
            sd = torch.load(init, map_location="cpu")
            for enc in (self.clip_texture, self.clip_depth):
                enc.load_state_dict(sd, strict=True)

        # --view_size: larger image input; the patch-position embedding is bicubically interpolated to the new grid
        vs = getattr(args, 'view_size', 224)
        for enc in (self.clip_texture, self.clip_depth):
            v = enc.visual; p = v.conv1.kernel_size[0]; pe = v.positional_embedding.data
            g0, g1 = int(round((pe.shape[0] - 1) ** 0.5)), vs // p
            if g1 != g0:
                grid = pe[1:].reshape(1, g0, g0, -1).permute(0, 3, 1, 2)
                grid = F.interpolate(grid, size=(g1, g1), mode='bicubic', align_corners=False).permute(0, 2, 3, 1).reshape(g1 * g1, -1)
                v.positional_embedding = nn.Parameter(torch.cat([pe[:1], grid], 0))
                v.input_resolution = vs
        if getattr(args, 'lora', 0):  # parameter-efficient fine-tuning of both CLIP image encoders (base weights stay frozen)
            for enc in (self.clip_texture, self.clip_depth):
                add_lora(enc.visual, args.lora, getattr(args, 'lora_alpha', 0) or 2 * args.lora)
        self.dtype = self.clip_texture.dtype

        self.quality_prompt_learner = PromptLearner(self.device, args, quality_classes, self.clip_texture)
        self.quality_tokenized_prompts = self.quality_prompt_learner.tokenized_prompts
        self.quality_text_encoder = TextEncoder(self.clip_texture)
        self.quality_logit_scale = self.clip_texture.logit_scale
        clip_feature_dim = self.clip_texture.visual.output_dim

        # Geometry branch G: project hand-crafted patch descriptors into the
        # CLIP image feature space, then fuse both streams before text matching.
        self.geom_mlp = nn.Sequential(
            nn.LayerNorm(GEOM_FEATURE_DIM),
            nn.Linear(GEOM_FEATURE_DIM, clip_feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(clip_feature_dim // 2, clip_feature_dim),
        )
        self.geom_fusion = nn.Sequential(
            nn.LayerNorm(clip_feature_dim * 2),
            nn.Linear(clip_feature_dim * 2, clip_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(clip_feature_dim, clip_feature_dim),
        )

        # splat-scale response branch: change of the color embedding when the same views are rendered
        # with a smaller splat radius. Last layer zero-initialised -> starts exactly as the baseline.
        # color2_mode "abs": the second rendering is a geometry-evidence map, use its own embedding
        # --aux_deg: auxiliary degradation head on the pooled image feature, classifies the distortion type and
        # regresses its parameters (QP, level, ...) parsed from file names (utils/deg_labels.py). Training only.
        # --tam: text-guided token pooling (after PMP's TAM, soft version): each view's 196 tokens are weighted by
        # their similarity to the mean quality-prompt embedding instead of averaged, so local defects are not
        # diluted. Temperature starts at 1 (cos sims in [-1,1] -> almost uniform weights = the baseline mean).
        # --fca: PMP-style fine/coarse cross-modal alignment, but with the KPA query tokens as the 3D side: each
        # query point is projected into the 6 views (exact render camera + crop, utils/pmp_camera.py) and its KPA
        # token is contrasted against the fused image token it lands in (InfoNCE over the sample's other queries),
        # plus a coarse sample-level InfoNCE over the batch. Training-only loss; the score path is unchanged.
        self.fca = getattr(args, "fca", 0) > 0
        self.last_fca = None
        if self.fca:
            from utils.pmp_camera import view_RT
            from models.kpa_net import D_TOKEN
            R, T = view_RT()
            self.register_buffer('fca_R', R, persistent=False); self.register_buffer('fca_T', T, persistent=False)
            self.geom_fca_proj = nn.Sequential(nn.LayerNorm(D_TOKEN), nn.Linear(D_TOKEN, clip_feature_dim))
            self.fca_tau, self.fca_nq = 0.07, 256
        self.tam = getattr(args, "tam", False)
        if self.tam:
            self.geom_tam_logscale = nn.Parameter(torch.zeros(()))
        self.aux_ntypes = getattr(args, "aux_ntypes", 0)
        self.last_aux = None
        if self.aux_ntypes:
            from utils.deg_labels import P_MAX
            self.geom_aux_head = nn.Sequential(nn.LayerNorm(clip_feature_dim), nn.Linear(clip_feature_dim, 256), nn.GELU(),
                                               nn.Linear(256, getattr(args, 'aux_out', 0) or self.aux_ntypes * (1 + P_MAX)))  # --aux_factor: factors + severity
        self.use_splat = bool(getattr(args, "data_dir_color2", ""))
        self.color2_mode = getattr(args, "color2_mode", "delta")
        self.splat_mlp = nn.Sequential(nn.LayerNorm(clip_feature_dim), nn.Linear(clip_feature_dim, clip_feature_dim),
                                       nn.ReLU(inplace=True), nn.Linear(clip_feature_dim, clip_feature_dim))
        nn.init.zeros_(self.splat_mlp[-1].weight); nn.init.zeros_(self.splat_mlp[-1].bias)
        # color2_mode "patch": per-16x16-patch statistics of the geometry map are embedded and added to the
        # texture ViT's patch tokens, so local geometry is fused position by position. Zero-init as above.
        # color2_mode "ladder": data_dir_color2 lists renders of the cloud resampled to fewer points;
        # the embedding shifts along that ladder are the self-anchor (where the cloud sits on its own
        # density curve). Zero-init as above.
        # (only built in that mode, so other configs keep their RNG stream and checkpoints unchanged)
        # color2_mode "anchor": same branch, but the shifts were precomputed with the frozen original CLIP
        # (a fixed measuring instrument) and arrive as one vector per sample
        # color2_mode "calib": the precomputed anchor only shifts the 5 quality-level logits (calibration),
        # leaving the CLIP image feature untouched. Small head, zero-init output.
        if self.color2_mode == "calib":
            z = np.load(getattr(args, "data_dir_color2"))
            d_in = (z[z.files[0]].shape[0] - 1) * z[z.files[0]].shape[1]
            self.splat_calib = nn.Sequential(nn.LayerNorm(d_in), nn.Dropout(0.2), nn.Linear(d_in, 64),
                                             nn.GELU(), nn.Linear(64, len(score_list)))
            nn.init.zeros_(self.splat_calib[-1].weight); nn.init.zeros_(self.splat_calib[-1].bias)
        # color2_mode "geonet": a sampling-sensitive point network on per-point fixed-radius features
        # (geo_feats.py); its pooled embedding shifts the 5 quality-level logits. Zero-init output.
        if self.color2_mode == "geonet":
            c2 = getattr(args, "data_dir_color2")
            d_in = np.load(os.path.join(c2, next(f for f in os.listdir(c2) if f.endswith('.npy')))).shape[1]
            d = 128
            self.geom_pnorm = nn.BatchNorm1d(d_in)
            self.geom_pmlp = nn.Sequential(nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
            self.geom_patt = nn.Linear(d, 1)
            self.geom_phead = nn.Sequential(nn.Dropout(0.2), nn.Linear(3 * d, len(score_list)))
            nn.init.zeros_(self.geom_phead[-1].weight); nn.init.zeros_(self.geom_phead[-1].bias)
        # color2_mode "kpa": kernel point attention network (train_kpa.KPANet) on fixed-radius neighbourhoods;
        # its pooled feature shifts the 5 quality-level logits. Zero-init output, no weight decay (see main.py).
        if self.color2_mode == "dse":  # density-preserving serialised encoder over the full-cloud voxels
            from models.dse_net import DSENet
            self.geom_dse = DSENet()
            self.geom_dse_out = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.geom_dse.out_dim, len(score_list)))
            nn.init.zeros_(self.geom_dse_out[-1].weight); nn.init.zeros_(self.geom_dse_out[-1].bias)
        if self.color2_mode == "kpa":
            import importlib.util, sys as _sys
            spec = importlib.util.spec_from_file_location("kpa_net", os.path.join(os.path.dirname(os.path.abspath(__file__)), "kpa_net.py"))
            kpa = importlib.util.module_from_spec(spec); spec.loader.exec_module(kpa)
            self.geom_kpa = kpa.KPANet()
            self.geom_kpa.head = nn.Identity()           # use the pooled feature, not the scalar head
            self.geom_kpa_out = nn.Sequential(nn.Dropout(0.2), nn.Linear(kpa.POOLED_DIM, len(score_list)))
            nn.init.zeros_(self.geom_kpa_out[-1].weight); nn.init.zeros_(self.geom_kpa_out[-1].bias)
            # --kpa_fuse: KPA feature joins the CLIP image feature BEFORE the text matching: unit-norm projection, learnable gate
            # starting at 0.1 (the CLIP feature is unit-norm too, so KPA starts as a ~10% residual and cannot swamp it)
            self.kpa_fuse = getattr(args, 'kpa_fuse', False)
            if self.kpa_fuse:
                self.geom_kpa_proj = nn.Sequential(nn.LayerNorm(kpa.POOLED_DIM), nn.Linear(kpa.POOLED_DIM, 512))
                self.geom_kpa_gate = nn.Parameter(torch.tensor(0.1))
            # --kpa_aux: the KPA branch also predicts the (0-1 scaled) MOS on its own, so it cannot hide behind CLIP
            if getattr(args, 'kpa_aux', 0):
                self.geom_kpa_auxhead = nn.Sequential(nn.Linear(kpa.POOLED_DIM, 128), nn.GELU(), nn.Linear(128, 1))
            self.last_kpa_pred = None
            # --kpa_gate: distortion-aware gate on the KPA logit correction. The aux head's distortion-type probabilities
            # (detached) give g = sigmoid(w . p + b) per sample; g starts at ~0.88 for every type (b = 2), so training
            # decides for which distortion types the geometry branch should speak.
            self.kpa_gate = getattr(args, 'kpa_gate', False)
            if self.kpa_gate:
                assert self.aux_ntypes, '--kpa_gate needs the distortion aux head (--aux_deg)'
                self.geom_kpa_gate_w = nn.Parameter(torch.zeros(self.aux_ntypes))
                self.geom_kpa_gate_b = nn.Parameter(torch.tensor(2.0))
            self.last_kpa_gate = None
            # --kpa_mix: KPA is a second quality expert with its own 5-level distribution head; the final distribution is
            # (1 - g) * p_CLIP + g * p_KPA with g = sigmoid(w . p_type + b) from the aux head's distortion-type probabilities
            # (detached), b = -1 so g starts at ~0.27. --kpa_aux then supervises the KPA expert's own expected score.
            # --clip_drop p: modality dropout. With probability p per training sample, the sample's final distribution
            # comes from a separate KPA-only head (its own 5-level distribution), so the KPA encoder must carry the score
            # alone; the logit-correction path is unchanged and is the only one used at test time.
            self.clip_drop = getattr(args, 'clip_drop', 0)
            if self.clip_drop:
                self.geom_kpa_drophead = nn.Sequential(nn.Linear(kpa.POOLED_DIM, 128), nn.GELU(), nn.Dropout(0.2),
                                                       nn.Linear(128, len(score_list)))
            self.kpa_mix = getattr(args, 'kpa_mix', False)
            if self.kpa_mix:
                assert self.aux_ntypes, '--kpa_mix needs the distortion aux head (--aux_deg)'
                self.geom_kpa_mixhead = nn.Sequential(nn.Linear(kpa.POOLED_DIM, 128), nn.GELU(), nn.Dropout(0.2),
                                                      nn.Linear(128, len(score_list)))
                self.geom_kpa_mix_w = nn.Parameter(torch.zeros(self.aux_ntypes))
                self.geom_kpa_mix_b = nn.Parameter(torch.tensor(-1.0))
        if self.color2_mode in ("ladder", "anchor"):
            c2 = getattr(args, "data_dir_color2")
            if c2.endswith('.npz'):  # [rungs + 1 (original), C] per sample -> rungs shifts
                z = np.load(c2)
                n_rungs = z[z.files[0]].shape[0] - 1
            else:
                n_rungs = len(c2.split(','))
            self.splat_ladder = nn.Sequential(nn.LayerNorm(n_rungs * clip_feature_dim),
                                              nn.Linear(n_rungs * clip_feature_dim, clip_feature_dim),
                                              nn.ReLU(inplace=True), nn.Linear(clip_feature_dim, clip_feature_dim))
            nn.init.zeros_(self.splat_ladder[-1].weight); nn.init.zeros_(self.splat_ladder[-1].bias)
        if self.color2_mode == "patch":
            vit = self.clip_texture.visual
            self.geom_patch = vit.conv1.kernel_size[0]
            self.geom_patch_embed = nn.Sequential(nn.Linear(7, 64), nn.GELU(), nn.Linear(64, vit.conv1.out_channels))
            nn.init.zeros_(self.geom_patch_embed[-1].weight); nn.init.zeros_(self.geom_patch_embed[-1].bias)
            self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
            self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def _geom_patch_tokens(self, g):
        """g: normalized geometry maps [N, 3, H, W] -> [N, (H/p)*(W/p), width] token offsets."""
        g = (g.float() * self.img_std + self.img_mean).clamp(0, 1)
        bg = (g.min(dim=1, keepdim=True).values > 0.999).float()
        u = F.unfold(torch.cat([g, bg], 1), self.geom_patch, stride=self.geom_patch)  # [N, 4*p*p, L]
        u = u.view(g.shape[0], 4, self.geom_patch ** 2, -1)
        fg = 1 - u[:, 3:4]
        n = fg.sum(2).clamp_min(1)
        mean = (u[:, :3] * fg).sum(2) / n
        std = (((u[:, :3] - mean.unsqueeze(2)) ** 2 * fg).sum(2) / n).sqrt()
        stats = torch.cat([mean, std, u[:, 3].mean(1, keepdim=True)], 1).transpose(1, 2)  # [N, L, 7]
        return self.geom_patch_embed(stats)

    def _pool(self, f, batch_size, num_views):
        f = f / f.norm(dim=-1, keepdim=True)
        return f.mean(dim=1).reshape(batch_size, num_views, -1).mean(dim=1)

    def _fca_loss(self, xyz, crop_meta):
        """xyz [B,Q,3] KPA query points (render frame), crop_meta [B,V,3] -> fine + coarse InfoNCE."""
        from utils.pmp_camera import points_to_tokens
        tok = self._tok                                                    # [B,V,P,C], unit norm
        B, V, P, C = tok.shape
        sel = torch.randperm(xyz.shape[1], device=xyz.device)[:self.fca_nq]
        idx, valid = points_to_tokens(xyz[:, sel], crop_meta, self.fca_R, self.fca_T)   # [B,q,V]
        g = torch.stack([tok[:, m].gather(1, idx[:, :, m, None].expand(-1, -1, C)) for m in range(V)], 2)  # [B,q,V,C]
        vf = valid.float()[..., None]
        bridge = F.normalize((g * vf).sum(2) / vf.sum(2).clamp_min(1), dim=-1)             # mean over visible views
        k3 = F.normalize(self.geom_fca_proj(self.geom_kpa.last_tokens[:, sel]), dim=-1)    # [B,q,C]
        keep = valid.any(2)
        fine = []
        for b in range(B):
            kb, hb = k3[b][keep[b]], bridge[b][keep[b]]
            if len(kb) < 2:
                continue
            lg = kb @ hb.t() / self.fca_tau
            tgt = torch.arange(len(kb), device=lg.device)
            fine.append((F.cross_entropy(lg, tgt) + F.cross_entropy(lg.t(), tgt)) / 2)
        lg = F.normalize(k3.mean(1), dim=-1) @ F.normalize(tok.mean((1, 2)), dim=-1).t() / self.fca_tau
        tgt = torch.arange(B, device=lg.device)
        coarse = (F.cross_entropy(lg, tgt) + F.cross_entropy(lg.t(), tgt)) / 2
        return (torch.stack(fine).mean() if fine else coarse * 0) + coarse

    def _tam_pool(self, f, batch_size, num_views):
        """f: [B*V, P, C] fused tokens -> [B, C], tokens weighted by similarity to the mean quality prompt."""
        t = f / f.norm(dim=-1, keepdim=True)
        g = self.quality_text_encoder(self.quality_prompt_learner(), self.quality_tokenized_prompts)
        g = (g / g.norm(dim=-1, keepdim=True)).mean(0)
        w = torch.softmax((t @ g.to(t.dtype)) * self.geom_tam_logscale.exp(), dim=1)   # over the tokens of a view
        return (w.unsqueeze(-1) * t).sum(1).reshape(batch_size, num_views, -1).mean(dim=1)

    def forward(self, texture_imgs, depth_imgs, point_patches=None, geom_features=None, color2_imgs=None):
        
        # preprocess
        batch_size, num_views, channels, image_height, image_width = texture_imgs.shape
        texture_imgs = texture_imgs.reshape(-1, channels, image_height, image_width).type(self.dtype)
        depth_imgs = depth_imgs.reshape(-1, channels, image_height, image_width).type(self.dtype)

        # feature extraction
        if self.use_splat and self.color2_mode == "patch" and color2_imgs is not None:
            g = color2_imgs.reshape(-1, channels, image_height, image_width)
            texture_f = self.clip_texture.encode_image(texture_imgs, self._geom_patch_tokens(g))
        else:
            texture_f = self.clip_texture.encode_image(texture_imgs)  # B*num_views, patches, C;
        texture_f = texture_f / texture_f.norm(dim=-1, keepdim=True)
        _, patches_num, C = texture_f.shape

        depth_f = self.clip_depth.encode_image(depth_imgs)
        depth_f = depth_f / depth_f.norm(dim=-1, keepdim=True) 

        # normalize
        image_f = (texture_f + depth_f) / 2
        if self.fca:
            self._tok = (image_f / image_f.norm(dim=-1, keepdim=True)).reshape(batch_size, num_views, patches_num, C)
        if self.tam:
            image_f = self._tam_pool(image_f, batch_size, num_views)
        else:
            image_f = (image_f / image_f.norm(dim=-1, keepdim=True)).mean(dim=1, keepdim=False).reshape(batch_size, num_views, C).mean(dim=1, keepdim=False)

        if self.use_splat and self.color2_mode == "anchor" and color2_imgs is not None:
            image_f = image_f + self.splat_ladder(color2_imgs.to(image_f.dtype))
            image_f = image_f / image_f.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        elif self.use_splat and self.color2_mode == "ladder" and color2_imgs is not None:
            base = self._pool(texture_f, batch_size, num_views)
            deltas = []
            with torch.no_grad():  # the ladder renders are measurements, like the small-splat branch
                for j in range(color2_imgs.shape[2]):
                    r = color2_imgs[:, :, j].reshape(-1, channels, image_height, image_width).type(self.dtype)
                    deltas.append(self._pool(self.clip_texture.encode_image(r), batch_size, num_views))
            delta = torch.cat([d - base for d in deltas], dim=-1)
            image_f = image_f + self.splat_ladder(delta)
            image_f = image_f / image_f.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        elif self.use_splat and self.color2_mode not in ("patch", "ladder", "anchor", "calib", "geonet", "kpa", "dse") and color2_imgs is not None:
            c2 = color2_imgs.reshape(-1, channels, image_height, image_width).type(self.dtype)
            with torch.no_grad():  # the small-splat render is a measurement; gradients reach the encoder via texture_f
                c2_pooled = self._pool(self.clip_texture.encode_image(c2), batch_size, num_views)
            delta = c2_pooled if self.color2_mode == "abs" else c2_pooled - self._pool(texture_f, batch_size, num_views)
            image_f = image_f + self.splat_mlp(delta)
            image_f = image_f / image_f.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        if self.use_geometry and (point_patches is not None or geom_features is not None):
            if geom_features is None:
                # point_patches accepts [B, N, 3] or [B, P, N, 3]. Multiple
                # patches per sample are averaged into one geometry descriptor.
                if point_patches.dim() == 4:
                    bsz, patch_count, point_count, coords = point_patches.shape
                    geom_features = compute_geom_features(point_patches.reshape(-1, point_count, coords), ablation_mask=self.ablation_mask)
                    geom_features = geom_features.reshape(bsz, patch_count, -1).mean(dim=1)
                else:
                    geom_features = compute_geom_features(point_patches, ablation_mask=self.ablation_mask)
            geom_features = geom_features.to(image_f.device).type(image_f.dtype)
            geom_f = self.geom_mlp(geom_features)
            geom_f = geom_f / geom_f.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            image_f = self.geom_fusion(torch.cat([image_f, geom_f], dim=1))
            image_f = image_f / image_f.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        kpa_feat = None
        # main.py --train_crops K: the images hold K crops per sample (crop-major within a sample) while color2_imgs holds
        # every sample's KPA input ONCE; the KPA heads run once per sample (one dropout mask, as without K) and their
        # outputs are repeated for the sample's K crops. Training only; eval batches are one crop set per sample.
        kpa_rep = getattr(self, 'kpa_repeat', 1) if self.training else 1
        if self.use_splat and self.color2_mode == "kpa" and color2_imgs is not None:
            q, nb, cnt = (t.float() for t in color2_imgs[:3])
            kpa_feat = self.geom_kpa(q, nb, cnt)
            if hasattr(self, 'geom_kpa_auxhead'):
                self.last_kpa_pred = self.geom_kpa_auxhead(kpa_feat).squeeze(-1).repeat_interleave(kpa_rep, 0)
            if self.kpa_fuse:
                image_f = image_f + self.geom_kpa_gate * F.normalize(self.geom_kpa_proj(kpa_feat), dim=-1).to(image_f.dtype)
                image_f = image_f / image_f.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        if self.aux_ntypes:
            self.last_aux = self.geom_aux_head(image_f)

        # prompts
        quality_prompts = self.quality_prompt_learner()
        quality_tokenized_prompts = self.quality_tokenized_prompts
        quality_text_f = self.quality_text_encoder(quality_prompts, quality_tokenized_prompts)
        quality_text_f = quality_text_f / quality_text_f.norm(dim=-1, keepdim=True)
        quality_logit_scale = self.quality_logit_scale.exp()
        quality_logits = quality_logit_scale * image_f @ quality_text_f.t() # B, quality_classes
        self.last_clip_dist = F.softmax(quality_logits.detach().float(), dim=1)  # CLIP-only distribution (for --ogm)
        if self.use_splat and self.color2_mode == "dse" and color2_imgs is not None:
            quality_logits = quality_logits + self.geom_dse_out(self.geom_dse(color2_imgs)).to(quality_logits.dtype)
        if self.use_splat and self.color2_mode == "kpa" and color2_imgs is not None:
            if self.kpa_mix:
                pass  # mixed at the distribution level below
            elif not self.kpa_fuse:  # default: late fusion, KPA adds a correction to the quality logits
                corr = self.geom_kpa_out(kpa_feat).repeat_interleave(kpa_rep, 0)
                if self.kpa_gate:
                    p = torch.softmax(self.last_aux[:, :self.aux_ntypes].detach().float(), -1)
                    g = torch.sigmoid(p @ self.geom_kpa_gate_w + self.geom_kpa_gate_b)
                    self.last_kpa_gate = g.detach()
                    corr = g[:, None] * corr
                quality_logits = quality_logits + corr.to(quality_logits.dtype)
            self.last_fca = self._fca_loss(q[..., :3], color2_imgs[3].float()) if self.fca and self.training and len(color2_imgs) > 3 else None
        if self.use_splat and self.color2_mode == "geonet" and color2_imgs is not None:
            x = color2_imgs.float()
            B, Q, F_ = x.shape
            h = self.geom_pmlp(self.geom_pnorm(x.reshape(-1, F_)).reshape(B, Q, F_))
            w = torch.softmax(self.geom_patt(h), 1)
            g = torch.cat([h.mean(1), h.max(1).values, (w * h).sum(1)], -1)
            quality_logits = quality_logits + self.geom_phead(g).to(quality_logits.dtype)
        if self.use_splat and self.color2_mode == "calib" and color2_imgs is not None:
            quality_logits = quality_logits + self.splat_calib(color2_imgs.to(quality_logits.dtype))

        # quality_cdf and quality_score
        pred_distribution = F.softmax(quality_logits, dim=1).to(self.device)
        bin_tensor = torch.tensor(self.score_list).to(self.device).reshape(1, self.num_quality_classes)
        if getattr(self, 'kpa_mix', False) and kpa_feat is not None:
            p_kpa = F.softmax(self.geom_kpa_mixhead(kpa_feat).float(), dim=1)
            p_type = torch.softmax(self.last_aux[:, :self.aux_ntypes].detach().float(), -1)
            g = torch.sigmoid(p_type @ self.geom_kpa_mix_w + self.geom_kpa_mix_b)[:, None]
            self.last_kpa_gate = g.detach().squeeze(1)
            lo, hi = float(self.score_list[0]), float(self.score_list[-1])
            self.last_kpa_pred = ((p_kpa * bin_tensor).sum(1) - lo) / (hi - lo)  # KPA expert alone, scaled to 0-1
            pred_distribution = ((1 - g) * pred_distribution.float() + g * p_kpa).to(self.device)
        if getattr(self, 'clip_drop', 0) and self.training and kpa_feat is not None:
            drop = torch.rand(pred_distribution.shape[0], device=pred_distribution.device) < self.clip_drop
            if drop.any():
                p_alone = F.softmax(self.geom_kpa_drophead(kpa_feat).float(), dim=1).to(pred_distribution.dtype)
                pred_distribution = torch.where(drop[:, None], p_alone, pred_distribution)
        pred_CDF = torch.cumsum(pred_distribution, dim=1)
        quality_score = (pred_distribution * bin_tensor).sum(1, keepdim=True).to(self.device)

        texture_f = texture_f.flatten(start_dim=1)
        depth_f = depth_f.flatten(start_dim=1)
        return texture_f, depth_f, quality_score, pred_distribution, pred_CDF
