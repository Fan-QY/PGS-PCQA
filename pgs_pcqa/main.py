import os, re, argparse, time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torchvision import transforms
import random
import torch.backends.cudnn as cudnn
import scipy
from scipy import stats
from scipy.optimize import curve_fit
from models.CLIPPCQA_Net import CLIPPCQA_Net, plain_state_dict
from utils.CLIPPCQA_Dataset import CLIPPCQA_Dataset, pcqa_collate, to_device


def object_of(database, name):
    """Reference-object id of a distorted sample name."""
    import re
    if database == 'SJTU':
        return name.rsplit('_', 1)[0]
    if database == 'BASICS':
        return name.split('_')[0]
    if database == 'WPC':
        return re.split(r'_(?:level|gQP|gsigma|pqs|tsl)_', name)[0]
    return name[:-4].rsplit('_', 2)[0]   # LS-PCQA style: obj_Type_level


def distortion_partners(dataset, database):
    """index -> indices of training samples with the same distortion (name minus object prefix) on other objects"""
    names = list(dataset.ply_name.iloc[:, 0]); key = {}
    for i, n in enumerate(names):
        key.setdefault(n[len(object_of(database, n)):], []).append(i)
    return [[j for j in key[n[len(object_of(database, n)):]] if j != i] for i, n in enumerate(names)]


def object_batches(dataset, database, bs):
    """One epoch of batches, each drawn from a single reference object (shuffled within and across objects)."""
    groups = {}
    for i, n in enumerate(dataset.ply_name.iloc[:, 0]):
        groups.setdefault(object_of(database, n), []).append(i)
    batches = []
    for idx in groups.values():
        idx = list(np.random.permutation(idx))
        batches += [idx[j:j + bs] for j in range(0, len(idx), bs)]
    return [batches[i] for i in np.random.permutation(len(batches))]
from utils.loss import EMD_Quan_Con_Loss
import math
from scipy.stats import pearsonr, spearmanr, kendalltau
from torch.nn import functional as F
from math import cos, pi
from thop import profile

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

def project_path(*parts):
    return os.path.join(PROJECT_DIR, *parts)

def resolve_project_path(path):
    if not path or os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(PROJECT_DIR, path))

def set_rand_seed(seed=1998):
    print("Random Seed: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)       
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True   # fix the random seed

def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat
    
def fit_function(y_label, y_output):
    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(logistic_func, y_output, \
        y_label, p0=beta, maxfev=100000000)
    y_output_logistic = logistic_func(y_output, *popt)
    return y_output_logistic

def parse_args():
    """Parse input arguments. """
    parser = argparse.ArgumentParser(description="training")
    parser.add_argument('--gpu', help="GPU device id to use [0]", default=0, type=int)
    parser.add_argument('--num_epochs',  help='Maximum number of training epochs.', default=60, type=int)
    parser.add_argument('--batch_size', help='Batch size.', default=16, type=int)
    parser.add_argument('--learning_rate', default=0.000004, type=float, help='learning rate in training')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--data_dir_color', default='', type=str, help = 'path to the color map images')
    parser.add_argument('--data_dir_depth', default='', type=str, help = 'path to the depth images')
    parser.add_argument('--data_dir_pc', default='', type=str, help = 'path to point cloud files for geometry branch')
    parser.add_argument('--use_geometry', action='store_true', help='enable geometry feature branch G')
    parser.add_argument('--geom_num_points', default=2048, type=int, help='number of xyz points sampled per patch')
    parser.add_argument('--ablation_mask', default=None, type=str, help=("Comma-separated feature groups to zero out for ablation study. Valid: density,radial,bbox,eigenvalues,shape,normal. For example: --ablation_mask density,normal"))
    parser.add_argument('--img_length_read', default=6, type=int, help = 'number of the using images')
    parser.add_argument('--n_ctx', default=16, type=int, help = 'number of context vectors')
    parser.add_argument('--database', default='SJTU', type=str)
    parser.add_argument('--class_token_position', default='middle', type=str, help = "'middle' or 'end' or 'front'")
    parser.add_argument('--k_fold_num', default=9, type=int, help='9 for the SJTU-PCQA, 5 for the LS-PCQA and BASICS')
    parser.add_argument('--data_dir_color2', default='', type=str, help='second color rendering (e.g. smaller splat radius)')
    parser.add_argument('--init_ckpt', default='', type=str, help='initialise the model from this checkpoint before training')
    parser.add_argument('--branch_lr', default=0, type=float, help='lr for added geom_/splat_ branches (0 = same as --learning_rate)')
    parser.add_argument('--color2_mode', default='delta', type=str, help="'delta' (splat response) or 'abs' (geometry map)")
    parser.add_argument('--seed', default=1998, type=int, help='random seed')
    parser.add_argument('--aux_deg', default=0, type=float, help='weight of the auxiliary degradation type/parameter loss (0 = off)')
    parser.add_argument('--rank_w', default=0, type=float, help='weight of the in-batch pairwise ranking loss on the predicted score (0 = off)')
    parser.add_argument('--aux_factor', action='store_true', help='aux head predicts dataset-agnostic degradation factors '
                        '(multi-hot, BCE) + severity in [0,1] (MSE) instead of the database type ids (utils/deg_labels.build_factors)')
    parser.add_argument('--synth_w', default=0, type=float, help='weight of the within-object severity ranking loss on synthetic '
                        'codec distortions of the training references (no MOS; 0 = off)')
    parser.add_argument('--synth_csv', default='', help='synthetic samples: csv with name,mos where mos = severity level (1 = mildest); '
                        'names <ref>_<type>_L<level>.ply, ranked only within the same <ref>_<type>')
    parser.add_argument('--synth_dir', default='', help='synthetic renders root (6view, 6view_depth) ')
    parser.add_argument('--synth_kpa', default='', help='synthetic KPA neighbourhood dir (npz), used as data_dir_color2')
    parser.add_argument('--synth_pairs', default=2, type=int, help='synthetic (milder, stronger) pairs per training step')
    parser.add_argument('--synth_anneal', default=0, type=int, help='>0: the synthetic ranking weight decays linearly from --synth_w to 0 '
                        'over the first N epochs (afterwards MOS only); 0 = constant weight')
    parser.add_argument('--synth_teacher', default='', help='csv name,deep: render-space perceptual distance of each synthetic sample to its '
                        'reference (bank/perceptual_distance.py); enables cross-type pairs within a reference')
    parser.add_argument('--synth_xtype', default=0.5, type=float, help='with --synth_teacher: share of the pairs drawn across distortion '
                        'types of the same reference, ordered by the teacher with a soft target; the rest stay within-type (hard)')
    parser.add_argument('--lora', default=0, type=int, help='>0: LoRA rank; the CLIP image encoders stay frozen and only low-rank adapters '
                        '(attention in/out projections and MLP matrices of every block) are trained, at --lora_lr; saved weights are merged')
    parser.add_argument('--lora_alpha', default=0, type=float, help='LoRA scale numerator (0 = 2 * rank)')
    parser.add_argument('--lora_lr', default=1e-4, type=float)
    parser.add_argument('--save_every', default=0, type=int, help='>0: also save the weights after every N-th epoch')
    parser.add_argument('--synth_gate', action='store_true', help='with --synth_teacher: within-type pairs keep the level order but '
                        'their target is sigma(max(d_stronger - d_milder, 0) / tau) (teacher distance gap), so levels the teacher '
                        'cannot tell apart get ~0.5 instead of a hard 1')
    parser.add_argument('--rank_soft', action='store_true', help='--rank_w pairs use the observers\' preference probability '
                        'Phi(dMOS / SE) from the raw scores as soft target, so pairs humans cannot tell apart carry ~no signal')
    parser.add_argument('--con_w', default=0.08, type=float, help='weight of the colour<->depth NT-Xent loss (0 = off; CLIP-PCQA uses 0.08)')
    parser.add_argument('--con_tau', default=0.07, type=float, help='NT-Xent temperature')
    parser.add_argument('--inv_w', default=0, type=float, help='content invariance: cosine pull of each sample\'s view features to a '
                        'stop-gradient copy of the SAME distortion on another training object (DSP-style; 0 = off)')
    parser.add_argument('--plcc_w', default=0, type=float, help='weight of the in-batch (1 - PLCC)/2 loss on the predicted score')
    parser.add_argument('--view_pool', default='', type=str, help='extra rendered views for training-time viewpoint augmentation: '
                        '"<colour dir>,<depth dir>,<n views>"; each training sample draws img_length_read views from the canonical ones + these')
    parser.add_argument('--clip_init', default='', type=str, help='state dict of a CLIP model (same architecture as PCQA_CLIP) to '
                        'initialise both encoders, e.g. the quality-pretrained LIQE ViT-B/32')
    parser.add_argument('--center_obj', default=0, type=float, help='alpha in [0,1]: shift the training MOS and raw scores of every '
                        'reference object by alpha * (train mean - object mean), removing per-object scale offsets from the targets')
    parser.add_argument('--kpa_fuse', action='store_true', help='KPA feature fused into the CLIP image feature (gated) instead of added to the logits')
    parser.add_argument('--kpa_aux', default=0, type=float, help='weight of the KPA-only MOS regression loss (0 = off)')
    parser.add_argument('--kpa_mix', action='store_true', help='KPA as a second expert: final distribution = gated mixture of CLIP and KPA distributions')
    parser.add_argument('--kpa_gate', action='store_true', help='gate the KPA logit correction by the aux head\'s distortion-type probabilities')
    parser.add_argument('--clip_drop', default=0, type=float, help='modality dropout: per training sample with this probability the '
                        'final distribution comes from a KPA-only head instead of CLIP + correction (test: never dropped)')
    parser.add_argument('--ogm', default=0, type=float, help='alpha of OGM-style modulation (CVPR 2022): when the CLIP path fits the '
                        'training batch better than the KPA-only head (error ratio rho > 1), the CLIP parameters step with lr * (1 - tanh(alpha*(rho-1))). '
                        'Applied to the learning rate, not the gradient, because Adam cancels a constant gradient scale. Needs --kpa_aux')
    parser.add_argument('--test_tta', default=1, type=int, help='test-time crops per view, averaged: 1 = centre crop (CLIP-PCQA '
                        'default); 5 = centre + 4 corner 224 crops; 9 / 16 = 3x3 / 4x4 grid of 224 windows covering the whole view '
                        '(the views are ~640-900 px, a centre crop sees only ~15-25 %% of the object)')
    parser.add_argument('--tta_batched', action='store_true', help='test-time crops computed from ONE load of every view '
                        '(all crops of a sample in one forward batch) instead of one dataset pass per crop; same predictions, faster')
    parser.add_argument('--tta_chunk', default=16, type=int, help='crops per forward batch with --tta_batched')
    parser.add_argument('--view_mode', default='crop', choices=['crop', 'resize', 'fragment'], help='how a rendered view '
                        '(~640-900 px) becomes the encoder input: crop = 224 crop (CLIP-PCQA); resize = whole view padded square and '
                        'resized to --view_size; fragment = 7x7 grid of native 32x32 patches stitched to 224 (FAST-VQA style)')
    parser.add_argument('--view_size', default=224, type=int, help='encoder input size; != 224 interpolates the CLIP position embedding')
    parser.add_argument('--tta_fg_weight', action='store_true', help='weight every test crop by its share of object (non-white) pixels '
                        'in the colour views instead of a plain mean')
    parser.add_argument('--train_last', default=0, type=int, help='fine-tune only the last N transformer blocks (+ ln_post/proj) of '
                        'both CLIP image encoders (0 = all, the CLIP-PCQA default)')
    parser.add_argument('--csv_suffix', default='', type=str, help='split files train_<k><suffix>.csv / test_<k><suffix>.csv '
                        '(e.g. _in2 = inner split 2 of outer fold k, for cross-fitted stacking; see make_inner_splits.py)')
    parser.add_argument('--kpa_init', default='', type=str, help='pretrained standalone KPA weights per fold, path with {k} (train_kpa.py --save)')
    parser.add_argument('--hflip', action='store_true', help='random horizontal mirror of training views')
    parser.add_argument('--tam', action='store_true', help='text-guided token pooling instead of the token mean')
    parser.add_argument('--fca', default=0, type=float, help='weight of the KPA-token <-> image-token alignment loss (needs --color2_mode kpa and --fca_bbox)')
    parser.add_argument('--fca_bbox', default='', type=str, help='npz with names + per-view render mask bboxes (y0,y1,x0,x1)')
    parser.add_argument('--shared_crop', action='store_true', help='same random crop window for colour and depth in training')
    parser.add_argument('--group_by_object', action='store_true', help='every training batch holds samples of ONE reference object, '
                        'so in-batch pairs (e.g. --rank_w) are all within-object comparisons')
    parser.add_argument('--train_crops', default=1, type=int, help='K independent random training crops per view of every sample, '
                        'run as one forward batch of B*K (1 = CLIP-PCQA default)')
    parser.add_argument('--train_crops_mode', default='avg', choices=['avg', 'indep'], help="with --train_crops K: 'avg' = the K crops' "
                        "outputs are averaged per sample before the losses (the test-time crop average, seen in training); 'indep' = "
                        "every crop is its own training sample (same crops, no averaging: the augmentation-only control)")
    parser.add_argument('--resume', action='store_true', help='save the full training state after every epoch to '
                        '<ckpt_dir>/<DB>_<k>_resume.pth and, if that file exists at start, continue from the next epoch '
                        '(the file is removed when the fold finishes)')
    parser.add_argument('--resume_dir', default='', type=str, help='where --resume keeps its state file (default: --ckpt_dir); e.g. '
                        'a node-local scratch directory so the multi-GB per-epoch state is not written to the shared file system')
    parser.add_argument('--fold', default=0, type=int, help='run only this fold (1-based); 0 = all folds')
    parser.add_argument('--ckpt_dir', default='', type=str, help='where to save checkpoints')
    args = parser.parse_args()
    assert not args.synth_gate or args.synth_teacher, '--synth_gate needs --synth_teacher'

    return args

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def extend_args(args):
    """
    Add new config variables.
    """
    # args.n_ctx = 16  # number of context vectors
    args.csc = True  # class-specific context
    args.ctx_init = ""  # initialization words
    args.prec = "fp32"  # fp16, fp32, amp
    # args.class_token_position = "middle"  # 'middle' or 'end' or 'front'
    args.subsample_classes = "all"  # all, base or new

if __name__=='__main__':
    print('*************************************************************************************************************************')
    args = parse_args()
    extend_args(args)

    if args.ablation_mask: args.ablation_mask = [g.strip() for g in args.ablation_mask.split(",")]
    else: args.ablation_mask = None

    set_rand_seed(args.seed)
    # init torch.linalg lazily-loaded kernels on every GPU before DataParallel threads use them
    for _d in range(torch.cuda.device_count()):
        torch.linalg.eigh(torch.eye(3, device=f'cuda:{_d}'))

    gpu = args.gpu
    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    database = args.database
    img_length_read = args.img_length_read
    data_dir_color = resolve_project_path(args.data_dir_color)
    data_dir_depth = resolve_project_path(args.data_dir_depth)
    data_dir_pc = resolve_project_path(args.data_dir_pc)
    best_all = np.zeros([args.k_fold_num, 4])
    ckpt_dir = args.ckpt_dir or project_path('ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)

    for k_fold_id in range(1,args.k_fold_num + 1):
        if args.fold and k_fold_id != args.fold:
            continue

        print('The current k_fold_id is ' + str(k_fold_id))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # if torch.cuda.is_available():
        #     os.environ["CUDA_VISIBLE_DEVICES"] = '5'

        quality_classes =['bad', 'poor', 'fair', 'good', 'excellent'] 
        if database == 'SJTU':           
            train_filename_list = project_path('csvfiles', 'sjtu_data_info', 'train_'+str(k_fold_id)+'.csv')
            test_filename_list = project_path('csvfiles', 'sjtu_data_info', 'test_'+str(k_fold_id)+'.csv')
            raw_score_path = project_path('csvfiles', 'sjtu_data_info', 'total_raw_score.csv')
            score_list = [2.0, 4.0, 6.0, 8.0, 10.0]


        elif database == 'LS_PCQA_part':
            train_filename_list = project_path('csvfiles', 'ls_pcqa_data_info', 'train_'+str(k_fold_id)+'.csv')
            test_filename_list = project_path('csvfiles', 'ls_pcqa_data_info', 'test_'+str(k_fold_id)+'.csv')
            raw_score_path = project_path('csvfiles', 'ls_pcqa_data_info', 'total_raw_score.csv')
            score_list = [1.0, 2.0, 3.0, 4.0, 5.0]

        elif database == 'BASICS':
            train_filename_list = project_path('csvfiles', 'basics_data_info', 'train_'+str(k_fold_id)+'.csv')
            test_filename_list = project_path('csvfiles', 'basics_data_info', 'test_'+str(k_fold_id)+'.csv')
            raw_score_path = project_path('csvfiles', 'basics_data_info', 'total_raw_score.csv')
            score_list = [0.0, 1.0, 2.0, 3.0, 4.0]

        elif database == 'WPC':
            train_filename_list = project_path('csvfiles', 'wpc_data_info', 'train_'+str(k_fold_id)+'.csv')
            test_filename_list = project_path('csvfiles', 'wpc_data_info', 'test_'+str(k_fold_id)+'.csv')
            raw_score_path = project_path('csvfiles', 'wpc_data_info', 'total_raw_score.csv')
            score_list = [20.0, 40.0, 60.0, 80.0, 100.0]
        elif database == 'LSX':  # LS-PCQA minus every SJTU reference content (cross-dataset pretraining for SJTU)
            train_filename_list = project_path('csvfiles', 'lsx_data_info', 'train_'+str(k_fold_id)+'.csv')
            test_filename_list = project_path('csvfiles', 'lsx_data_info', 'test_'+str(k_fold_id)+'.csv')
            raw_score_path = project_path('csvfiles', 'lsx_data_info', 'total_raw_score.csv')
            score_list = [1.0, 2.0, 3.0, 4.0, 5.0]
        elif database == 'MIX':  # WPC + BASICS rescaled to the SJTU 0-10 scale (cross-dataset pretraining)
            train_filename_list = project_path('csvfiles', 'mix_data_info', 'train_'+str(k_fold_id)+'.csv')
            test_filename_list = project_path('csvfiles', 'mix_data_info', 'test_'+str(k_fold_id)+'.csv')
            raw_score_path = project_path('csvfiles', 'mix_data_info', 'total_raw_score.csv')
            score_list = [2.0, 4.0, 6.0, 8.0, 10.0]
        else:
            raise ValueError(f"Unsupported database: {database}")

        transformations_train = transforms.Compose([transforms.RandomCrop(224),transforms.ToTensor(),\
                transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
        transformations_test = transforms.Compose([transforms.CenterCrop(224),transforms.ToTensor(),\
                transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

        if args.csv_suffix:  # inner (nested) split of the outer fold's training objects
            train_filename_list = train_filename_list[:-4] + args.csv_suffix + '.csv'
            test_filename_list = test_filename_list[:-4] + args.csv_suffix + '.csv'
        print('Trainging set: ' + train_filename_list)
        if args.view_mode != 'crop':  # the dataset already produces the final view; no crop in the transforms
            assert args.test_tta == 1, '--test_tta only applies to --view_mode crop'
            transformations_train = transformations_test = transforms.Compose([transforms.ToTensor(),
                transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
        
        if args.aux_deg:  # degradation labels from file names (free side information, utils/deg_labels.py)
            from utils.deg_labels import build as build_deg, P_MAX
            total_names = pd.read_csv(os.path.join(os.path.dirname(train_filename_list), 'total.csv')).name
            if args.aux_factor:
                from utils.deg_labels import build_factors, FACTORS
                aux_labels, aux_types = build_factors(database, list(total_names)), FACTORS
                args.aux_out = len(FACTORS) + 1
            else:
                aux_labels, aux_types = build_deg(database, list(total_names))
            args.aux_ntypes = len(aux_types)
            assert torch.cuda.device_count() <= 1, '--aux_deg reads model.last_aux, which DataParallel replicas do not set'
            print(f'aux_deg: {len(aux_types)} types {aux_types}, weight {args.aux_deg}')
        # load the network
        model = CLIPPCQA_Net(device, args, score_list, quality_classes)
        if args.kpa_init:  # standalone-pretrained KPA (trained on this fold's training split only); its scalar head is dropped
            ksd = torch.load(args.kpa_init.format(k=k_fold_id), map_location='cpu')
            r = model.geom_kpa.load_state_dict({kk: v for kk, v in ksd.items() if not kk.startswith('head.')}, strict=True)
            print('kpa_init loaded', args.kpa_init.format(k=k_fold_id), r)
        if args.init_ckpt:  # start from a (cross-dataset) pretrained checkpoint
            sd = torch.load(args.init_ckpt, map_location='cpu')
            sd = {kk[7:] if kk.startswith('module.') else kk: v for kk, v in sd.items()}
            own = model.state_dict()  # e.g. the aux head of a MIX checkpoint has 9 types, SJTU has 7: re-initialise it
            skipped = [kk for kk, v in sd.items() if kk in own and own[kk].shape != v.shape]
            sd = {kk: v for kk, v in sd.items() if kk not in skipped}
            if skipped:
                print(f'init: skipped {len(skipped)} tensors with a different shape: {skipped}')
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f'init from {args.init_ckpt}: missing={len(missing)} unexpected={len(unexpected)}')
        model = model.to(device)
       
        criterion = EMD_Quan_Con_Loss(score_list, align_loss_weight=args.con_w, temperature=args.con_tau).to(device)

        for name, param in model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)
            if "visual" in name:
                param.requires_grad_(True)
                if args.train_last:  # freeze the stem and the first (12 - N) blocks
                    mb = re.search(r'visual\.transformer\.resblocks\.(\d+)\.', name)
                    nblk = len(model.clip_texture.visual.transformer.resblocks)
                    if (mb is None and not re.search(r'visual\.(ln_post|proj)', name)) or (mb and int(mb.group(1)) < nblk - args.train_last):
                        param.requires_grad_(False)
            if "geom_" in name or "splat_" in name:
                param.requires_grad_(True)
            if args.lora and "visual" in name:  # LoRA: frozen encoders, adapters trainable
                param.requires_grad_("lora_" in name)

        # added branches (geom_/splat_) start from zero and need a larger step than the CLIP fine-tuning lr
        branch = [p for n, p in model.named_parameters() if p.requires_grad and ("geom_" in n or "splat_" in n)]
        lora = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
        others = [p for n, p in model.named_parameters() if p.requires_grad and not ("geom_" in n or "splat_" in n or "lora_" in n)]
        # no weight decay on the branches: their output layer is zero-initialised, so the inner layers get
        # no loss gradient at first and Adam's coupled L2 term alone would shrink them to zero (dead branch)
        optimizer = torch.optim.Adam([{'params': others, 'lr': args.learning_rate, 'weight_decay': args.decay_rate},
                                      {'params': branch, 'lr': args.branch_lr or args.learning_rate, 'weight_decay': 0.0}]
                                     + ([{'params': lora, 'lr': args.lora_lr, 'weight_decay': 0.0}] if lora else []))
        if args.lora:
            print('LoRA rank %d: %d adapter tensors, %.2fM trainable parameters in total' % (args.lora, len(lora),
                  sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6))
        print('Using Adam optimizer, initial learning rate: ' + str(args.learning_rate))

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        print(f'Using dataset: {database}')

        print("Ready to train network")
        print('*************************************************************************************************************************')
        best = np.zeros(4)

        print(f'Total trainable parameters: {count_parameters(model)}')

        min_training_loss = 10000
        best_test_srcc = -2.0; ogm_e = []; ogm_log = []  # per-sample predictions of the best-test epoch are also saved (industry-style evaluation)
        
        train_dataset = CLIPPCQA_Dataset(data_dir_color = data_dir_color, data_dir_depth = data_dir_depth, datainfo_path = train_filename_list, raw_score_path = raw_score_path, transform = transformations_train, img_length_read = img_length_read, data_dir_pc = data_dir_pc, use_geometry = args.use_geometry, geom_num_points = args.geom_num_points, data_dir_color2 = resolve_project_path(args.data_dir_color2))
        test_dataset = CLIPPCQA_Dataset(data_dir_color = data_dir_color, data_dir_depth = data_dir_depth, datainfo_path = test_filename_list, raw_score_path = raw_score_path, transform = transformations_test, img_length_read = img_length_read, is_train = False, data_dir_pc = data_dir_pc, use_geometry = args.use_geometry, geom_num_points = args.geom_num_points, data_dir_color2 = resolve_project_path(args.data_dir_color2))
        # --test_tta: extra test sets whose views are cropped at the 4 corners (same window for colour and depth, as in
        # eval_folds.py); predictions of all crops are averaged in the test loop
        class _TTAViews(torch.utils.data.Dataset):
            """All test-time crops of one test sample from a single read of its views: colour / depth [C, V, 3, 224, 224]
            (C = crop positions in eval_pos order, same windows as _crop_at), plus the sample's point-branch input and MOS
            taken from the centre-crop test set."""
            def __init__(self, base, pos):
                self.base, self.pos = base, pos
                self.norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            def __len__(self):
                return len(self.base)
            def __getitem__(self, i):
                item = self.base[i]
                name = self.base.ply_name.iloc[i, 0].split('.')[0]
                cols, deps = [], []
                for v in range(self.base.img_length_read):
                    c = Image.open(os.path.join(self.base.data_dir_color, name + '_view_' + str(v) + '.png')).convert('RGB')
                    d = Image.open(os.path.join(self.base.data_dir_depth, name + '_view_' + str(v) + '.png')).convert('RGB')
                    w, h = c.size
                    assert d.size == c.size, 'colour and depth views differ in size'
                    if min(w, h) < 224:  # as the dataset at test time: centre square resized to 224 (every crop is then this image)
                        sq = transforms.Compose([transforms.CenterCrop(min(w, h)), transforms.Resize((224, 224))])
                        c, d = sq(c), sq(d); w, h = c.size
                    cc, dd = [], []
                    for py, px in self.pos:
                        t0, l0 = int(round((h - 224) * py)), int(round((w - 224) * px))
                        cc.append(self.norm(transforms.functional.crop(c, t0, l0, 224, 224)))
                        dd.append(self.norm(transforms.functional.crop(d, t0, l0, 224, 224)))
                    cols.append(torch.stack(cc)); deps.append(torch.stack(dd))
                c2 = item[2] if self.base.kpa_dir is not None else None
                return torch.stack(cols, 1), torch.stack(deps, 1), c2, item[-2]

        def _crop_at(py, px):
            def f(im):
                w, h = im.size
                return transforms.functional.crop(im, int(round((h - 224) * py)), int(round((w - 224) * px)), 224, 224)
            return transforms.Compose([transforms.Lambda(f), transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        if args.test_tta in (9, 16):  # full grid; the centre-crop test set is then not used for scoring
            g = np.linspace(0, 1, int(round(args.test_tta ** 0.5)))
            tta_pos = [(float(y), float(x)) for y in g for x in g]
        else:
            tta_pos = [(0, 0), (0, 1), (1, 0), (1, 1)][:max(args.test_tta - 1, 0)]
        tta_datasets = [CLIPPCQA_Dataset(data_dir_color = data_dir_color, data_dir_depth = data_dir_depth, datainfo_path = test_filename_list, raw_score_path = raw_score_path, transform = _crop_at(py, px), img_length_read = img_length_read, is_train = False, data_dir_pc = data_dir_pc, use_geometry = args.use_geometry, geom_num_points = args.geom_num_points, data_dir_color2 = resolve_project_path(args.data_dir_color2))
                        for py, px in tta_pos]
        eval_pos = tta_pos if args.test_tta in (9, 16) else [(0.5, 0.5)] + tta_pos
        # per-sample crop weights: object (non-white) pixel share of the colour crop, averaged over the views
        test_names = list(test_dataset.ply_name.iloc[:, 0])
        tta_w = np.ones((len(test_names), len(eval_pos)))
        if args.tta_fg_weight:
            for i_n, nm in enumerate(test_names):
                for i_v in range(img_length_read):
                    im = Image.open(os.path.join(data_dir_color, nm.split('.')[0] + '_view_' + str(i_v) + '.png')).convert('RGB')
                    if min(im.size) < 224:  # narrow view: the model gets the centre square resized to 224 for every crop
                        im = transforms.Resize((224, 224))(transforms.CenterCrop(min(im.size))(im))
                    im = np.asarray(im)
                    hh, ww = im.shape[:2]
                    for c_i, (py, px) in enumerate(eval_pos):
                        t0, l0 = int(round((hh - 224) * py)), int(round((ww - 224) * px))
                        tta_w[i_n, c_i] += (im[t0:t0 + 224, l0:l0 + 224].min(-1) < 240).mean() / img_length_read - (1.0 / img_length_read)
            tta_w = tta_w + 1e-6
        tta_w = tta_w / tta_w.sum(1, keepdims=True)
        print(f'test-time crops: {len(eval_pos)} per view, fg weighting {args.tta_fg_weight}')
        for d_ in [train_dataset, test_dataset] + tta_datasets:
            d_.view_mode = args.view_mode
            if args.view_mode == 'resize':
                d_.crop_size = args.view_size
        if args.view_mode != 'crop':
            print(f'view input: {args.view_mode}, size {args.view_size}')
        
        if args.aux_deg:
            train_dataset.aux = aux_labels
        train_dataset.shared_crop = args.shared_crop
        train_dataset.hflip = args.hflip
        if args.center_obj:  # per-object offset removal on the TRAINING targets only (test labels untouched)
            names = train_dataset.ply_name.iloc[:, 0]
            objs = names.map(lambda n: object_of(database, n))
            mos = train_dataset.ply_mos.astype(float)
            shift = args.center_obj * (mos.mean() - mos.groupby(objs.values).transform('mean'))
            train_dataset.ply_mos = mos + shift.values
            raw = train_dataset.raw_score.copy(); cols = raw.columns[1:]
            lo, hi = np.nanmin(raw[cols].values), np.nanmax(raw[cols].values)
            sh = dict(zip(names, shift.values))
            d = raw['name'].map(sh).fillna(0).values[:, None]
            raw[cols] = np.clip(raw[cols].values.astype(float) + d, lo, hi)
            train_dataset.raw_score = raw
            print('center_obj %.2f: per-object shifts %s' % (args.center_obj, np.round(sorted(set(np.round(shift.values, 3))), 2)))
        if args.view_pool:
            vc, vd, vn = args.view_pool.split(',')
            train_dataset.view_pool = [(data_dir_color, data_dir_depth, i) for i in range(img_length_read)] + \
                                      [(resolve_project_path(vc), resolve_project_path(vd), i) for i in range(int(vn))]
        if args.fca:
            assert args.color2_mode == 'kpa' and args.fca_bbox, '--fca needs the KPA branch and --fca_bbox'
            zb = np.load(args.fca_bbox)
            train_dataset.fca_bbox = dict(zip([str(n) for n in zb['names']], zb['bbox']))
            assert torch.cuda.device_count() <= 1, '--fca reads model.last_fca, which DataParallel replicas do not set'
        assert not args.inv_w or args.group_by_object, '--inv_w reads the batch indices of --group_by_object'
        partners = distortion_partners(train_dataset, database) if args.inv_w else None
        if args.train_crops > 1:
            assert not (args.use_geometry or args.inv_w or args.fca or args.ogm or args.view_mode != 'crop' or args.kpa_fuse
                        or args.kpa_mix or args.kpa_gate or args.clip_drop), \
                '--train_crops is implemented for the image path + late-fusion KPA (kpa_aux / aux_deg / rank / plcc) only'
            model.kpa_repeat = args.train_crops  # KPA heads once per sample, repeated for its crops (model forward)

            class _KCrops(torch.utils.data.Dataset):
                """K draws of the same training sample: colour / depth stacked to [K, V, 3, H, W] (each draw makes its
                own random crops); the remaining fields (KPA input, MOS, raw scores, aux labels) come from the first draw."""
                def __init__(self, base, k):
                    self.base, self.k = base, k
                def __len__(self):
                    return len(self.base)
                def __getattr__(self, name):  # object_batches() etc. read the base dataset's attributes
                    if name == 'base':
                        raise AttributeError(name)
                    return getattr(self.base, name)
                def __getitem__(self, idx):
                    items = [self.base[idx] for _ in range(self.k)]
                    out = list(items[0])
                    out[0], out[1] = torch.stack([it[0] for it in items]), torch.stack([it[1] for it in items])
                    return tuple(out)
            train_dataset = _KCrops(train_dataset, args.train_crops)

        start_epoch = 0
        resume_path = os.path.join(args.resume_dir or ckpt_dir, '%s_%d_resume.pth' % (database, k_fold_id))
        if args.resume:
            os.makedirs(os.path.dirname(resume_path), exist_ok=True)
        if args.resume and os.path.exists(resume_path):  # continue an interrupted run after its last completed epoch
            st = torch.load(resume_path, map_location='cpu')
            model.load_state_dict(st['model']); optimizer.load_state_dict(st['optimizer']); scheduler.load_state_dict(st['scheduler'])
            best_test_srcc, min_training_loss, best, ogm_e = st['best_test_srcc'], st['min_training_loss'], st['best'], st['ogm_e']
            random.setstate(st['rng'][0]); np.random.set_state(st['rng'][1]); torch.set_rng_state(st['rng'][2])
            if torch.cuda.is_available():
                torch.cuda.set_rng_state_all(st['rng'][3])
            start_epoch = st['epoch']
            print('RESUMED from %s: epochs 1-%d done, continuing at epoch %d (best test SRCC so far %.4f, min train loss %.4f)'
                  % (resume_path, start_epoch, start_epoch + 1, best_test_srcc, min_training_loss), flush=True)
        for epoch in range(start_epoch, num_epochs):
            if args.group_by_object:
                epoch_batches = object_batches(train_dataset, database, batch_size)
                batch_idx_cache = [list(b) for b in epoch_batches]  # --inv_w needs the sample indices of each batch
                train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=8, collate_fn=pcqa_collate,
                                                           batch_sampler=epoch_batches)
            else:
                train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=pcqa_collate)
            test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8, collate_fn=pcqa_collate)
            if args.synth_w:  # (milder, stronger) pairs of the same reference and codec type, adjacent in each batch
                assert args.train_crops == 1 and not args.use_geometry
                if epoch == start_epoch:
                    synth_ds = CLIPPCQA_Dataset(data_dir_color=os.path.join(resolve_project_path(args.synth_dir), '6view'),
                                                data_dir_depth=os.path.join(resolve_project_path(args.synth_dir), '6view_depth'),
                                                datainfo_path=resolve_project_path(args.synth_csv), raw_score_path=raw_score_path,
                                                transform=transformations_train, img_length_read=img_length_read,
                                                data_dir_color2=resolve_project_path(args.synth_kpa) if args.data_dir_color2 else '')
                    # only references of training objects (k-fold: never the test fold's objects); SJTU spells one 'UBL_unicorn'
                    norm = lambda s: re.sub(r'[^a-z0-9]', '', s.lower()).replace('ublunicorn', 'ulbunicorn')
                    train_objs = {norm(object_of(database, n)) for n in train_dataset.ply_name.iloc[:, 0]}
                    groups = {}
                    for j, n in enumerate(synth_ds.ply_name.iloc[:, 0]):
                        grp = re.sub(r'_L\d+\.ply$', '', n)
                        if norm(grp.rsplit('_', 1)[0]) in train_objs:
                            groups.setdefault(grp, []).append((int(synth_ds.ply_mos.iloc[j]), j))
                    synth_groups = [g for g in groups.values() if len(g) > 1]
                    synth_refs, synth_teach, teach_tau = {}, None, 1.0
                    if args.synth_teacher:  # perceptual distance per synthetic sample (lower = closer to its reference = better)
                        tdf = pd.read_csv(resolve_project_path(args.synth_teacher)); tmap = dict(zip(tdf.name, tdf.deep))
                        synth_teach = np.array([tmap[n] for n in synth_ds.ply_name.iloc[:, 0]], dtype=np.float64)
                        for grp, g in groups.items():
                            synth_refs.setdefault(grp.rsplit('_', 1)[0], []).extend(j for _, j in g)
                        synth_refs = [v for v in synth_refs.values() if len(v) > 1]
                        teach_tau = 0.1 * float(np.std(synth_teach))   # soft-target temperature on the teacher scale
                        print('synthetic teacher: %d refs, tau %.4f, cross-type share %.2f' % (len(synth_refs), teach_tau, args.synth_xtype))
                    if args.aux_deg and args.aux_factor:  # exact factors / severity of the synthetic samples, appended to each item
                        synth_ds.aux = build_factors('SYNTH', list(synth_ds.ply_name.iloc[:, 0]))
                    print('synthetic ranking: %d groups from refs %s' % (len(synth_groups), sorted({k.rsplit('_', 1)[0] for k in groups})))
                    assert synth_groups, 'no synthetic sample matches a training object'

                synth_targets = []   # per yielded batch: the target P(first of each pair is better), consumed in order
                class _PairSampler(torch.utils.data.Sampler):
                    def __init__(self, n):
                        self.n = n
                    def __len__(self):
                        return self.n
                    def __iter__(self):
                        for _ in range(self.n):
                            out, tgt = [], []
                            for g in random.sample(synth_groups, min(args.synth_pairs, len(synth_groups))):
                                if synth_teach is not None and random.random() < args.synth_xtype:
                                    a, b = random.sample(random.choice(synth_refs), 2)      # any two versions of one reference
                                    if synth_teach[a] > synth_teach[b]: a, b = b, a          # a = closer to the reference
                                    tgt.append(1 / (1 + math.exp(-(synth_teach[b] - synth_teach[a]) / teach_tau)))
                                else:
                                    (_, a), (_, b) = sorted(random.sample(g, 2))   # a = milder level
                                    tgt.append(1 / (1 + math.exp(-max(synth_teach[b] - synth_teach[a], 0.0) / teach_tau))
                                               if args.synth_gate else 1.0)
                                out += [a, b]
                            synth_targets.append(tgt)
                            yield out
                synth_iter = iter(torch.utils.data.DataLoader(synth_ds, batch_sampler=_PairSampler(len(train_loader) + 1),
                                                              num_workers=4, collate_fn=pcqa_collate))
            if args.tta_batched and len(eval_pos) > 1:
                assert not args.use_geometry and args.view_mode == 'crop'
                tta_loader = torch.utils.data.DataLoader(_TTAViews(test_dataset, eval_pos), batch_size=None, shuffle=False, num_workers=8)
            test_loaders = ([] if args.test_tta in (9, 16) else [test_loader]) + [torch.utils.data.DataLoader(dataset=d, batch_size=1, shuffle=False, num_workers=2, collate_fn=pcqa_collate) for d in tta_datasets]
            if args.tta_batched and len(eval_pos) > 1:
                test_loaders = []  # the batched path above replaces the per-crop loaders
            n_train = len(train_dataset)
            n_test = len(test_dataset)

            model.train()
            start = time.time()
            batch_losses = []
            batch_losses_each_disp = []

            aux_losses = []
            synth_losses = []
            inv_losses = []
            fca_losses = []
            for i, batch in enumerate(train_loader):
                if args.aux_deg:
                    *batch, aux_t, aux_p = batch
                if args.use_geometry:
                    imgs, depth_imgs, point_patches, mos, original_scores = batch
                    point_patches = point_patches.to(device)
                elif args.data_dir_color2:
                    imgs, depth_imgs, color2_imgs, mos, original_scores = batch
                    point_patches = None
                else:
                    imgs, depth_imgs, mos, original_scores = batch
                    point_patches = None
                c2 = to_device(color2_imgs, device) if args.data_dir_color2 else None
                imgs = imgs.to(device)
                depth_imgs = depth_imgs.to(device)
                mos = mos[:,np.newaxis]
                mos = mos.to(device)
                K = args.train_crops
                if K > 1:  # [B, K, V, ...] -> one forward batch of B*K samples, the K crops of a sample adjacent;
                    B = imgs.shape[0]  # the KPA input c2 stays one per sample (model.kpa_repeat)
                    imgs, depth_imgs = imgs.flatten(0, 1), depth_imgs.flatten(0, 1)
                if point_patches is not None and isinstance(model, nn.DataParallel):
                    # compute descriptors in the main thread: torch.linalg is not safe in DataParallel replica threads
                    from utils.geometry_features import compute_geom_features
                    geom_features = compute_geom_features(point_patches, ablation_mask=args.ablation_mask)
                    texture_f, depth_f, quality_score, pred_dis, pred_CDF = model(imgs, depth_imgs, geom_features=geom_features)
                else:
                    texture_f, depth_f, quality_score, pred_dis, pred_CDF = model(imgs, depth_imgs, point_patches=point_patches, color2_imgs=c2)

                # compute loss
                if K > 1:
                    # texture_f / depth_f hold one row per (sample, crop, view); the colour<->depth NT-Xent is taken per
                    # crop draw (the colour and depth crops of one draw, as without K) and averaged over the draws.
                    # 'avg': the quality terms see the crop-averaged distribution
                    # (identical for every crop); 'indep': every crop's own distribution, its own score in the in-batch terms
                    tv, dv = texture_f.reshape(B, K, -1, texture_f.shape[-1]), depth_f.reshape(B, K, -1, depth_f.shape[-1])
                    cdf = pred_CDF.reshape(B, K, -1)
                    if args.train_crops_mode == 'avg':
                        cdf = cdf.mean(1, keepdim=True).expand(-1, K, -1)
                        quality_score = quality_score.reshape(B, K, -1).mean(1)
                        for att in ('last_aux', 'last_kpa_pred'):
                            v = getattr(model, att, None)
                            if v is not None:
                                setattr(model, att, v.reshape(B, K, *v.shape[1:]).mean(1))
                    loss = sum(criterion(tv[:, k].flatten(0, 1), dv[:, k].flatten(0, 1), original_scores, cdf[:, k]) for k in range(K)) / K
                    if args.train_crops_mode == 'indep':
                        mos, original_scores = mos.repeat_interleave(K, 0), original_scores.repeat_interleave(K, 0)
                        if args.aux_deg:
                            aux_t, aux_p = aux_t.repeat_interleave(K, 0), aux_p.repeat_interleave(K, 0)
                else:
                    loss = criterion(texture_f, depth_f, original_scores, pred_CDF)
                q_loss = loss.detach()  # model selection keeps using the quality loss only
                if args.rank_w:  # logistic pairwise ranking over the batch; tau = 5% of the score range
                    sc = quality_score.squeeze(1).float(); mm = mos.squeeze(1).float()
                    sg = torch.sign(mm[:, None] - mm[None]); keep = sg != 0
                    if keep.any():
                        tau = 0.05 * (score_list[-1] - score_list[0])
                        if args.rank_soft:  # target = P(a better than b) from the observer scores (two-sample z on the means)
                            r = original_scores.float().to(device); n = (~torch.isnan(r)).sum(1).clamp(min=2).float()
                            mu = torch.nanmean(r, 1); var = torch.nanmean((r - mu[:, None]) ** 2, 1) * n / (n - 1)
                            se = (var / n)[:, None] + (var / n)[None]
                            z = (mu[:, None] - mu[None]) / se.clamp(min=1e-8).sqrt()
                            tgt = torch.where(se > 1e-8, 0.5 * (1 + torch.erf(z / math.sqrt(2))), (sg + 1) / 2)
                            d = (sc[:, None] - sc[None]) / tau
                            loss = loss + args.rank_w * F.binary_cross_entropy_with_logits(d[keep], tgt[keep])
                        else:
                            loss = loss + args.rank_w * F.softplus(-(sc[:, None] - sc[None])[keep] * sg[keep] / tau).mean()
                if args.aux_deg:
                    out = model.last_aux.float(); nt = args.aux_ntypes
                    aux_t, aux_p = aux_t.to(device), aux_p.to(device)
                    if args.aux_factor:  # aux_t = severity, aux_p = factor multi-hot
                        l_type = F.binary_cross_entropy_with_logits(out[:, :nt], aux_p.float())
                        l_par = F.mse_loss(torch.sigmoid(out[:, nt]), aux_t.float())
                    else:
                        l_type = F.cross_entropy(out[:, :nt], aux_t)
                        par = out[:, nt:].reshape(-1, nt, P_MAX)[torch.arange(len(aux_t), device=device), aux_t]
                        m = ~torch.isnan(aux_p)
                        l_par = ((par - torch.nan_to_num(aux_p)) ** 2)[m].mean() if m.any() else out.sum() * 0
                    aux_losses.append((l_type.item(), l_par.item()))
                    loss = loss + args.aux_deg * (l_type + l_par)
                if args.kpa_aux and model.last_kpa_pred is not None:  # KPA alone regresses the MOS (scaled to 0-1)
                    tgt = (mos.squeeze(1).float() - score_list[0]) / (score_list[-1] - score_list[0])
                    loss = loss + args.kpa_aux * F.mse_loss(model.last_kpa_pred.float(), tgt)
                if args.fca and model.last_fca is not None:
                    fca_losses.append(model.last_fca.item())
                    loss = loss + args.fca * model.last_fca
                if args.plcc_w and len(mos) > 2:
                    sc = quality_score.squeeze(1).float(); mm = mos.squeeze(1).float()
                    sc, mm = sc - sc.mean(), mm - mm.mean()
                    loss = loss + args.plcc_w * (1 - (sc * mm).sum() / (sc.norm() * mm.norm() + 1e-8)) / 2
                if args.inv_w:  # partner = same distortion, other object; its features are a fixed target
                    bidx = batch_idx_cache.pop(0)
                    pick = [random.choice(partners[b]) if partners[b] else b for b in bidx]
                    pb = pcqa_collate([train_dataset[j] for j in pick])
                    with torch.no_grad():
                        pt, pd_, *_ = model(pb[0].to(device), pb[1].to(device))
                    l_inv = (2 - F.cosine_similarity(texture_f, pt, dim=1).mean() - F.cosine_similarity(depth_f, pd_, dim=1).mean()) / 2
                    inv_losses.append(l_inv.item())
                    loss = loss + args.inv_w * l_inv
                if args.synth_w:  # within-object severity ranking on synthetic codec distortions (no MOS)
                    sb = next(synth_iter)
                    s_c2 = to_device(sb[2], device) if args.data_dir_color2 else None
                    s_score = model(sb[0].to(device), sb[1].to(device), point_patches=None, color2_imgs=s_c2)[2].squeeze(1).float()
                    tgt = torch.tensor(synth_targets.pop(0), device=device, dtype=torch.float32)  # 1 = hard within-type order
                    l_syn = F.binary_cross_entropy_with_logits((s_score[0::2] - s_score[1::2]) / (0.05 * (score_list[-1] - score_list[0])), tgt)
                    synth_losses.append((l_syn.item(), (s_score[0::2] > s_score[1::2]).float().mean().item()))
                    w_syn = args.synth_w * (max(0.0, 1.0 - epoch / args.synth_anneal) if args.synth_anneal else 1.0)
                    loss = loss + w_syn * l_syn
                    if args.aux_deg and args.aux_factor:
                        so = model.last_aux.float(); sev, fac = sb[-2].float().to(device), sb[-1].float().to(device)
                        loss = loss + args.aux_deg * (F.binary_cross_entropy_with_logits(so[:, :args.aux_ntypes], fac)
                                                      + F.mse_loss(torch.sigmoid(so[:, args.aux_ntypes]), sev))
                # Add NaN loss check during training
                if torch.isnan(loss):
                    print(f"WARNING: NaN loss at epoch {epoch+1}, batch {i}. Skipping this batch.")
                    optimizer.zero_grad()
                    continue
                loss.requires_grad_(True)
                batch_losses.append(q_loss.item())
                batch_losses_each_disp.append(q_loss.item())
                optimizer.zero_grad()   # clear gradients for next train

                torch.autograd.backward(loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #add gradient clipping to prevent NaN from happening
                if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                    print(f'WARNING: non-finite gradient at epoch {epoch+1}, batch {i}. Skipping this step.')
                    optimizer.zero_grad()  # a single non-finite step would turn every weight into NaN
                    continue
                if args.ogm and model.last_kpa_pred is not None:  # OGM-style: slow the dominant CLIP path down
                    with torch.no_grad():
                        lo_s, hi_s = score_list[0], score_list[-1]
                        tgt01 = (mos.squeeze(1).float() - lo_s) / (hi_s - lo_s)
                        bins01 = (torch.tensor(score_list, device=device) - lo_s) / (hi_s - lo_s)
                        e_clip = ((model.last_clip_dist.float() * bins01).sum(1) - tgt01).abs().mean().item()
                        e_kpa = (model.last_kpa_pred.float() - tgt01).abs().mean().item()
                    ogm_e = [0.9 * ogm_e[0] + 0.1 * e_clip, 0.9 * ogm_e[1] + 0.1 * e_kpa] if ogm_e else [e_clip, e_kpa]
                    rho = ogm_e[1] / max(ogm_e[0], 1e-6)
                    k_ogm = 1 - math.tanh(args.ogm * max(rho - 1, 0))
                    ogm_log.append(k_ogm)
                    g0 = optimizer.param_groups[0]; base_lr = g0['lr']; g0['lr'] = base_lr * k_ogm
                    optimizer.step()
                    g0['lr'] = base_lr
                else:
                    optimizer.step()

            # an epoch whose batches were all skipped must never win the min-train-loss model selection
            avg_loss = sum(batch_losses) / len(batch_losses) if batch_losses else float('inf')
            print('Epoch %d averaged training loss: %.4f' % (epoch + 1, avg_loss))
            if ogm_log:
                print('Epoch %d OGM CLIP lr factor: mean %.3f min %.3f' % (epoch + 1, sum(ogm_log) / len(ogm_log), min(ogm_log))); ogm_log = []
            if fca_losses:
                print('Epoch %d FCA loss: %.4f' % (epoch + 1, np.mean(fca_losses)))
            if aux_losses:
                print('Epoch %d aux loss: type CE %.4f, param MSE %.4f' % (epoch + 1, *np.mean(aux_losses, 0)))
            if synth_losses:
                print('Epoch %d synthetic ranking loss %.4f, pair accuracy %.3f' % (epoch + 1, *np.mean(synth_losses, 0)))
            if inv_losses:
                print('Epoch %d invariance loss: %.4f' % (epoch + 1, np.mean(inv_losses)))

            scheduler.step()
            lr_current = scheduler.get_last_lr()
            print('The current learning rate is {:.10f}'.format(lr_current[0]))

            end = time.time()
            print('Epoch %d training time cost: %.4f seconds' % (epoch + 1, end-start))

            # Test texture_f
            model.eval()
            y_output = np.zeros(n_test)
            y_test = np.zeros(n_test)

            t_test0 = time.time()
            if args.tta_batched and len(eval_pos) > 1:  # all crops of a sample from one read, one forward batch
                eval_model = model.module if isinstance(model, nn.DataParallel) else model
                with torch.no_grad():
                    for i, (cols, deps, c2s, mos) in enumerate(tta_loader):
                        sc = []
                        for c0 in range(0, cols.shape[0], args.tta_chunk):
                            cb, db = cols[c0:c0 + args.tta_chunk].to(device), deps[c0:c0 + args.tta_chunk].to(device)
                            c2b = None if c2s is None else [t[None].repeat(cb.shape[0], *([1] * t.dim())).to(device) for t in c2s]
                            sc.append(eval_model(cb, db, point_patches=None, color2_imgs=c2b)[2].squeeze(1).float().cpu())
                        y_output[i] = float((torch.cat(sc).numpy() * tta_w[i]).sum())
                        y_test[i] = float(mos)
            with torch.no_grad():  # per-crop path (skipped when test_loaders is empty, i.e. --tta_batched); then the metrics
                for i, batches in enumerate(zip(*test_loaders)):  # same sample from every crop set
                  for c_i, batch in enumerate(batches):
                    if args.use_geometry:
                        imgs, depth_imgs, point_patches, mos, original_scores = batch
                        point_patches = point_patches.to(device)
                    elif args.data_dir_color2:
                        imgs, depth_imgs, color2_imgs, mos, original_scores = batch
                        point_patches = None
                    else:
                        imgs, depth_imgs, mos, original_scores = batch
                        point_patches = None
                    c2 = to_device(color2_imgs, device) if args.data_dir_color2 else None
                    imgs = imgs.to(device)
                    depth_imgs = depth_imgs.to(device)
                    y_test[i] = mos.item()
                    eval_model = model.module if isinstance(model, nn.DataParallel) else model
                    texture_f, depth_f, quality_score, pred_dis, pred_CDF = eval_model(imgs, depth_imgs, point_patches=point_patches, color2_imgs=c2)
                    y_output[i] += tta_w[i, c_i] * quality_score.item()

                print('Epoch %d test time: %.1f seconds' % (epoch + 1, time.time() - t_test0))
                # Add NaN guard
                if np.any(np.isnan(y_output)) or np.any(np.isinf(y_output)):
                    print("WARNING: NaN/Inf in predictions, skipping logistic fit")
                    y_output_logistic = np.zeros_like(y_output)
                    test_PLCC, test_SROCC, test_RMSE, test_KROCC = 0.0, 0.0, 0.0, 0.0
                else:
                    y_output_logistic = fit_function(y_test, y_output)
                    test_PLCC = stats.pearsonr(y_output_logistic, y_test)[0]
                    test_SROCC = stats.spearmanr(y_output, y_test)[0]
                    test_RMSE = np.sqrt(((y_output_logistic-y_test) ** 2).mean())
                    test_KROCC = scipy.stats.kendalltau(y_output, y_test)[0]
                print("Test results: PLCC={:.4f}, SROCC={:.4f}, RMSE={:.4f}, KROCC={:.4f}".format(test_PLCC, test_SROCC, test_RMSE, test_KROCC))

                if test_SROCC > best_test_srcc:  # best test epoch: keep its per-sample predictions for per-type / noise-aware analysis
                    best_test_srcc = test_SROCC
                    pd.DataFrame({'fold': k_fold_id, 'epoch': epoch + 1, 'name': list(test_dataset.ply_name.iloc[:, 0]),
                                  'mos': y_test, 'pred': y_output}).to_csv(
                        os.path.join(ckpt_dir, 'preds_besttest_fold%d.csv' % k_fold_id), index=False)
                    # weights of the best-test epoch too, so the reported (industry-protocol) numbers can be reproduced
                    torch.save(plain_state_dict(model) if args.lora else model.state_dict(), os.path.join(ckpt_dir, database + '_' + str(k_fold_id) + '_besttest_model.pth'))

                if avg_loss < min_training_loss:
                    print('--------------------------------')
                    print("Update best model using best_val_criterion ")
                    torch.save(plain_state_dict(model) if args.lora else model.state_dict(), os.path.join(ckpt_dir, database + '_' + str(k_fold_id) + '_best_model.pth'))
                    best[0:4] = [test_PLCC, test_SROCC, test_RMSE, test_KROCC]
                    min_training_loss = avg_loss

                    print("Update min_trainloss results: PLCC={:.4f}, SROCC={:.4f}, RMSE={:.4f}, KROCC={:.4f}".format(test_PLCC, test_SROCC, test_RMSE, test_KROCC))
                    print('=========================================================================')

            if args.save_every and (epoch + 1) % args.save_every == 0:  # weights every N epochs (epoch-sensitivity analysis)
                torch.save(plain_state_dict(model) if args.lora else model.state_dict(), os.path.join(ckpt_dir, database + '_' + str(k_fold_id) + '_ep%d.pth' % (epoch + 1)))
            if args.resume:  # full state after this epoch; written to a temp file and renamed, so a kill never leaves a torn file
                torch.save({'epoch': epoch + 1, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(), 'best_test_srcc': best_test_srcc, 'min_training_loss': min_training_loss,
                            'best': best.copy(), 'ogm_e': ogm_e,
                            'rng': (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)}, resume_path + '.tmp')
                os.replace(resume_path + '.tmp', resume_path)
                print('Epoch %d resume state saved' % (epoch + 1), flush=True)

        if args.resume and os.path.exists(resume_path):  # the fold is complete: the restart state is no longer needed
            os.remove(resume_path)
        best_all[k_fold_id-1, :] = best
        print('---------------------------------------------------------------')
        print("The best Val results: PLCC={:.4f}, SROCC={:.4f}, RMSE={:.4f}, KROCC={:.4f}".format(best[0], best[1], best[2], best[3]))
        print('*************************************************************************************************************************')
    
    # average score
    best_mean = np.mean(best_all, 0)
    print('------------------------------', database, '-------------------------------')
    print('num_epochs:', num_epochs, ', batchsize:', batch_size, ', img_num:', img_length_read, ', lr:', args.learning_rate, 'n_ctx:', args.n_ctx, 'class_token_position:', args.class_token_position)
    print('*************************************************************************************************************************')
    print("The mean val results: PLCC={:.4f}, SROCC={:.4f}, RMSE={:.4f}, KROCC={:.4f}".format(best_mean[0], best_mean[1], best_mean[2], best_mean[3]))
    print('*************************************************************************************************************************')
