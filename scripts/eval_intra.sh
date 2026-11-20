#!/bin/bash
# Re-inference of k-fold checkpoints on each fold's test split (centre crop or foreground-weighted 4x4 crops), e.g. to
# check saved checkpoints against the predictions written during training.
# usage: bash scripts/eval_intra.sh <DB> <ckpt_dir> <centre|fg4x4> [besttest|best]
#   besttest = best-test-epoch checkpoints (Table 1), best = min-training-loss checkpoints
set -e; source "$(dirname "$0")/common.sh"
db=$1; ck=$2; mode=$3; kind=${4:-besttest}; K=$(cfg "c['intra']['$db']['k_fold_num']")
ex=""; [ $mode = fg4x4 ] && ex="--tta 16 --tta_fg_weight --tta_batched"
pat="$ck/${db}_{k}_${kind}_model.pth"; [ $kind = best ] && pat="$ck/${db}_{k}_best_model.pth"
$PYTHON -W ignore $ROOT/tools/eval_folds.py --code $CODE --ckpt_pattern "$pat" --folds 1-$K --database $db \
  --color $DATA_ROOT/$db/6view --depth $DATA_ROOT/$db/6view_depth --color2 $DATA_ROOT/$db/kpa --color2_mode kpa \
  $ex --out $ck/reinfer_${kind}_$mode.csv
$PYTHON -c "import sys, pandas as pd; sys.path.insert(0, '$ROOT'); from metrics import cv_metrics; m=cv_metrics(pd.read_csv('$ck/reinfer_${kind}_$mode.csv')); print('$db $kind $mode: SRCC %.4f PLCC %.4f RMSE %.4f' % (m['SRCC'], m['PLCC'], m['RMSE']))"
