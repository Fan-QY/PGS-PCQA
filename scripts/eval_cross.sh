#!/bin/bash
# Cross-dataset test of one source model on complete target datasets (target split file test_1_xtgt.csv: all samples;
# LS-PCQA without the references shared with SJTU-PCQA). Writes <out_dir>/<target>_centre.csv and
# <out_dir>/<target>_fg4x4.csv (columns fold,name,mos,pred,...); Table 2 uses the fg4x4 files (SRCC over all samples).
# usage: bash scripts/eval_cross.sh <source DB> <checkpoint.pth> <out_dir> <target DB> [<target DB> ...]
set -e; source "$(dirname "$0")/common.sh"
src=$1; ck=$2; O=$3; shift 3; mkdir -p $O
for t in "$@"; do for mode in centre fg4x4; do
  ex=""; [ $mode = fg4x4 ] && ex="--tta 16 --tta_fg_weight --tta_batched"
  $PYTHON -W ignore $ROOT/tools/eval_folds.py --code $CODE --ckpt_pattern "$ck" --folds 1-1 --database $t --score_db $src \
    --csv_suffix _xtgt --color $DATA_ROOT/$t/6view --depth $DATA_ROOT/$t/6view_depth --color2 $DATA_ROOT/$t/kpa --color2_mode kpa \
    $ex --out $O/${t}_$mode.csv > $O/${t}_$mode.log 2>&1
  $PYTHON -c "import pandas as pd; from scipy.stats import spearmanr; d=pd.read_csv('$O/${t}_$mode.csv'); print('$src -> $t ($mode): SRCC %.4f over %d samples' % (spearmanr(d.mos, d.pred)[0], len(d)))"
done; done
