# Shared settings of the evaluation scripts. Paths are configurable through environment variables:
#   DATA_ROOT  rendered datasets: $DATA_ROOT/<DB>/{6view,6view_depth,kpa}        (default: ./datasets)
#   BANK_ROOT  rendered synthetic banks: $BANK_ROOT/<DB>/{6view,6view_depth,kpa} (default: ./banks)
#   RUN_ROOT   checkpoints, logs and predictions: $RUN_ROOT/...                  (default: ./runs)
#   PYTHON     python interpreter (default: python)
# <DB> is one of SJTU, WPC, BASICS, LS_PCQA_part.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CODE=$ROOT/pgs_pcqa
DATA_ROOT=$(realpath -m "${DATA_ROOT:-$ROOT/datasets}")
BANK_ROOT=$(realpath -m "${BANK_ROOT:-$ROOT/banks}")
RUN_ROOT=$(realpath -m "${RUN_ROOT:-$ROOT/runs}")
PYTHON=${PYTHON:-python}
export PYTHONNOUSERSITE=1
declare -A BANK_CSV=([SJTU]=sjtu [WPC]=wpc [BASICS]=basics [LS_PCQA_part]=ls)
cfg() { $PYTHON -c "import json,sys; c=json.load(open('$ROOT/configs/reported_runs.json')); print(eval(sys.argv[1]))" "$1"; }
