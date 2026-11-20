# Synthetic G-PCC distortions (MPEG TMC13 v23, CTC-style settings) of the SOURCE dataset's own references, for the
# within-object severity ranking loss. 4 types x 4-5 levels; level 1 = mildest. Light single-core CPU work (nice).
# usage: TMC3=<path to the tmc3 binary> bash bank/encode_gpcc.sh <ref_dir> <out_dir>     (refs: *.ply)
# Writes <out_dir>/vox (10-bit voxelised references, the input of every bank distortion), <out_dir>/ply (reconstructions)
# and <out_dir>/log. Run bank/clean_ply.py on <out_dir>/ply afterwards.
T=${TMC3:?set TMC3 to the tmc3 binary of TMC13 v23}; PY="${PYTHON:-python}"; HERE=$(cd "$(dirname "$0")" && pwd)
R=$1; O=$2; mkdir -p $O/vox $O/ply $O/log
COMMON="--mode=0 --convertPlyColourspace=1 --attribute=color --qpChromaOffset=-2 --bitdepth=8 --mergeDuplicatedPoints=1"
PQS=(0.75 0.5 0.375 0.25 0.125); QPO=(28 34 40 46 51); TRI=(2 3 4 5); QPT=(22 28 34 40)
for f in $R/*.ply; do b=$(basename $f .ply)
  [ -s $O/vox/$b.ply ] || PYTHONNOUSERSITE=1 nice $PY $HERE/prep_refs.py $f $O/vox/$b.ply
  enc() { local n=$1; shift; [ -s $O/ply/$n.ply ] && return
    nice $T $COMMON --uncompressedDataPath=$O/vox/$b.ply --compressedStreamPath=/dev/null --reconstructedDataPath=$O/ply/$n.ply "$@" > $O/log/$n.log 2>&1 \
      || echo "FAILED $n"; }
  for i in 0 1 2 3 4; do L=$((i+1))
    enc ${b}_otr_L$L --positionQuantizationScale=${PQS[$i]} --transformType=0 --qp=${QPO[$i]}
    enc ${b}_otl_L$L --positionQuantizationScale=${PQS[$i]} --transformType=2 --qp=${QPO[$i]}
    enc ${b}_atr_L$L --positionQuantizationScale=1 --transformType=0 --qp=${QPO[$i]}
  done
  for i in 0 1 2 3; do enc ${b}_tri_L$((i+1)) --trisoupNodeSizeLog2=${TRI[$i]} --transformType=0 --qp=${QPT[$i]}; done
  echo "$(date +%H:%M) $b: $(ls $O/ply | grep -c "^${b}_")"
done
