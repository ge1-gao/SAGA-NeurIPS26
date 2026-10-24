#!/bin/bash
# SAGA on one MPEG MIV CTC perspective sequence (D01, E01, E02): 65 frames, one GOP. As in the CTC objective
# evaluation, every source view is coded and evaluated (--test-cams all). The Neu3D time knobs are rescaled to 65 frames.
#   bash configs/miv.sh <MIV_texture_65f/seq> <S|M|L> <λ|none>     e.g. bash configs/miv.sh $MIV/D01 L 0.004
# λ = none: representation only (no codec). Extra train.py flags can be passed in EXTRA. Finished steps are skipped.
set -eo pipefail
RAW=$1; TIER=$2; LAM=$3
NAME=$(basename $RAW)
SCENE=data/miv/$NAME  # converted to the Neu3D layout
POINTS=data/points_miv/$NAME
WARM=results/miv/$NAME/stg_g0
COMMON="--scene-dir $SCENE --points-dir $POINTS --test-cams all --near-plane 0.2 --downscale 2 --gop-len 65"

[ -d $SCENE/png ] || python -m saga.data.prep_miv --seq-dir $RAW --out-dir $SCENE --n-frames 65
python -m saga.data.prep_points --scene-dir $SCENE --out-dir $POINTS --frames $(seq 0 5 64) --test-cams all
[ -f $WARM/model.pt ] || python -m saga.train --model stg $COMMON --out-dir $WARM \
    --max-steps 4000 --refine-stop 3500 --reset-every 100000 --eval-every 100000

case $TIER in  # fine / coarse anchors, MLP width
    S) ARGS="--tree.n-fine 45000 --tree.n-coarse 6000 --saga.hidden 64" ;;
    M) ARGS="--tree.n-fine 90000 --tree.n-coarse 12000 --saga.hidden 96" ;;
    L) ARGS="--tree.n-fine 180000 --tree.n-coarse 24000 --saga.hidden 128" ;;
esac
[ "$LAM" = none ] || ARGS="$ARGS --codec.enable --codec.lambda-rate $LAM --saga.gauss-mask"
python -m saga.train --model saga $COMMON --proposal-ckpt $WARM/model.pt \
    --tree.max-depth 14 --tree.lambda-tau 1.3 --tree.temporal-split-std 0.115385 --tree.min-half-tau 0.0480769 \
    --eval-every 100000 --log-every 1000 $ARGS ${EXTRA:-} --out-dir results/miv/$NAME/${TIER}_l$LAM
