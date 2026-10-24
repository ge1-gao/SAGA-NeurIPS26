#!/bin/bash
# SAGA on one Neu3D scene: one model for all 300 frames, proposals from 6 STG warm-ups of 50 frames.
#   bash configs/neu3d.sh <scene_dir> <S|M|L> <λ|none>       e.g. bash configs/neu3d.sh $N3V/flame_steak L 0.004
# λ = none: representation only (no codec). Extra train.py flags can be passed in EXTRA. Finished steps are skipped.
set -eo pipefail
SCENE=$1; TIER=$2; LAM=$3
NAME=$(basename $SCENE)
POINTS=data/points/$NAME
WARM=results/neu3d/$NAME/stg

[ "$(ls $SCENE/png/cam20 2>/dev/null | wc -l)" -ge 300 ] || python -m saga.data.prep_neu3d --scene-dir $SCENE --n-frames 300
python -m saga.data.prep_points --scene-dir $SCENE --out-dir $POINTS  # every 5th frame
for g in 0 50 100 150 200 250; do
    [ -f ${WARM}_g$g/model.pt ] || python -m saga.train --model stg --scene-dir $SCENE --points-dir $POINTS \
        --gop-start $g --out-dir ${WARM}_g$g --max-steps 4000 --refine-stop 3500 --reset-every 100000 --eval-every 100000
done

case $TIER in  # fine / coarse anchors, MLP width
    S) ARGS="--tree.n-fine 45000 --tree.n-coarse 6000 --saga.hidden 64" ;;
    M) ARGS="--tree.n-fine 90000 --tree.n-coarse 12000 --saga.hidden 96" ;;
    L) ARGS="--tree.n-fine 180000 --tree.n-coarse 24000 --saga.hidden 128" ;;
esac
[ "$LAM" = none ] || ARGS="$ARGS --codec.enable --codec.lambda-rate $LAM --saga.gauss-mask"
python -m saga.train --model saga --scene-dir $SCENE --points-dir $POINTS \
    --gop-len 300 --proposal-gop-len 50 --proposal-ckpt "${WARM}_g{g}/model.pt" \
    --tree.max-depth 14 --tree.lambda-tau 6 --tree.temporal-split-std 0.025 --tree.min-half-tau 0.0104 \
    --eval-every 100000 --log-every 1000 $ARGS ${EXTRA:-} --out-dir results/neu3d/$NAME/${TIER}_l$LAM
