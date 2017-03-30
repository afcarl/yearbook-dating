EXP='TEST'
GENDER='men'
WEIGHTS='../models/VGG_ILSVRC_16_layers.caffemodel'
GPU=0
LR=0.001
BATCH=64
FROZEN=1
stdbuf -oL python src/solve.py $EXP $GENDER $WEIGHTS $GPU $LR $BATCH $FROZEN 2>&1 | tee output/$EXP.log
