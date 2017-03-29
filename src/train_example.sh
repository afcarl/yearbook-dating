EXP='TEST'
GENDER='men'
#WEIGHTS='../output/aligned-men-softmax-frozen/snapshots/train_iter_1000.caffemodel'
WEIGHTS='../models/VGG_ILSVRC_16_layers.caffemodel'
GPU=3
LR=0.001
BATCH=32
stdbuf -oL python solve.py $EXP $GENDER $WEIGHTS $GPU $LR $BATCH 2>&1 | tee ../output/$EXP.log
