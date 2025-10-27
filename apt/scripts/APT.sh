#!/bin/bash

# custom config
DATA=kaggle/input/
TRAINER=APT

DATASET=$1
CFG=$2  # config file
CTP=$3  # class token position (end or middle)
NCTX=$4  # number of context tokens
SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
CSC=$6  # class-specific context (False or True)
EPS=$7  # epsilon for AT
ALPHA=$8  # alpha or step size for AT
STEPS=$9  # number of steps for AT
SEED=${10}
ATP=${11}
PALPHA=${12}
IS_RESUME=false
for arg in "$@"; do
    if [[ "$arg" == "--resume" ]]; then
        IS_RESUME=true
        break
    fi
done
RESUME=${13}


#pertubed
if [ ${ATP} == 'perturbed' ]
then
    DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/eps${EPS}_alpha${ALPHA}_step${STEPS}_${ATP}_${PALPHA}/seed${SEED}

#constant
elif [ ${ATP} == 'constant' ]
then
    DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/eps${EPS}_alpha${ALPHA}_step${STEPS}_${ATP}/seed${SEED}

#onfly
else
    DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/eps${EPS}_alpha${ALPHA}_step${STEPS}/seed${SEED}
fi

#results existed
if [ "$IS_RESUME" = false ] && [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"

#train is implemented
else
    python train.py \
    --root ${DATA} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    TRAIN.CHECKPOINT_FREQ 1 \
    "$@"  # <--- CHỈ GIỮ LẠI DÒNG NÀY
fi
