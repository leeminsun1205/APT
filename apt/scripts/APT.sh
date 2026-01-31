#!/bin/bash

# custom config
DATA=kaggle/input/
TRAINER=APT

# 1. Đọc 12 tham số vị trí CỐ ĐỊNH
DATASET=$1
CFG=$2
CTP=$3
NCTX=$4
SHOTS=$5
CSC=$6
EPS=$7
ALPHA=$8
STEPS=$9
SEED=${10}
ATP=${11}
PALPHA=${12}
RUN_MODE=${13} # $13 là "resume" hoặc "train"

EPOCHS=""
# Check if $14 is an integer (for EPOCHS)
if [[ "${14}" =~ ^[0-9]+$ ]]; then
    EPOCHS=${14}
    EXTRA_ARGS="${@:15}"
else
    # If $14 is not an integer (e.g. empty or starts with --), assume it's part of EXTRA_ARGS
    EXTRA_ARGS="${@:14}"
fi

# Prepare EPOCH_FLAG
EPOCH_FLAG=""
if [ ! -z "$EPOCHS" ]; then
    EPOCH_FLAG="--max-epoch ${EPOCHS}"
fi

# 3. Tạo đường dẫn DIR (Một lần duy nhất)
if [ ${ATP} == 'perturbed' ]
then
    DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/eps${EPS}_alpha${ALPHA}_step${STEPS}_${ATP}_${PALPHA}/seed${SEED}
elif [ ${ATP} == 'constant' ]
then
    DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/eps${EPS}_alpha${ALPHA}_step${STEPS}_${ATP}/seed${SEED}
else
    DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/eps${EPS}_alpha${ALPHA}_step${STEPS}/seed${SEED}
fi

# 4. Xử lý resume
RESUME_FLAG=""
IS_RESUME=false
if [ "$RUN_MODE" == "resume" ]; then
    IS_RESUME=true
    RESUME_FLAG="--resume ${DIR}"
fi

# 5. Kiểm tra thư mục
if [ "$IS_RESUME" = false ] && [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"

# 6. Chạy python
else
    python train.py \
    --root ${DATA} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --eps ${EPS} \
    --alpha ${ALPHA} \
    --steps ${STEPS} \
    --adv-prompt ${ATP} \
    --prompt-alpha ${PALPHA} \
    ${EPOCH_FLAG} \
    ${RESUME_FLAG} \
    ${EXTRA_ARGS} \
    TRAINER.COOP.N_CTX ${NCTX} \
    TRAINER.COOP.CSC ${CSC} \
    TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
    DATASET.NUM_SHOTS ${SHOTS} \
    TRAIN.CHECKPOINT_FREQ 1 
fi