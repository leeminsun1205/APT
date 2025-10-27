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

# 2. Xử lý các tham số TÙY CHỌN ($13 trở đi)
RESUME=""
EXTRA_ARGS=""

if [ -n "$13" ] && [[ "$13" != -* ]]; then
    RESUME=$13
    EXTRA_ARGS="${@:14}"
else
    RESUME=""
    EXTRA_ARGS="${@:13}"
fi

# 3. Kiểm tra resume (dựa trên biến RESUME)
IS_RESUME=false
if [ -n "$RESUME" ]; then
    IS_RESUME=true
fi

# 4. Tạo đường dẫn DIR
if [ ${ATP} == 'perturbed' ]
then
    DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/eps${EPS}_alpha${ALPHA}_step${STEPS}_${ATP}_${PALPHA}/seed${SEED}
elif [ ${ATP} == 'constant' ]
then
    DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/eps${EPS}_alpha${ALPHA}_step${STEPS}_${ATP}/seed${SEED}
else
    DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/eps${EPS}_alpha${ALPHA}_step${STEPS}/seed${SEED}
fi

# 5. Kiểm tra thư mục
if [ "$IS_RESUME" = false ] && [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"

# 6. Chạy python
else
    RESUME_FLAG=""
    if [ "$IS_RESUME" = true ]; then
        RESUME_FLAG="--resume ${RESUME}"
    fi
    
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
    ${RESUME_FLAG} ${EXTRA_ARGS} \
    TRAINER.COOP.N_CTX ${NCTX} \
    TRAINER.COOP.CSC ${CSC} \
    TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
    DATASET.NUM_SHOTS ${SHOTS} \
    TRAIN.CHECKPOINT_FREQ 1 
fi