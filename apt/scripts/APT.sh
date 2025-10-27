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
# Logic này sẽ kiểm tra $13, nếu nó là đường dẫn thì dùng làm RESUME
# nếu không (rỗng hoặc là cờ -*), nó sẽ cho vào EXTRA_ARGS
RESUME=""
EXTRA_ARGS=""

if [ -n "$13" ] && [[ "$13" != -* ]]; then
    RESUME=$13
    EXTRA_ARGS="${@:14}" # Lấy từ $14 (ví dụ: --no-backbone)
else
    RESUME=""
    EXTRA_ARGS="${@:13}" # Lấy từ $13 (bao gồm $13 nếu nó là cờ)
fi

# 3. Kiểm tra resume (dựa trên biến RESUME)
IS_RESUME=false
if [ -n "$RESUME" ]; then
    IS_RESUME=true
fi

# 4. Tạo đường dẫn DIR (Logic này của bạn đã đúng)
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

# 5. Kiểm tra thư mục (Logic này của bạn đã đúng)
if [ "$IS_RESUME" = false ] && [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"

# 6. Chạy python (đã sửa lỗi)
else
    # Tạo cờ (flag) resume một cách an toàn
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
    ${RESUME_FLAG} ${EXTRA_ARGS} \ # <--- SỬA LỖI: Đặt 2 biến an toàn trên CÙNG 1 DÒNG
    TRAINER.COOP.N_CTX ${NCTX} \
    TRAINER.COOP.CSC ${CSC} \
    TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
    DATASET.NUM_SHOTS ${SHOTS} \
    TRAIN.CHECKPOINT_FREQ 1 
fi