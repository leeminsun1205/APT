#!/bin/bash

# Navigate to apt directory to ensure imports work as expected by train.py
cd apt

# Set data directory
DATA=../data

# Create data directory if it doesn't exist
mkdir -p $DATA

echo "Verifying CIFAR-10..."
python train.py \
    --root ${DATA} \
    --trainer APT \
    --dataset-config-file configs/datasets/cifar10.yaml \
    --config-file configs/trainers/APT/rn50.yaml \
    --output-dir ../output/test_cifar10 \
    --no-train \
    DATASET.NUM_SHOTS 1

echo "Verifying CIFAR-100..."
python train.py \
    --root ${DATA} \
    --trainer APT \
    --dataset-config-file configs/datasets/cifar100.yaml \
    --config-file configs/trainers/APT/rn50.yaml \
    --output-dir ../output/test_cifar100 \
    --no-train \
    DATASET.NUM_SHOTS 1

echo "Verification complete."
