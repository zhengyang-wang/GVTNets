#!/bin/bash -x

NPZ_DATASET_DIR="../CSBDeep/data/Denoising_Tribolium/train_data/"
SAVE_DIR="saved_models/Tribolium/"
BATCH_SIZE=16
LEARNING_RATE=0.0004
LR_DECAY_RATE=0.7
NUM_ITERS=50000
SAVE_CHECKPOINT_ITER=2000
GPU_ID=0
LOSS_TYPE="MAE"

python train.py \
        --npz_dataset_dir ${NPZ_DATASET_DIR} \
        --already_cropped \
        --batch_size ${BATCH_SIZE} \
        --learning_rate ${LEARNING_RATE} \
        --lr_decay_rate ${LR_DECAY_RATE} \
        --num_iters ${NUM_ITERS} \
        --save_checkpoints_iter ${SAVE_CHECKPOINT_ITER} \
        --model_dir ${SAVE_DIR} \
        --gpu_id ${GPU_ID} \
        --loss_type ${LOSS_TYPE} \
        --offset \
        --probalistic 