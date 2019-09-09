#!/bin/bash -x

DATASET=${1:-dna}
GPU_ID=${2:-4}
NPZ_DATASET_DIR="/mnt/dive/shared/zhengyang/label-free/${DATASET}/datasets/train"
NUM_TRAIN_PAIRS=30
BATCH_SIZE=16
LOSS_TYPE="MSE"
LEARNING_RATE=0.001
NUM_ITERS=100000
SAVE_CHECKPOINTS_ITER=5000
MODEL_NAME=${3:-"test_model"}
MODEL_DIR="/mnt/dive/shared/zhengyang/label-free/${DATASET}/models/${MODEL_NAME}"
PATCH_SIZE_D=32
PATCH_SIZE_H=64
PATCH_SIZE_W=64


if [ -f network_configures/${MODEL_NAME}.py ]
then
	cp network_configures/${MODEL_NAME}.py network_configure.py
else
	cp network_configure.py network_configures/${MODEL_NAME}.py
fi

# train
python train.py \
		--gpu_id ${GPU_ID} \
		--npz_dataset_dir ${NPZ_DATASET_DIR} \
		--train_patch_size ${PATCH_SIZE_D} ${PATCH_SIZE_H} ${PATCH_SIZE_W} \
		--num_train_pairs ${NUM_TRAIN_PAIRS} \
		--batch_size ${BATCH_SIZE} \
		--loss_type ${LOSS_TYPE} \
		--learning_rate ${LEARNING_RATE} \
		--num_iters ${NUM_ITERS} \
		--save_checkpoints_iter ${SAVE_CHECKPOINTS_ITER} \
		--model_dir ${MODEL_DIR}
