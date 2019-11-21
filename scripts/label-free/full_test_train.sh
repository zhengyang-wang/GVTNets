#!/bin/bash -x

DATASET=${1:-dna}
GPU_ID=${2:-4}
CSV_DATASET_DIR="datasets/label-free/csvs/${DATASET}/"
RAW_DATASET_DIR="/mnt/dive/shared/yaochen.xie/Label_free_prediction"
NPZ_DATASET_DIR="/mnt/dive/shared/zhengyang/label-free/${DATASET}/datasets/train"
NUM_TRAIN_PAIRS=30
BATCH_SIZE=16
LOSS_TYPE="MSE"
LEARNING_RATE=0.001
NUM_ITERS=100000
SAVE_CHECKPOINTS_ITER=5000
MODEL_NAME=${3:-"gvtnet_label-free"}
MODEL_DIR="/mnt/dive/shared/zhengyang/label-free/${DATASET}/models/${MODEL_NAME}"
TRAIN_PATCH_SIZE_D=32
TRAIN_PATCH_SIZE_H=64
TRAIN_PATCH_SIZE_W=64

# Pre-process the training data and save them into the npz format.
python datasets/label-free/generate_npz_or_tiff.py \
		--csv_dataset_dir ${CSV_DATASET_DIR} \
		--raw_dataset_dir ${RAW_DATASET_DIR} \
		--npz_dataset_dir ${NPZ_DATASET_DIR} \
		--num_train_pairs ${NUM_TRAIN_PAIRS}

# Build the network according to MODEL_NAME
# If network_configures/${MODEL_NAME}.py exists, copy it to network_configure.py
# If not, save the current network_configure.py to network_configures/${MODEL_NAME}.py
if [ -f network_configures/${MODEL_NAME}.py ]
then
	cp network_configures/${MODEL_NAME}.py network_configure.py
else
	cp network_configure.py network_configures/${MODEL_NAME}.py
fi

# Train the GVTNet
python train.py \
		--gpu_id ${GPU_ID} \
		--npz_dataset_dir ${NPZ_DATASET_DIR} \
		--train_patch_size ${TRAIN_PATCH_SIZE_D} ${TRAIN_PATCH_SIZE_H} ${TRAIN_PATCH_SIZE_W} \
		--num_train_pairs ${NUM_TRAIN_PAIRS} \
		--batch_size ${BATCH_SIZE} \
		--loss_type ${LOSS_TYPE} \
		--learning_rate ${LEARNING_RATE} \
		--num_iters ${NUM_ITERS} \
		--save_checkpoints_iter ${SAVE_CHECKPOINTS_ITER} \
		--model_dir ${MODEL_DIR}
