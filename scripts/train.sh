#!/bin/bash -x

DATASET=${1:-dna}
CSV_DATASET_DIR="csvs/${DATASET}/"
RAW_DATASET_DIR="/mnt/dive/shared/yaochen.xie/Label_free_prediction"
SAVE_DIR="/mnt/dive/shared/zhengyang/label-free/${DATASET}"
mkdir ${SAVE_DIR}
TF_DATASET_DIR="${SAVE_DIR}/tfrecord"
NUM_TRAIN_PAIRS=30
BATCH_SIZE=16
BUFFER_SIZE=30
LEARNING_RATE=0.001
NUM_ITERS=100000
SAVE_CHECKPOINTS_ITER=5000
MODEL_NAME=${3:-"model_unet"}
mkdir ${SAVE_DIR}/models
MODEL_DIR="${SAVE_DIR}/models/${MODEL_NAME}"
PATCH_SIZE_D=32
PATCH_SIZE_H=64
PATCH_SIZE_W=64
GPU_ID=${2:-0}

if [ -f network_configures/${MODEL_NAME}.py ]
then
	cp network_configures/${MODEL_NAME}.py network_configure.py
else
	cp network_configure.py network_configures/${MODEL_NAME}.py
fi

# generate tfrecord files
python unet/data/generate_tfrecord.py \
		--raw_dataset_dir ${RAW_DATASET_DIR} \
		--csv_dataset_dir ${CSV_DATASET_DIR} \
		--tf_dataset_dir ${TF_DATASET_DIR} \
		--num_train_pairs ${NUM_TRAIN_PAIRS}

# train
python train.py \
		--tf_dataset_dir ${TF_DATASET_DIR} \
		--num_train_pairs ${NUM_TRAIN_PAIRS} \
		--batch_size ${BATCH_SIZE} \
		--buffer_size ${BUFFER_SIZE} \
		--learning_rate ${LEARNING_RATE} \
		--num_iters ${NUM_ITERS} \
		--save_checkpoints_iter ${SAVE_CHECKPOINTS_ITER} \
		--model_dir ${MODEL_DIR} \
		--patch_size ${PATCH_SIZE_D} ${PATCH_SIZE_H} ${PATCH_SIZE_W} \
		--gpu_id ${GPU_ID}
