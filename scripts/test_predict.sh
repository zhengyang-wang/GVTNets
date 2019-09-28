#!/bin/bash -x

DATASET=${1:-dna}
GPU_ID=${2:-4}
TIFF_DATASET_DIR="/mnt/dive/shared/zhengyang/label-free/${DATASET}/datasets/test/sources"
NUM_TEST_PAIRS=20
MODEL_NAME=${3:-"test_model"}
CHECKPOINT_NUM=${4:-5000}
RESULT_DIR="/mnt/dive/shared/zhengyang/label-free/${DATASET}/results/${MODEL_NAME}"
MODEL_DIR="/mnt/dive/shared/zhengyang/label-free/${DATASET}/models/${MODEL_NAME}"

cp network_configures/${MODEL_NAME}.py network_configure.py

# predict
python predict.py \
		--gpu_id ${GPU_ID} \
		--tiff_dataset_dir ${TIFF_DATASET_DIR} \
		--num_test_pairs ${NUM_TEST_PAIRS} \
		--result_dir ${RESULT_DIR} \
		--model_dir ${MODEL_DIR} \
		--checkpoint_num ${CHECKPOINT_NUM}
		