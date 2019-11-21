#!/bin/bash -x

DATASET=${1:-dna}
GPU_ID=${2:-4}
CSV_DATASET_DIR="datasets/label-free/csvs/${DATASET}/"
RAW_DATASET_DIR="/mnt/dive/shared/yaochen.xie/Label_free_prediction"
TIFF_DATASET_DIR="/mnt/dive/shared/zhengyang/label-free/${DATASET}/datasets/test"
NUM_TEST_PAIRS=20
MODEL_NAME=${3:-"gvtnet_label-free"}
CHECKPOINT_NUM=${4:-5000}
RESULT_DIR="/mnt/dive/shared/zhengyang/label-free/${DATASET}/results/${MODEL_NAME}"
MODEL_DIR="/mnt/dive/shared/zhengyang/label-free/${DATASET}/models/${MODEL_NAME}"

# Pre-process the training data and save them into the npz format.
python datasets/label-free/generate_npz_or_tiff.py \
		--csv_dataset_dir ${CSV_DATASET_DIR} \
		--raw_dataset_dir ${RAW_DATASET_DIR} \
		--tiff_dataset_dir ${TIFF_DATASET_DIR} \
		--num_test_pairs ${NUM_TEST_PAIRS}

# Load the network configures according to MODEL_NAME	
cp network_configures/${MODEL_NAME}.py network_configure.py

# Predict using trained GVTNet
python predict.py \
		--gpu_id ${GPU_ID} \
		--tiff_dataset_dir ${TIFF_DATASET_DIR} \
		--num_test_pairs ${NUM_TEST_PAIRS} \
		--result_dir ${RESULT_DIR} \
		--model_dir ${MODEL_DIR} \
		--checkpoint_num ${CHECKPOINT_NUM}
		