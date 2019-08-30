#!/bin/bash -x

DATASET=${1:-dna}
CSV_DATASET_DIR="csvs/${DATASET}/"
RAW_DATASET_DIR="/mnt/dive/shared/yaochen.xie/Label_free_prediction"
SAVE_DIR="/mnt/dive/shared/zhengyang/label-free/${DATASET}"
TF_DATASET_DIR="${SAVE_DIR}/tfrecord"
NUM_TEST_PAIRS=20
TEST_PATCH_SIZE_D=0
TEST_PATCH_SIZE_H=0
TEST_PATCH_SIZE_W=0
TEST_STEP_SIZE_D=0
TEST_STEP_SIZE_H=0
TEST_STEP_SIZE_W=0
RESULT_DIR="${SAVE_DIR}/results"
BATCH_SIZE=1
CHECKPOINT_NUM=${4:-100000}
MODEL_NAME=${3:-"model1"}
MODEL_DIR="${SAVE_DIR}/models/${MODEL_NAME}"
GPU_ID=${2:-0}

cp network_configures/${MODEL_NAME}.py network_configure.py

# generate tfrecord files
python unet/data/generate_tfrecord.py \
		--raw_dataset_dir ${RAW_DATASET_DIR} \
		--csv_dataset_dir ${CSV_DATASET_DIR} \
		--tf_dataset_dir ${TF_DATASET_DIR} \
		--num_test_pairs ${NUM_TEST_PAIRS} \
		--test_patch_size ${TEST_PATCH_SIZE_D} ${TEST_PATCH_SIZE_H} ${TEST_PATCH_SIZE_W} \
		--test_step_size ${TEST_STEP_SIZE_D} ${TEST_STEP_SIZE_H} ${TEST_STEP_SIZE_W} \
		--result_dir ${RESULT_DIR}

# predict
python predict.py \
		--tf_dataset_dir ${TF_DATASET_DIR} \
		--num_test_pairs ${NUM_TEST_PAIRS} \
		--test_patch_size ${TEST_PATCH_SIZE_D} ${TEST_PATCH_SIZE_H} ${TEST_PATCH_SIZE_W} \
		--test_step_size ${TEST_STEP_SIZE_D} ${TEST_STEP_SIZE_H} ${TEST_STEP_SIZE_W} \
		--result_dir ${RESULT_DIR} \
		--batch_size ${BATCH_SIZE} \
		--model_dir ${MODEL_DIR} \
		--checkpoint_num ${CHECKPOINT_NUM} \
		--gpu_id ${GPU_ID}
