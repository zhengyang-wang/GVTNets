#!/bin/bash -x

# Provide the full path to the folder that stores the data extracted from
# Denoising_Tribolium.tar.gz downloaded from https://publications.mpi-cbg.de/publications-sites/7207/
# TEST_DATASET_DIR should refer to test_data.
TEST_DATASET_DIR="/mnt/dive/shared/yaochen.xie/CSBDeep/Denoising_Tribolium/test_data/"
# Provide the condition name.
CONDITION=${1:-condition_1}
# Provide the GPU id. Use -1 for CPU only.
GPU_ID=${2:-4}
# Provide the name of your model. Use 'gvtnet_care_pretrained' for provided pretrained model.
MODEL_NAME=${3:-"gvtnet_care"}
# Provide the number of saved checkpoint. Use 'pretrained' for provided pretrained model.
CHECKPOINT_NUM=${4:-25000}
# Provide the path to the main folder that saves transformed datasets, checkpoints and results.
SAVE_DIR="save_dir/care_denoising/tribolium"

TIFF_DATASET_DIR="${TEST_DATASET_DIR}/${CONDITION}/"
RESULT_DIR="${SAVE_DIR}/results/${MODEL_NAME}/${CONDITION}/"
MODEL_DIR="${SAVE_DIR}/models/${MODEL_NAME}"
NUM_TEST_PAIRS=6
TEST_PATCH_SIZE_D=128
TEST_PATCH_SIZE_H=256
TEST_PATCH_SIZE_W=256
TEST_BATCH_SIZE=1

# Pre-process the testing data and save them into the required format.
python datasets/care/format_data.py \
        --raw_test_dataset_dir ${TEST_DATASET_DIR}

# Load the network configures according to MODEL_NAME   
cp network_configures/${MODEL_NAME}.py network_configure.py

# Predict using trained GVTNet
python predict.py \
        --gpu_id ${GPU_ID} \
        --tiff_dataset_dir ${TIFF_DATASET_DIR} \
        --num_test_pairs ${NUM_TEST_PAIRS} \
        --result_dir ${RESULT_DIR} \
        --model_dir ${MODEL_DIR} \
        --checkpoint_num ${CHECKPOINT_NUM} \
        --cropped_prediction \
        --predict_patch_size ${TEST_PATCH_SIZE_D} ${TEST_PATCH_SIZE_H} ${TEST_PATCH_SIZE_W} \
        --test_batch_size ${TEST_BATCH_SIZE} \
        --offset \
        --CARE_normalize
        