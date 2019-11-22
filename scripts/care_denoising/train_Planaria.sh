#!/bin/bash -x

# Provide the full path to the folder that stores the data extracted from
# Denoising_Planaria.tar.gz downloaded from https://publications.mpi-cbg.de/publications-sites/7207/
# NPZ_DATASET_DIR should refer to train_data.
NPZ_DATASET_DIR="/mnt/dive/shared/yaochen.xie/CSBDeep/Denoising_Planaria/train_data"
# Provide the GPU id. Use -1 for CPU only.
GPU_ID=${1:-4}
# Provide the name of your model.
MODEL_NAME=${2:-"gvtnet_care"}
# Provide the path to the main folder that saves transformed datasets, checkpoints and results.
SAVE_DIR="save_dir/care_denoising/planaria"

MODEL_DIR="${SAVE_DIR}/models/${MODEL_NAME}"
BATCH_SIZE=16
LOSS_TYPE="MAE"
LEARNING_RATE=0.0004
LR_DECAY_RATE=0.7
NUM_ITERS=55000
SAVE_CHECKPOINTS_ITER=5000

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
        --already_cropped \
        --batch_size ${BATCH_SIZE} \
        --loss_type ${LOSS_TYPE} \
        --learning_rate ${LEARNING_RATE} \
        --lr_decay_rate ${LR_DECAY_RATE} \
        --num_iters ${NUM_ITERS} \
        --save_checkpoints_iter ${SAVE_CHECKPOINTS_ITER} \
        --model_dir ${MODEL_DIR} \
        --offset \
        --probalistic
