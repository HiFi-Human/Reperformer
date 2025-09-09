#!/bin/bash

# Example training script to run different training modes

# Set the source data and model paths
SOURCE_DIR="hu_flute1"
MODEL_DIR="Reperformer_hu_flute1_relative"
TRAINING_DEVICE="3"
# Define the different training modes
# You can pass multiple modes as a space-separated list
# Available modes:
# 0 - All training (motion, preprocessing, and UNet)
# 1 - Motion only (train only the motion gaussian)
# 2 - Preprocessing (train the preprocessing part)
# 3 - UNet only (train the UNet model)
# 4 - Evaluate only (only evaluation step)
TRAINING_MODES="4"  # Modify this to run specific modes (e.g., "0", "1", "2", "3", or "4")
# Set the common arguments
FRAME_START=30  # Starting frame index
FRAME_END=990  # Ending frame index
ITERATIONS=30000  # Total number of iterations to train the model
SUBSEQ_ITERS=15000  # Number of iterations for each subsequence
MAP_RESOLUTION=256  # The resolution of the map for training

# Loop through all training modes and run them sequentially
for MODE in $TRAINING_MODES; do
    echo "Starting training with training_mode=${MODE}..."

    # Execute the training script for the current mode
    CUDA_VISIBLE_DEVICES=${TRAINING_DEVICE} python train.py \
        -s ${SOURCE_DIR} \
        -m ${MODEL_DIR} \
        --frame_st ${FRAME_START} \
        --frame_ed ${FRAME_END} \
        --iterations ${ITERATIONS} \
        --subseq_iters ${SUBSEQ_ITERS} \
        --training_mode ${MODE} \
        --parallel_load \
        --map_resolution ${MAP_RESOLUTION} \
        -r 1
done

echo "Training for all modes completed!"