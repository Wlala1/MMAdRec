#!/bin/bash

# Set environment variables for memory management optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export CUDA_LAUNCH_BLOCKING=1

# Show runtime script directory
echo ${RUNTIME_SCRIPT_DIR}

# Enter train workspace
cd ${RUNTIME_SCRIPT_DIR}

pip3 install orjson

# Write your code below
python -u main.py \
    --batch_size 128 \
    --lr 0.0005 \
    --maxlen 101 \
    --num_workers 4 \
    --hidden_units 256 \
    --num_blocks 8 \
    --num_epochs 5 \
    --num_heads 8 \
    --dropout_rate 0.00 \
    --l2_emb 0.0 \
    --device cuda \
    --norm_first \
    --use_hstu \
    --hstu_hidden_dim 32 \
    --hstu_attn_dim 32 \
    --hstu_use_silu \
    --hstu_use_causal_mask \
    --hstu_use_padding_mask \
    --mm_emb_id 82 81 \
    --seed 3407 \
    --use_rope \
    --use_autoencoder \
    --autoencoder_dim 32 \
    --autoencoder_epochs 4 \
    --autoencoder_batch_size 8192 \

echo "Training completed!"
