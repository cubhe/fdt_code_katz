#!/bin/bash

# Training beads with fs=128, dz=0.2, layers=60, center_mask=200, 2 GPUs, 3k iters
PYTHON=/global/scratch/users/cubhe/conda_envs/lim/bin/python
cd /global/home/users/cubhe/FDT

# Run 1: b2b=0.5
CUDA_VISIBLE_DEVICES=1,2 $PYTHON -u run_nerf.py \
    --fs 128 \
    --dz 0.2 \
    --layers 60 \
    --b2b 0.5 \
    --N_iters 3000 \
    --num_gpu 2 \
    --center_mask_enable 1 \
    --center_mask_size 200 \
    --object_category_ori auto \
    2>&1 | tee log_beads_fs128_b2b05.txt

# Run 2: b2b=0
CUDA_VISIBLE_DEVICES=1,2 $PYTHON -u run_nerf.py \
    --fs 128 \
    --dz 0.2 \
    --layers 60 \
    --b2b 0 \
    --N_iters 3000 \
    --num_gpu 2 \
    --center_mask_enable 1 \
    --center_mask_size 200 \
    --object_category_ori auto \
    2>&1 | tee log_beads_fs128_b2b0.txt
