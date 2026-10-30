#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python train_model.py \
    --epoch 20 \
    --lr 1e-4 \
    --eta_min 1e-6 \
    --batch_size 1 \
    --num_worker 8 \
    --dataset gopro \
    --voxel_bins 128 \
    --nb_of_flow 16 \
    --save_dir train/gopro \
    --fix_query \
    --fix_warp \
    --fix_voxel \
    --lpips_normalize \
    --fwd_loss_weight 0.1 \
    --seed 1234 \
    "$@"
