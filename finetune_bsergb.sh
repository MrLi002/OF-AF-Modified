#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Must start from a GOPRO checkpoint trained with the SAME --fix_* flags.
python train_model.py \
    --dataset bsergb \
    --epoch 5 \
    --lr 3e-5 \
    --eta_min 1e-6 \
    --batch_size 1 \
    --num_worker 8 \
    --voxel_bins 128 \
    --nb_of_flow 16 \
    --weights_path train/gopro/epoch19.pth \
    --only_weights \
    --save_dir train/bsergb_ft \
    --fix_query \
    --fix_warp \
    --fix_voxel \
    --lpips_normalize \
    --fwd_loss_weight 0.1 \
    "$@"
