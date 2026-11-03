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
    --save_dir train/dact_v1 \
    --flow_mode traj \
    --traj_degree 4 \
    --traj_iters 6 \
    --traj_reversal fixed_point \
    --fix_warp \
    --fix_voxel \
    --fix_mask_grad \
    --lpips_normalize \
    --fwd_loss_weight 0.1 \
    --w_ph 0.5 --w_end 0.2 --w_cyc 0.02 --w_sm 0.05 --w_cm 0 \
    --traj_warmup_iters 2000 \
    --train_query_count 4 \
    --seed 1234 \
    "$@"
