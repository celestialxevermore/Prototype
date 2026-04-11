#!/bin/bash
# ================================================================================
# Exp A — Single 누락 (heart, alpha 0.8, seed 66~72)
# 2026.04.10
# ================================================================================
gpu_id=4
random_seeds="66 68 70 72"
base_dir="expA_EEE_20260407"

echo "=== [Single 누락] heart alpha=0.8 seed 66~72 ==="

for seed in $random_seeds; do
    echo "[Single] source=heart, alpha=0.8, seed=${seed}"
    CUDA_VISIBLE_DEVICES=$gpu_id OMP_NUM_THREADS=10 python main_EEE.py \
        --exp_mode single_source \
        --eval_source heart \
        --sampling_alpha 0.8 \
        --source_data heart \
        --target_data heart \
        --random_seed $seed \
        --base_dir $base_dir \
        --alpha 0.7 \
        --fgw_alpha 1 \
        --tau 0.2 \
        --soft_tau 0.005 \
        --vq_beta 0.2 \
        --entropy_reg 0.01 \
        --dropout_rate 0.3 \
        --source_lr 0.001 \
        --source_lr_few 0.0001 \
        --struct_hidden_dim 192 \
        --num_basis_layers 2
done

echo "=== 완료 ==="
