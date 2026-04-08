#!/bin/bash
# ================================================================================
# Exp A — main_EEE.py 누락분 (alpha=0.7, seed 46/48/50)
# 2026.04.08
# ================================================================================
gpu_id=1
base_dir="expA_EEE_20260407"
alphas="0.7"
random_seeds="46 48 50"

ALL_SOURCES="Medicaldataset Cardiovascular_Disease_Dataset Heart_disease_statlog Erbil_Cardiovascular_Health_Dataset cardio_SAheart heart_failure_clinical_records heart"

echo "=== [EEE 누락분] alpha=0.7 seed 46,48,50 ==="
for alpha in $alphas; do
    for seed in $random_seeds; do
        echo "[Case2] alpha=${alpha}, seed=${seed}"
        CUDA_VISIBLE_DEVICES=$gpu_id OMP_NUM_THREADS=10 python main_EEE.py \
            --exp_mode case2 \
            --sampling_alpha $alpha \
            --source_data $ALL_SOURCES \
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
done

echo "=== 완료 ==="
