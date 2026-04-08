#!/bin/bash
# ================================================================================
# Exp A — main_EEE.py 누락분 (alpha=0.8 seed 48/50, alpha=1.0 seed 44)
# 2026.04.08
# ================================================================================
gpu_id=3
base_dir="expA_EEE_20260407"

ALL_SOURCES="Medicaldataset Cardiovascular_Disease_Dataset Heart_disease_statlog Erbil_Cardiovascular_Health_Dataset cardio_SAheart heart_failure_clinical_records heart"

run_one() {
    local alpha=$1
    local seed=$2
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
}

echo "=== [EEE 누락분] alpha=0.8 seed 48,50, alpha=1.0 seed 44 ==="
run_one 0.8 48
run_one 0.8 50
run_one 1.0 44

echo "=== 완료 ==="
