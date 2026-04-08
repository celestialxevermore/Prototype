#!/bin/bash
# ================================================================================
# Exp A — Single-Source Baseline (alpha 0.1 ~ 0.5)
# 2026.04.02
# ================================================================================
gpu_id=5
random_seeds="42 44 46 48 50"
sampling_alphas="0.1 0.2 0.3 0.4 0.5"
base_dir="expA_20260402"

sources="Medicaldataset Cardiovascular_Disease_Dataset Heart_disease_statlog Erbil_Cardiovascular_Health_Dataset cardio_SAheart heart_failure_clinical_records heart"

echo "=== [Exp A] Single-Source Baseline (alpha 0.1~0.5) ==="

for src in $sources; do
    for sampling_alpha in $sampling_alphas; do
        for seed in $random_seeds; do
            echo "[Single] source=${src}, alpha=${sampling_alpha}, seed=${seed}"
            CUDA_VISIBLE_DEVICES=$gpu_id OMP_NUM_THREADS=10 python main_E.py \
                --exp_mode single_source \
                --eval_source $src \
                --sampling_alpha $sampling_alpha \
                --source_data $src \
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
done

echo "=== [Exp A] Single-Source Baseline (alpha 0.1~0.5) 완료 ==="
