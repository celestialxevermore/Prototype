#!/bin/bash
gpu_id=4
#export CUDA_VISIBLE_DEVICES=$gpu_id
random_seeds="42"
few_shots="0 4 8 16 32 64"
num_basis_layers="2"
struct_dims="192"

echo "=== 시나리오 1 VQVAE: 'Medicaldataset Cardiovascular_Disease_Dataset Heart_disease_statlog Erbil_Cardiovascular_Health_Dataset cardio_SAheart heart_failure_clinical_records' -> heart ==="
for random_seed in $random_seeds; do
    for few_shot in $few_shots; do
        for struct_dim in $struct_dims; do
            base_dir="20260102_cardio_SAheart"
            echo "Running experiment - seed:${random_seed}, few_shot:${few_shot}"
            CUDA_VISIBLE_DEVICES=$gpu_id OMP_NUM_THREADS=10 python main_S.py \
                --random_seed $random_seed \
                --source_data Medicaldataset Cardiovascular_Disease_Dataset Heart_disease_statlog Erbil_Cardiovascular_Health_Dataset heart_failure_clinical_records heart \
                --target_data cardio_SAheart \
                --base_dir $base_dir \
                --few_shot $few_shot \
                --alpha 0.9 \
                --tau 1 \
                --soft_tau 0.02 \
                --vq_beta 0.1 \
                --entropy_reg 0.01 \
                --dropout_rate 0.3 \
                --source_lr 0.0001 \
                --source_lr_few 0.0001 \
                --struct_hidden_dim $struct_dim \
                --num_basis_layers $num_basis_layers \
                --run_tag 20251223_053405
        done
    done
done

for random_seed in $random_seeds; do
    for few_shot in $few_shots; do
        for struct_dim in $struct_dims; do
            base_dir="20260102_cardio_SAheart"
            echo "Running experiment - seed:${random_seed}, few_shot:${few_shot}"
            CUDA_VISIBLE_DEVICES=$gpu_id OMP_NUM_THREADS=10 python main_S.py \
                --random_seed $random_seed \
                --source_data Medicaldataset Cardiovascular_Disease_Dataset Heart_disease_statlog Erbil_Cardiovascular_Health_Dataset heart_failure_clinical_records heart \
                --target_data cardio_SAheart \
                --base_dir $base_dir \
                --few_shot $few_shot \
                --alpha 0.8 \
                --tau 0.4 \
                --soft_tau 0.05 \
                --vq_beta 0.2 \
                --entropy_reg 0.01 \
                --dropout_rate 0.3 \
                --source_lr 0.0001 \
                --source_lr_few 0.00001 \
                --struct_hidden_dim $struct_dim \
                --num_basis_layers $num_basis_layers \
                --run_tag 20251227_061623
        done
    done
done

for random_seed in $random_seeds; do
    for few_shot in $few_shots; do
        for struct_dim in $struct_dims; do
            base_dir="20260102_cardio_SAheart"
            echo "Running experiment - seed:${random_seed}, few_shot:${few_shot}"
            CUDA_VISIBLE_DEVICES=$gpu_id OMP_NUM_THREADS=10 python main_S.py \
                --random_seed $random_seed \
                --source_data Medicaldataset Cardiovascular_Disease_Dataset Heart_disease_statlog Erbil_Cardiovascular_Health_Dataset heart_failure_clinical_records heart \
                --target_data cardio_SAheart \
                --base_dir $base_dir \
                --few_shot $few_shot \
                --alpha 0.8 \
                --tau 0.3 \
                --soft_tau 0.05 \
                --vq_beta 0.3 \
                --entropy_reg 0.01 \
                --dropout_rate 0.3 \
                --source_lr 0.0001 \
                --source_lr_few 0.00001 \
                --struct_hidden_dim $struct_dim \
                --num_basis_layers $num_basis_layers \
                --run_tag 20251225_052243
        done
    done
done

echo "=== 모든 실험 완료 ==="
