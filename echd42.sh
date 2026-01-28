#!/bin/bash
gpu_id=5
#export CUDA_VISIBLE_DEVICES=$gpu_id
random_seeds="42"
few_shots="0 4 8 16 32 64"
num_basis_layers="2"
struct_dims="192"

# target_data = Erbil_Cardiovascular_Health_Dataset 이므로 base_dir도 그에 맞게 표기
echo "=== 시나리오 1 VQVAE: 'Medicaldataset Cardiovascular_Disease_Dataset Heart_disease_statlog heart cardio_SAheart heart_failure_clinical_records' -> Erbil_Cardiovascular_Health_Dataset ==="
for random_seed in $random_seeds; do
    for few_shot in $few_shots; do
        for struct_dim in $struct_dims; do
            base_dir="20260102_Erbil_Cardiovascular_Health_Dataset"
            echo "Running experiment - seed:${random_seed}, few_shot:${few_shot}"
            CUDA_VISIBLE_DEVICES=$gpu_id OMP_NUM_THREADS=10 python main_S.py \
                --random_seed $random_seed \
                --source_data Medicaldataset Cardiovascular_Disease_Dataset Heart_disease_statlog heart cardio_SAheart heart_failure_clinical_records \
                --target_data Erbil_Cardiovascular_Health_Dataset \
                --base_dir $base_dir \
                --few_shot $few_shot \
                --alpha 0.9 \
                --tau 0.5 \
                --soft_tau 0.02 \
                --vq_beta 0.3 \
                --entropy_reg 0.01 \
                --dropout_rate 0.2 \
                --source_lr 0.0001 \
                --source_lr_few 0.00001 \
                --struct_hidden_dim $struct_dim \
                --num_basis_layers $num_basis_layers \
                --run_tag 20251230_134759
        done
    done
done

echo "=== 시나리오 1 VQVAE: 'Medicaldataset Cardiovascular_Disease_Dataset Heart_disease_statlog heart cardio_SAheart heart_failure_clinical_records' -> Erbil_Cardiovascular_Health_Dataset ==="
for random_seed in $random_seeds; do
    for few_shot in $few_shots; do
        for struct_dim in $struct_dims; do
            base_dir="20260102_Erbil_Cardiovascular_Health_Dataset"
            echo "Running experiment - seed:${random_seed}, few_shot:${few_shot}"
            CUDA_VISIBLE_DEVICES=$gpu_id OMP_NUM_THREADS=10 python main_S.py \
                --random_seed $random_seed \
                --source_data Medicaldataset Cardiovascular_Disease_Dataset Heart_disease_statlog heart cardio_SAheart heart_failure_clinical_records \
                --target_data Erbil_Cardiovascular_Health_Dataset \
                --base_dir $base_dir \
                --few_shot $few_shot \
                --alpha 0.9 \
                --tau 0.5 \
                --soft_tau 0.05 \
                --vq_beta 0.4 \
                --entropy_reg 0.01 \
                --dropout_rate 0 \
                --source_lr 0.0001 \
                --source_lr_few 0.0001 \
                --struct_hidden_dim $struct_dim \
                --num_basis_layers $num_basis_layers \
                --run_tag 20251231_115414
        done
    done
done

echo "=== 시나리오 1 VQVAE: 'Medicaldataset Cardiovascular_Disease_Dataset Heart_disease_statlog heart cardio_SAheart heart_failure_clinical_records' -> Erbil_Cardiovascular_Health_Dataset ==="
for random_seed in $random_seeds; do
    for few_shot in $few_shots; do
        for struct_dim in $struct_dims; do
            base_dir="20260102_Erbil_Cardiovascular_Health_Dataset"
            echo "Running experiment - seed:${random_seed}, few_shot:${few_shot}"
            CUDA_VISIBLE_DEVICES=$gpu_id OMP_NUM_THREADS=10 python main_S.py \
                --random_seed $random_seed \
                --source_data Medicaldataset Cardiovascular_Disease_Dataset Heart_disease_statlog heart cardio_SAheart heart_failure_clinical_records \
                --target_data Erbil_Cardiovascular_Health_Dataset \
                --base_dir $base_dir \
                --few_shot $few_shot \
                --alpha 0.9 \
                --tau 0.5 \
                --soft_tau 0.05 \
                --vq_beta 0.4 \
                --entropy_reg 0.01 \
                --dropout_rate 0 \
                --source_lr 0.0001 \
                --source_lr_few 0.0001 \
                --struct_hidden_dim $struct_dim \
                --num_basis_layers $num_basis_layers \
                --run_tag 20251230_133949
        done
    done
done

echo "=== 모든 실험 완료 ==="
