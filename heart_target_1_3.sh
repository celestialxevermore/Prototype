#!/bin/bash
gpu_id=4
#export CUDA_VISIBLE_DEVICES=$gpu_id
random_seeds="46"
few_shots="0 4 8 16 32 64"
num_basis_layers="2"
num_basis_heads="6"

echo "=== 시나리오 1: Case1 - target: heart ==="
for random_seed in $random_seeds; do
    for few_shot in $few_shots; do
        base_dir="test20251220_LOCAL_GAT_ONLY_Case1_head_6"
        echo "Running experiment - seed:${random_seed}, few_shot:${few_shot}"
        CUDA_VISIBLE_DEVICES=$gpu_id OMP_NUM_THREADS=10 python main_S.py \
            --random_seed $random_seed \
            --source_data Medicaldataset Cardiovascular_Disease_Dataset Heart_disease_statlog Erbil_Cardiovascular_Health_Dataset cardio_SAheart heart_failure_clinical_records \
            --target_data heart \
            --base_dir $base_dir \
            --few_shot $few_shot \
            --num_basis_heads $num_basis_heads \
            --num_basis_layers $num_basis_layers \
            --des LOCAL_GAT_ONLY_20251220_Case1_
    done
done

echo "=== 시나리오 2: Case2 - target: Medicaldataset ==="
for random_seed in $random_seeds; do
    for few_shot in $few_shots; do
        base_dir="test20251220_LOCAL_GAT_ONLY_Case1_head_6"
        echo "Running experiment - seed:${random_seed}, few_shot:${few_shot}"
        CUDA_VISIBLE_DEVICES=$gpu_id OMP_NUM_THREADS=10 python main_S.py \
            --random_seed $random_seed \
            --source_data heart Cardiovascular_Disease_Dataset Heart_disease_statlog Erbil_Cardiovascular_Health_Dataset cardio_SAheart heart_failure_clinical_records \
            --target_data Medicaldataset \
            --base_dir $base_dir \
            --few_shot $few_shot \
            --num_basis_heads $num_basis_heads \
            --num_basis_layers $num_basis_layers \
            --des LOCAL_GAT_ONLY_20251220_Case2_
    done
done

echo "=== 시나리오 3: Case3 - target: Cardiovascular_Disease_Dataset ==="
for random_seed in $random_seeds; do
    for few_shot in $few_shots; do
        base_dir="test20251220_LOCAL_GAT_ONLY_Case1_head_6"
        echo "Running experiment - seed:${random_seed}, few_shot:${few_shot}"
        CUDA_VISIBLE_DEVICES=$gpu_id OMP_NUM_THREADS=10 python main_S.py \
            --random_seed $random_seed \
            --source_data heart Medicaldataset Erbil_Cardiovascular_Health_Dataset Heart_disease_statlog cardio_SAheart heart_failure_clinical_records \
            --target_data Cardiovascular_Disease_Dataset \
            --base_dir $base_dir \
            --few_shot $few_shot \
            --num_basis_heads $num_basis_heads \
            --num_basis_layers $num_basis_layers \
            --des LOCAL_GAT_ONLY_20251220_Case3_
    done
done

echo "=== 시나리오 4: Case4 - target: Erbil_Cardiovascular_Health_Dataset ==="
for random_seed in $random_seeds; do
    for few_shot in $few_shots; do
        base_dir="test20251220_LOCAL_GAT_ONLY_Case1_head_6"
        echo "Running experiment - seed:${random_seed}, few_shot:${few_shot}"
        CUDA_VISIBLE_DEVICES=$gpu_id OMP_NUM_THREADS=10 python main_S.py \
            --random_seed $random_seed \
            --source_data heart Medicaldataset Cardiovascular_Disease_Dataset Heart_disease_statlog cardio_SAheart heart_failure_clinical_records \
            --target_data Erbil_Cardiovascular_Health_Dataset \
            --base_dir $base_dir \
            --few_shot $few_shot \
            --num_basis_heads $num_basis_heads \
            --num_basis_layers $num_basis_layers \
            --des LOCAL_GAT_ONLY_20251220_Case4_
    done
done

echo "=== 시나리오 5: Case5 - target: cardio_SAheart ==="
for random_seed in $random_seeds; do
    for few_shot in $few_shots; do
        base_dir="test20251220_LOCAL_GAT_ONLY_Case1_head_6"
        echo "Running experiment - seed:${random_seed}, few_shot:${few_shot}"
        CUDA_VISIBLE_DEVICES=$gpu_id OMP_NUM_THREADS=10 python main_S.py \
            --random_seed $random_seed \
            --source_data heart Medicaldataset Cardiovascular_Disease_Dataset Heart_disease_statlog Erbil_Cardiovascular_Health_Dataset heart_failure_clinical_records \
            --target_data cardio_SAheart \
            --base_dir $base_dir \
            --few_shot $few_shot \
            --num_basis_heads $num_basis_heads \
            --num_basis_layers $num_basis_layers \
            --des LOCAL_GAT_ONLY_20251220_Case5_
    done
done

echo "=== 시나리오 6: Case6 - target: heart_failure_clinical_records ==="
for random_seed in $random_seeds; do
    for few_shot in $few_shots; do
        base_dir="test20251220_LOCAL_GAT_ONLY_Case1_head_6"
        echo "Running experiment - seed:${random_seed}, few_shot:${few_shot}"
        CUDA_VISIBLE_DEVICES=$gpu_id OMP_NUM_THREADS=10 python main_S.py \
            --random_seed $random_seed \
            --source_data heart Medicaldataset Cardiovascular_Disease_Dataset Heart_disease_statlog Erbil_Cardiovascular_Health_Dataset cardio_SAheart \
            --target_data heart_failure_clinical_records \
            --base_dir $base_dir \
            --few_shot $few_shot \
            --num_basis_heads $num_basis_heads \
            --num_basis_layers $num_basis_layers \
            --des LOCAL_GAT_ONLY_20251220_Case6_
    done
done

echo "=== 시나리오 7: Case6 - target: Heart_disease_statlog ==="
for random_seed in $random_seeds; do
    for few_shot in $few_shots; do
        base_dir="test20251220_LOCAL_GAT_ONLY_Case1_head_6"
        echo "Running experiment - seed:${random_seed}, few_shot:${few_shot}"
        CUDA_VISIBLE_DEVICES=$gpu_id OMP_NUM_THREADS=10 python main_S.py \
            --random_seed $random_seed \
            --source_data heart Medicaldataset Cardiovascular_Disease_Dataset Erbil_Cardiovascular_Health_Dataset heart_failure_clinical_records cardio_SAheart \
            --target_data Heart_disease_statlog \
            --base_dir $base_dir \
            --few_shot $few_shot \
            --num_basis_heads $num_basis_heads \
            --num_basis_layers $num_basis_layers \
            --des LOCAL_GAT_ONLY_20251220_Case7_
    done
done

echo "=== 모든 실험 완료 ==="