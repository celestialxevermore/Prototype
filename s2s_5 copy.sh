


# gpu_id=5
# source_datasets="heart diabetes adult"
# few_shots="4 8 16 32 64"
# model_types="TabularFLM"

# for few_shot in $few_shots; do 
#     for source_dataset in $source_datasets; do

#         echo "Running experiment for $source_dataset with $shot few-shot samples in random_seed:5678"
#         CUDA_VISIBLE_DEVICES=$gpu_id python main_G2.py \
#         --random_seed 5678 \
#         --source_dataset_name $source_dataset \
#         --base_dir 'Experiment_TabularFLM_G4' \
#         --few_shot $few_shot \
#         --train_epochs 1000 \
#         --model_type "TabularFLM"
#     done
# done

#!/bin/bash

gpu_id=5
source_datasets="adult"
few_shots="4 8 16 32 64"
model_types="TabularFLM"

# 4000부터 4199까지 200개의 랜덤 시드로 반복
for random_seed in $(seq 4000 4199); do
    for few_shot in $few_shots; do 
        for source_dataset in $source_datasets; do
            echo "Running experiment for $source_dataset with $few_shot few-shot samples in random_seed:$random_seed"
            
            CUDA_VISIBLE_DEVICES=$gpu_id python main_G3.py \
            --random_seed $random_seed \
            --source_dataset_name $source_dataset \
            --base_dir 'Experiment_TabularFLM_G5' \
            --use_gmm \
            --few_shot $few_shot \
            --train_epochs 1000 \
            --model_type "TabularFLM"
        done
    done
done