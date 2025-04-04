
# gpu_id=0
# source_datasets="heart diabetes adult"
# few_shots="4 8 16 32 64"
# model_types="TabularFLM"

# for few_shot in $few_shots; do 
#     for source_dataset in $source_datasets; do

#         echo "Running experiment for $source_dataset with $shot few-shot samples in random_seed:42"
#         CUDA_VISIBLE_DEVICES=$gpu_id python main_G2.py \
#         --random_seed 42 \
#         --source_dataset_name $source_dataset \
#         --base_dir 'Experiment_TabularFLM_G4' \
#         --few_shot $few_shot \
#         --train_epochs 1000 \
#         --model_type "TabularFLM"
#     done
# done



#!/bin/bash

gpu_id=5
source_datasets="heart diabetes adult"
few_shots="4 8 16 32 64"
model_types="TabularFLM"

# 200개의 랜덤 시드로 반복
for random_seed in $(seq 42 241); do
    for few_shot in $few_shots; do 
        for source_dataset in $source_datasets; do
            echo "Running experiment for $source_dataset with $few_shot few-shot samples in random_seed:$random_seed"
            
            CUDA_VISIBLE_DEVICES=$gpu_id python main_G.py \
            --random_seed $random_seed \
            --source_dataset_name $source_dataset \
            --base_dir 'Experiment_TabularFLM_G' \
            --use_gmm \
            --few_shot $few_shot \
            --train_epochs 1000 \
            --model_type "TabularFLM"
        done
    done
done
