#!/bin/bash

gpu_id=0
source_datasets="diabetes"
random_seeds="42 44 46 48 50"
embed_types="carte"
edge_types="mlp"
attn_types="gat_v2"
few_shots="4 8 16 32 64"

for random_seed in $random_seeds; do
    for few_shot in $few_shots; do
        for embed_type in $embed_types; do
            for attn_type in $attn_types; do
                for edge_type in $edge_types; do 
                    for source_dataset in $source_datasets; do
                        # # del_feat 조합 0: Whole feature
                        echo "Running experiment - del_feat: Whole feature"
                        CUDA_VISIBLE_DEVICES=$gpu_id python main_A.py \
                        --random_seed $random_seed \
                        --source_data $source_dataset \
                        --base_dir 'test20250726' \
                        --embed_type $embed_type \
                        --edge_type $edge_type \
                        --attn_type $attn_type \
                        --few_shot $few_shot \
                        --train_epochs 1000 \
                        --model_type "TabularFLM" \

                        # # del_feat 조합 1: Glucose
                        # echo "Running experiment - del_feat: Glucose"
                        # CUDA_VISIBLE_DEVICES=$gpu_id python main_G.py \
                        # --random_seed $random_seed \
                        # --source_data $source_dataset \
                        # --base_dir 'test20250721_hidden_dim_1024' \
                        # --embed_type $embed_type \
                        # --edge_type $edge_type \
                        # --attn_type $attn_type \
                        # --few_shot $few_shot \
                        # --train_epochs 1000 \
                        # --model_type "TabularFLM" \
                        # --del_feat Glucose BloodPressure BMI
                        
                        # # del_feat 조합 2: SkinThickness Age
                        # echo "Running experiment - del_feat: Insulin DiabetesPredigreeFunction"
                        # CUDA_VISIBLE_DEVICES=$gpu_id python main_G.py \
                        # --random_seed $random_seed \
                        # --source_data $source_dataset \
                        # --base_dir 'test20250721_hidden_dim_1024' \
                        # --embed_type $embed_type \
                        # --edge_type $edge_type \
                        # --attn_type $attn_type \
                        # --few_shot $few_shot \
                        # --train_epochs 1000 \
                        # --model_type "TabularFLM" \
                        # --del_feat Glucose BloodPressure BMI SkinThickness
                        
                        # # del_feat 조합 3: 안중요한 변수만 제거 
                        # echo "Running experiment - del_feat: SkinThickness Insulin DiabetesPedigreeFunction"
                        # CUDA_VISIBLE_DEVICES=$gpu_id python main_G.py \
                        # --random_seed $random_seed \
                        # --source_data $source_dataset \
                        # --base_dir 'test20250721_hidden_dim_1024' \
                        # --embed_type $embed_type \
                        # --edge_type $edge_type \
                        # --attn_type $attn_type \
                        # --few_shot $few_shot \
                        # --train_epochs 1000 \
                        # --model_type "TabularFLM" \
                        # --del_feat SkinThickness Insulin DiabetesPedigreeFunction
                        
                        # # del_feat 조합 4: 긴 조합
                        # echo "Running experiment - del_feat: SkinThickness Insulin DiabetesPedigreeFunction Age"
                        # CUDA_VISIBLE_DEVICES=$gpu_id python main_G.py \
                        # --random_seed $random_seed \
                        # --source_data $source_dataset \
                        # --base_dir 'test20250721_hidden_dim_1024' \
                        # --embed_type $embed_type \
                        # --edge_type $edge_type \
                        # --attn_type $attn_type \
                        # --few_shot $few_shot \
                        # --train_epochs 1000 \
                        # --model_type "TabularFLM" \
                        # --del_feat SkinThickness Insulin DiabetesPedigreeFunction Age 

                        #                         # del_feat 조합 3: BloodPressure Insulin
                        # echo "Running experiment - del_feat: SkinThickness Insulin DiabetesPedigreeFunction Age"
                        # CUDA_VISIBLE_DEVICES=$gpu_id python main_G.py \
                        # --random_seed $random_seed \
                        # --source_data $source_dataset \
                        # --base_dir 'test20250721_hidden_dim_1024' \
                        # --embed_type $embed_type \
                        # --edge_type $edge_type \
                        # --attn_type $attn_type \
                        # --few_shot $few_shot \
                        # --train_epochs 1000 \
                        # --model_type "TabularFLM" \
                        # --del_feat SkinThickness Insulin DiabetesPedigreeFunction Age Pregnancies                    
                    done
                done
            done
        done
    done
done