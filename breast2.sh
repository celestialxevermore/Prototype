#!/bin/bash

gpu_id=4
source_datasets="breast"
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
                       
                       # del_feat 조합 1: Sex
                       echo "Running experiment - del_feat: ST_Slope Oldpeak Cholesterol"
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
                       #--del_feat ST_Slope Oldpeak Cholesterol
                       
                    # #    # del_feat 조합 2: ChestPainType
                    #    echo "Running experiment - del_feat: ST_Slope Oldpeak Cholesterol FastingBS"
                    #    CUDA_VISIBLE_DEVICES=$gpu_id python main_G.py \
                    #    --random_seed $random_seed \
                    #    --source_data $source_dataset \
                    #    --base_dir 'test20250721_hidden_dim_1024' \
                    #    --embed_type $embed_type \
                    #    --edge_type $edge_type \
                    #    --attn_type $attn_type \
                    #    --few_shot $few_shot \
                    #    --train_epochs 1000 \
                    #    --model_type "TabularFLM" \
                    #    --del_feat ST_Slope Oldpeak Cholesterol FastingBS
                       
                    #    # del_feat 조합 3: RestingECG
                    #    echo "Running experiment - del_feat: Sex Age MaxHR"
                    #    CUDA_VISIBLE_DEVICES=$gpu_id python main_G.py \
                    #    --random_seed $random_seed \
                    #    --source_data $source_dataset \
                    #    --base_dir 'test20250721_hidden_dim_1024' \
                    #    --embed_type $embed_type \
                    #    --edge_type $edge_type \
                    #    --attn_type $attn_type \
                    #    --few_shot $few_shot \
                    #    --train_epochs 1000 \
                    #    --model_type "TabularFLM" \
                    #    --del_feat Sex Age MaxHR
                       
                    #    # del_feat 조합 4: ExerciseAngina
                    #    echo "Running experiment - del_feat: ST_Slope Oldpeak Cholesterol Age "
                    #    CUDA_VISIBLE_DEVICES=$gpu_id python main_G.py \
                    #    --random_seed $random_seed \
                    #    --source_data $source_dataset \
                    #    --base_dir 'test20250721_hidden_dim_1024' \
                    #    --embed_type $embed_type \
                    #    --edge_type $edge_type \
                    #    --attn_type $attn_type \
                    #    --few_shot $few_shot \
                    #    --train_epochs 1000 \
                    #    --model_type "TabularFLM" \
                    #    --del_feat ST_Slope Oldpeak Cholesterol Age 
                       
                    #    # del_feat 조합 5: ST_Slope
                    #    echo "Running experiment - del_feat: Sex ChestPainType RestingECG ExerciseAngina Age RestingBP FastingBS MaxHR"
                    #    CUDA_VISIBLE_DEVICES=$gpu_id python main_G.py \
                    #    --random_seed $random_seed \
                    #    --source_data $source_dataset \
                    #    --base_dir 'test20250721_hidden_dim_1024' \
                    #    --embed_type $embed_type \
                    #    --edge_type $edge_type \
                    #    --attn_type $attn_type \
                    #    --few_shot $few_shot \
                    #    --train_epochs 1000 \
                    #    --model_type "TabularFLM" \
                    #    --del_feat Sex ChestPainType RestingECG ExerciseAngina Age RestingBP FastingBS MaxHR
                       
                   done
               done
           done
       done
   done
done