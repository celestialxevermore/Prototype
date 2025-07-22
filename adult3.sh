#!/bin/bash

gpu_id=4
source_datasets="adult"
random_seeds="2011 1023 4005 3007 4017"
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
                       echo "Running experiment - del_feat: educational-num capital-gain fnlwgt"
                       CUDA_VISIBLE_DEVICES=$gpu_id python main_G.py \
                       --random_seed $random_seed \
                       --source_data $source_dataset \
                       --base_dir 'test20250719' \
                       --embed_type $embed_type \
                       --edge_type $edge_type \
                       --attn_type $attn_type \
                       --few_shot $few_shot \
                       --train_epochs 1000 \
                       --model_type "TabularFLM" \
                       --del_feat educational-num capital-gain fnlwgt
                       
                       # del_feat 조합 2: ChestPainType
                       echo "Running experiment - del_feat: educational-num capital-gain fnlwgt capital-loss"
                       CUDA_VISIBLE_DEVICES=$gpu_id python main_G.py \
                       --random_seed $random_seed \
                       --source_data $source_dataset \
                       --base_dir 'test20250719' \
                       --embed_type $embed_type \
                       --edge_type $edge_type \
                       --attn_type $attn_type \
                       --few_shot $few_shot \
                       --train_epochs 1000 \
                       --model_type "TabularFLM" \
                       --del_feat educational-num capital-gain fnlwgt capital-loss
                       
                       # del_feat 조합 3: RestingECG
                       echo "Running experiment - del_feat: marital-status occupation race"
                       CUDA_VISIBLE_DEVICES=$gpu_id python main_G.py \
                       --random_seed $random_seed \
                       --source_data $source_dataset \
                       --base_dir 'test20250719' \
                       --embed_type $embed_type \
                       --edge_type $edge_type \
                       --attn_type $attn_type \
                       --few_shot $few_shot \
                       --train_epochs 1000 \
                       --model_type "TabularFLM" \
                       --del_feat marital-status occupation race
                       
                   done
               done
           done
       done
   done
done