#!/bin/bash

gpu_id=4
source_datasets="heart"
random_seeds="2095"
embed_types="carte_desc"
attn_types="gat att"
few_shots="4"

for random_seed in $random_seeds; do
   for embed_type in $embed_types; do
       for attn_type in $attn_types; do
           for few_shot in $few_shots; do 
               for source_dataset in $source_datasets; do
                   echo "Running experiment for $source_dataset with $few_shot few-shot samples, embed_type:$embed_type, attn_type:$attn_type, random_seed:$random_seed"
                   
                   CUDA_VISIBLE_DEVICES=$gpu_id python main_G.py \
                   --random_seed $random_seed \
                   --source_data $source_dataset \
                   --base_dir 'Experiment_TabularFLM_G' \
                   --embed_type $embed_type \
                   --attn_type $attn_type \
                   --use_edge_attr \
                   --few_shot $few_shot \
                   --train_epochs 1000 \
                   --model_type "TabularFLM"
               done
           done
       done
   done
done