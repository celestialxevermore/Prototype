#!/bin/bash

gpu_id=4
source_datasets="diabetes"
random_seeds="50"
embed_types="carte"
edge_types="mlp"
attn_types="gat_v1"
few_shots="4 8 16 32 64"

for random_seed in $random_seeds; do
   for few_shot in $few_shots; do
       for embed_type in $embed_types; do
           for attn_type in $attn_types; do
               for edge_type in $edge_types; do 
                   for source_dataset in $source_datasets; do
                       
                       # exp1 : (아무것도 제거하지 않았을 때) 
                       echo "Running experiment - del_feat: Full_Features"
                       CUDA_VISIBLE_DEVICES=$gpu_id python main_G.py \
                       --random_seed $random_seed \
                       --source_data $source_dataset \
                       --base_dir 'test20250723' \
                       --embed_type $embed_type \
                       --edge_type $edge_type \
                       --attn_type $attn_type \
                       --few_shot $few_shot \
                       --train_epochs 1000 \
                       --model_type "TabularFLM" \
                       --del_exp "exp1"

                       
                    #    # exp2 : (중요한 소수의 변수를 제거했을 때) 
                       echo "Running experiment - del_feat: Glucose BMI DiabetesPedigreeFunction"
                       CUDA_VISIBLE_DEVICES=$gpu_id python main_G.py \
                       --random_seed $random_seed \
                       --source_data $source_dataset \
                       --base_dir 'test20250723' \
                       --embed_type $embed_type \
                       --edge_type $edge_type \
                       --attn_type $attn_type \
                       --few_shot $few_shot \
                       --train_epochs 1000 \
                       --model_type "TabularFLM" \
                       --del_feat Glucose BMI DiabetesPedigreeFunction \
                       --del_exp "exp2"
                       
                       # exp3 : (중요한 소수의 변수와 안중요한 변수 몇 개를 제거했을 때) 
                       echo "Running experiment - del_feat: Glucose BMI DiabetesPedigreeFunction Pregnancies Insulin"
                       CUDA_VISIBLE_DEVICES=$gpu_id python main_G.py \
                       --random_seed $random_seed \
                       --source_data $source_dataset \
                       --base_dir 'test20250723' \
                       --embed_type $embed_type \
                       --edge_type $edge_type \
                       --attn_type $attn_type \
                       --few_shot $few_shot \
                       --train_epochs 1000 \
                       --model_type "TabularFLM" \
                       --del_feat Glucose BMI DiabetesPedigreeFunction Pregnancies Insulin \
                       --del_exp "exp3"
                       
                       # exp4 : (안중요한 소수 변수만 제거했을 때) 
                       echo "Running experiment - del_feat: Pregnancies Insulin BloodPressure"
                       CUDA_VISIBLE_DEVICES=$gpu_id python main_G.py \
                       --random_seed $random_seed \
                       --source_data $source_dataset \
                       --base_dir 'test20250723' \
                       --embed_type $embed_type \
                       --edge_type $edge_type \
                       --attn_type $attn_type \
                       --few_shot $few_shot \
                       --train_epochs 1000 \
                       --model_type "TabularFLM" \
                       --del_feat Pregnancies Insulin BloodPressure \
                       --del_exp "exp4"
                       
                       # exp5 : (중요한 변수만 남기고 대다수 변수를 모두 제거했을 때)
                       echo "Running experiment - del_feat: Pregnancies Insulin BloodPressure SkinThickness"
                       CUDA_VISIBLE_DEVICES=$gpu_id python main_G.py \
                       --random_seed $random_seed \
                       --source_data $source_dataset \
                       --base_dir 'test20250723' \
                       --embed_type $embed_type \
                       --edge_type $edge_type \
                       --attn_type $attn_type \
                       --few_shot $few_shot \
                       --train_epochs 1000 \
                       --model_type "TabularFLM" \
                       --del_feat Pregnancies Insulin BloodPressure SkinThickness \
                       --del_exp "exp5"
                       
                   done
               done
           done
       done
   done
done

#!/bin/bash

gpu_id=4
source_datasets="diabetes"
random_seeds="50"
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
                       
                       # exp1 : (아무것도 제거하지 않았을 때) 
                       echo "Running experiment - del_feat: Full_Features"
                       CUDA_VISIBLE_DEVICES=$gpu_id python main_G.py \
                       --random_seed $random_seed \
                       --source_data $source_dataset \
                       --base_dir 'test20250723' \
                       --embed_type $embed_type \
                       --edge_type $edge_type \
                       --attn_type $attn_type \
                       --few_shot $few_shot \
                       --train_epochs 1000 \
                       --model_type "TabularFLM" \
                       --del_exp "exp1"
                       
                    #    # exp2 : (중요한 소수의 변수를 제거했을 때) 
                       echo "Running experiment - del_feat: Glucose BMI Age"
                       CUDA_VISIBLE_DEVICES=$gpu_id python main_G.py \
                       --random_seed $random_seed \
                       --source_data $source_dataset \
                       --base_dir 'test20250723' \
                       --embed_type $embed_type \
                       --edge_type $edge_type \
                       --attn_type $attn_type \
                       --few_shot $few_shot \
                       --train_epochs 1000 \
                       --model_type "TabularFLM" \
                       --del_feat Glucose BMI Age \
                       --del_exp "exp2"
                       
                       # exp3 : (중요한 소수의 변수와 안중요한 변수 몇 개를 제거했을 때) 
                       echo "Running experiment - del_feat: Glucose BMI Age SkinThickness Insulin"
                       CUDA_VISIBLE_DEVICES=$gpu_id python main_G.py \
                       --random_seed $random_seed \
                       --source_data $source_dataset \
                       --base_dir 'test20250723' \
                       --embed_type $embed_type \
                       --edge_type $edge_type \
                       --attn_type $attn_type \
                       --few_shot $few_shot \
                       --train_epochs 1000 \
                       --model_type "TabularFLM" \
                       --del_feat Glucose BMI Age SkinThickness Insulin \
                       --del_exp "exp3"
                       
                       # exp4 : (안중요한 소수 변수만 제거했을 때) 
                       echo "Running experiment - del_feat: SkinThickness Insulin Pregnancies"
                       CUDA_VISIBLE_DEVICES=$gpu_id python main_G.py \
                       --random_seed $random_seed \
                       --source_data $source_dataset \
                       --base_dir 'test20250723' \
                       --embed_type $embed_type \
                       --edge_type $edge_type \
                       --attn_type $attn_type \
                       --few_shot $few_shot \
                       --train_epochs 1000 \
                       --model_type "TabularFLM" \
                       --del_feat SkinThickness Insulin Pregnancies \
                       --del_exp "exp4"
                       
                       # exp5 : (중요한 변수만 남기고 대다수 변수를 모두 제거했을 때)
                       echo "Running experiment - del_feat: SkinThickness Insulin Pregnancies DiabetesPedigreeFunction BloodPressure"
                       CUDA_VISIBLE_DEVICES=$gpu_id python main_G.py \
                       --random_seed $random_seed \
                       --source_data $source_dataset \
                       --base_dir 'test20250723' \
                       --embed_type $embed_type \
                       --edge_type $edge_type \
                       --attn_type $attn_type \
                       --few_shot $few_shot \
                       --train_epochs 1000 \
                       --model_type "TabularFLM" \
                       --del_feat SkinThickness Insulin Pregnancies DiabetesPedigreeFunction BloodPressure \
                       --del_exp "exp5"
                       
                   done
               done
           done
       done
   done
done