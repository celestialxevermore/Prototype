gpu_id=0
source_datasets="diabetes"
random_seeds="42"
embed_types="carte"
edge_types="mlp"
attn_types="gat_v1"
few_shots="4 8 16 32 64"

# Diabetes에서 절대 제거하면 안 되는 변수들
# Glucose BMI DiabetesPedigreeFunction

# 제거 가능한 변수들 (random하게 선택)
del_feat_configs=(
    "Pregnancies BloodPressure"
    "SkinThickness Insulin"
    "Age"
    "Pregnancies SkinThickness"
    "BloodPressure Insulin"
    "Pregnancies Age"
    "BloodPressure Age"
    "SkinThickness Age"
    "Pregnancies BloodPressure SkinThickness"
    "BloodPressure SkinThickness Insulin"
    "Pregnancies SkinThickness Age"
    "BloodPressure Insulin Age"
    "Pregnancies BloodPressure Insulin"
    "SkinThickness Insulin Age"
    "Pregnancies BloodPressure SkinThickness Insulin"
    "BloodPressure SkinThickness Insulin Age"
    "Pregnancies BloodPressure SkinThickness Age"
    "Pregnancies BloodPressure SkinThickness Insulin Age"
)

for random_seed in $random_seeds; do
   for few_shot in $few_shots; do
       for embed_type in $embed_types; do
           for attn_type in $attn_types; do
               for edge_type in $edge_types; do 
                   for source_dataset in $source_datasets; do
                       for del_feat in "${del_feat_configs[@]}"; do
                           
                           # Self-loop 유지 버전
                           echo "Running experiment - WITH self-loop, del_feat: $del_feat"
                           CUDA_VISIBLE_DEVICES=$gpu_id python main_G.py \
                           --random_seed $random_seed \
                           --source_data $source_dataset \
                           --base_dir 'test20250812_perturbation_self_loop' \
                           --embed_type $embed_type \
                           --edge_type $edge_type \
                           --attn_type $attn_type \
                           --few_shot $few_shot \
                           --train_epochs 1000 \
                           --model_type "TabularFLM" \
                           --del_feat $del_feat

                           # Self-loop 제거 버전  
                           echo "Running experiment - WITHOUT self-loop, del_feat: $del_feat"
                           CUDA_VISIBLE_DEVICES=$gpu_id python main_G.py \
                           --random_seed $random_seed \
                           --source_data $source_dataset \
                           --base_dir 'test20250812_perturbation_no_self_loop' \
                           --embed_type $embed_type \
                           --edge_type $edge_type \
                           --attn_type $attn_type \
                           --few_shot $few_shot \
                           --train_epochs 1000 \
                           --model_type "TabularFLM" \
                           --no_self_loop \
                           --del_feat $del_feat
                           
                       done
                   done
               done
           done
       done
   done
done







# gpu_id=4
# source_datasets="diabetes"
# random_seeds="4078 73 96 218 4142"
# embed_types="carte"
# edge_types="mlp"
# attn_types="gat_v1"
# few_shots="4 8 16 32 64"

# for random_seed in $random_seeds; do
#   for few_shot in $few_shots; do
#       for embed_type in $embed_types; do
#           for attn_type in $attn_types; do
#               for edge_type in $edge_types; do 
#                   for source_dataset in $source_datasets; do
                      
#                       # Self-loop 유지 버전
#                       echo "Running experiment - WITH self-loop"
#                       CUDA_VISIBLE_DEVICES=$gpu_id python main_A.py \
#                       --random_seed $random_seed \
#                       --source_data $source_dataset \
#                       --base_dir 'test20250726_scenario12345_with_self_loop_H_D_L_2' \
#                       --embed_type $embed_type \
#                       --edge_type $edge_type \
#                       --attn_type $attn_type \
#                       --dropout_rate 0.5 \
#                       --n_heads 8 \
#                       --num_layers 2 \
#                       --few_shot $few_shot \
#                       --train_epochs 1000 \
#                       --model_type "TabularFLM"

#                       # Self-loop 제거 버전  
#                       echo "Running experiment - WITHOUT self-loop"
#                       CUDA_VISIBLE_DEVICES=$gpu_id python main_A.py \
#                       --random_seed $random_seed \
#                       --source_data $source_dataset \
#                       --base_dir 'test20250726_scenario12345_no_self_loop_H_D_L_2' \
#                       --embed_type $embed_type \
#                       --edge_type $edge_type \
#                       --attn_type $attn_type \
#                       --dropout_rate 0.5 \
#                       --n_heads 8 \
#                       --num_layers 2 \
#                       --few_shot $few_shot \
#                       --train_epochs 1000 \
#                       --model_type "TabularFLM" \
#                       --no_self_loop
                      
#                   done
#               done
#           done
#       done
#   done
# done