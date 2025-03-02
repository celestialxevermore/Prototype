<<<<<<< Updated upstream
gpu_id=4
source_datasets="heart diabetes adult"
few_shots="4 8 16 32 64"
model_types="TabularFLM"
aggr_types="flatten attn"
enc_types="ind shared"
meta_types="meta_mlp meta_attn"
labels="add no"
for few_shot in $few_shots; do 
    for source_dataset in $source_datasets; do
        for agg_type in $aggr_types; do 
            for enc_type in $enc_types; do
                for meta_type in $meta_types; do
                    for label in $labels; do
                            echo "Running experiment for $source_dataset with $shot few-shot samples in random_seed:2025"
                            CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
                            --random_seed 2025 \
                            --source_dataset_name $source_dataset \
                            --base_dir 'Experiment_TabularFLM3' \
                            --few_shot $few_shot \
                            --train_epochs 1000 \
                            --label $label \
                            --aggr_type $agg_type \
                            --enc_type $enc_type \
                            --meta_type $meta_type \
                            --model_type "TabularFLM"
=======


gpu_id=2
source_datasets="heart diabetes adult"
few_shots="4 8 16 32 64"
batch_sizes="32 64"
model_types="GAT_edge_4"
FDs="N D ND"
center_types="CM CA_m CA_f CA_p"
random_seeds="1234"
for random_seed in $random_seeds; do
    for source_dataset in $source_datasets; do
        for shot in $few_shots; do
            for batch_size in $batch_sizes; do
                for center_type in $center_types; do
                    for model_type in $model_types; do
                        for FD in $FDs; do
                            echo "Running experiment for $source_dataset with $shot few-shot samples in random_seed:$random_seed"
                            CUDA_VISIBLE_DEVICES=$gpu_id python main_s2s_few.py \
                            --random_seed $random_seed \
                            --source_dataset_name $source_dataset \
                            --few_shot $shot \
                            --train_epochs 200 \
                            --num_layers 3 \
                            --heads 4 \
                            --dropout_rate 0.2 \
                            --batch_size $batch_size \
                            --graph_type 'star' \
                            --FD $FD \
                            --center_type $center_type \
                            --model_type $model_type
                        done
                    done
                done
            done
        done
    done
done

gpu_id=2
source_datasets="heart diabetes adult"
few_shots="4 8 16 32 64"
batch_sizes="32 64"
model_types="GAT_edge_4"
FDs="N D ND"
center_types="CM CA_m CA_f CA_p"
random_seeds="1234"
for random_seed in $random_seeds; do
    for source_dataset in $source_datasets; do
        for shot in $few_shots; do
            for batch_size in $batch_sizes; do
                for center_type in $center_types; do
                    for model_type in $model_types; do
                        for FD in $FDs; do
                            echo "Running experiment for $source_dataset with $shot few-shot samples in random_seed:$random_seed"
                            CUDA_VISIBLE_DEVICES=$gpu_id python main_s2s_few.py \
                            --random_seed $random_seed \
                            --source_dataset_name $source_dataset \
                            --few_shot $shot \
                            --train_epochs 200 \
                            --batch_size $batch_size \
                            --graph_type 'star' \
                            --FD $FD \
                            --center_type $center_type \
                            --model_type $model_type
                        done
>>>>>>> Stashed changes
                    done
                done
            done
        done
    done
done

# gpu_id=2
# source_datasets="heart diabetes adult"
# few_shots="4 8 16 32 64"
# batch_sizes="32 64"
# model_types="GAT_edge_4"
# FDs="N D ND"
# center_types="CM CA_m CA_f CA_p"
# random_seeds="1234"
# for random_seed in $random_seeds; do
#     for source_dataset in $source_datasets; do
#         for shot in $few_shots; do
#             for batch_size in $batch_sizes; do
#                 for center_type in $center_types; do
#                     for model_type in $model_types; do
#                         for FD in $FDs; do
#                             echo "Running experiment for $source_dataset with $shot few-shot samples in random_seed:$random_seed"
#                             CUDA_VISIBLE_DEVICES=$gpu_id python main_s2s.py \
#                             --random_seed $random_seed \
#                             --source_dataset_name $source_dataset \
#                             --few_shot $shot \
#                             --train_epochs 200 \
#                             --num_layers 3 \
#                             --heads 4 \
#                             --dropout_rate 0.2 \
#                             --batch_size $batch_size \
#                             --graph_type 'star' \
#                             --FD $FD \
#                             --center_type $center_type \
#                             --model_type $model_type
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done

# gpu_id=2
# source_datasets="heart diabetes adult"
# few_shots="4 8 16 32 64"
# batch_sizes="32 64"
# model_types="GAT_edge_4"
# FDs="N D ND"
# center_types="CM CA_m CA_f CA_p"
# random_seeds="1234"
# for random_seed in $random_seeds; do
#     for source_dataset in $source_datasets; do
#         for shot in $few_shots; do
#             for batch_size in $batch_sizes; do
#                 for center_type in $center_types; do
#                     for model_type in $model_types; do
#                         for FD in $FDs; do
#                             echo "Running experiment for $source_dataset with $shot few-shot samples in random_seed:$random_seed"
#                             CUDA_VISIBLE_DEVICES=$gpu_id python main_s2s.py \
#                             --random_seed $random_seed \
#                             --source_dataset_name $source_dataset \
#                             --few_shot $shot \
#                             --train_epochs 200 \
#                             --batch_size $batch_size \
#                             --graph_type 'star' \
#                             --FD $FD \
#                             --center_type $center_type \
#                             --model_type $model_type
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done

