
gpu_id=4
source_datasets="heart diabetes adult"
few_shots="4 8 16 32 64"
batch_sizes="32 64"
model_types="GAT_edge_4"
num_layers="2"
heads="4 8"
FDs="N D"
center_types="CA_m CA_f CM"
random_seeds="5678"
base_dir="20250215_ablation_few"

for source_dataset in $source_datasets; do
    for random_seed in $random_seeds; do
        for shot in $few_shots; do
            for center_type in $center_types; do
                for batch_size in $batch_sizes; do
                    for num_layers in $num_layers; do
                        for head in $heads; do
                            for FD in $FDs; do
                                echo "Running experiment for $source_dataset with $shot few-shot samples in random_seed:$random_seed"
                                CUDA_VISIBLE_DEVICES=$gpu_id python main_s2s_few.py \
                                --base_dir $base_dir \
                                --random_seed $random_seed \
                                --source_dataset_name $source_dataset \
                                --few_shot $shot \
                                --train_epochs 200 \
                                --num_layers $num_layers \
                                --heads $head \
                                --dropout_rate 0.3 \
                                --batch_size $batch_size \
                                --graph_type 'star' \
                                --FD $FD \
                                --center_type $center_type \
                                --model_type 'GAT_edge_5'
                                
                            done
                        done
                    done
                done
            done
        done
    done
done

gpu_id=4
source_datasets="heart diabetes adult"
few_shots="4 8 16 32 64"
batch_sizes="32 64"
model_types="GAT_edge_2"
num_layers="2"
heads="2 4 8"
FDs="N D"
center_types="CA_m CA_f CM"
random_seeds="5678"
base_dir="20250215_ablation_few"

for source_dataset in $source_datasets; do
    for random_seed in $random_seeds; do
        for shot in $few_shots; do
            for center_type in $center_types; do
                for batch_size in $batch_sizes; do
                    for num_layers in $num_layers; do
                        for head in $heads; do
                            for FD in $FDs; do
                                echo "Running experiment for $source_dataset with $shot few-shot samples in random_seed:$random_seed"
                                CUDA_VISIBLE_DEVICES=$gpu_id python main_s2s_few.py \
                                --base_dir $base_dir \
                                --random_seed $random_seed \
                                --source_dataset_name $source_dataset \
                                --few_shot $shot \
                                --train_epochs 200 \
                                --num_layers $num_layers \
                                --heads $head \
                                --dropout_rate 0.3 \
                                --batch_size $batch_size \
                                --graph_type 'star' \
                                --FD $FD \
                                --center_type $center_type \
                                --model_type 'GAT_edge_4'
                                
                            done
                        done
                    done
                done
            done
        done
    done
done

for source_dataset in $source_datasets; do
    for random_seed in $random_seeds; do
        for shot in $few_shots; do
            for center_type in $center_types; do
                for batch_size in $batch_sizes; do
                    for num_layers in $num_layers; do
                        for head in $heads; do
                            for FD in $FDs; do
                                echo "Running experiment for $source_dataset with $shot few-shot samples in random_seed:$random_seed"
                                CUDA_VISIBLE_DEVICES=$gpu_id python main_s2s_few.py \
                                --base_dir $base_dir \
                                --random_seed $random_seed \
                                --source_dataset_name $source_dataset \
                                --few_shot $shot \
                                --train_epochs 200 \
                                --num_layers $num_layers \
                                --heads $head \
                                --dropout_rate 0.3 \
                                --batch_size $batch_size \
                                --graph_type 'star' \
                                --FD $FD \
                                --center_type $center_type \
                                --model_type 'GAT_edge_3'
                                
                            done
                        done
                    done
                done
            done
        done
    done
done

gpu_id=4
source_datasets="heart diabetes adult"
few_shots="4 8 16 32 64"
batch_sizes="32 64"
model_types="GAT_edge_2"
num_layers="2"
heads="2 4 8"
FDs="N D"
center_types="CA_m CA_f CM"
random_seeds="5678"
base_dir="20250215_ablation_few"

for source_dataset in $source_datasets; do
    for random_seed in $random_seeds; do
        for shot in $few_shots; do
            for center_type in $center_types; do
                for batch_size in $batch_sizes; do
                    for num_layers in $num_layers; do
                        for head in $heads; do
                            for FD in $FDs; do
                                echo "Running experiment for $source_dataset with $shot few-shot samples in random_seed:$random_seed"
                                CUDA_VISIBLE_DEVICES=$gpu_id python main_s2s_few.py \
                                --base_dir $base_dir \
                                --random_seed $random_seed \
                                --source_dataset_name $source_dataset \
                                --few_shot $shot \
                                --train_epochs 200 \
                                --num_layers $num_layers \
                                --heads $head \
                                --dropout_rate 0.3 \
                                --batch_size $batch_size \
                                --graph_type 'star' \
                                --FD $FD \
                                --center_type $center_type \
                                --model_type 'GAT_edge_2'
                                
                            done
                        done
                    done
                done
            done
        done
    done
done