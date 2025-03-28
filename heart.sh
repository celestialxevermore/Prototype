gpu_id=4
source_datasets="heart"
few_shots="4 8 16 32 64"
model_types="TabularFLM"
random_seeds="2095 3192 3155 67 1045 "
# 2000부터 2199까지 200개의 랜덤 시드로 반복
for random_seed in $random_seeds; do
    for few_shot in $few_shots; do 
        for source_dataset in $source_datasets; do
            echo "Running experiment for $source_dataset with $few_shot few-shot samples in random_seed:$random_seed"
            
            CUDA_VISIBLE_DEVICES=$gpu_id python main_G2.py \
            --random_seed $random_seed \
            --source_dataset_name $source_dataset \
            --base_dir 'Experiment_TabularFLM_G6' \
            --use_gmm \
            --few_shot $few_shot \
            --train_epochs 1000 \
            --model_type "TabularFLM"
        done
    done
done

gpu_id=4
source_datasets="heart"
few_shots="4 8 16 32 64"
model_types="TabularFLM"
random_seeds="2095 3192 3155 67 1045 "
# 2000부터 2199까지 200개의 랜덤 시드로 반복
for random_seed in $random_seeds; do
    for few_shot in $few_shots; do 
        for source_dataset in $source_datasets; do
            echo "Running experiment for $source_dataset with $few_shot few-shot samples in random_seed:$random_seed"
            
            CUDA_VISIBLE_DEVICES=$gpu_id python main_G2.py \
            --random_seed $random_seed \
            --source_dataset_name $source_dataset \
            --base_dir 'Experiment_TabularFLM_G7' \
            --use_gmm2 \
            --few_shot $few_shot \
            --train_epochs 1000 \
            --model_type "TabularFLM"
        done
    done
done