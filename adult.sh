gpu_id=0
source_datasets="adult"
few_shots="4"
model_types="TabularFLM"
random_seeds="2011 3007 4000 4005 1019"
# 200개의 랜덤 시드로 반복
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


gpu_id=0
source_datasets="adult"
few_shots="4"
model_types="TabularFLM"
random_seeds="2011 3007 4000 4005 1019"
# 200개의 랜덤 시드로 반복
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