#!/bin/bash

# OpenMP 스레드 수 설정
export OMP_NUM_THREADS=10

# 기본 경로 설정
base_path="/storage/personal/eungyeop/experiments/checkpoints/gpt2_mean"
datasets=("diabetes" "heart")
modes=("Full" "Few")
seeds=("42" "44" "46" "48" "50")

# 출력 디렉토리 설정
output_base_dir="clustering_analysis_results"

# 각 데이터셋, 모드, 시드에 대해 실행
for dataset in "${datasets[@]}"; do
    for mode in "${modes[@]}"; do
        for seed in "${seeds[@]}"; do
            checkpoint_dir="${base_path}/${dataset}/${mode}/${seed}"
            
            # 체크포인트 디렉토리가 존재하는지 확인
            if [ ! -d "$checkpoint_dir" ]; then
                echo "Directory not found: $checkpoint_dir"
                continue
            fi
            
            # .pt 파일들을 하나씩 찾아서 실행
            for pt_file in "$checkpoint_dir"/*.pt; do
                # 파일이 존재하는지 확인
                if [ ! -f "$pt_file" ]; then
                    continue
                fi
                
                # 파일명에서 정보 추출
                filename=$(basename "$pt_file")
                echo "Processing: $filename"
                
                # 출력 디렉토리 생성
                output_dir="${output_base_dir}/${dataset}/${mode}/${seed}"
                mkdir -p "$output_dir"
                
                # clustering1_analysis.py 실행
                echo "Running clustering analysis for: $filename"
                python analysis/clustering1_analysis.py \
                    --checkpoint_dir "$pt_file" \
                    --mode "$mode" \
                    --layer_idx 2 \
                    --n_clusters 3 \
                    --max_samples 5 \
                    --output_dir "$output_dir" \
                    --save_attention_maps \
                    --viz_graph
                
                echo "Completed: $filename"
                echo "----------------------------------------"
            done
        done
    done
done

echo "All clustering analysis completed!"