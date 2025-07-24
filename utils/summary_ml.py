import json
import csv
import glob
import os
import numpy as np
from collections import defaultdict
import argparse

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def calculate_mean_std(values):
    values = np.array(values)
    mean = np.mean(values)
    std = np.std(values, ddof=0)
    return mean, std

def create_ml_combined_summary(results_by_dataset, full_results_by_dataset, base_directory, dataset):
    """모든 ML 모델의 모든 메트릭을 하나의 TSV 표로 생성"""
    
    summary_dir = os.path.join(base_directory, dataset, "summary")
    create_directory(summary_dir)
    
    # 전체 ML 결과를 담을 TSV 파일
    combined_file = os.path.join(summary_dir, 'all_models_combined.tsv')
    
    # 모델 순서 정의
    model_order = ['rf', 'lr', 'xgb', 'mlp', 'cat']
    
    with open(combined_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        
        # 각 모델의 각 메트릭별로 행 생성
        for model in model_order:
            if model not in results_by_dataset[dataset]:
                continue
            
            model_data = results_by_dataset[dataset][model]
            few_shot_keys = sorted(model_data.keys())
            
            # 5개 메트릭 각각에 대해 행 생성
            for metric in ['auc', 'acc', 'precision', 'recall', 'f1']:
                row = []
                
                # Few-shot 결과들 (4, 8, 16, 32, 64)
                for few_shot in few_shot_keys:
                    values = model_data[few_shot][metric]
                    if values:
                        mean, std = calculate_mean_std(values)
                        row.append(f"{mean:.4f}({std:.4f})")
                    else:
                        row.append("")
                
                # Full dataset 결과
                if model in full_results_by_dataset[dataset]:
                    values = full_results_by_dataset[dataset][model][metric]
                    if values:
                        mean, std = calculate_mean_std(values)
                        row.append(f"{mean:.4f}({std:.4f})")
                    else:
                        row.append("")
                else:
                    row.append("")
                
                writer.writerow(row)

def process_ml_results(base_directory):
    results_by_dataset = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    full_results_by_dataset = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # 시드별 full 결과 저장
    
    for dataset in ['heart', 'diabetes']:
        dataset_path = os.path.join(base_directory, dataset)
        if not os.path.exists(dataset_path):
            continue
            
        json_pattern = os.path.join(dataset_path, "**/f*.json")
        json_files = glob.glob(json_pattern, recursive=True)
        
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            few_shot = data['hyperparameters']['few_shot']
            batch_size = data['hyperparameters']['batch_size']
            
            # few-shot 결과 저장
            for model in ['rf', 'lr', 'xgb', 'mlp', 'cat']:
                if model in data['results']['Best_results']['few']:
                    # Few-shot 결과
                    metrics_few = data['results']['Best_results']['few'][model]
                    for metric_name, value in metrics_few.items():
                        clean_metric = metric_name.replace(f'{model}_best_few_', '')
                        results_by_dataset[dataset][model][few_shot][clean_metric].append(value)
                    
                    # Full dataset 결과 (시드별로 저장)
                    metrics_full = data['results']['Best_results']['full'][model]
                    for metric_name, value in metrics_full.items():
                        clean_metric = metric_name.replace(f'{model}_best_full_', '')
                        full_results_by_dataset[dataset][model][clean_metric].append(value)
        
        # 결과 저장
        for model in ['rf', 'lr', 'xgb', 'mlp', 'cat']:
            model_dir = os.path.join(base_directory, dataset, "summary", model)
            create_directory(model_dir)
            
            # Few-shot 결과 저장
            for few_shot in sorted(results_by_dataset[dataset][model].keys()):
                few_output_file = os.path.join(model_dir, f'f{few_shot}_b{batch_size}.csv')
                
                with open(few_output_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Model Type', model])
                    writer.writerow(['Few Shot', few_shot])
                    writer.writerow(['Batch Size', batch_size])
                    writer.writerow([''])
                    
                    for metric in ['auc', 'acc', 'precision', 'recall', 'f1']:
                        values = results_by_dataset[dataset][model][few_shot][metric]
                        mean, std = calculate_mean_std(values)
                        writer.writerow([f"{mean:.4f}({std:.4f})"])
            
            # Full dataset 결과 저장 (모델별로 한 번만)
            full_output_file = os.path.join(model_dir, 'full_results.csv')
            with open(full_output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Model Type', model])
                writer.writerow(['Full Dataset Results'])
                writer.writerow(['Batch Size', batch_size])
                writer.writerow([''])
                
                for metric in ['auc', 'acc', 'precision', 'recall', 'f1']:
                    values = full_results_by_dataset[dataset][model][metric]
                    mean, std = calculate_mean_std(values)
                    writer.writerow([f"{mean:.4f}({std:.4f})"])
            
            print(f"Results saved to {few_output_file} and {full_output_file}")
        
        # 🔥 새로 추가: 모든 모델을 하나로 합친 TSV 파일 생성
        create_ml_combined_summary(results_by_dataset, full_results_by_dataset, base_directory, dataset)
        print(f"Combined TSV saved for dataset: {dataset}")

def main():
    parser = argparse.ArgumentParser(description='Summarize ML results')
    parser.add_argument('--base_dir', type=str, default='/home/eungyeop/LLM/tabular/ProtoLLM/experiments/ml_baselines_ML_results_20250322',
                      help='Base directory containing the results')
    
    args = parser.parse_args()
    process_ml_results(args.base_dir)

if __name__ == "__main__":
    main()