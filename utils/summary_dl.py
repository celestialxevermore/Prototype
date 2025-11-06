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

def create_combined_summary_by_model(results_by_config, full_results, dataset, base_dir):
    """각 model_config별로 모든 설정의 모든 메트릭을 하나의 TSV 표로 생성"""
    
    # model_config별로 그룹화
    results_by_model = defaultdict(dict)
    full_results_by_model = defaultdict(dict)
    
    for key, data in results_by_config.items():
        model_config, few_shot, batch_size = key
        results_by_model[model_config][(few_shot, batch_size)] = data
    
    for key, data in full_results.items():
        model_config, batch_size = key
        full_results_by_model[model_config][batch_size] = data
    
    # 각 model_config별로 TSV 파일 생성
    for model_config in results_by_model.keys():
        config_data = results_by_model[model_config]
        full_data = full_results_by_model.get(model_config, {})
        
        # few_shot과 batch_size별로 정렬
        sorted_configs = sorted(config_data.items(), key=lambda x: (x[0][0], x[0][1]))  # few_shot, batch_size 순으로 정렬
        
        summary_dir = os.path.join(base_dir, f"{dataset}_summary", model_config)
        create_directory(summary_dir)
        
        # 전체 결과를 담을 TSV 파일
        combined_file = os.path.join(summary_dir, 'all_configs_combined.tsv')
        
        with open(combined_file, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            
            # 헤더 작성 (설정별 컬럼명)
            header = []
            for (few_shot, batch_size), _ in sorted_configs:
                header.append(f"f{few_shot}_b{batch_size}")
            
            # Full dataset 컬럼 추가 (batch_size별로)
            unique_batch_sizes = sorted(set([bs for (fs, bs), _ in sorted_configs]))
            for batch_size in unique_batch_sizes:
                header.append(f"full_b{batch_size}")
            
            writer.writerow(header)
            
            # 각 메트릭별로 행 생성 (5개 메트릭)
            for metric in ['auc', 'acc', 'precision', 'recall', 'f1']:
                row = []
                
                # Few-shot 결과들
                for (few_shot, batch_size), data in sorted_configs:
                    values = data['few_shot'][metric]
                    if values:
                        mean, std = calculate_mean_std(values)
                        row.append(f"{mean:.4f}({std:.4f})")
                    else:
                        row.append("")
                
                # Full dataset 결과들
                for batch_size in unique_batch_sizes:
                    if batch_size in full_data and metric in full_data[batch_size]:
                        values = full_data[batch_size][metric]
                        if values:
                            mean, std = calculate_mean_std(values)
                            row.append(f"{mean:.4f}({std:.4f})")
                        else:
                            row.append("")
                    else:
                        row.append("")
                
                writer.writerow(row)

def process_json_files(directory_path, selected_datasets=None, selected_seeds=None):
    if selected_datasets:
        print(f"처리할 데이터셋: {selected_datasets}")
    
    if selected_seeds:
        selected_seeds = [str(seed) for seed in selected_seeds]  # 문자열로 변환
    
    datasets_to_process = selected_datasets if selected_datasets else ['heart']
    
    for dataset in datasets_to_process:
        results_by_config = {}
        
        # ✅ base_dir이 이미 dataset 수준이거나 더 깊은 경우 그대로 사용
        if os.path.isdir(directory_path) and any(name.startswith("args_seed:") for name in os.listdir(directory_path)):
            # 이미 args_seed: 폴더가 포함된 경로일 경우
            dataset_path = directory_path
        elif os.path.isdir(directory_path) and "ngraphs-" in directory_path:
            # ngraphs- 폴더 내부 경로인 경우
            dataset_path = os.path.dirname(directory_path)
        else:
            # 기존 구조 (base_dir + dataset)
            dataset_path = os.path.join(directory_path, dataset)
            
        if not os.path.exists(dataset_path):
            print(f"경로를 찾을 수 없음: {dataset_path}")
            continue
        
        json_files = []
        processed_seeds = []
        
        # 선택된 시드가 있으면 해당 시드 경로만 처리
        if selected_seeds:
            for seed in selected_seeds:
                seed_path = os.path.join(dataset_path, f"args_seed:{seed}")
                if os.path.exists(seed_path):
                    # 패턴을 실제 폴더 구조에 맞게 수정
                    seed_json_pattern = os.path.join(seed_path, "TabularFLM/*/f*.json")
                    seed_json_files = glob.glob(seed_json_pattern, recursive=True)
                    if seed_json_files:
                        json_files.extend(seed_json_files)
                        processed_seeds.append(seed)
                else:
                    print(f"시드 {seed}의 경로가 존재하지 않음: {seed_path}")
        else:
            # 모든 시드 처리 (기존 방식) - 패턴 수정
            #json_pattern = os.path.join(dataset_path, "args_seed:*/TabularFLM/*/f*.json")
            json_pattern = os.path.join(dataset_path, "args_seed:*/ngraphs-*/f*.json")
            json_files = glob.glob(json_pattern, recursive=True)
        
        if not json_files:
            print(f"데이터셋 {dataset}에서 JSON 파일을 찾을 수 없음")
            continue
        
        # Full dataset 결과를 저장할 별도의 딕셔너리
        full_results = defaultdict(lambda: {'auc': [], 'acc': [], 'precision': [], 'recall': [], 'f1': []})
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    
                path_parts = json_file.split('/')
                model_config = path_parts[-2]
                
                config_key = (
                    model_config,
                    data['hyperparameters']['few_shot'],
                    data['hyperparameters']['batch_size']
                )
                
                if config_key not in results_by_config:
                    results_by_config[config_key] = {
                        'few_shot': {'auc': [], 'acc': [], 'precision': [], 'recall': [], 'f1': []}
                    }
                
                if 'Ours_few' in data['results']:
                    results = data['results']['Ours_few']
                    metrics = {
                        'auc': results['Ours_best_few_auc'],
                        'acc': results['Ours_best_few_acc'],
                        'precision': results['Ours_best_few_precision'],
                        'recall': results['Ours_best_few_recall'],
                        'f1': results['Ours_best_few_f1']
                    }
                    for metric_name, value in metrics.items():
                        results_by_config[config_key]['few_shot'][metric_name].append(value)
                
                if data['hyperparameters']['few_shot'] == 4 and isinstance(data['results']['Ours'], dict):
                    results = data['results']['Ours']
                    full_key = (model_config, data['hyperparameters']['batch_size'])
                    metrics = {
                        'auc': results['Ours_best_full_auc'],
                        'acc': results['Ours_best_full_acc'],
                        'precision': results['Ours_best_full_precision'],
                        'recall': results['Ours_best_full_recall'],
                        'f1': results['Ours_best_full_f1']
                    }
                    for metric_name, value in metrics.items():
                        full_results[full_key][metric_name].append(value)
                            
            except Exception as e:
                print(f"파일 처리 오류 {json_file}: {str(e)}")
        
        # 기존 개별 CSV 파일 저장
        for config_key in results_by_config.keys():
            model_config, few_shot, batch_size = config_key
            
            # 변경: 데이터셋 이름_summary 폴더 생성
            summary_dir = os.path.join(directory_path, f"{dataset}_summary")
            model_dir = os.path.join(summary_dir, model_config)
            create_directory(model_dir)
            
            # 시드 정보 없이 간단한 파일명 사용
            output_file = os.path.join(model_dir, f'f{few_shot}_b{batch_size}.csv')
            
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Few-shot 결과
                writer.writerow(['Few-shot Results:'])
                for metric in ['auc', 'acc', 'precision', 'recall', 'f1']:
                    values = results_by_config[config_key]['few_shot'][metric]
                    if values:
                        mean, std = calculate_mean_std(values)
                        writer.writerow([f"{mean:.4f}({std:.4f})"])
                
                # Full dataset 결과
                writer.writerow([''])
                writer.writerow(['Full Dataset Results:'])
                full_key = (model_config, batch_size)
                if full_key in full_results:
                    for metric in ['auc', 'acc', 'precision', 'recall', 'f1']:
                        values = full_results[full_key][metric]
                        if values:
                            mean, std = calculate_mean_std(values)
                            writer.writerow([f"{mean:.4f}({std:.4f})"])
        
        # 🔥 새로 추가: 각 model_config별로 모든 설정을 하나로 합친 TSV 파일 생성
        create_combined_summary_by_model(results_by_config, full_results, dataset, directory_path)
        
        print(f"데이터셋 {dataset} 처리 완료!")
        print(f"총 {len(results_by_config)}개 설정 조합 처리됨")

def main():
    parser = argparse.ArgumentParser(description='Summarize DL results')
    parser.add_argument('--base_dir', type=str, #/home/eungyeop/LLM/tabular/ProtoLLM/experiments/source_to_source_tabular_embedding_new_bio-clinical-bert
                       default="/home/eungyeop/LLM/tabular/ProtoLLM/experiments/source_to_source_tabular_embedding_new_bio-clinical-bert",
                       help='Base directory containing the results')
    parser.add_argument('--datasets', nargs='+', type=str,
                       help='Specific datasets to process (e.g., --datasets adult diabetes)')
    parser.add_argument('--seeds', nargs='+', type=str,
                       help='Specific seeds to include in analysis (e.g., --seeds 42 123 456)')
    parser.add_argument('--best_seed', nargs='+', type=str,
                       help='Specific best seeds to include in analysis (alias for --seeds)')
    
    args = parser.parse_args()
    
    # --best_seed를 --seeds 매개변수로 처리
    selected_seeds = args.seeds if args.seeds else args.best_seed
    
    process_json_files(args.base_dir, args.datasets, selected_seeds)

if __name__ == "__main__":
    main()