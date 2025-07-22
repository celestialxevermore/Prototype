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

def process_json_files(directory_path, selected_datasets=None, selected_seeds=None):
    if selected_datasets:
        print(f"처리할 데이터셋: {selected_datasets}")
    
    if selected_seeds:
        selected_seeds = [str(seed) for seed in selected_seeds]  # 문자열로 변환
    
    datasets_to_process = selected_datasets if selected_datasets else ['heart']
    
    for dataset in datasets_to_process:
        # 🔥 구조 변경: del_exp별로 그룹화
        results_by_del_exp = defaultdict(lambda: defaultdict(lambda: {
            'few_shot': {'auc': [], 'acc': [], 'precision': [], 'recall': [], 'f1': []},
            'full': {'auc': [], 'acc': [], 'precision': [], 'recall': [], 'f1': []}
        }))
        
        dataset_path = os.path.join(directory_path, dataset)
        
        if not os.path.exists(dataset_path):
            print(f"경로를 찾을 수 없음: {dataset_path}")
            continue
        
        json_files = []
        
        # 선택된 시드가 있으면 해당 시드 경로만 처리
        if selected_seeds:
            for seed in selected_seeds:
                seed_path = os.path.join(dataset_path, f"args_seed:{seed}")
                if os.path.exists(seed_path):
                    seed_json_pattern = os.path.join(seed_path, "TabularFLM/Embed:*_Edge:*_A:*/f*.json")
                    seed_json_files = glob.glob(seed_json_pattern, recursive=True)
                    if seed_json_files:
                        json_files.extend(seed_json_files)
                else:
                    print(f"시드 {seed}의 경로가 존재하지 않음: {seed_path}")
        else:
            # 모든 시드 처리 (기존 방식)
            json_pattern = os.path.join(dataset_path, "args_seed:*/TabularFLM/Embed:*_Edge:*_A:*/f*.json")
            json_files = glob.glob(json_pattern, recursive=True)
        
        if not json_files:
            print(f"데이터셋 {dataset}에서 JSON 파일을 찾을 수 없음")
            continue
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    
                path_parts = json_file.split('/')
                model_config = path_parts[-2]  # Embed:carte_Edge:mlp_A:gat_v1
                
                # 🔥 del_exp 정보 추출
                del_exp = data.get('del_exp', 'unknown')
                
                few_shot = data['hyperparameters']['few_shot']
                
                # 🔥 del_exp와 model_config 조합으로 키 생성
                key = (model_config, del_exp)
                
                # Few-shot 결과 저장
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
                        results_by_del_exp[key][few_shot]['few_shot'][metric_name].append(value)
                
                # Full dataset 결과 저장 (few_shot=4일 때만)
                if few_shot == 4 and isinstance(data['results']['Ours'], dict):
                    results = data['results']['Ours']
                    metrics = {
                        'auc': results['Ours_best_full_auc'],
                        'acc': results['Ours_best_full_acc'],
                        'precision': results['Ours_best_full_precision'],
                        'recall': results['Ours_best_full_recall'],
                        'f1': results['Ours_best_full_f1']
                    }
                    for metric_name, value in metrics.items():
                        results_by_del_exp[key]['full']['full'][metric_name].append(value)
                            
            except Exception as e:
                print(f"파일 처리 오류 {json_file}: {str(e)}")
        
        # 🔥 결과 저장: del_exp별로 각 few-shot마다 별도 CSV 파일
        for key, few_shot_data in results_by_del_exp.items():
            model_config, del_exp = key
            
            summary_dir = os.path.join(directory_path, f"{dataset}_summary_experiments")
            model_dir = os.path.join(summary_dir, model_config, f"{del_exp}")
            create_directory(model_dir)
            
            # 실험 설명을 파일에 저장
            desc_file = os.path.join(model_dir, 'experiment_description.txt')
            exp_descriptions = {
                'exp1': 'Baseline: 아무것도 제거하지 않았을 때',
                'exp2': '중요한 소수의 변수를 제거했을 때',
                'exp3': '중요한 소수의 변수와 안중요한 변수 몇 개를 제거했을 때',
                'exp4': '안중요한 소수 변수만 제거했을 때',
                'exp5': '중요한 변수만 남기고 대다수 변수를 모두 제거했을 때'
            }
            with open(desc_file, 'w') as f:
                f.write(f"{del_exp}: {exp_descriptions.get(del_exp, 'Unknown experiment')}\n")
            
            # 🔥 각 few-shot별로 별도 파일 생성 (원래 방식)
            for few_shot in few_shot_data.keys():
                if few_shot != 'full':  # 'full' 키 제외
                    output_file = os.path.join(model_dir, f'f{few_shot}_b32.csv')
                    
                    with open(output_file, 'w', newline='') as f:
                        writer = csv.writer(f)
                        
                        # Few-shot 결과 (원래 형태: 세로로 나열)
                        writer.writerow(['Few-shot Results:'])
                        for metric in ['auc', 'acc', 'precision', 'recall', 'f1']:
                            values = few_shot_data[few_shot]['few_shot'][metric]
                            if values:
                                mean, std = calculate_mean_std(values)
                                writer.writerow([f"{mean:.4f}({std:.4f})"])
                        
                        # Full dataset 결과 (few_shot=4일 때만)
                        if few_shot == 4:
                            writer.writerow([''])
                            writer.writerow(['Full Dataset Results:'])
                            if 'full' in few_shot_data:
                                for metric in ['auc', 'acc', 'precision', 'recall', 'f1']:
                                    values = few_shot_data['full']['full'][metric]
                                    if values:
                                        mean, std = calculate_mean_std(values)
                                        writer.writerow([f"{mean:.4f}({std:.4f})"])
            
            # 🔥 모든 few-shot을 한눈에 보는 요약 테이블도 생성
            summary_output_file = os.path.join(model_dir, 'summary_all_fewshots.csv')
            with open(summary_output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # 🔥 헤더 추가 (실험 정보 포함)
                writer.writerow([f"{del_exp}: {exp_descriptions.get(del_exp, 'Unknown experiment')}"])
                writer.writerow([''])
                
                few_shot_keys = sorted([k for k in few_shot_data.keys() if isinstance(k, int)])
                header = [f'f{fs}' for fs in few_shot_keys] + ['full']
                writer.writerow(['Metric'] + header)
                
                # 🔥 각 metric별로 행 생성
                for metric in ['auc', 'acc', 'precision', 'recall', 'f1']:
                    row = [metric.upper()]
                    
                    # Few-shot 결과들
                    for few_shot in few_shot_keys:
                        values = few_shot_data[few_shot]['few_shot'][metric]
                        if values:
                            mean, std = calculate_mean_std(values)
                            row.append(f"{mean:.4f}({std:.4f})")
                        else:
                            row.append("N/A")
                    
                    # Full dataset 결과
                    if 'full' in few_shot_data:
                        values = few_shot_data['full']['full'][metric]
                        if values:
                            mean, std = calculate_mean_std(values)
                            row.append(f"{mean:.4f}({std:.4f})")
                        else:
                            row.append("N/A")
                    else:
                        row.append("N/A")
                    
                    writer.writerow(row)
        
        print(f"데이터셋 {dataset} 처리 완료!")
        print(f"총 {len(results_by_del_exp)}개 실험 조합 생성됨")

def main():
    parser = argparse.ArgumentParser(description='Summarize experiment results by del_exp types')
    parser.add_argument('--base_dir', type=str, 
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