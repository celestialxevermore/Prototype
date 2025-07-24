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

def determine_scenario_from_del_feature(del_feature, important_features, all_features):
    """
    삭제된 변수 리스트를 보고 시나리오를 역추적하는 함수
    """
    if not del_feature:  # 삭제된 변수가 없으면
        return 0, "Baseline (No feature removal)"
    
    del_set = set(del_feature)
    important_set = set(important_features) if important_features else set()
    unimportant_features = [f for f in all_features if f not in important_set] if all_features else []
    unimportant_set = set(unimportant_features)
    
    # 삭제된 변수 중 중요한 변수와 비중요한 변수의 개수 계산
    deleted_important = del_set & important_set
    deleted_unimportant = del_set & unimportant_set
    
    # 시나리오 1: 중요한 3개 제거
    if len(deleted_important) == len(important_features) and len(deleted_unimportant) == 0:
        return 1, "Remove top-3 important features"
    
    # 시나리오 2: 중요한 3개 + 비중요한 1개 제거
    elif len(deleted_important) == len(important_features) and len(deleted_unimportant) == 1:
        return 2, "Remove top-3 important + 1 unimportant features"
    
    # 시나리오 3: 비중요한 3개 제거
    elif len(deleted_important) == 0 and len(deleted_unimportant) <= 3:
        return 3, "Remove bottom-3 unimportant features"
    
    # 시나리오 4: 중요한 3개만 유지 (나머지 모두 제거)
    elif len(deleted_important) == 0 and len(deleted_unimportant) == len(unimportant_features):
        return 4, "Keep only top-3 important features"
    
    # 알 수 없는 패턴
    else:
        return -1, f"Unknown pattern (del: {len(del_feature)})"

def extract_important_features_from_json(data):
    """
    JSON 파일에서 important_features와 all_features 정보 추출
    """
    # JSON에서 important_features 정보 찾기
    important_features = data.get('important_features', [])
    all_features = data.get('all_features', [])
    
    # 만약 위 키가 없다면 다른 경로에서 찾기 시도
    if not important_features and 'ablation_info' in data:
        important_features = data['ablation_info'].get('important_features', [])
        all_features = data['ablation_info'].get('all_features', [])
    
    return important_features, all_features

def create_combined_summary_by_model(results_by_scenario, dataset, base_dir):
    """각 model_config별로 모든 시나리오의 모든 메트릭을 하나의 표로 생성"""
    
    # model_config별로 그룹화
    results_by_model = defaultdict(dict)
    
    for key, few_shot_data in results_by_scenario.items():
        model_config, scenario_id, scenario_desc = key
        results_by_model[model_config][(scenario_id, scenario_desc)] = few_shot_data
    
    # 각 model_config별로 TSV 파일 생성
    for model_config, scenario_data in results_by_model.items():
        # 시나리오별로 정렬
        sorted_scenarios = sorted(scenario_data.items(), key=lambda x: x[0][0])  # scenario_id로 정렬
        
        summary_dir = os.path.join(base_dir, f"{dataset}_summary_scenarios", model_config)
        create_directory(summary_dir)
        
        # 각 model_config별 전체 결과를 담을 TSV 파일
        combined_file = os.path.join(summary_dir, 'all_scenarios_combined.tsv')
        
        with open(combined_file, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            
            # 각 시나리오의 각 메트릭별로 행 생성
            for (scenario_id, scenario_desc), few_shot_data in sorted_scenarios:
                few_shot_keys = sorted([k for k in few_shot_data.keys() if isinstance(k, int)])
                
                # 5개 메트릭 각각에 대해 행 생성
                for metric in ['auc', 'acc', 'precision', 'recall', 'f1']:
                    row = []
                    
                    # Few-shot 결과들 (4, 8, 16, 32, 64)
                    for few_shot in few_shot_keys:
                        values = few_shot_data[few_shot]['few_shot'][metric]
                        if values:
                            mean, std = calculate_mean_std(values)
                            row.append(f"{mean:.4f}({std:.4f})")
                        else:
                            row.append("")
                    
                    # Full dataset 결과
                    if 'full' in few_shot_data:
                        values = few_shot_data['full']['full'][metric]
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
        # 🔥 구조 변경: 시나리오별로 그룹화
        results_by_scenario = defaultdict(lambda: defaultdict(lambda: {
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
                    seed_json_pattern = os.path.join(seed_path, "TabularFLM/Embed:*_Edge:*_A:*/*scenario*.json")
                    seed_json_files = glob.glob(seed_json_pattern, recursive=True)
                    if seed_json_files:
                        json_files.extend(seed_json_files)
                else:
                    print(f"시드 {seed}의 경로가 존재하지 않음: {seed_path}")
        else:
            # 모든 시드 처리 (기존 방식)
            json_pattern = os.path.join(dataset_path, "args_seed:*/TabularFLM/Embed:*_Edge:*_A:*/*scenario*.json")
            json_files = glob.glob(json_pattern, recursive=True)
        
        if not json_files:
            print(f"데이터셋 {dataset}에서 시나리오 JSON 파일을 찾을 수 없음")
            continue
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    
                path_parts = json_file.split('/')
                model_config = path_parts[-2]  # Embed:carte_Edge:mlp_A:gat_v1
                
                # 🔥 JSON에서 시나리오 정보 직접 추출
                scenario_info = data.get('scenario_info', {})
                scenario_id = scenario_info.get('scenario_id', 'unknown')
                scenario_desc = scenario_info.get('scenario_description', 'Unknown scenario')
                
                few_shot = data['hyperparameters']['few_shot']
                
                # 🔥 시나리오와 model_config 조합으로 키 생성
                key = (model_config, scenario_id, scenario_desc)
                
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
                        results_by_scenario[key][few_shot]['few_shot'][metric_name].append(value)
                
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
                        results_by_scenario[key]['full']['full'][metric_name].append(value)
                            
            except Exception as e:
                print(f"파일 처리 오류 {json_file}: {str(e)}")
        
        # 🔥 결과 저장: 시나리오별로 각 few-shot마다 별도 CSV 파일
        for key, few_shot_data in results_by_scenario.items():
            model_config, scenario_id, scenario_desc = key
            
            summary_dir = os.path.join(directory_path, f"{dataset}_summary_scenarios")
            model_dir = os.path.join(summary_dir, model_config, f"scenario_{scenario_id}")
            create_directory(model_dir)
            
            # 시나리오 설명을 파일에 저장
            desc_file = os.path.join(model_dir, 'scenario_description.txt')
            with open(desc_file, 'w') as f:
                f.write(f"Scenario {scenario_id}: {scenario_desc}\n")
            
            # 🔥 각 few-shot별로 별도 파일 생성
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
            
            # 🔥 개별 시나리오별 요약 테이블 생성 (TSV, 숫자만)
            summary_output_file = os.path.join(model_dir, 'summary_all_fewshots.tsv')
            with open(summary_output_file, 'w', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                
                few_shot_keys = sorted([k for k in few_shot_data.keys() if isinstance(k, int)])
                
                # 🔥 숫자만 나오는 표 생성 (헤더와 메트릭 이름 모두 제거)
                for metric in ['auc', 'acc', 'precision', 'recall', 'f1']:
                    row = []  # 메트릭 이름 제거
                    
                    # Few-shot 결과들
                    for few_shot in few_shot_keys:
                        values = few_shot_data[few_shot]['few_shot'][metric]
                        if values:
                            mean, std = calculate_mean_std(values)
                            row.append(f"{mean:.4f}({std:.4f})")
                        else:
                            row.append("")
                    
                    # Full dataset 결과
                    if 'full' in few_shot_data:
                        values = few_shot_data['full']['full'][metric]
                        if values:
                            mean, std = calculate_mean_std(values)
                            row.append(f"{mean:.4f}({std:.4f})")
                        else:
                            row.append("")
                    else:
                        row.append("")
                    
                    writer.writerow(row)
        
        # 🔥 각 model_config별로 모든 시나리오를 하나로 합친 TSV 파일 생성
        create_combined_summary_by_model(results_by_scenario, dataset, directory_path)
        
        print(f"데이터셋 {dataset} 처리 완료!")
        print(f"총 {len(results_by_scenario)}개 시나리오 조합 생성됨")

def main():
    parser = argparse.ArgumentParser(description='Summarize Ablation Study results by scenarios')
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