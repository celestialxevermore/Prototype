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

def process_json_files(directory_path):
    # 실험 설정별로 결과를 모을 딕셔너리
    results_by_config = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    
    # JSON 파일 찾기
    for dataset in ['adult', 'diabetes', 'heart']:
        dataset_path = os.path.join(directory_path, dataset)
        if not os.path.exists(dataset_path):
            continue
            
        json_pattern = os.path.join(dataset_path, "**/GAT_edge*/star_*/f*.json")
        json_files = glob.glob(json_pattern, recursive=True)
        
        if not json_files:
            print(f"No JSON files found for dataset {dataset}")
            continue
        
        # 먼저 모든 결과를 수집
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    
                    # 파일 경로에서 필요한 정보 추출
                    path_parts = json_file.split('/')
                    model_type = path_parts[-3]  # GAT_edge
                    graph_center_type = path_parts[-2]  # star_D_CM
                    
                    # 실험 설정을 키로 사용
                    config_key = (
                        model_type,
                        graph_center_type,
                        data['hyperparameters']['few_shot'],
                        data['hyperparameters']['batch_size']
                    )
                    
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
                            results_by_config[dataset][config_key][metric_name].append(value)
                            
            except Exception as e:
                print(f"Error processing file {json_file}: {str(e)}")
        
        # 결과를 새로운 디렉토리 구조로 저장
        for config_key in results_by_config[dataset].keys():
            model_type, graph_center_type, few_shot, batch_size = config_key
            
            # 디렉토리 구조 생성
            model_dir = os.path.join(directory_path, dataset, "summary", model_type, graph_center_type)
            create_directory(model_dir)
            
            # CSV 파일 생성
            output_file = os.path.join(model_dir, f'f{few_shot}_b{batch_size}.csv')
            
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # 헤더 작성
                writer.writerow(['Model Type', model_type])
                writer.writerow(['Graph Type', graph_center_type.split('_')[0]])
                writer.writerow(['FD Type', graph_center_type.split('_')[1]])
                writer.writerow(['Center Type', '_'.join(graph_center_type.split('_')[2:])])
                writer.writerow(['Few Shot', few_shot])
                writer.writerow(['Batch Size', batch_size])
                writer.writerow([''])
                
                # 메트릭 값 작성
                for metric in ['auc', 'acc', 'precision', 'recall', 'f1']:
                    values = results_by_config[dataset][config_key][metric]
                    mean, std = calculate_mean_std(values)
                    writer.writerow([f"{mean:.4f}({std:.4f})"])
            
            print(f"Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Summarize DL results')
    parser.add_argument('--base_dir', type=str, default="/home/eungyeop/LLM/tabular/ProtoLLM/experiments/source_to_source_SEEDS/adult",
                      help='Base directory containing the results')
    
    args = parser.parse_args()
    process_json_files(args.base_dir)

if __name__ == "__main__":
    main()