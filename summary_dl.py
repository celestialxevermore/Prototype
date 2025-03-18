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
    results_by_config = {}
    
    for dataset in ['adult', 'diabetes', 'heart']:
        results_by_config[dataset] = {}
        dataset_path = os.path.join(directory_path, dataset)
        if not os.path.exists(dataset_path):
            continue
            
        json_pattern = os.path.join(dataset_path, "args_seed:*/TabularFLM/A:*_L:*_E:*_M:*/f*.json")
        json_files = glob.glob(json_pattern, recursive=True)
        
        if not json_files:
            print(f"No JSON files found for dataset {dataset}")
            continue
        
        # Full dataset 결과를 저장할 별도의 딕셔너리
        full_results = defaultdict(lambda: {'auc': [], 'acc': [], 'precision': [], 'recall': [], 'f1': []})
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    
                path_parts = json_file.split('/')
                model_config = path_parts[-2]
                seed = path_parts[-4].split(':')[1]
                
                config_parts = model_config.split('_')
                aggr_type = config_parts[0].split(':')[1]
                label_type = config_parts[1].split(':')[1]
                enc_type = config_parts[2].split(':')[1]
                meta_type = config_parts[3].split(':')[1]
                
                config_key = (
                    model_config,
                    data['hyperparameters']['few_shot'],
                    data['hyperparameters']['batch_size']
                )
                
                if config_key not in results_by_config[dataset]:
                    results_by_config[dataset][config_key] = {
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
                        results_by_config[dataset][config_key]['few_shot'][metric_name].append(value)
                
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
                print(f"Error processing file {json_file}: {str(e)}")
        
        for config_key in results_by_config[dataset].keys():
            model_config, few_shot, batch_size = config_key
            
            model_dir = os.path.join(directory_path, dataset, "summary", model_config)
            create_directory(model_dir)
            
            output_file = os.path.join(model_dir, f'f{few_shot}_b{batch_size}.csv')
            
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Few-shot 결과
                writer.writerow(['Few-shot Results:'])
                for metric in ['auc', 'acc', 'precision', 'recall', 'f1']:
                    values = results_by_config[dataset][config_key]['few_shot'][metric]
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
            
            print(f"Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Summarize DL results')
    parser.add_argument('--base_dir', type=str, 
                       default="/home/eungyeop/LLM/tabular/ProtoLLM/experiments/source_to_source_Experiment_TabularFLM4",
                       help='Base directory containing the results')
    
    args = parser.parse_args()
    process_json_files(args.base_dir)

if __name__ == "__main__":
    main()