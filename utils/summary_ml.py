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

    combined_file = os.path.join(summary_dir, 'all_models_combined.tsv')
    model_order = ['rf', 'lr', 'xgb', 'mlp', 'cat']
    METRICS = ['auc', 'acc', 'precision', 'recall', 'f1', 'auprc']

    with open(combined_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')

        for model in model_order:
            if model not in results_by_dataset[dataset]:
                continue

            model_data = results_by_dataset[dataset][model]
            few_shot_keys = sorted(model_data.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))

            for metric in METRICS:
                row = []

                for few_shot in few_shot_keys:
                    values = model_data[few_shot].get(metric, [])
                    if values:
                        mean, std = calculate_mean_std(values)
                        row.append(f"{mean:.4f}({std:.4f})")
                    else:
                        row.append("")

                if model in full_results_by_dataset[dataset]:
                    values = full_results_by_dataset[dataset][model].get(metric, [])
                    if values:
                        mean, std = calculate_mean_std(values)
                        row.append(f"{mean:.4f}({std:.4f})")
                    else:
                        row.append("")
                else:
                    row.append("")

                writer.writerow(row)

def create_ml_numbers_only_tsv_by_model(results_by_dataset, full_results_by_dataset, base_directory, dataset, model, METRICS):
    """
    ✅ DL summary 스타일:
    - 헤더 없음
    - 메트릭 이름 없음
    - 값만 (행=METRICS 순서 고정, 열=few-shot 오름차순 + full)
    """
    model_dir = os.path.join(base_directory, dataset, "summary", model)
    create_directory(model_dir)

    out_path = os.path.join(model_dir, "all_configs_combined.tsv")

    model_data = results_by_dataset[dataset][model]
    few_shot_keys = sorted(model_data.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))

    # few-shot 결과가 하나도 없으면 파일 생성 스킵
    if len(few_shot_keys) == 0:
        return

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")

        for metric in METRICS:
            row = []

            # few-shot columns
            for few_shot in few_shot_keys:
                values = model_data[few_shot].get(metric, [])
                if values:
                    mean, std = calculate_mean_std(values)
                    row.append(f"{mean:.4f}({std:.4f})")
                else:
                    row.append("")

            # full column (맨 마지막)
            full_vals = full_results_by_dataset[dataset][model].get(metric, [])
            if full_vals:
                mean, std = calculate_mean_std(full_vals)
                row.append(f"{mean:.4f}({std:.4f})")
            else:
                row.append("")

            writer.writerow(row)

def process_ml_results(base_directory):
    results_by_dataset = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    full_results_by_dataset = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    METRICS = ['auc', 'acc', 'precision', 'recall', 'f1', 'auprc']
    MODEL_LIST = ['rf', 'lr', 'xgb', 'mlp', 'cat']

    dataset_list = [
        'heart', 'diabetes', 'heart_statlog', 'hungarian', 'switzerland',
        'Heart_disease_statlog', 'Cardiovascular_Disease_Dataset',
        'Heart_disease_statlog', 'Medicaldataset', 'heart_failure_clinical_records',
        'cardio_SAheart', 'Erbil_Cardiovascular_Health_Dataset'
    ]
    # ✅ 중복 제거(값 누적 2번 되는 거 방지) - 순서 유지
    dataset_list = list(dict.fromkeys(dataset_list))

    for dataset in dataset_list:
        dataset_path = os.path.join(base_directory, dataset)
        if not os.path.exists(dataset_path):
            continue

        json_pattern = os.path.join(dataset_path, "**/f*.json")
        json_files = glob.glob(json_pattern, recursive=True)

        # dataset 안에 json이 없으면 스킵
        if not json_files:
            continue

        batch_size_last = None

        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)

            few_shot = data['hyperparameters']['few_shot']
            batch_size = data['hyperparameters']['batch_size']
            batch_size_last = batch_size

            for model in MODEL_LIST:
                if model in data['results']['Best_results']['few']:
                    metrics_few = data['results']['Best_results']['few'][model]
                    for metric_name, value in metrics_few.items():
                        clean_metric = metric_name.replace(f'{model}_best_few_', '')
                        results_by_dataset[dataset][model][few_shot][clean_metric].append(value)

                    metrics_full = data['results']['Best_results']['full'][model]
                    for metric_name, value in metrics_full.items():
                        clean_metric = metric_name.replace(f'{model}_best_full_', '')
                        full_results_by_dataset[dataset][model][clean_metric].append(value)

        # 결과 저장 (기존 CSV들)
        for model in MODEL_LIST:
            model_dir = os.path.join(base_directory, dataset, "summary", model)
            create_directory(model_dir)

            # Few-shot 결과 저장
            for few_shot in sorted(results_by_dataset[dataset][model].keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
                few_output_file = os.path.join(model_dir, f'f{few_shot}_b{batch_size_last}.csv')

                with open(few_output_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Model Type', model])
                    writer.writerow(['Few Shot', few_shot])
                    writer.writerow(['Batch Size', batch_size_last])
                    writer.writerow([''])

                    for metric in METRICS:
                        values = results_by_dataset[dataset][model][few_shot].get(metric, [])
                        if values:
                            mean, std = calculate_mean_std(values)
                            writer.writerow([f"{mean:.4f}({std:.4f})"])
                        else:
                            writer.writerow([""])

            # Full dataset 결과 저장
            full_output_file = os.path.join(model_dir, 'full_results.csv')
            with open(full_output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Model Type', model])
                writer.writerow(['Full Dataset Results'])
                writer.writerow(['Batch Size', batch_size_last])
                writer.writerow([''])

                for metric in METRICS:
                    values = full_results_by_dataset[dataset][model].get(metric, [])
                    if values:
                        mean, std = calculate_mean_std(values)
                        writer.writerow([f"{mean:.4f}({std:.4f})"])
                    else:
                        writer.writerow([""])

        # ✅ (추가) 모델별 숫자-only TSV 생성 (네가 원하는 “복붙용”)
        for model in MODEL_LIST:
            create_ml_numbers_only_tsv_by_model(
                results_by_dataset, full_results_by_dataset, base_directory, dataset, model, METRICS
            )

        # 기존 “전체 모델 합친 TSV”도 유지
        create_ml_combined_summary(results_by_dataset, full_results_by_dataset, base_directory, dataset)
        print(f"Combined TSV saved for dataset: {dataset}")

def main():
    parser = argparse.ArgumentParser(description='Summarize ML results')
    parser.add_argument('--base_dir', type=str,
                        default='/home/eungyeop/LLM/tabular/ProtoLLM/experiments/ml_baselines_ML_results_20250322',
                        help='Base directory containing the results')
    args = parser.parse_args()
    process_ml_results(args.base_dir)

if __name__ == "__main__":
    main()
