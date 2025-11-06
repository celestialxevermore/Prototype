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


def process_json_files(b):
    """
    b 예시:
    /storage/.../args_seed:42/ngraphs-8_nnodes-8_gdim-128_nheads-4_...
    -> 이 폴더 내부의 f*.json 파일을 전부 읽고 summary 생성
    """

    json_files = glob.glob(os.path.join(b, "f*.json"))
    if not json_files:
        print(f"❌ JSON 파일을 찾을 수 없음: {b}")
        return

    print(f"📁 처리 중: {b}")
    results = defaultdict(lambda: defaultdict(list))
    full_results = defaultdict(list)
    batch_size = None

    for json_file in sorted(json_files):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            few_shot = data["hyperparameters"]["few_shot"]
            batch_size = data["hyperparameters"]["batch_size"]

            few_results = data["results"].get("Ours_few", {})
            metrics = [k.replace("Ours_best_few_", "") for k in few_results.keys() if k.startswith("Ours_best_few_")]

            for m in metrics:
                val = few_results.get(f"Ours_best_few_{m}")
                if val is not None:
                    results[m][few_shot].append(val)

            # full 결과 (few_shot == 4일 때만)
            if few_shot == 4 and isinstance(data["results"].get("Ours"), dict):
                full_data = data["results"]["Ours"]
                for m in metrics:
                    val = full_data.get(f"Ours_best_full_{m}")
                    if val is not None:
                        full_results[m].append(val)

        except Exception as e:
            print(f"❌ 파일 처리 오류: {json_file}\n   {e}")
            continue

    # summary 폴더 생성
    summary_dir = os.path.join(b, "summary")
    create_directory(summary_dir)
    combined_tsv = os.path.join(summary_dir, "combined_summary.tsv")

    # few-shot 정렬
    all_few_shots = sorted({k for metric_dict in results.values() for k in metric_dict.keys()})
    headers = [f"f{fs}_b{batch_size}" for fs in all_few_shots]
    if full_results:
        headers.append(f"full_b{batch_size}")

    # TSV 작성 (metric 이름 없이 수치만)
    with open(combined_tsv, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(headers)

        # metric 종류만큼 행 생성
        for metric_name, few_data in results.items():
            row = []
            for fs in all_few_shots:
                vals = few_data.get(fs, [])
                if vals:
                    mean, std = calculate_mean_std(vals)
                    row.append(f"{mean:.4f}({std:.4f})")
                else:
                    row.append("")
            if full_results and metric_name in full_results:
                vals = full_results[metric_name]
                if vals:
                    mean, std = calculate_mean_std(vals)
                    row.append(f"{mean:.4f}({std:.4f})")
                else:
                    row.append("")
            writer.writerow(row)

    print(f"✅ Summary TSV 생성 완료 → {combined_tsv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--b", type=str, required=True,
                        help="ngraphs-* 폴더 경로 (args_seed 밑 폴더 기준)")
    args = parser.parse_args()
    process_json_files(args.b)


if __name__ == "__main__":
    main()
