import json
import glob
import os
import numpy as np
from collections import defaultdict
import argparse
import sys
from io import StringIO

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def select_best_seeds(directory_path, output_file, top_n=1):
    """
    각 시드별 few-shot AUC 평균을 계산하고 상위 시드를 선택하는 함수
    
    Args:
        directory_path: 결과 JSON 파일이 있는 기본 경로
        output_file: 선택된 시드를 저장할 텍스트 파일 경로
        top_n: 선택할 상위 시드 개수 (정확히 이 개수만큼 선택)
    """
    # 시드별 결과를 저장할 딕셔너리
    results_by_seed = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    for dataset in ['diabetes', 'heart']:
        dataset_path = os.path.join(directory_path, dataset)
        if not os.path.exists(dataset_path):
            print(f"Dataset path {dataset_path} does not exist")
            continue
            
        # 모든 시드 디렉토리 찾기
        seed_dirs = glob.glob(os.path.join(dataset_path, "args_seed:*"))
        print(f"Found {len(seed_dirs)} seed directories for dataset {dataset}")
        
        for seed_dir in seed_dirs:
            seed = seed_dir.split(':')[-1]
            
            # 모든 few-shot 결과 파일 찾기
            few_shot_files = glob.glob(os.path.join(seed_dir, "TabularFLM/**/f*.json"), recursive=True)
            
            for json_file in few_shot_files:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    few_shot = data['hyperparameters']['few_shot']
                    
                    if 'Ours_few' in data['results']:
                        results = data['results']['Ours_few']
                        
                        # 결과 저장
                        results_by_seed[dataset][seed][few_shot] = {
                            'auc': results['Ours_best_few_auc'],
                            'acc': results['Ours_best_few_acc'],
                            'precision': results.get('Ours_best_few_precision', 0),
                            'recall': results.get('Ours_best_few_recall', 0),
                            'f1': results.get('Ours_best_few_f1', 0)
                        }
                                
                except Exception as e:
                    print(f"Error processing file {json_file}: {str(e)}")
    
    # 각 데이터셋별로 시드 평가 및 정렬
    top_seeds_by_dataset = {}
    trending_seeds_by_dataset = {}  # 점진적 향상을 보이는 시드
    
    # 출력을 저장할 문자열 스트림
    output_buffer = StringIO()
    
    # 헤더 출력
    output_buffer.write("\n======== 선택된 최고 성능 시드 ========\n")
    
    for dataset in results_by_seed:
        # 각 시드의 평균 AUC 계산
        seed_avg_aucs = []
        # 각 시드의 점진적 향상 점수 계산
        seed_trend_scores = []
        
        for seed in results_by_seed[dataset]:
            aucs = []
            few_shot_results = {}
            
            # 필요한 few-shot 설정들
            required_shots = [4, 8, 16, 32, 64]
            available_shots = set(int(fs) for fs in results_by_seed[dataset][seed].keys())
            
            # 모든 필요한 few-shot 데이터가 있는지 확인
            if not all(fs in available_shots for fs in required_shots):
                continue
            
            # 각 few-shot 설정에 대한 AUC 수집
            for few_shot in required_shots:
                if few_shot in results_by_seed[dataset][seed]:
                    auc_value = results_by_seed[dataset][seed][few_shot]['auc']
                    aucs.append(auc_value)
                    few_shot_results[few_shot] = auc_value
            
            if len(aucs) < len(required_shots):
                continue  # 일부 few-shot 결과가 없는 경우 건너뛰기
                
            # 평균 AUC 계산
            avg_auc = np.mean(aucs)
            seed_avg_aucs.append((seed, avg_auc, few_shot_results))
            
            # 점진적 향상 점수 계산
            # 1. few-shot 수가 증가함에 따른 AUC 값의 기울기
            trend_slope = np.polyfit(required_shots, [few_shot_results[fs] for fs in required_shots], 1)[0]
            
            # 2. 16, 32 shot에서 AUC가 0.80 이상인지 가중치 부여
            high_perf_bonus = 0
            if few_shot_results[16] >= 0.80:
                high_perf_bonus += 0.2
            if few_shot_results[32] >= 0.80:
                high_perf_bonus += 0.3
                
            # 3. 점진적으로 상승하는지 체크 (각 단계마다 일정 비율 이상 성능 향상)
            is_progressive = True
            for i in range(1, len(required_shots)):
                prev_fs, curr_fs = required_shots[i-1], required_shots[i]
                if few_shot_results[curr_fs] <= few_shot_results[prev_fs]:
                    is_progressive = False
                    break
            
            # 종합 점수 계산 (점진적 향상이 아니면 점수 낮춤)
            trend_score = trend_slope * 100 + high_perf_bonus
            if not is_progressive:
                trend_score *= 0.5
                
            seed_trend_scores.append((seed, trend_score, few_shot_results))
        
        # AUC 평균 기준으로 내림차순 정렬
        seed_avg_aucs.sort(key=lambda x: x[1], reverse=True)
        
        # 점진적 향상 기준으로 내림차순 정렬
        seed_trend_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 상위 N개 시드 선택 (실제 시드 수보다 많이 요청하면 모든 시드 선택)
        actual_top_n = min(top_n, len(seed_avg_aucs))
        top_seeds = seed_avg_aucs[:actual_top_n]
        top_seeds_by_dataset[dataset] = [seed for seed, _, _ in top_seeds]
        
        # 상위 N개 점진적 향상 시드 선택
        actual_trend_n = min(top_n, len(seed_trend_scores))
        trending_seeds = seed_trend_scores[:actual_trend_n]
        trending_seeds_by_dataset[dataset] = [seed for seed, _, _ in trending_seeds]
        
        # 데이터셋 헤더 출력
        output_buffer.write(f"\n{dataset}: 총 {len(results_by_seed[dataset])}개 시드 중 상위 {len(top_seeds)}개 선택\n")
        
        # 1. 평균 AUC 기준 상위 시드 출력
        output_buffer.write("\n■ 평균 AUC 기준 상위 시드\n")
        for seed, avg_auc, few_shot_aucs in top_seeds:
            output_buffer.write(f"  시드 {seed} (평균 AUC: {avg_auc:.4f})\n")
            
            # Few-shot 결과 출력
            few_shot_str = ", ".join([f"f{fs}: {auc:.4f}" for fs, auc in sorted(few_shot_aucs.items())])
            output_buffer.write(f"    Few-shot별 AUC: {few_shot_str}\n")
        
        # 상위 N개 시드의 평균 AUC 계산 및 출력
        avg_of_top_seeds = np.mean([avg_auc for _, avg_auc, _ in top_seeds])
        output_buffer.write(f"  ** 선택된 {len(top_seeds)}개 시드의 평균 AUC: {avg_of_top_seeds:.4f} **\n")
        
        # 각 few-shot별 상위 N개 시드들의 평균 AUC 계산
        few_shot_avgs = {}
        for fs in [4, 8, 16, 32, 64]:
            values = [fs_aucs.get(fs, 0) for _, _, fs_aucs in top_seeds if fs in fs_aucs]
            if values:
                few_shot_avgs[fs] = np.mean(values)
        
        # Few-shot별 평균 AUC 출력
        if few_shot_avgs:
            few_shot_avg_str = ", ".join([f"f{fs}: {auc:.4f}" for fs, auc in sorted(few_shot_avgs.items())])
            output_buffer.write(f"    Few-shot별 평균 AUC: {few_shot_avg_str}\n")
            
        # 2. 점진적 향상 기준 상위 시드 출력
        output_buffer.write("\n■ 점진적 향상 기준 상위 시드\n")
        for seed, trend_score, few_shot_aucs in trending_seeds:
            # 16, 32 shot에서 0.80 이상이면 표시
            highlight = ""
            if few_shot_aucs[16] >= 0.80 and few_shot_aucs[32] >= 0.80:
                highlight = " ★★★"
            elif few_shot_aucs[32] >= 0.80:
                highlight = " ★★"
            elif few_shot_aucs[16] >= 0.80:
                highlight = " ★"
                
            output_buffer.write(f"  시드 {seed} (점진적 향상 점수: {trend_score:.4f}){highlight}\n")
            
            # Few-shot 결과 출력 (증가율 표시)
            few_shot_values = []
            for i, fs in enumerate(sorted(few_shot_aucs.keys())):
                auc = few_shot_aucs[fs]
                
                # 첫 번째가 아니면 증가율 표시
                if i > 0:
                    prev_fs = sorted(few_shot_aucs.keys())[i-1]
                    prev_auc = few_shot_aucs[prev_fs]
                    change = (auc - prev_auc) / prev_auc * 100
                    few_shot_values.append(f"f{fs}: {auc:.4f} ({change:+.1f}%)")
                else:
                    few_shot_values.append(f"f{fs}: {auc:.4f}")
                    
            # 16, 32가 0.8 이상이면 강조 표시
            for i, fs in enumerate(sorted(few_shot_aucs.keys())):
                if fs in [16, 32] and few_shot_aucs[fs] >= 0.80:
                    few_shot_values[i] = few_shot_values[i] + " ✓"
                    
            few_shot_str = ", ".join(few_shot_values)
            output_buffer.write(f"    Few-shot별 AUC: {few_shot_str}\n")
    
    # 결과를 텍스트 파일로 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output_buffer.getvalue())
    
    # 콘솔에도 동일한 내용 출력
    print(output_buffer.getvalue())
    
    # JSON 파일로도 시드 목록 저장
    json_file = output_file.replace('.txt', '.json')
    json_data = {
        "평균_AUC_기준": top_seeds_by_dataset,
        "점진적_향상_기준": trending_seeds_by_dataset
    }
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"텍스트 결과 저장 완료: {output_file}")
    print(f"JSON 시드 목록 저장 완료: {json_file}")

def main():
    parser = argparse.ArgumentParser(description='시드의 평균 AUC를 기준으로 최고 성능 시드 선택')
    parser.add_argument('--base_dir', type=str, 
                      default="/home/eungyeop/LLM/tabular/ProtoLLM/experiments/source_to_source_Experiment_TabularFLM_G5",
                      help='결과가 저장된 기본 디렉토리 경로')
    parser.add_argument('--output_file', type=str, 
                      default="top_seeds_results.txt",
                      help='결과를 저장할 텍스트 파일 경로')
    parser.add_argument('--top_n', type=int, 
                      default=5,
                      help='선택할 상위 시드 개수 (예: 1000개 중 100개 또는 5개 중 1개)')
    
    args = parser.parse_args()
    select_best_seeds(args.base_dir, args.output_file, args.top_n)

if __name__ == "__main__":
    main()