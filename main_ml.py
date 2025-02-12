import os
import time
import argparse
import numpy as np
from utils.util import setup_logger, format_time, fix_seed
from utils.util import prepare_ml_results, save_ml_results
from dataset.data_dataloaders import prepare_tabular_dataloaders, get_few_shot_tabular_samples
from models.XGBoost import xgboost_benchmark
from models.LogReg import logistic_regression_benchmark
from models.MLP import mlp_benchmark
from models.CatBoost import catboost_benchmark
from models.RF import random_forest_benchmark
import psutil 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
p = psutil.Process()

p.cpu_affinity(range(1, 80))
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

logger = setup_logger()

def get_args():
    parser = argparse.ArgumentParser(description='ML Baselines For Tabular Task')

    parser.add_argument('--random_seed', type=int, default=42, help='random_seed')
    parser.add_argument('--dataset_seed', type=int, default=4)
    
    parser.add_argument('--source_dataset_name', type=str, default='cleveland', 
                        choices=['adult','bank','blood','car','communities','credit-g',
                                'diabetes','heart','myocardial','cleveland', 
                                'heart_statlog','hungarian','switzerland'])
    parser.add_argument('--few_shot', type=int, default=4, help='the number of shot')
    parser.add_argument('--table_path', type=str, default="/storage/personal/eungyeop/dataset/table")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hidden_dim', type = int, default = 128)
    parser.add_argument('--dropout_rate', type=float, default=0.3)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--train_epochs', type=int, default=200)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--base_dir', type=str, required=True)
    parser.add_argument('--baseline', nargs='*', 
                       default=['lr', 'xgb', 'mlp', 'cat', 'rf'],
                       choices=['lr', 'xgb', 'mlp', 'cat', 'rf'],
                       help='List of baselines to use')
    parser.add_argument('--des', type=str, help='experimental memo')
    
    args = parser.parse_args()
    return args

def main():
    start_time = time.time()
    args = get_args()
    fix_seed(args.random_seed)
    
    logger.info(f"Starting experiment with dataset: {args.source_dataset_name}")
    logger.info(f"Preparing Tabular datasets...")

    # 데이터셋 준비
    (X_train_full, X_val_full, X_test_full, 
     y_train_full, y_val_full, y_test_full), _ = prepare_tabular_dataloaders(
        args, args.source_dataset_name, args.random_seed
    )

    X_train_few, y_train_few = get_few_shot_tabular_samples(X_train_full, y_train_full, args)
    X_val_few, y_val_few = X_val_full, y_val_full
    X_test_few, y_test_few = X_test_full, y_test_full
    
    logger.info(f"Datasets prepared, source dataset names : {args.source_dataset_name}")
    
    # 이진 분류 여부 확인
    is_binary = (len(np.unique(y_train_full)) == 2)
    
    full_baseline_results = {}
    few_baseline_results = {}

    # baseline 반복
    for baseline in args.baseline:
        if baseline == "rf":
            # --------------------------------------------------------
            # 1) Random Forest만을 위한 데이터 사본 생성
            # --------------------------------------------------------
            X_train_rf = X_train_full.copy()
            X_val_rf = X_val_full.copy()
            X_test_rf = X_test_full.copy()

            # 범주형 열을 선택
            categorical_columns = X_train_rf.select_dtypes(include=['object', 'category']).columns

            # Label Encoding (RF 전용)
            for col in categorical_columns:
                le = LabelEncoder()
                X_train_rf[col] = le.fit_transform(X_train_rf[col])
                X_val_rf[col] = le.transform(X_val_rf[col])
                X_test_rf[col] = le.transform(X_test_rf[col])
            
            # Random Forest 학습 및 평가
            full_baseline_results[baseline] = random_forest_benchmark(
                args,
                X_train_rf, X_val_rf, X_test_rf, 
                y_train_full, y_val_full, y_test_full,
                is_binary=is_binary
            )
            few_baseline_results[baseline] = random_forest_benchmark(
                args,
                X_train_few.copy(), X_val_few.copy(), X_test_few.copy(), 
                y_train_few, y_val_few, y_test_few,
                is_binary=is_binary
            )

        elif baseline == "lr":
            # Logistic Regression
            full_baseline_results[baseline] = logistic_regression_benchmark(
                args,
                X_train_full, X_val_full, X_test_full,
                y_train_full, y_val_full, y_test_full,
                is_binary=is_binary,
            )
            few_baseline_results[baseline] = logistic_regression_benchmark(
                args,
                X_train_few, X_val_few, X_test_few,
                y_train_few, y_val_few, y_test_few,
                is_binary=is_binary,
            )

        elif baseline == "xgb":
            # XGBoost
            full_baseline_results[baseline] = xgboost_benchmark(
                args,
                X_train_full, X_val_full, X_test_full,
                y_train_full, y_val_full, y_test_full,
                is_binary=is_binary,
            )
            few_baseline_results[baseline] = xgboost_benchmark(
                args,
                X_train_few, X_val_few, X_test_few,
                y_train_few, y_val_few, y_test_few,
                is_binary=is_binary,
            )

        elif baseline == "mlp":
            # MLP
            full_baseline_results[baseline] = mlp_benchmark(
                args,
                X_train_full, X_val_full, X_test_full,
                y_train_full, y_val_full, y_test_full,
                is_binary=is_binary,
            )
            few_baseline_results[baseline] = mlp_benchmark(
                args,
                X_train_few, X_val_few, X_test_few,
                y_train_few, y_val_few, y_test_few,
                is_binary=is_binary,
            )

        elif baseline == "cat":
            # CatBoost
            full_baseline_results[baseline] = catboost_benchmark(
                args,
                X_train_full, X_val_full, X_test_full,
                y_train_full, y_val_full, y_test_full,
                is_binary=is_binary,
            )
            few_baseline_results[baseline] = catboost_benchmark(
                args,
                X_train_few, X_val_few, X_test_few,
                y_train_few, y_val_few, y_test_few,
                is_binary=is_binary,
            )

        else:
            logger.warning(f"Invalid baseline specified: {baseline}. Skipping.")

    # 각 모델 결과 정리
    results = prepare_ml_results(args, full_baseline_results, few_baseline_results)

    # 결과 저장
    save_ml_results(args, results)
    logger.info(f"Results saved")
    
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Total experiment time: {format_time(total_time)}")


if __name__ == "__main__":
    main()