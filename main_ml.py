import os
import time
import argparse
import numpy as np
from utils.util import setup_logger, format_time, fix_seed
from utils.util import prepare_ml_results, save_ml_results
from dataset.data_dataloaders import ml_prepare_tabular_dataloaders, get_few_shot_tabular_samples
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
os.environ["CUDA_VISIBLE_DEVICES"]="4"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

logger = setup_logger()

def get_args():
    parser = argparse.ArgumentParser(description='ML Baselines For Tabular Task')

    parser.add_argument('--random_seed', type=int, default=42, help='random_seed')
    parser.add_argument('--dataset_seed', type=int, default=4)
    
    parser.add_argument('--source_data', type=str, default='heart', 
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
    parser.add_argument('--del_feat', nargs='+', default = [], help = 'List  of features to remove from dataset')
    
    args = parser.parse_args()
    return args
def main():
    start_time = time.time()
    args = get_args()
    fix_seed(args.random_seed)

    logger.info(f"Starting experiment with dataset: {args.source_data}")
    logger.info(f"Preparing Tabular datasets...")

    # ✅ 50/50 split — val=None
    (X_train_full, _, X_test_full,
     y_train_full, _, y_test_full), _ = ml_prepare_tabular_dataloaders(
        args, args.source_data, args.random_seed
    )
    logger.info(f"[Target] Train pool: {len(X_train_full)} | Test: {len(X_test_full)}")

    # ✅ K-shot 샘플링 (train pool에서만)
    X_train_few, y_train_few = get_few_shot_tabular_samples(X_train_full, y_train_full, args)
    logger.info(f"[Few-shot] Support set size: {len(X_train_few)}")

    is_binary = (len(np.unique(y_train_full)) == 2)

    full_baseline_results = {}
    few_baseline_results = {}

    for baseline in args.baseline:
        logger.info(f"\n>>> Running baseline: {baseline}")

        if baseline == "rf":
            X_train_rf   = X_train_full.copy()
            X_test_rf    = X_test_full.copy()
            X_few_rf     = X_train_few.copy()

            categorical_columns = X_train_rf.select_dtypes(include=['object', 'category']).columns
            for col in categorical_columns:
                le = LabelEncoder()
                all_values = pd.concat([X_train_rf[col], X_test_rf[col]]).unique()
                le.fit(all_values)
                X_train_rf[col] = le.transform(X_train_rf[col])
                X_test_rf[col]  = le.transform(X_test_rf[col])
                # few-shot은 unseen value 있을 수 있으니 safe transform
                X_few_rf[col]   = X_few_rf[col].map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )

            full_baseline_results[baseline] = random_forest_benchmark(
                args,
                X_train_rf, None, X_test_rf,
                y_train_full, None, y_test_full,
                is_binary=is_binary
            )
            few_baseline_results[baseline] = random_forest_benchmark(
                args,
                X_few_rf, None, X_test_rf,
                y_train_few, None, y_test_full,
                is_binary=is_binary
            )

        elif baseline == "lr":
            full_baseline_results[baseline] = logistic_regression_benchmark(
                args,
                X_train_full, None, X_test_full,
                y_train_full, None, y_test_full,
                is_binary=is_binary,
            )
            few_baseline_results[baseline] = logistic_regression_benchmark(
                args,
                X_train_few, None, X_test_full,
                y_train_few, None, y_test_full,
                is_binary=is_binary,
            )

        elif baseline == "xgb":
            full_baseline_results[baseline] = xgboost_benchmark(
                args,
                X_train_full, None, X_test_full,
                y_train_full, None, y_test_full,
                is_binary=is_binary,
            )
            few_baseline_results[baseline] = xgboost_benchmark(
                args,
                X_train_few, None, X_test_full,
                y_train_few, None, y_test_full,
                is_binary=is_binary,
            )

        elif baseline == "mlp":
            full_baseline_results[baseline] = mlp_benchmark(
                args,
                X_train_full, None, X_test_full,
                y_train_full, None, y_test_full,
                is_binary=is_binary,
            )
            few_baseline_results[baseline] = mlp_benchmark(
                args,
                X_train_few, None, X_test_full,
                y_train_few, None, y_test_full,
                is_binary=is_binary,
            )

        elif baseline == "cat":
            full_baseline_results[baseline] = catboost_benchmark(
                args,
                X_train_full, None, X_test_full,
                y_train_full, None, y_test_full,
                is_binary=is_binary,
            )
            few_baseline_results[baseline] = catboost_benchmark(
                args,
                X_train_few, None, X_test_full,
                y_train_few, None, y_test_full,
                is_binary=is_binary,
            )

        else:
            logger.warning(f"Invalid baseline: {baseline}. Skipping.")

    results = prepare_ml_results(args, full_baseline_results, few_baseline_results)
    save_ml_results(args, results)
    logger.info(f"Results saved")
    logger.info(f"Total experiment time: {format_time(time.time() - start_time)}")


if __name__ == "__main__":
    main()