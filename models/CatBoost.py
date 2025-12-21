from catboost import CatBoostClassifier
from sklearn.metrics import log_loss
from utils.metrics import compute_overall_accuracy
import numpy as np
import logging

def catboost_benchmark(
    args,
    X_train, X_valid, X_test,
    y_train, y_valid, y_test,
    is_binary=True,
    depth_list=[1, 2],
    iterations_list=[100]
):
    # ============================================================
    # [Leakage check] Suspicious columns like index/id
    # - 여기는 CatBoost 직전에 확인하므로, "모델 탓인지 데이터 탓인지" 빠르게 분리 가능
    # ============================================================
    suspicious_cols = [c for c in X_train.columns if c.lower() in ["index", "unnamed: 0", "level_0", "id"]]
    print(f"[CatBoost][LeakCheck] suspicious cols in X_train: {suspicious_cols}")
    if suspicious_cols:
        print("[CatBoost][LeakCheck] WARNING: These columns can cause leakage-like behavior.")

    # feature 제거 후 현재 데이터에서 컬럼 타입을 다시 확인
    categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_columns = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

    print(f"[CatBoost] Input columns: {X_train.columns.tolist()}")
    print(f"[CatBoost] After feature removal - Categorical: {categorical_columns}, Numeric: {numeric_columns}")

    # Determine number of classes
    num_class = len(np.unique(y_train))
    is_binary = num_class == 2

    print(f"[CatBoost] Number of classes: {num_class}, Is binary: {is_binary}")
    print(f"[CatBoost] Data shapes - Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}")

    # 범주형 컬럼이 있는지 확인하고 CatBoost에 전달
    if len(categorical_columns) > 0:
        print(f"[CatBoost] Using {len(categorical_columns)} categorical features: {categorical_columns}")
        cat_features = categorical_columns
    else:
        print("[CatBoost] No categorical features found, using numeric features only")
        cat_features = None

    # Find the best depth and iterations
    best_loss = float('inf')
    best_params = None

    print(f"[CatBoost] Starting hyperparameter search...")

    for depth in depth_list:
        for iterations in iterations_list:
            model = CatBoostClassifier(
                depth=depth,
                iterations=iterations,
                loss_function='Logloss' if is_binary else 'MultiClass',
                eval_metric='AUC' if is_binary else 'MultiClass',
                cat_features=cat_features,  # None if no categorical features
                verbose=0,
                random_seed=args.random_seed
            )

            try:
                model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True)

                # Make predictions on validation set
                y_valid_pred_proba = model.predict_proba(X_valid)[:, 1] if is_binary else model.predict_proba(X_valid)
                valid_loss = log_loss(y_valid, y_valid_pred_proba)

                valid_acc, valid_auc, valid_auprc, valid_f1, valid_recall, valid_precision = compute_overall_accuracy(
                    y_valid_pred_proba, y_valid, num_class,
                    threshold=args.threshold, activation=False
                )

                print(
                    f"[CatBoost] depth: {depth}, iterations: {iterations}, "
                    f"Validation Loss: {valid_loss:.4f}, Validation Acc: {valid_acc:.4f}, "
                    f"Validation AUC: {valid_auc:.4f}, Validation AUPRC: {valid_auprc:.4f}, "
                    f"Validation F1: {valid_f1:.4f}, Validation Recall: {valid_recall:.4f}, "
                    f"Validation Precision: {valid_precision:.4f}"
                )
                logging.info(
                    f"[CatBoost] depth: {depth}, iterations: {iterations}, "
                    f"Validation Loss: {valid_loss:.4f}, Validation Acc: {valid_acc:.4f}, "
                    f"Validation AUC: {valid_auc:.4f}, Validation AUPRC: {valid_auprc:.4f}, "
                    f"Validation F1: {valid_f1:.4f}, Validation Recall: {valid_recall:.4f}, "
                    f"Validation Precision: {valid_precision:.4f}"
                )

                if valid_loss < best_loss:
                    best_loss = valid_loss
                    best_params = (depth, iterations)

            except Exception as e:
                print(f"[CatBoost] Error with depth={depth}, iterations={iterations}: {str(e)}")
                logging.error(f"[CatBoost] Error with depth={depth}, iterations={iterations}: {str(e)}")
                continue

    if best_params is None:
        raise ValueError("[CatBoost] No valid hyperparameters found. All combinations failed.")

    # Train the final model with the best parameters
    best_depth, best_iterations = best_params
    print(f"[CatBoost] Best depth: {best_depth}, Best iterations: {best_iterations} with Validation Loss: {best_loss:.4f}")
    logging.info(f"[CatBoost] Best depth: {best_depth}, Best iterations: {best_iterations} with Validation Loss: {best_loss:.4f}")

    final_model = CatBoostClassifier(
        depth=best_depth,
        iterations=best_iterations,
        loss_function='Logloss' if is_binary else 'MultiClass',
        eval_metric='AUC' if is_binary else 'MultiClass',
        cat_features=cat_features,
        verbose=0,
        random_seed=args.random_seed
    )

    final_model.fit(X_train, y_train)

    # Make predictions on test set
    y_test_pred_proba = final_model.predict_proba(X_test)[:, 1] if is_binary else final_model.predict_proba(X_test)
    test_loss = log_loss(y_test, y_test_pred_proba)

    test_acc, test_auc, test_auprc, test_f1, test_recall, test_precision = compute_overall_accuracy(
        y_test_pred_proba, y_test, num_class,
        threshold=args.threshold, activation=False
    )

    print(f"[CatBoost] Final test results - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, AUC: {test_auc:.4f}, F1: {test_f1:.4f}")

    total_results = {
        'test_cat_loss': test_loss,
        'test_cat_acc': test_acc,
        'test_cat_auc': test_auc,
        'test_cat_auprc': test_auprc,
        'test_cat_f1': test_f1,
        'test_cat_recall': test_recall,
        'test_cat_precision': test_precision,
        'best_depth': best_depth,
        'best_iterations': best_iterations
    }

    return total_results
