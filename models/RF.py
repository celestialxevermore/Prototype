import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, log_loss
from utils.metrics import compute_overall_accuracy
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

def random_forest_benchmark(args, X_train, X_valid, X_test, y_train, y_valid, y_test,
                            is_binary=True,
                            n_estimators_list=[100, 200, 300, 500],
                            max_depth_list=[3, 5, 7, 10, None],
                            min_samples_split_list=[2, 5, 10],
                            random_state=42):
    """
    RandomForestClassifier를 활용해 하이퍼파라미터 탐색 후 최종 성능을 반환하는 함수.

    매개변수:
    - X_train, X_valid, X_test: 각각 학습/검증/테스트용 특징 데이터 (판다스 DataFrame 등)
    - y_train, y_valid, y_test: 각각 학습/검증/테스트용 라벨 데이터 (numpy array 등)
    - is_binary: 이진분류 여부 (True/False)
    - n_estimators_list: 탐색할 트리 개수 리스트
    - max_depth_list: 탐색할 트리 깊이 리스트
    - min_samples_split_list: 탐색할 min_samples_split 값 리스트
    - random_state: 재현성 유지를 위한 랜덤시드
    """

    num_class = len(np.unique(y_train))
    is_binary = (num_class == 2)

    # 범주형 열과 수치형 열 구분
    categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns
    numeric_columns = X_train.select_dtypes(include=['int64', 'float64']).columns

    # 수치형 데이터 처리
    X_train_numeric = X_train[numeric_columns].copy()
    X_valid_numeric = X_valid[numeric_columns].copy()
    X_test_numeric = X_test[numeric_columns].copy()

    # 수치형 결측치를 중앙값으로 대체
    for col in numeric_columns:
        median_val = X_train_numeric[col].median()
        X_train_numeric[col] = X_train_numeric[col].fillna(median_val)
        X_valid_numeric[col] = X_valid_numeric[col].fillna(median_val)
        X_test_numeric[col] = X_test_numeric[col].fillna(median_val)

    # 범주형 데이터 처리
    if len(categorical_columns) > 0:
        X_train_cat = X_train[categorical_columns].fillna('missing').astype(str)
        X_valid_cat = X_valid[categorical_columns].fillna('missing').astype(str)
        X_test_cat = X_test[categorical_columns].fillna('missing').astype(str)

        encoder = OneHotEncoder(handle_unknown='ignore', sparse=False, dtype=np.float32)
        X_train_encoded = encoder.fit_transform(X_train_cat)
        X_valid_encoded = encoder.transform(X_valid_cat)
        X_test_encoded = encoder.transform(X_test_cat)

        X_train_final = np.hstack((X_train_numeric, X_train_encoded))
        X_valid_final = np.hstack((X_valid_numeric, X_valid_encoded))
        X_test_final = np.hstack((X_test_numeric, X_test_encoded))
    else:
        X_train_final = X_train_numeric
        X_valid_final = X_valid_numeric
        X_test_final = X_test_numeric

    best_loss = float('inf')
    best_params = None

    # 하이퍼파라미터 탐색
    for n_est in n_estimators_list:
        for depth in max_depth_list:
            for min_split in min_samples_split_list:
                model = RandomForestClassifier(
                    n_estimators=n_est,
                    max_depth=depth,
                    min_samples_split=min_split,
                    max_features='sqrt',  # 일반적으로 분류에서 좋은 성능
                    random_state=random_state,
                    n_jobs=-1  # 병렬 처리로 속도 향상
                )
                
                # 학습 (Train)
                model.fit(X_train_final, y_train)
                
                # 검증 (Validation)
                y_valid_pred_proba = model.predict_proba(X_valid_final)
                
                # 이진분류면 y_valid_pred_proba[:, 1], 다중분류면 전체 배열
                if is_binary:
                    y_valid_pred = (y_valid_pred_proba[:, 1] > args.threshold).astype(int)
                    valid_loss = log_loss(y_valid, y_valid_pred_proba[:, 1])
                else:
                    y_valid_pred = y_valid_pred_proba.argmax(axis=1)
                    valid_loss = log_loss(y_valid, y_valid_pred_proba)
                
                # compute_overall_accuracy 함수를 이용해 AUC, ACC, F1 등 계산
                valid_acc, valid_auc, valid_auprc, valid_f1, valid_recall, valid_precision = \
                    compute_overall_accuracy(y_valid_pred, y_valid, num_class,
                                             threshold=args.threshold, activation=False)
                
                logging.info(
                    f"[RandomForest] n_estimators={n_est}, max_depth={depth}, "
                    f"min_samples_split={min_split}, Valid Loss={valid_loss:.4f}, Valid Acc={valid_acc:.4f}, "
                    f"Valid AUC={valid_auc:.4f}, Valid AUPRC={valid_auprc:.4f}, "
                    f"Valid F1={valid_f1:.4f}, Valid Recall={valid_recall:.4f}, Valid Precision={valid_precision:.4f}"
                )
                print(
                    f"[RandomForest] n_estimators={n_est}, max_depth={depth}, "
                    f"min_samples_split={min_split}, Valid Loss={valid_loss:.4f}, Valid Acc={valid_acc:.4f}, "
                    f"Valid AUC={valid_auc:.4f}, Valid AUPRC={valid_auprc:.4f}, "
                    f"Valid F1={valid_f1:.4f}, Valid Recall={valid_recall:.4f}, Valid Precision={valid_precision:.4f}"
                )

                # Validation Loss가 더 좋으면(더 작으면) 갱신
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    best_params = (n_est, depth, min_split)
    
    best_n_est, best_depth, best_min_split = best_params
    logging.info(f"[RandomForest] Best params: n_estimators={best_n_est}, max_depth={best_depth}, min_samples_split={best_min_split}, Valid Loss={best_loss:.4f}")
    print(f"[RandomForest] Best params: n_estimators={best_n_est}, max_depth={best_depth}, min_samples_split={best_min_split}, Valid Loss={best_loss:.4f}")

    # 최적 파라미터로 모델 재학습 (Train + Valid 전체를 묶어서 쓰고 싶다면 수정)
    model = RandomForestClassifier(
        n_estimators=best_n_est,
        max_depth=best_depth,
        min_samples_split=best_min_split,
        max_features='sqrt',
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train_final, y_train)

    # 최종 Test 세트 평가
    y_test_pred_proba = model.predict_proba(X_test_final)
    
    if is_binary:
        y_test_pred = (y_test_pred_proba[:, 1] > args.threshold).astype(int)
        test_loss = log_loss(y_test, y_test_pred_proba[:, 1])
    else:
        y_test_pred = y_test_pred_proba.argmax(axis=1)
        test_loss = log_loss(y_test, y_test_pred_proba)

    test_acc, test_auc, test_auprc, test_f1, test_recall, test_precision = \
        compute_overall_accuracy(y_test_pred, y_test, num_class,
                                 threshold=args.threshold, activation=False)

    total_results = {
        'test_rf_loss': test_loss,
        'test_rf_acc': test_acc,
        'test_rf_auc': test_auc,
        'test_rf_auprc': test_auprc,
        'test_rf_f1': test_f1,
        'test_rf_recall': test_recall,
        'test_rf_precision': test_precision,
        'best_n_estimators': best_n_est,
        'best_max_depth': best_depth,
        'best_min_samples_split': best_min_split
    }

    return total_results