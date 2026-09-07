import logging
import copy
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
import pandas as pd
from utils.metrics import compute_overall_accuracy


def random_forest_benchmark(
    args,
    X_train, X_valid, X_test,
    y_train, y_valid, y_test,
    is_binary=True,
    n_estimators_list=[100, 200, 300],
    max_depth_list=[3, 5, 7, None],
    min_samples_split_list=[2, 5],
    random_state=42,
):
    """
    Val-free Random Forest benchmark.
    HP selection: cross-validation on support (train) set.
    X_valid / y_valid 은 무시됨 (50/50 프로토콜).
    """
    num_class = len(np.unique(y_train))
    is_binary = (num_class == 2)

    # ── 전처리 ──────────────────────────────────────────────────
    categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns
    numeric_columns     = X_train.select_dtypes(include=['int64', 'float64']).columns

    print(f"[RF] Categorical: {categorical_columns.tolist()} | Numeric: {numeric_columns.tolist()}")

    # 수치형: 중앙값으로 결측 대체
    X_train_num = X_train[numeric_columns].copy()
    X_test_num  = X_test[numeric_columns].copy()
    for col in numeric_columns:
        med = X_train_num[col].median()
        X_train_num[col] = X_train_num[col].fillna(med)
        X_test_num[col]  = X_test_num[col].fillna(med)

    # 범주형: OHE
    if len(categorical_columns) > 0:
        X_train_cat = X_train[categorical_columns].astype(str).fillna('missing')
        X_test_cat  = X_test[categorical_columns].astype(str).fillna('missing')

        encoder = OneHotEncoder(handle_unknown='ignore', sparse=False, dtype=np.float32)
        X_train_enc = encoder.fit_transform(X_train_cat)
        X_test_enc  = encoder.transform(X_test_cat)

        X_train_final = np.hstack([X_train_num.values, X_train_enc])
        X_test_final  = np.hstack([X_test_num.values,  X_test_enc])
    else:
        X_train_final = X_train_num.values.astype(np.float32)
        X_test_final  = X_test_num.values.astype(np.float32)

    print(f"[RF] Train: {X_train_final.shape} | Test: {X_test_final.shape}")

    # ── HP 탐색: support set CV ──────────────────────────────────
    # few-shot (N < 10) 이면 CV 생략하고 default 사용
    n_splits = min(3, len(y_train))
    use_cv   = (len(y_train) >= 6)   # 클래스당 최소 2개 이상

    scoring  = 'roc_auc' if is_binary else 'roc_auc_ovr_weighted'
    best_score  = -1.0
    best_params = (100, 5, 2)   # fallback default

    if use_cv:
        for n_est in n_estimators_list:
            for depth in max_depth_list:
                for min_split in min_samples_split_list:
                    clf = RandomForestClassifier(
                        n_estimators=n_est,
                        max_depth=depth,
                        min_samples_split=min_split,
                        max_features='sqrt',
                        random_state=random_state,
                        n_jobs=-1,
                        class_weight='balanced',
                    )
                    try:
                        scores = cross_val_score(
                            clf, X_train_final, y_train,
                            cv=n_splits, scoring=scoring, n_jobs=-1
                        )
                        score = float(np.mean(scores))
                    except Exception as e:
                        logging.warning(f"[RF] CV failed ({n_est},{depth},{min_split}): {e}")
                        score = -1.0

                    print(f"[RF] n_est={n_est}, depth={depth}, min_split={min_split} → CV AUC={score:.4f}")
                    if score > best_score:
                        best_score  = score
                        best_params = (n_est, depth, min_split)
    else:
        print(f"[RF] Too few samples ({len(y_train)}) for CV → using default HP")

    best_n_est, best_depth, best_min_split = best_params
    print(f"[RF] Best HP: n_est={best_n_est}, depth={best_depth}, min_split={best_min_split} (CV AUC={best_score:.4f})")
    logging.info(f"[RF] Best HP: n_est={best_n_est}, depth={best_depth}, min_split={best_min_split}")

    # ── 최종 학습 & 테스트 평가 ──────────────────────────────────
    model = RandomForestClassifier(
        n_estimators=best_n_est,
        max_depth=best_depth,
        min_samples_split=best_min_split,
        max_features='sqrt',
        random_state=random_state,
        n_jobs=-1,
        class_weight='balanced',
    )
    model.fit(X_train_final, y_train)

    y_test_pred_proba = model.predict_proba(X_test_final)
    if is_binary:
        y_prob = y_test_pred_proba[:, 1]
        test_loss = log_loss(y_test, y_prob)
    else:
        y_prob = y_test_pred_proba
        test_loss = log_loss(y_test, y_prob)

    test_acc, test_auc, test_auprc, test_f1, test_recall, test_precision = \
        compute_overall_accuracy(y_prob, y_test, num_class,
                                 threshold=args.threshold, activation=False)

    print(f"[RF] Test → Loss:{test_loss:.4f} AUC:{test_auc:.4f} AUPRC:{test_auprc:.4f} "
          f"ACC:{test_acc:.4f} F1:{test_f1:.4f}")

    return {
        'test_rf_loss':              test_loss,
        'test_rf_acc':               test_acc,
        'test_rf_auc':               test_auc,
        'test_rf_auprc':             test_auprc,
        'test_rf_f1':                test_f1,
        'test_rf_recall':            test_recall,
        'test_rf_precision':         test_precision,
        'best_n_estimators':         best_n_est,
        'best_max_depth':            best_depth,
        'best_min_samples_split':    best_min_split,
    }