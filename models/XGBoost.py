import logging
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from utils.metrics import compute_overall_accuracy


def xgboost_benchmark(
    args,
    X_train, X_valid, X_test,
    y_train, y_valid, y_test,
    is_binary=True,
    max_depth_list=[3, 4, 5],
    n_estimators_list=[50, 100, 200],
    oh_max_categories=10,
):
    """
    Val-free XGBoost benchmark.
    - X_valid / y_valid 무시 (50/50 프로토콜)
    - 범주형: OHE (handle_unknown='ignore') → few-shot unseen category 대응
    - HP 탐색: support set CV (샘플 부족 시 default)
    """
    categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns
    numeric_columns     = X_train.select_dtypes(include=['int64', 'float64']).columns

    num_classes = len(np.unique(y_train))
    is_binary   = (num_classes == 2)

    print(f"[XGB] Categorical: {categorical_columns.tolist()} | Numeric: {numeric_columns.tolist()}")
    print(f"[XGB] num_classes={num_classes}, is_binary={is_binary}")
    print(f"[XGB] Train: {X_train.shape} | Test: {X_test.shape}")

    # ── 전처리: OHE (handle_unknown='ignore' → unseen category → 0벡터) ───
    if len(categorical_columns) > 0:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32)
        X_tr_cat = ohe.fit_transform(X_train[categorical_columns].astype(str))
        X_te_cat = ohe.transform(X_test[categorical_columns].astype(str))

        X_tr_num = X_train[numeric_columns].values.astype(np.float32)
        X_te_num = X_test[numeric_columns].values.astype(np.float32)

        X_tr_final = np.hstack([X_tr_num, X_tr_cat])
        X_te_final = np.hstack([X_te_num, X_te_cat])
    else:
        X_tr_final = X_train[numeric_columns].values.astype(np.float32)
        X_te_final = X_test[numeric_columns].values.astype(np.float32)

    print(f"[XGB] Processed → Train: {X_tr_final.shape} | Test: {X_te_final.shape}")

    # ── HP 탐색: support set CV ──────────────────────────────────────────────
    n_splits = min(3, len(y_train))
    use_cv   = (len(y_train) >= 6)
    scoring  = 'roc_auc' if is_binary else 'roc_auc_ovr_weighted'

    best_score  = -1.0
    best_params = (3, 100)   # fallback default

    if use_cv:
        for max_depth in max_depth_list:
            for n_est in n_estimators_list:
                clf = XGBClassifier(
                    max_depth=max_depth,
                    n_estimators=n_est,
                    tree_method='hist',
                    device='cuda',
                    use_label_encoder=False,
                    eval_metric='logloss' if is_binary else 'mlogloss',
                    verbosity=0,
                    random_state=args.random_seed,
                )
                try:
                    scores = cross_val_score(
                        clf, X_tr_final, y_train,
                        cv=n_splits, scoring=scoring, n_jobs=1
                    )
                    score = float(np.mean(scores))
                except Exception as e:
                    logging.warning(f"[XGB] CV failed (depth={max_depth}, n_est={n_est}): {e}")
                    score = -1.0

                print(f"[XGB] depth={max_depth}, n_est={n_est} → CV AUC={score:.4f}")
                if score > best_score:
                    best_score  = score
                    best_params = (max_depth, n_est)
    else:
        print(f"[XGB] Too few samples ({len(y_train)}) for CV → using default HP")

    best_max_depth, best_n_est = best_params
    print(f"[XGB] Best HP: depth={best_max_depth}, n_est={best_n_est} (CV AUC={best_score:.4f})")
    logging.info(f"[XGB] Best HP: depth={best_max_depth}, n_est={best_n_est}")

    # ── 최종 학습 & 테스트 평가 ─────────────────────────────────────────────
    base_params = {
        'tree_method': 'hist',
        'device': 'cuda',
        'max_depth': best_max_depth,
        'verbosity': 0,
    }
    if is_binary:
        base_params['objective']   = 'binary:logistic'
        base_params['eval_metric'] = 'logloss'
    else:
        base_params['objective']   = 'multi:softprob'
        base_params['num_class']   = num_classes
        base_params['eval_metric'] = 'mlogloss'

    dtrain = xgb.DMatrix(X_tr_final, y_train)
    dtest  = xgb.DMatrix(X_te_final, y_test)

    bst = xgb.train(
        base_params, dtrain, best_n_est,
        evals=[(dtrain, 'train')],
        verbose_eval=False,
    )

    y_test_pred_proba = bst.predict(dtest)
    test_loss = log_loss(y_test, y_test_pred_proba)
    test_acc, test_auc, test_auprc, test_f1, test_recall, test_precision = \
        compute_overall_accuracy(y_test_pred_proba, y_test, num_classes,
                                 threshold=0.5, activation=False)

    print(f"[XGB] Test → Loss:{test_loss:.4f} AUC:{test_auc:.4f} AUPRC:{test_auprc:.4f} "
          f"ACC:{test_acc:.4f} F1:{test_f1:.4f}")

    return {
        'test_xgb_loss':      test_loss,
        'test_xgb_acc':       test_acc,
        'test_xgb_auc':       test_auc,
        'test_xgb_auprc':     test_auprc,
        'test_xgb_f1':        test_f1,
        'test_xgb_recall':    test_recall,
        'test_xgb_precision': test_precision,
        'best_max_depth':     best_max_depth,
        'best_n_estimators':  best_n_est,
    }