import logging
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from utils.metrics import compute_overall_accuracy


def catboost_benchmark(
    args,
    X_train, X_valid, X_test,
    y_train, y_valid, y_test,
    is_binary=True,
    depth_list=[3, 4, 5, 6],
    iterations_list=[100, 200, 300],
):
    """
    Val-free CatBoost benchmark.
    HP selection: cross-validation on support (train) set.
    X_valid / y_valid 은 무시됨 (50/50 프로토콜).
    use_best_model=False (val 없으므로).
    """
    categorical_columns = X_train.select_dtypes(
        include=['object', 'category']
    ).columns.tolist()
    numeric_columns = X_train.select_dtypes(
        include=['int64', 'float64']
    ).columns.tolist()

    num_class = len(np.unique(y_train))
    is_binary = (num_class == 2)

    print(f"[CatBoost] Categorical: {categorical_columns} | Numeric: {numeric_columns}")
    print(f"[CatBoost] num_classes={num_class}, is_binary={is_binary}")
    print(f"[CatBoost] Train: {X_train.shape} | Test: {X_test.shape}")

    cat_features = categorical_columns if categorical_columns else None

    # ── HP 탐색: support set CV ──────────────────────────────────
    n_splits = min(3, len(y_train))
    use_cv   = (len(y_train) >= 6)

    best_score  = -1.0
    best_params = (4, 100)   # fallback default

    if use_cv:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                              random_state=args.random_seed)
        for depth in depth_list:
            for iterations in iterations_list:
                fold_aucs = []
                for tr_idx, vl_idx in skf.split(X_train, y_train):
                    X_tr_f = X_train.iloc[tr_idx]
                    X_vl_f = X_train.iloc[vl_idx]
                    y_tr_f = y_train[tr_idx]
                    y_vl_f = y_train[vl_idx]

                    clf = CatBoostClassifier(
                        depth=depth,
                        iterations=iterations,
                        loss_function='Logloss' if is_binary else 'MultiClass',
                        eval_metric='AUC' if is_binary else 'Accuracy',
                        cat_features=cat_features,
                        verbose=0,
                        random_seed=args.random_seed,
                        use_best_model=False,   # ✅ val 없으므로 반드시 False
                    )
                    try:
                        clf.fit(X_tr_f, y_tr_f)
                        if is_binary:
                            proba = clf.predict_proba(X_vl_f)[:, 1]
                            from sklearn.metrics import roc_auc_score
                            auc = roc_auc_score(y_vl_f, proba)
                        else:
                            proba = clf.predict_proba(X_vl_f)
                            from sklearn.metrics import roc_auc_score
                            from sklearn.preprocessing import label_binarize
                            y_bin = label_binarize(y_vl_f, classes=range(num_class))
                            auc = roc_auc_score(y_bin, proba, multi_class='ovr',
                                                average='macro')
                        fold_aucs.append(auc)
                    except Exception as e:
                        logging.warning(f"[CatBoost] fold failed: {e}")

                score = float(np.mean(fold_aucs)) if fold_aucs else -1.0
                print(f"[CatBoost] depth={depth}, iters={iterations} → CV AUC={score:.4f}")
                if score > best_score:
                    best_score  = score
                    best_params = (depth, iterations)
    else:
        print(f"[CatBoost] Too few samples ({len(y_train)}) for CV → using default HP")

    best_depth, best_iterations = best_params
    print(f"[CatBoost] Best HP: depth={best_depth}, iters={best_iterations} (CV AUC={best_score:.4f})")
    logging.info(f"[CatBoost] Best HP: depth={best_depth}, iters={best_iterations}")

    # ── 최종 학습 & 테스트 ──────────────────────────────────────
    final_model = CatBoostClassifier(
        depth=best_depth,
        iterations=best_iterations,
        loss_function='Logloss' if is_binary else 'MultiClass',
        eval_metric='AUC' if is_binary else 'Accuracy',
        cat_features=cat_features,
        verbose=0,
        random_seed=args.random_seed,
        use_best_model=False,   # ✅ val 없으므로 반드시 False
    )
    final_model.fit(X_train, y_train)

    y_test_pred_proba = (
        final_model.predict_proba(X_test)[:, 1]
        if is_binary else final_model.predict_proba(X_test)
    )
    test_loss = log_loss(y_test, y_test_pred_proba)
    test_acc, test_auc, test_auprc, test_f1, test_recall, test_precision = \
        compute_overall_accuracy(y_test_pred_proba, y_test, num_class,
                                 threshold=args.threshold, activation=False)

    print(f"[CatBoost] Test → Loss:{test_loss:.4f} AUC:{test_auc:.4f} AUPRC:{test_auprc:.4f} "
          f"ACC:{test_acc:.4f} F1:{test_f1:.4f}")

    return {
        'test_cat_loss':      test_loss,
        'test_cat_acc':       test_acc,
        'test_cat_auc':       test_auc,
        'test_cat_auprc':     test_auprc,
        'test_cat_f1':        test_f1,
        'test_cat_recall':    test_recall,
        'test_cat_precision': test_precision,
        'best_depth':         best_depth,
        'best_iterations':    best_iterations,
    }