import copy
import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import log_loss
from utils.metrics import compute_overall_accuracy


def logistic_regression_benchmark(
    args,
    X_train, X_valid, X_test,
    y_train, y_valid, y_test,
    is_binary=True,
    max_iter=1000,
):
    """
    Val-free Logistic Regression benchmark.
    HP selection: cross-validation on support (train) set.
    X_valid / y_valid 은 무시됨 (50/50 프로토콜).
    """
    categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns
    numeric_columns     = X_train.select_dtypes(include=['int64', 'float64']).columns

    num_classes = len(np.unique(y_train))
    is_binary   = (num_classes == 2)

    print(f"[LR] Categorical: {categorical_columns.tolist()} | Numeric: {numeric_columns.tolist()}")

    # ── 전처리 파이프라인 ───────────────────────────────────────
    transformers = []
    if len(numeric_columns) > 0:
        transformers.append(('num', Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('sc', StandardScaler()),
        ]), numeric_columns))
    if len(categorical_columns) > 0:
        transformers.append(('cat', Pipeline([
            ('imp', SimpleImputer(strategy='most_frequent')),
            ('oh', OneHotEncoder(handle_unknown='ignore')),
        ]), categorical_columns))

    preprocessor = ColumnTransformer(transformers=transformers) if transformers else None

    # ── HP 탐색: support set CV ──────────────────────────────────
    C_list       = [100, 10, 1, 0.1, 0.01, 0.001]
    penalty_list = ['l2']           # few-shot에서 l1 solver 불안정 → l2만
    n_splits     = min(3, len(y_train))
    use_cv       = (len(y_train) >= 6)
    scoring      = 'roc_auc' if is_binary else 'roc_auc_ovr_weighted'

    best_score  = -1.0
    best_params = {'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs'}   # fallback

    if use_cv:
        for penalty in penalty_list:
            for c in C_list:
                solver = 'liblinear' if penalty == 'l1' else 'lbfgs'
                clf = LogisticRegression(
                    C=c, penalty=penalty, solver=solver,
                    max_iter=max_iter, class_weight='balanced',
                    random_state=args.random_seed
                )
                if preprocessor is not None:
                    pipeline = Pipeline([('pre', preprocessor), ('clf', clf)])
                else:
                    pipeline = clf

                try:
                    scores = cross_val_score(
                        pipeline, X_train, y_train,
                        cv=n_splits, scoring=scoring, n_jobs=-1
                    )
                    score = float(np.mean(scores))
                except Exception as e:
                    logging.warning(f"[LR] CV failed (C={c}, {penalty}): {e}")
                    score = -1.0

                print(f"[LR] C={c}, penalty={penalty} → CV AUC={score:.4f}")
                if score > best_score:
                    best_score  = score
                    best_params = {'C': c, 'penalty': penalty, 'solver': solver}
    else:
        print(f"[LR] Too few samples ({len(y_train)}) for CV → using default HP")

    print(f"[LR] Best HP: {best_params} (CV AUC={best_score:.4f})")
    logging.info(f"[LR] Best HP: {best_params}")

    # ── 최종 학습 & 테스트 ──────────────────────────────────────
    final_clf = LogisticRegression(
        C=best_params['C'],
        penalty=best_params['penalty'],
        solver=best_params['solver'],
        max_iter=max_iter,
        class_weight='balanced',
        random_state=args.random_seed
    )
    if preprocessor is not None:
        final_pipeline = Pipeline([('pre', preprocessor), ('clf', final_clf)])
        final_pipeline.fit(X_train, y_train)
        y_test_pred_proba = (
            final_pipeline.predict_proba(X_test)[:, 1]
            if is_binary else final_pipeline.predict_proba(X_test)
        )
    else:
        final_clf.fit(X_train, y_train)
        y_test_pred_proba = (
            final_clf.predict_proba(X_test)[:, 1]
            if is_binary else final_clf.predict_proba(X_test)
        )

    test_loss = log_loss(y_test, y_test_pred_proba)
    test_acc, test_auc, test_auprc, test_f1, test_recall, test_precision = \
        compute_overall_accuracy(y_test_pred_proba, y_test, num_classes,
                                 threshold=0.5, activation=False)

    print(f"[LR] Test → Loss:{test_loss:.4f} AUC:{test_auc:.4f} AUPRC:{test_auprc:.4f} "
          f"ACC:{test_acc:.4f} F1:{test_f1:.4f}")

    return {
        'test_lr_loss':      test_loss,
        'test_lr_acc':       test_acc,
        'test_lr_auc':       test_auc,
        'test_lr_auprc':     test_auprc,
        'test_lr_f1':        test_f1,
        'test_lr_recall':    test_recall,
        'test_lr_precision': test_precision,
        'best_lr_c':         best_params['C'],
        'best_lr_penalty':   best_params['penalty'],
    }