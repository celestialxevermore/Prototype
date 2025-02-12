from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
import numpy as np
import pandas as pd
from utils.metrics import compute_overall_accuracy
import logging

def logistic_regression_benchmark(args, X_train, X_valid, X_test, y_train, y_valid, y_test, is_binary=True, max_iter=200):
    print("Input columns:", X_train.columns.tolist())
    categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns
    numeric_columns = X_train.select_dtypes(include=['int64', 'float64']).columns
    print(f"Categorical: {categorical_columns.tolist()}, Numeric: {numeric_columns.tolist()}")
    
    num_classes = len(np.unique(y_train))
    is_binary = num_classes == 2
    
    # Create preprocessing steps
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_columns),
            ('cat', categorical_transformer, categorical_columns)
        ])

    # FeatLLM에서 제안한 parameter grid
    param_grid = {
        'C': [100, 10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
        'penalty': ['l1', 'l2']
    }
    
    best_loss = float('inf')
    best_params = None
    
    # Grid Search
    for penalty in param_grid['penalty']:
        for c in param_grid['C']:
            # l1 penalty는 liblinear solver 필요
            solver = 'liblinear' if penalty == 'l1' else 'lbfgs'
            
            clf = LogisticRegression(
                max_iter=max_iter,
                C=c,
                penalty=penalty,
                solver=solver,
                random_state=args.random_seed
            )
            
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', clf)
            ])
            
            pipeline.fit(X_train, y_train)
            y_valid_pred_proba = pipeline.predict_proba(X_valid)[:, 1] if is_binary else pipeline.predict_proba(X_valid)
            
            valid_loss = log_loss(y_valid, y_valid_pred_proba)
            valid_acc, valid_auc, valid_auprc, valid_f1, valid_recall, valid_precision = compute_overall_accuracy(
                y_valid_pred_proba, y_valid, num_classes, threshold=0.5, activation=False
            )
            
            print(f"C: {c}, Penalty: {penalty}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}, "
                  f"Valid AUC: {valid_auc:.4f}, Valid AUPRC: {valid_auprc:.4f}")
            logging.info(f"C: {c}, Penalty: {penalty}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}, "
                        f"Valid AUC: {valid_auc:.4f}, Valid AUPRC: {valid_auprc:.4f}")
            
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_params = {'C': c, 'penalty': penalty, 'solver': solver}
    
    
    print(f"Best parameters: C={best_params['C']}, penalty={best_params['penalty']} "
          f"with Validation Loss: {best_loss:.4f}")
    logging.info(f"Best parameters: C={best_params['C']}, penalty={best_params['penalty']} "
                f"with Validation Loss: {best_loss:.4f}")
    
    final_clf = LogisticRegression(
        max_iter=max_iter,
        C=best_params['C'],
        penalty=best_params['penalty'],
        solver=best_params['solver'],
        random_state=args.random_seed
    )
    
    final_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', final_clf)
    ])
    
    final_pipeline.fit(X_train, y_train)
    
    # 테스트 데이터셋에서 성능 평가
    y_test_pred_proba = final_pipeline.predict_proba(X_test)[:, 1] if is_binary else final_pipeline.predict_proba(X_test)
    
    test_loss = log_loss(y_test, y_test_pred_proba)
    test_acc, test_auc, test_auprc, test_f1, test_recall, test_precision = compute_overall_accuracy(
        y_test_pred_proba, y_test, num_classes, threshold=0.5, activation=False
    )
    
    total_results = {
        'test_lr_loss': test_loss,
        'test_lr_acc': test_acc,
        'test_lr_auc': test_auc,
        'test_lr_auprc': test_auprc,
        'test_lr_f1': test_f1,
        'test_lr_recall': test_recall,
        'test_lr_precision': test_precision,
        'best_lr_c': best_params['C'],
        'best_lr_penalty': best_params['penalty'],
        'best_lr_loss': best_loss,
    }

    return total_results