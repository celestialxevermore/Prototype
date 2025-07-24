from xgboost import XGBClassifier, DMatrix
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from utils.metrics import calculate_binary_accuracy, calculate_multiclass_accuracy
import pandas as pd
import numpy as np
import xgboost as xgb

from xgboost import XGBClassifier, DMatrix
from sklearn.metrics import roc_auc_score, log_loss
from utils.metrics import compute_overall_accuracy
import xgboost as xgb
import numpy as np
import logging

def xgboost_benchmark(args, X_train, X_valid, X_test, y_train, y_valid, y_test, is_binary=True, max_depth_list=[1,2,3,4,5], n_estimators_list=[100, 200, 300, 400, 500], oh_max_categories=10):
    # feature 제거 후 현재 데이터에서 컬럼 타입을 다시 확인
    categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns
    numeric_columns = X_train.select_dtypes(include=['int64', 'float64']).columns
    
    print(f"[XGBoost] Input columns: {X_train.columns.tolist()}")
    print(f"[XGBoost] After feature removal - Categorical: {categorical_columns.tolist()}, Numeric: {numeric_columns.tolist()}")
    
    # 데이터 복사본 생성 (원본 수정 방지)
    X_train_copy = X_train.copy()
    X_valid_copy = X_valid.copy()
    X_test_copy = X_test.copy()
    
    # XGBoost는 내부 enable_categorical=True 사용
    for col in categorical_columns:
        X_train_copy[col] = X_train_copy[col].astype('category')
        X_valid_copy[col] = X_valid_copy[col].astype('category')
        X_test_copy[col] = X_test_copy[col].astype('category')
    
    # Determine number of classes
    num_classes = len(np.unique(y_train))
    is_binary = num_classes == 2
    
    print(f"[XGBoost] Number of classes: {num_classes}, Is binary: {is_binary}")
    
    # Set up parameters
    params = {
        'tree_method': 'hist',
        'n_jobs': -1,
        "device": "cuda",
        'max_cat_to_onehot': oh_max_categories,
    }
    
    if is_binary:
        params['objective'] = 'binary:logistic'
        params['eval_metric'] = ['logloss', 'auc']
    else:
        params['objective'] = 'multi:softprob'
        params['num_class'] = num_classes
        params['eval_metric'] = ['mlogloss', 'auc']
    

    # Create DMatrix objects with enable_categorical=True
    dtrain = xgb.DMatrix(X_train_copy, y_train, enable_categorical=True)
    dvalid = xgb.DMatrix(X_valid_copy, y_valid, enable_categorical=True)
    dtest = xgb.DMatrix(X_test_copy, y_test, enable_categorical=True)
    
    print(f"[XGBoost] DMatrix created - Train shape: {X_train_copy.shape}, Valid shape: {X_valid_copy.shape}, Test shape: {X_test_copy.shape}")
    
    # Find the best max_depth and n_estimators
    best_auc = 0
    best_loss = float('inf')
    best_params = None
    for max_depth in max_depth_list:
        for n_estimators in n_estimators_list:
            params['max_depth'] = max_depth
            evallist = [(dtrain, 'train'), (dvalid, 'valid')]
            evals_result = {}
            bst = xgb.train(params, dtrain, n_estimators, evals=evallist, evals_result=evals_result, verbose_eval=False)
            
            # Make predictions on validation set
            y_valid_pred_proba = bst.predict(dvalid)
            valid_loss = log_loss(y_valid, y_valid_pred_proba) if is_binary else log_loss(y_valid, y_valid_pred_proba)
            valid_acc, valid_auc, valid_auprc, valid_f1, valid_recall, valid_precision = compute_overall_accuracy(y_valid_pred_proba, y_valid, num_classes, threshold=0.5, activation=False)
            
            print(f"[XGBoost] max_depth: {max_depth}, n_estimators: {n_estimators}, Validation Loss: {valid_loss:.4f}, Validation Acc: {valid_acc:.4f}, Validation AUC: {valid_auc:.4f}, Validation AUPRC: {valid_auprc:.4f}, Validation F1: {valid_f1:.4f}, Validation Recall: {valid_recall:.4f}, Validation Precision: {valid_precision:.4f}")
            logging.info(f"[XGBoost] max_depth: {max_depth}, n_estimators: {n_estimators}, Validation Loss: {valid_loss:.4f}, Validation Acc: {valid_acc:.4f}, Validation AUC: {valid_auc:.4f}, Validation AUPRC: {valid_auprc:.4f}, Validation F1: {valid_f1:.4f}, Validation Recall: {valid_recall:.4f}, Validation Precision: {valid_precision:.4f}")
            
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_params = (max_depth, n_estimators)
            # if valid_auc > best_auc:
            #     best_auc = valid_auc
            #     best_params = (max_depth, n_estimators)
    
    # Train the final model with the best parameters
    best_max_depth, best_n_estimators = best_params
    print(f"[XGBoost] Best max_depth: {best_max_depth}, Best n_estimators: {best_n_estimators} with Validation Loss: {best_loss:.4f}")
    # print(f"Best max_depth: {best_max_depth}, Best n_estimators: {best_n_estimators} with Validation AUC: {best_auc:.4f}")
    logging.info(f"[XGBoost] Best max_depth: {best_max_depth}, Best n_estimators: {best_n_estimators} with Validation Loss: {best_loss:.4f}")
    # print(f"Best max_depth: {best_max_depth}, Best n_estimators: {best_n_estimators} with Validation AUC: {best_auc:.4f}")
    params['max_depth'] = best_max_depth
    evallist = [(dtrain, 'train'), (dtest, 'valid')]
    evals_result = {}
    bst = xgb.train(params, dtrain, best_n_estimators, evals=evallist, evals_result=evals_result, verbose_eval=False)
    
    # Make predictions on test set
    y_test_pred_proba = bst.predict(dtest)
    test_loss = log_loss(y_test, y_test_pred_proba) if is_binary else log_loss(y_test, y_test_pred_proba)
    test_acc, test_auc, test_auprc, test_f1, test_recall, test_precision = compute_overall_accuracy(y_test_pred_proba, y_test, num_classes, threshold=0.5, activation=False)
    
    total_results = {
        'test_xgb_loss': test_loss,
        'test_xgb_acc': test_acc,
        'test_xgb_auc': test_auc,
        'test_xgb_auprc': test_auprc,
        'test_xgb_f1': test_f1,
        'test_xgb_recall': test_recall,
        'test_xgb_precision': test_precision,
        'best_max_depth': best_max_depth,
        'best_n_estimators': best_n_estimators
    }

    return total_results