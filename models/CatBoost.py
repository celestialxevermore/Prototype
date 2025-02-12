from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, log_loss
from utils.metrics import compute_overall_accuracy
import numpy as np
import logging

def catboost_benchmark(args, X_train, X_valid, X_test, y_train, y_valid, y_test, is_binary=True, depth_list=[1, 2, 3, 4, 5], iterations_list=[100, 200, 300, 400, 500]):
    print("Input columns:", X_train.columns.tolist())
    categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_columns = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    print(f"Categorical: {categorical_columns}, Numeric: {numeric_columns}")
    
    # Determine number of classes
    num_class = len(np.unique(y_train))
    is_binary = num_class == 2
    
    # Find the best depth and iterations
    best_loss = float('inf')
    best_roc = 0
    best_params = None
    for depth in depth_list:
        for iterations in iterations_list:
            model = CatBoostClassifier(
                depth=depth,
                iterations=iterations,
                loss_function='Logloss' if is_binary else 'MultiClass',
                eval_metric='AUC' if is_binary else 'MultiClass',
                cat_features=categorical_columns,
                verbose=0
            )
            
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True)
            
            # Make predictions on validation set
            y_valid_pred_proba = model.predict_proba(X_valid)[:, 1] if is_binary else model.predict_proba(X_valid)
            valid_loss = log_loss(y_valid, y_valid_pred_proba)
            valid_acc, valid_auc, valid_auprc, valid_f1, valid_recall, valid_precision = compute_overall_accuracy(y_valid_pred_proba, y_valid, num_class, threshold=args.threshold, activation=False)
            

            print(f"depth: {depth}, iterations: {iterations}, Validation Loss: {valid_loss:.4f}, Validation Acc: {valid_acc:.4f}, Validation AUC: {valid_auc:.4f}, Validation AUPRC: {valid_auprc:.4f}, Validation F1: {valid_f1:.4f}, Validation Recall: {valid_recall:.4f}, Validation Precision: {valid_precision:.4f}")
            logging.info(f"depth: {depth}, iterations: {iterations}, Validation Loss: {valid_loss:.4f}, Validation Acc: {valid_acc:.4f}, Validation AUC: {valid_auc:.4f}, Validation AUPRC: {valid_auprc:.4f}, Validation F1: {valid_f1:.4f}, Validation Recall: {valid_recall:.4f}, Validation Precision: {valid_precision:.4f}")
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_params = (depth, iterations)
            # if valid_auc > best_roc:
            #     best_roc = valid_auc
            #     best_params = (depth, iterations)
    
    # Train the final model with the best parameters
    best_depth, best_iterations = best_params
    print(f"Best depth: {best_depth}, Best iterations: {best_iterations} with Validation Loss: {best_loss:.4f}")
    # print(f"Best depth: {best_depth}, Best iterations: {best_iterations} with Validation AUC: {best_roc:.4f}")
    logging.info(f"Best depth: {best_depth}, Best iterations: {best_iterations} with Validation Loss: {best_loss:.4f}")
    # print(f"Best depth: {best_depth}, Best iterations: {best_iterations} with Validation AUC: {best_roc:.4f}")
    model = CatBoostClassifier(
        depth=best_depth,
        iterations=best_iterations,
        loss_function='Logloss' if is_binary else 'MultiClass',
        eval_metric='AUC' if is_binary else 'MultiClass',
        cat_features=categorical_columns,
        verbose=0
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions on test set
    y_test_pred_proba = model.predict_proba(X_test)[:, 1] if is_binary else model.predict_proba(X_test)
    test_loss = log_loss(y_test, y_test_pred_proba)
    test_acc, test_auc, test_auprc, test_f1, test_recall, test_precision = compute_overall_accuracy(y_test_pred_proba, y_test, num_class, threshold=args.threshold, activation=False)
    
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