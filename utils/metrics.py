from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc, f1_score
from sklearn.preprocessing import label_binarize
import numpy as np
import pdb
from scipy.special import softmax
from scipy.special import expit as sigmoid
import torch
from sklearn.metrics import average_precision_score

# 이진 분류의 경우
def calculate_binary_accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        # 만약 y_pred가 확률 분포라면 (예: [[0.1, 0.9], [0.8, 0.2], ...])
        y_pred_class = y_pred.argmax(axis=1)
    else:
        # 만약 y_pred가 단일 열이라면
        y_pred = y_pred.flatten()  # 1차원으로 평탄화
        y_pred_class = (y_pred > 0.5).astype(int)  # 0.5를 임계값으로 사용
    
    return np.mean(y_pred_class == y_true)

# 다중 클래스 분류의 경우
def calculate_multiclass_accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if len(y_pred.shape) == 1 or y_pred.shape[1] == 1:
        # y_pred가 1차원 배열이거나 단일 열인 경우
        return np.mean(y_pred == y_true)
    else:
        # y_pred가 2차원 배열인 경우 (각 클래스에 대한 확률)
        y_pred_class = np.argmax(y_pred, axis=1)
        return np.mean(y_pred_class == y_true)

def get_best_performance(test_aucs, test_precisions, test_recalls, test_f1s, all_y_true, all_y_pred, is_binary):
    # AUC 기준으로 가장 좋은 Epoch 선택
    best_epoch = test_aucs.index(max(test_aucs))
    best_auc = test_aucs[best_epoch]

    # 해당 Epoch의 y_true와 y_pred 가져오기
    y_true = all_y_true[best_epoch]
    y_pred = all_y_pred[best_epoch]

    if is_binary:
        # Binary classification
        best_accuracy = calculate_binary_accuracy(y_true, y_pred)
        best_precision = test_precisions[best_epoch]
        best_recall = test_recalls[best_epoch]
        best_f1 = test_f1s[best_epoch]
    else:
        # Multiclass classification
        best_accuracy = calculate_multiclass_accuracy(y_true, np.argmax(y_pred, axis=1))
        best_precision = test_precisions[best_epoch]
        best_recall = test_recalls[best_epoch]
        best_f1 = test_f1s[best_epoch]

    return best_epoch, best_auc, best_accuracy, best_precision, best_recall, best_f1


def compute_overall_accuracy(probs, labels, num_classes, threshold=0.5, activation=False):
    if num_classes == 2:  # 이진 분류인 경우
        if activation:
            probs = torch.sigmoid(probs).cpu().numpy()
        pred = (probs >= threshold).astype(int)
        
        accuracy = accuracy_score(labels, pred)
        auc = roc_auc_score(labels, probs)
        auprc = average_precision_score(labels, probs)
        f1 = f1_score(labels, pred)
        recall = recall_score(labels, pred)
        precision = precision_score(labels, pred)
        
    else:  # 다중 분류인 경우
        if len(probs.shape) == 1:
            probs = probs.reshape(-1, 1)
        pred = np.argmax(probs, axis=1)
        
        accuracy = accuracy_score(labels, pred)
        auc = roc_auc_score(labels, probs, multi_class='ovr')
        auprc = average_precision_score(labels, probs)
        f1 = f1_score(labels, pred, average='macro')
        recall = recall_score(labels, pred, average='macro')
        precision = precision_score(labels, pred, average='macro')
    
    return accuracy, auc, auprc, f1, recall, precision