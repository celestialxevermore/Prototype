import copy
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.metrics import compute_overall_accuracy


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes=2,
                 dropout_rate=0.3, is_binary=True):
        super().__init__()
        output_dim = 1 if is_binary else num_classes
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
        )
        self.is_binary = is_binary

    def forward(self, x):
        return self.mlp(x)


def mlp_benchmark(
    args,
    X_train, X_valid, X_test,
    y_train, y_valid, y_test,
    is_binary=True,
):
    """
    Val-free MLP benchmark.
    - 고정 epoch 학습 (early stopping 없음)
    - deepcopy 버그 수정
    - X_valid / y_valid 은 무시됨 (50/50 프로토콜)

    버그 수정 목록:
      1. best_model = model.state_dict().copy()  →  shallow copy → 항상 마지막 epoch
         수정: copy.deepcopy(model.state_dict())
      2. compute_overall_accuracy(..., 1 if is_binary else num_classes, ...)
         → binary일 때 2 전달해야 함
      3. val 기반 early stopping 제거 → 고정 epoch
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── 데이터프레임 정리 ───────────────────────────────────────
    X_train = pd.DataFrame(X_train).reset_index(drop=True)
    X_test  = pd.DataFrame(X_test).reset_index(drop=True)

    print(f"[MLP] Train: {X_train.shape} | Test: {X_test.shape}")

    num_classes = len(np.unique(y_train))
    is_binary   = (num_classes == 2)

    # ── 범주형 인코딩 ────────────────────────────────────────────
    numeric_cols     = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X_train.select_dtypes(exclude=['int64', 'float64']).columns

    if len(categorical_cols) > 0:
        X_tr_cat = pd.get_dummies(X_train[categorical_cols].astype(str), drop_first=True)
        X_te_cat = pd.get_dummies(X_test[categorical_cols].astype(str),  drop_first=True)
        for col in X_tr_cat.columns:
            if col not in X_te_cat.columns:
                X_te_cat[col] = 0
        X_te_cat = X_te_cat[X_tr_cat.columns]
    else:
        X_tr_cat = pd.DataFrame(index=X_train.index)
        X_te_cat = pd.DataFrame(index=X_test.index)

    # ── 수치형 스케일링 ──────────────────────────────────────────
    if len(numeric_cols) > 0:
        scaler = StandardScaler()
        X_tr_num = pd.DataFrame(
            scaler.fit_transform(X_train[numeric_cols].fillna(X_train[numeric_cols].mean())),
            columns=numeric_cols, index=X_train.index
        )
        X_te_num = pd.DataFrame(
            scaler.transform(X_test[numeric_cols].fillna(X_train[numeric_cols].mean())),
            columns=numeric_cols, index=X_test.index
        )
    else:
        X_tr_num = pd.DataFrame(index=X_train.index)
        X_te_num = pd.DataFrame(index=X_test.index)

    X_tr_proc = pd.concat([X_tr_num, X_tr_cat], axis=1).astype(np.float32)
    X_te_proc = pd.concat([X_te_num, X_te_cat], axis=1).astype(np.float32)

    print(f"[MLP] Processed → Train: {X_tr_proc.shape} | Test: {X_te_proc.shape}")

    if X_tr_proc.shape[1] == 0:
        raise ValueError("[MLP] No features after preprocessing!")

    # ── 텐서 변환 ────────────────────────────────────────────────
    X_tr_t = torch.from_numpy(X_tr_proc.values).float().to(device)
    X_te_t = torch.from_numpy(X_te_proc.values).float().to(device)

    if is_binary:
        y_tr_t = torch.FloatTensor(y_train).view(-1, 1).to(device)
        y_te_t = torch.FloatTensor(y_test).view(-1, 1).to(device)
    else:
        y_tr_t = torch.LongTensor(y_train).to(device)
        y_te_t = torch.LongTensor(y_test).to(device)

    # ── 데이터로더 ───────────────────────────────────────────────
    train_loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t),
        batch_size=min(args.batch_size, len(y_train)),   # few-shot 대응
        shuffle=True
    )

    # ── 모델 초기화 ──────────────────────────────────────────────
    input_dim = X_tr_proc.shape[1]
    model     = MLPClassifier(
        input_dim, args.hidden_dim, num_classes,
        args.dropout_rate, is_binary
    ).to(device)

    criterion = nn.BCEWithLogitsLoss() if is_binary else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,
                           weight_decay=1e-4)

    # ── 고정 epoch 학습 (val / early stopping 없음) ─────────────
    n_epochs = args.train_epochs
    print(f"[MLP] Training {n_epochs} epochs (no val, no early stopping)")

    for epoch in range(n_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                tr_out = model(X_tr_t)
                tr_loss = criterion(tr_out, y_tr_t).item()
            logging.info(f"[MLP] Epoch {epoch+1}/{n_epochs} | Train Loss: {tr_loss:.4f}")
            model.train()

    # ── 테스트 평가 ──────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        te_out  = model(X_te_t)
        te_loss = criterion(te_out, y_te_t).item()

        if is_binary:
            te_probs = torch.sigmoid(te_out).cpu().numpy()          # (N, 1)
        else:
            te_probs = torch.softmax(te_out, dim=1).cpu().numpy()   # (N, C)

    # ✅ 수정: binary일 때 num_classes=2 전달
    test_acc, test_auc, test_auprc, test_f1, test_recall, test_precision = \
        compute_overall_accuracy(
            te_probs, y_test, num_classes,   # ← 2 (binary) or C
            threshold=args.threshold, activation=False
        )

    print(f"[MLP] Test → Loss:{te_loss:.4f} AUC:{test_auc:.4f} AUPRC:{test_auprc:.4f} "
          f"ACC:{test_acc:.4f} F1:{test_f1:.4f}")

    return {
        'test_mlp_loss':      te_loss,
        'test_mlp_acc':       test_acc,
        'test_mlp_auc':       test_auc,
        'test_mlp_auprc':     test_auprc,
        'test_mlp_f1':        test_f1,
        'test_mlp_recall':    test_recall,
        'test_mlp_precision': test_precision,
    }