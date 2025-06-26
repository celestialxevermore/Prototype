import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from utils.metrics import compute_overall_accuracy
import numpy as np
import logging
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes=2, dropout_rate=0.3, is_binary=True):
        super(MLPClassifier, self).__init__()
        
        # is_binary=True일 때 out_features=1 -> sigmoid 사용
        # is_binary=False일 때 out_features=num_classes -> softmax 사용
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
            
            nn.Linear(hidden_dim, output_dim)
        )
        self.is_binary = is_binary

    def forward(self, x):
        return self.mlp(x)

def mlp_benchmark(args, X_train, X_valid, X_test, y_train, y_valid, y_test, is_binary=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 데이터프레임으로 변환
    X_train = pd.DataFrame(X_train).reset_index(drop=True)
    X_valid = pd.DataFrame(X_valid).reset_index(drop=True)
    X_test = pd.DataFrame(X_test).reset_index(drop=True)
    
    print("Initial data types:", X_train.dtypes)
    print("Any NaN in X_train:", X_train.isna().sum().sum())
    
    # 2. 수치형/범주형 컬럼 분리
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X_train.select_dtypes(exclude=['int64', 'float64']).columns
    
    # 3. 범주형 데이터 처리
    if len(categorical_cols) > 0:
        X_train_cat = pd.get_dummies(X_train[categorical_cols], drop_first=True)
        X_valid_cat = pd.get_dummies(X_valid[categorical_cols], drop_first=True)
        X_test_cat = pd.get_dummies(X_test[categorical_cols], drop_first=True)
        
        # 모든 데이터셋에 동일한 컬럼 확보
        all_columns = X_train_cat.columns
        for col in all_columns:
            if col not in X_valid_cat.columns:
                X_valid_cat[col] = 0
            if col not in X_test_cat.columns:
                X_test_cat[col] = 0
        X_valid_cat = X_valid_cat[all_columns]
        X_test_cat = X_test_cat[all_columns]
    else:
        X_train_cat = pd.DataFrame(index=X_train.index)
        X_valid_cat = pd.DataFrame(index=X_valid.index)
        X_test_cat = pd.DataFrame(index=X_test.index)
    
    # 4. 수치형 데이터 처리
    if len(numeric_cols) > 0:
        # NaN 처리
        X_train_num = X_train[numeric_cols].fillna(X_train[numeric_cols].mean())
        X_valid_num = X_valid[numeric_cols].fillna(X_train[numeric_cols].mean())
        X_test_num = X_test[numeric_cols].fillna(X_train[numeric_cols].mean())
        
        scaler = StandardScaler()
        X_train_num = pd.DataFrame(
            scaler.fit_transform(X_train_num),
            columns=numeric_cols,
            index=X_train.index
        )
        X_valid_num = pd.DataFrame(
            scaler.transform(X_valid_num),
            columns=numeric_cols,
            index=X_valid.index
        )
        X_test_num = pd.DataFrame(
            scaler.transform(X_test_num),
            columns=numeric_cols,
            index=X_test.index
        )
    else:
        X_train_num = pd.DataFrame(index=X_train.index)
        X_valid_num = pd.DataFrame(index=X_valid.index)
        X_test_num = pd.DataFrame(index=X_test.index)
    
    # 5. 수치형과 범주형 데이터 결합
    X_train_processed = pd.concat([X_train_num, X_train_cat], axis=1)
    X_valid_processed = pd.concat([X_valid_num, X_valid_cat], axis=1)
    X_test_processed = pd.concat([X_test_num, X_test_cat], axis=1)
    
    # 명시적으로 float32로 변환
    X_train_processed = X_train_processed.astype(np.float32)
    X_valid_processed = X_valid_processed.astype(np.float32)
    X_test_processed = X_test_processed.astype(np.float32)
    
    print("Data type after conversion:", X_train_processed.dtypes.unique())
    
    # 6. 텐서 변환
    X_train_tensor = torch.from_numpy(X_train_processed.values).float().to(device)
    X_valid_tensor = torch.from_numpy(X_valid_processed.values).float().to(device)
    X_test_tensor = torch.from_numpy(X_test_processed.values).float().to(device)
    
    if is_binary:
        y_train_tensor = torch.FloatTensor(y_train).view(-1, 1).to(device)
        y_valid_tensor = torch.FloatTensor(y_valid).view(-1, 1).to(device)
        y_test_tensor = torch.FloatTensor(y_test).view(-1, 1).to(device)
    else:
        y_train_tensor = torch.LongTensor(y_train).to(device)
        y_valid_tensor = torch.LongTensor(y_valid).to(device)
        y_test_tensor = torch.LongTensor(y_test).to(device)
    
    # shape 확인을 위한 디버깅 출력
    print(f"X_test shape: {X_test_tensor.shape}")
    print(f"y_test shape: {y_test_tensor.shape}")
    
    print(f"Final tensor shapes - X_train: {X_train_tensor.shape}, y_train: {y_train_tensor.shape}")
    
    # 10. 데이터셋과 데이터로더 생성
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # 11. 모델 초기화
    input_dim = X_train_processed.shape[1]
    num_classes = 1 if is_binary else len(np.unique(y_train))
    model = MLPClassifier(input_dim, args.hidden_dim, num_classes, args.dropout_rate, is_binary).to(device)
    
    criterion = nn.BCEWithLogitsLoss() if is_binary else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # 학습 설정
    n_epochs = args.train_epochs
    best_valid_metrics = {
        'loss': float('inf'),
        'f1': 0
    }
    best_model = None
    patience = 10
    counter = 0
    
    # 학습
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 검증
        model.eval()
        with torch.no_grad():
            valid_outputs = model(X_valid_tensor)
            valid_loss = criterion(valid_outputs, y_valid_tensor)
            
            if is_binary:
                valid_probs = torch.sigmoid(valid_outputs).cpu().numpy()
            else:
                valid_probs = torch.softmax(valid_outputs, dim=1).cpu().numpy()
            
            valid_acc, valid_auc, valid_auprc, valid_f1, valid_recall, valid_precision = compute_overall_accuracy(
                valid_probs, 
                y_valid, 
                1 if is_binary else num_classes, 
                threshold=args.threshold, 
                activation=False
            )
            
            logging.info(f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, "
                        f"Valid Loss: {valid_loss:.4f}, Valid F1: {valid_f1:.4f}")
            
            # Early stopping 조건 개선 (손실과 F1 점수 모두 고려)
            if valid_loss < best_valid_metrics['loss'] and valid_f1 >= best_valid_metrics['f1']:
                best_valid_metrics['loss'] = valid_loss
                best_valid_metrics['f1'] = valid_f1
                best_model = model.state_dict().copy()
                counter = 0
            else:
                counter += 1
                
            if counter >= patience:
                logging.info("Early stopping triggered!")
                break
    
    # 최종 평가
    model.load_state_dict(best_model)
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_loss = criterion(test_outputs, y_test_tensor)
        
        if is_binary:
            test_probs = torch.sigmoid(test_outputs).cpu().numpy()
            test_preds = (test_probs >= args.threshold).astype(int)
        else:
            test_probs = torch.softmax(test_outputs, dim=1).cpu().numpy()
            test_preds = np.argmax(test_probs, axis=1)
            
        test_acc, test_auc, test_auprc, test_f1, test_recall, test_precision = compute_overall_accuracy(
            test_probs,
            y_test,
            1 if is_binary else num_classes,
            threshold=args.threshold,
            activation=False
        )
    
    
    total_results = {
        'test_mlp_loss': test_loss.item(),
        'test_mlp_acc': test_acc,
        'test_mlp_auc': test_auc,
        'test_mlp_auprc': test_auprc,
        'test_mlp_f1': test_f1,
        'test_mlp_recall': test_recall,
        'test_mlp_precision': test_precision
    }
    
    return total_results, model