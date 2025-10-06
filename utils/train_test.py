import torch 
import torch.nn as nn 
import numpy as np 
#from dataset.data_dataloaders import CombinedDataLoader

def binary_train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0 
    for step, batch in enumerate(train_loader):
        

        optimizer.zero_grad()
        loss = model(batch, batch['y'])
        #output = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss.backward()
        # u_params = [p for n,p in model.basis_affinity.named_parameters()
        #     if ('U_param' in n) and (p.grad is not None)]
        # if u_params:
        #     import torch
        #     g_norm = torch.linalg.vector_norm(
        #         torch.cat([p.grad.reshape(-1) for p in u_params])
        #     ).item()
        # else:
        #     g_norm = 0.0
        # print(f"[grad] ||dL/dU_param|| = {g_norm:.3e}")
        optimizer.step()
        total_loss += loss.item() * len(batch['y'])
        #print(f"Step [{step+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    return total_loss / len(train_loader.dataset)

def binary_evaluate(model, loader, criterion, device):
    model.eval()
    test_loss = 0
    y_true, y_pred = [], []
    
    #print(f"Dataloader length: {len(loader)}")
    
    with torch.no_grad():
        for batch in loader:
            pred = model.predict(batch)
            loss = model(batch, batch['y'])
            
            test_loss += loss.item() * len(batch['y'])
            
            y_true.extend(batch['y'].cpu().numpy())
            y_pred.extend(torch.sigmoid(pred).cpu().numpy())
            
            #print(f"Current batch - y_true: {len(y_true)}, y_pred: {len(y_pred)}")
    
    test_loss /= len(loader.dataset)
    return test_loss, np.array(y_true), np.array(y_pred)

def multi_train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0 
    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()
        loss = model(batch, batch['y'])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch['y'])
    return total_loss / len(train_loader.dataset)

def multi_evaluate(model, loader, criterion, device):
    model.eval()
    test_loss = 0
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for batch in loader:
            pred = model.predict(batch)  # 모델의 예측값
            loss = model(batch, batch['y'])  # 손실 계산
            
            test_loss += loss.item() * len(batch['y'])
            
            y_true.extend(batch['y'].cpu().numpy())
            # multi-class이므로 softmax 적용
            y_pred.extend(torch.softmax(pred, dim=1).cpu().numpy())
    
    test_loss /= len(loader.dataset)
    return test_loss, np.array(y_true), np.array(y_pred)

def _binary_log_loss(y_true, y_prob, eps=1e-7):
    p = np.clip(np.asarray(y_prob), eps, 1 - eps)
    y = np.asarray(y_true).astype(np.float32)
    return float(-np.mean(y * np.log(p) + (1-y) * np.log(1-p)))

def _multiclass_log_loss(y_true, y_prob, eps=1e-7):
    P = np.asarray(y_prob)
    P = np.clip(P, eps, 1 - eps)
    P = P / P.sum(axis = 1, keepdims=True)
    y = np.asarray(y_true).astype(int)
    return float(-np.mean(np.log(P[np.arange(len(y)), y])))