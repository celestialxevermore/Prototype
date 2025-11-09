import torch 
import torch.nn as nn 
import numpy as np 
#from dataset.data_dataloaders import CombinedDataLoader

def binary_train(model, train_loader, criterion, optimizer, device, debug_first=False):
    model.train()
    total_loss = 0 
    
    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()
        
        # ✅ 첫 step만 디버깅 - Before forward
        if debug_first and step == 0:
            print(f"\n{'='*70}")
            print(f"🔍 [TRAINING] Before forward pass")
            print(f"{'='*70}")
            emb_before = model.latent_graph.node_embeddings.detach().clone()
            print(f"  LCG embeddings (before):")
            print(f"    mean: {emb_before.mean().item():.8f}")
            print(f"    std: {emb_before.std().item():.8f}")
            print(f"    min/max: {emb_before.min().item():.6f} / {emb_before.max().item():.6f}")
        
        loss = model(batch, batch['y'])
        
        # ✅ 첫 step만 디버깅 - After forward, before backward
        if debug_first and step == 0:
            print(f"\n{'='*70}")
            print(f"🔍 [TRAINING] After forward, before backward")
            print(f"{'='*70}")
            print(f"  total_loss: {loss.item():.6f}")
            if hasattr(model, 'fgw_loss'):
                print(f"  fgw_loss: {model.fgw_loss.item():.6f}")
                print(f"  fgw_loss.requires_grad: {model.fgw_loss.requires_grad}")
        
        loss.backward()
        
        # ✅ 첫 step만 디버깅 - After backward
        if debug_first and step == 0:
            print(f"\n{'='*70}")
            print(f"🔍 [TRAINING] After backward")
            print(f"{'='*70}")
            
            lcg_grad = model.latent_graph.node_embeddings.grad
            if lcg_grad is None:
                print(f"  ❌❌❌ CRITICAL ERROR: node_embeddings.grad is None!")
                print(f"  → Gradient is NOT flowing to LCG!")
            else:
                print(f"  ✅ node_embeddings.grad exists!")
                print(f"    norm: {lcg_grad.norm().item():.6f}")
                print(f"    mean: {lcg_grad.mean().item():.8f}")
                print(f"    std: {lcg_grad.std().item():.8f}")
                print(f"    min/max: {lcg_grad.min().item():.6f} / {lcg_grad.max().item():.6f}")
                
                if lcg_grad.norm() < 1e-6:
                    print(f"  ⚠️ WARNING: Gradient norm too small (vanishing)!")
        
        optimizer.step()
        
        # ✅ 첫 step만 디버깅 - After optimizer step
        if debug_first and step == 0:
            print(f"\n{'='*70}")
            print(f"🔍 [TRAINING] After optimizer.step")
            print(f"{'='*70}")
            
            emb_after = model.latent_graph.node_embeddings.detach()
            diff = (emb_after - emb_before).abs()
            
            print(f"  LCG embeddings (after):")
            print(f"    mean: {emb_after.mean().item():.8f}")
            print(f"    std: {emb_after.std().item():.8f}")
            print(f"    min/max: {emb_after.min().item():.6f} / {emb_after.max().item():.6f}")
            
            print(f"\n  Update magnitude:")
            print(f"    mean diff: {diff.mean().item():.8f}")
            print(f"    max diff: {diff.max().item():.8f}")
            
            if diff.max() < 1e-8:
                print(f"  ❌ PROBLEM: Weights barely changed!")
            elif diff.max() < 1e-5:
                print(f"  ⚠️ WARNING: Small update (might be ok)")
            else:
                print(f"  ✅ SUCCESS: Weights updated!")
            
            print(f"{'='*70}\n")
        
        total_loss += loss.item() * len(batch['y'])
    
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