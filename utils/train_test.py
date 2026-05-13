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
        
        optimizer.step()
        total_loss += loss.item() * len(batch['y'])
        #print(f"Step [{step+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    return total_loss / len(train_loader.dataset)

def binary_evaluate(model, loader, criterion, device):
    model.eval()
    
    # 저장소 초기화
    loss_g_sum, loss_l_sum = 0.0, 0.0
    y_true = []
    y_pred_g, y_pred_l = [], []
    
    has_global = False # Global 결과가 있는지 확인용 플래그
    
    with torch.no_grad():
        for batch in loader:
            # 1. 모델 예측 호출
            # Phase 1이면 1개(Local), Phase 2면 2개(Global, Local) 리턴됨
            preds = model.predict(batch, return_all=True)
            
            target = batch['y'].to(device).view(-1, 1).float()
            
            # 2. 리턴값 개수에 따른 분기 처리
            if isinstance(preds, tuple) and len(preds) == 2:
                # [Phase 2] Global & Local 둘 다 있음
                global_pred, local_pred = preds
                has_global = True
                
                loss_g = criterion(global_pred, target)
                loss_g_sum += loss_g.item() * len(target)
                y_pred_g.extend(torch.sigmoid(global_pred).cpu().numpy())
                
            else:
                # [Phase 1] Local만 있음 (preds 자체가 tensor)
                local_pred = preds
                has_global = False
                # Global 쪽은 더미 값 처리 (안 씀)
            
            # Local은 항상 있음
            loss_l = criterion(local_pred, target)
            loss_l_sum += loss_l.item() * len(target)
            y_pred_l.extend(torch.sigmoid(local_pred).cpu().numpy())
            
            y_true.extend(target.cpu().numpy())
            
    # 3. 결과 정리
    dataset_len = len(loader.dataset)
    loss_l_avg = loss_l_sum / dataset_len
    
    res_l = (loss_l_avg, np.array(y_true), np.array(y_pred_l))
    
    if has_global:
        loss_g_avg = loss_g_sum / dataset_len
        res_g = (loss_g_avg, np.array(y_true), np.array(y_pred_g))
    else:
        # Global이 없으면 Local 값을 복사해서 리턴 (형식 맞추기 위해)
        # 또는 None을 리턴하고 밖에서 처리 (여기선 복사가 안전함)
        res_g = res_l 
        
    # 항상 2개 튜플 리턴 (Unpacking 에러 방지)
    return res_g, res_l

def binary_evaluate_(model, loader, criterion, device):
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

def regression_train(model, train_loader, criterion, optimizer, device):
    """KF regression train step. Mirrors binary_train; loss comes from model.forward (MSELoss inside)."""
    model.train()
    total_loss = 0
    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()
        loss = model(batch, batch['y'])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch['y'])
    return total_loss / len(train_loader.dataset)


def regression_evaluate(model, loader, criterion, device):
    """KF regression eval. Returns (res_g, res_l) tuple matching binary_evaluate signature.
    Each res is (loss, y_true, y_pred) — y_pred is raw model output (no sigmoid/softmax).
    """
    model.eval()
    loss_g_sum, loss_l_sum = 0.0, 0.0
    y_true = []
    y_pred_g, y_pred_l = [], []
    has_global = False

    with torch.no_grad():
        for batch in loader:
            preds = model.predict(batch, return_all=True)
            target = batch['y'].to(device).view(-1, 1).float()

            if isinstance(preds, tuple) and len(preds) == 2:
                global_pred, local_pred = preds
                has_global = True
                loss_g = criterion(global_pred, target)
                loss_g_sum += loss_g.item() * len(target)
                y_pred_g.extend(global_pred.detach().cpu().numpy())
            else:
                local_pred = preds
                has_global = False

            loss_l = criterion(local_pred, target)
            loss_l_sum += loss_l.item() * len(target)
            y_pred_l.extend(local_pred.detach().cpu().numpy())
            y_true.extend(target.cpu().numpy())

    dataset_len = len(loader.dataset)
    loss_l_avg = loss_l_sum / dataset_len
    res_l = (loss_l_avg, np.array(y_true), np.array(y_pred_l))
    if has_global:
        loss_g_avg = loss_g_sum / dataset_len
        res_g = (loss_g_avg, np.array(y_true), np.array(y_pred_g))
    else:
        res_g = res_l
    return res_g, res_l


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


# def binary_train(model, train_loader, criterion, optimizer, device):
#     model.train()
#     total_loss = 0 
#     for step, batch in enumerate(train_loader):
        
#         optimizer.zero_grad()
#         loss = model(batch, batch['y'])
#         loss.backward()

#         # --- [ ❗️❗️ 결정적인 증거 ❗️❗️ ] ---
#         # 100 스텝마다 (또는 원하는 주기로) optimizer.step()의 작동을 직접 확인
#         if (step + 1) % 100 == 0:
#             try:
#                 # 1. step() 전의 가중치 저장
#                 weights_before_step = model.latent_graph.node_embeddings.detach().clone()
                
#                 # 2. .grad 값 확인
#                 grad = model.latent_graph.node_embeddings.grad
#                 grad_norm = grad.norm().item() if grad is not None else 0.0

#                 # 3. 옵티마이저 스텝 실행 (이것이 루프의 원본 스텝입니다)
#                 optimizer.step()
                
#                 # 4. step() 후의 가중치 가져오기
#                 weights_after_step = model.latent_graph.node_embeddings.detach().clone()
                
#                 # 5. 한 스텝 동안의 실제 변화량 계산
#                 weight_diff_step = torch.norm(weights_after_step - weights_before_step).item()

#                 print("\n" + "*"*50)
#                 print(f"--- 🩺 OPTIMIZER STEP CHECK (Step: {step+1}) ---")
#                 print(f"   .grad L2-Norm (Before step): {grad_norm:.8f}")
#                 print(f"   Weight L2-Norm (After step): {weight_diff_step:.8f}")

#                 if grad_norm > 1e-8 and weight_diff_step < 1e-8:
#                     print("   -> ❗️❗️❗️ 치명적 오류: 그래디언트가 0이 아닌데도,")
#                     print("         optimizer.step()이 파라미터를 업데이트하지 않았습니다!")
#                 elif grad_norm < 1e-8:
#                      print("   -> ❗️WARNING: 그래디언트가 0이라 업데이트가 없습니다.")
#                 else:
#                     print(f"   -> ✅ 파라미터가 성공적으로 업데이트되었습니다. (변화량: {weight_diff_step:.8f})")
#                 print("*"*50 + "\n")
                
#                 # total_loss 계산 (정상 로직)
#                 total_loss += loss.item() * len(batch['y'])
                
#                 # 100번째 스텝에서는 optimizer.step()이 이미 호출되었으므로,
#                 # 루프의 맨 아래로 가서 다시 호출되는 것을 막기 위해 continue 사용
#                 continue # ----------------> ❗️ 중요

#             except Exception as e:
#                 print(f"🚨 LCG .grad 확인 실패: {e}")
#         # --- [ 추가 끝 ] ---

#         optimizer.step() # -----------------> 100번째 스텝이 아닌 경우 정상 실행
#         total_loss += loss.item() * len(batch['y'])

#     return total_loss / len(train_loader.dataset)