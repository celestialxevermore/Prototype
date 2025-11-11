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
        loss.backward()

        # --- [ ❗️❗️ 결정적인 증거 ❗️❗️ ] ---
        # 100 스텝마다 (또는 원하는 주기로) optimizer.step()의 작동을 직접 확인
        if (step + 1) % 1 == 0:
            try:
                # 1. step() 전의 가중치 저장
                weights_before_step = model.latent_graph.node_embeddings.detach().clone()
                
                # 2. .grad 값 확인
                grad = model.latent_graph.node_embeddings.grad
                grad_norm = grad.norm().item() if grad is not None else 0.0

                # 3. 옵티마이저 스텝 실행 (이것이 루프의 원본 스텝입니다)
                optimizer.step()
                
                # 4. step() 후의 가중치 가져오기
                weights_after_step = model.latent_graph.node_embeddings.detach().clone()
                
                # 5. 한 스텝 동안의 실제 변화량 계산
                weight_diff_step = torch.norm(weights_after_step - weights_before_step).item()

                print("\n" + "*"*50)
                print(f"--- 🩺 OPTIMIZER STEP CHECK (Step: {step+1}) ---")
                print(f"   .grad L2-Norm (Before step): {grad_norm:.8f}")
                print(f"   Weight L2-Norm (After step): {weight_diff_step:.8f}")

                if grad_norm > 1e-8 and weight_diff_step < 1e-8:
                    print("   -> ❗️❗️❗️ 치명적 오류: 그래디언트가 0이 아닌데도,")
                    print("         optimizer.step()이 파라미터를 업데이트하지 않았습니다!")
                elif grad_norm < 1e-8:
                     print("   -> ❗️WARNING: 그래디언트가 0이라 업데이트가 없습니다.")
                else:
                    print(f"   -> ✅ 파라미터가 성공적으로 업데이트되었습니다. (변화량: {weight_diff_step:.8f})")
                print("*"*50 + "\n")
                
                # total_loss 계산 (정상 로직)
                total_loss += loss.item() * len(batch['y'])
                
                # 100번째 스텝에서는 optimizer.step()이 이미 호출되었으므로,
                # 루프의 맨 아래로 가서 다시 호출되는 것을 막기 위해 continue 사용
                continue # ----------------> ❗️ 중요

            except Exception as e:
                print(f"🚨 LCG .grad 확인 실패: {e}")
        # --- [ 추가 끝 ] ---

        optimizer.step() # -----------------> 100번째 스텝이 아닌 경우 정상 실행
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