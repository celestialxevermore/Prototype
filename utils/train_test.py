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
    for step, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        #output = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs 
        #print(f"Step [{step+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    return total_loss / len(train_loader.dataset)

def multi_evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0 
    y_true = []
    y_pred = [] 
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data.x, data.edge_index, data.batch)
            #output = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = criterion(output, data.y)
            total_loss += loss.item() * data.num_graphs
            y_true.extend(data.y.cpu().numpy())
            y_pred.extend(torch.softmax(output, dim=1).cpu().numpy())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if y_pred.ndim == 1: 
        y_pred = y_pred.reshape(-1,1)
    return total_loss / len(loader.dataset), y_true, y_pred
