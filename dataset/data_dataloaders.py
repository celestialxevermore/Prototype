import torch
import torch.nn as nn 
import pandas as pd
import numpy as np
import sys
import pdb
sys.path.append('..')
from dataset.data import get_dataset, get_dataset_ml
from utils.table_to_graph import Table2GraphTransformer
from utils.table_to_embedding import Table2EmbeddingTransformer
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
#from torch_geometric.data import DataLoader
#from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DataLoader, Dataset
from collections import Counter
# from torch_geometric.loader import DataLoader as PyGDataLoader
# from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader
import os, pickle
import os, pickle
import random
import warnings
def preprocessing_heart_datasets(DATASETS: pd.DataFrame, data_name: str):
    """
    cleveland, hungarian, switzerland, heart_statlog 데이터셋을 위한 전처리 함수
    """
    heart_datasets = ['cleveland', 'hungarian', 'switzerland', 'heart_statlog']
    assert data_name in heart_datasets, f"{data_name} is not a valid heart dataset name"
    
    # target_binary 컬럼이 이미 'no'/'yes'로 되어있음
    X = DATASETS[data_name][0].drop('target_binary', axis=1)
    X = X.reset_index(drop=True)

    y = DATASETS[data_name][0]['target_binary']
    # 'no'/'yes'를 0/1로 변환
    y = (y == 'yes').astype(int)
    y = y.reset_index(drop=True)
    
    return X, y

def preprocessing(DATASETS : pd.DataFrame, data_name : str):
    # 기존 데이터셋을 위한 전처리
    dataset_and_class = {
            "adult" : ['income', ['no','yes']],
            "bank" : ['Class', ['no','yes']],
            "blood" : ['Class',['no','yes']],
            "car" : ['class',['unacceptable','acceptable','good','very good']],
            "communities" : ['ViolentCrimesPerPop',['high','medium','low']],
            "credit-g" : ['class', ['no','yes']],
            "diabetes": ['Outcome',['no','yes']],
            "myocardial" : ['ZSN',['no','yes']],
            "cleveland": ['target_binary', ['no','yes']],
            "hungarian": ['target_binary', ['no','yes']],
            "switzerland": ['target_binary', ['no','yes']],
            "heart_statlog": ['target_binary', ['no','yes']],
            "heart": ['target_binary', ['no','yes']]
        }
    
    # heart 데이터셋들은 새로운 전처리 함수로 처리
    heart_datasets = ['cleveland', 'hungarian', 'switzerland', 'heart_statlog']
    if data_name in heart_datasets:
        return preprocessing_heart_datasets(DATASETS, data_name)
        
    assert data_name in dataset_and_class, f"{data_name} is not a valid dataset name"

    class_name = dataset_and_class[data_name][0]
    class_values = dataset_and_class[data_name][1]
    
    class_mapping = {label: idx for idx, label in enumerate(class_values)}
    
    X = DATASETS[data_name][0].drop(class_name, axis=1)
    X = X.reset_index()

    y = DATASETS[data_name][0][class_name]
    y = y.map(class_mapping).astype(int)
    y = y.reset_index(drop=True)
    
    return X, y

'''
    Prepare each datasets 
    Table -> Graph
'''
def ml_prepare_tabular_dataloaders(args, dataset_name, random_seed):
    """데이터셋을 로드하고 train/val/test로 분할"""
    DATASETS = {} 
    DATASETS[dataset_name] = get_dataset_ml(args, dataset_name, random_seed)

    # heart 데이터셋들 처리
    heart_datasets = ['cleveland', 'hungarian', 'switzerland', 'heart_statlog']
    if dataset_name in heart_datasets:
        X = DATASETS[dataset_name][0].drop('target_binary', axis=1)
        y = (DATASETS[dataset_name][0]['target_binary'] == 'yes').astype(int)
    else:
        X, y = preprocessing(DATASETS, dataset_name)

    le = LabelEncoder()
    y = le.fit_transform(y)
    num_classes = len(le.classes_)
    
    # Train/val/test split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=args.random_seed, 
        stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, 
        test_size=0.25,  # 0.25 * 0.8 = 0.2 for validation
        random_state=args.random_seed, 
        stratify=y_temp
    )
    
    return (X_train, X_val, X_test, y_train, y_val, y_test), num_classes



'''
    주석된 tabular 코드: 딥러닝과 동일한 "총 K개 분배" 방식

    N=2, K=4 → 총 4개 샘플
'''


def get_few_shot_tabular_samples(X_train, y_train, args):
    """이미 분할된 train set에서 few-shot 샘플링을 수행"""
    #np.random.seed(4)
    num_classes = len(np.unique(y_train))
    shot = args.few_shot
    samples_per_class = shot // num_classes
    remainder = shot % num_classes
    
    support_X, support_y = [], []
    for cls in range(num_classes):
        cls_indices = np.where(y_train == cls)[0]
        sample_num = samples_per_class + (1 if remainder > 0 else 0)
        if remainder > 0:
            remainder -= 1
        
        selected_indices = np.random.choice(
            cls_indices, 
            size=min(sample_num, len(cls_indices)), 
            replace=len(cls_indices) < sample_num
        )
        
        support_X.append(X_train.iloc[selected_indices])
        support_y.extend([cls] * len(selected_indices))
    
    X_train_few = pd.concat(support_X, ignore_index=True)
    y_train_few = np.array(support_y)
    
    print(f"Few-shot 학습 데이터 크기: {len(X_train_few)}")
    print(f"클래스별 분포: {Counter(y_train_few)}")
    
    return X_train_few, y_train_few



'''
    이 아래 코드가 원래 쓰던 버전인데, 이는 각 클래스당 K 개 샘플을 할당하고 있음.
    반면에, 딥러닝 모델은 K가 주어지면, 각 클래스의 개수만큼 나눈 몫을 각 클래스에 할당하고 있음. 
    결과적으로 내가 지금까지 불리한 조건에서 경쟁하고 있음. 따라서, 위에 구현된 버전으로 다시 학습. 
    2025.07.19에 이 아래 코드 주석처리함. 
    현재 tabular 코드: "클래스당 K개" 방식

    N=2, K=4 → 총 8개 샘플

    딥러닝 코드: "총 K개 분배" 방식

    N=2, K=4 → 총 4개 샘플
'''
# def get_few_shot_tabular_samples(X_train, y_train, args):
#     """이미 분할된 train set에서 few-shot 샘플링을 수행 (표준 K-shot)"""
#     num_classes = len(np.unique(y_train))
#     shot_per_class = args.few_shot  # 각 클래스당 shot 개수
    
#     support_X, support_y = [], []
#     for cls in range(num_classes):
#         cls_indices = np.where(y_train == cls)[0]
        
#         selected_indices = np.random.choice(
#             cls_indices, 
#             size=min(shot_per_class, len(cls_indices)), 
#             replace=len(cls_indices) < shot_per_class
#         )
        
#         support_X.append(X_train.iloc[selected_indices])
#         support_y.extend([cls] * len(selected_indices))
    
#     X_train_few = pd.concat(support_X, ignore_index=True)
#     y_train_few = np.array(support_y)
    
#     total_samples = num_classes * shot_per_class
#     print(f"Few-shot 학습 데이터 크기: {len(X_train_few)} (={num_classes} classes × {shot_per_class} shots)")
#     print(f"클래스별 분포: {Counter(y_train_few)}")
    
#     return X_train_few, y_train_few



def collate_fn(batch):
    # batch에서 X와 y를 분리
    X = pd.DataFrame([item[0] for item in batch])  # Series들을 DataFrame으로 변환
    y = torch.stack([item[1] for item in batch])
    return X, y

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = X  # DataFrame 그대로 저장
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # DataFrame row를 Series로 반환
        return self.X.iloc[idx], self.y[idx]

def prepare_tabular_dataloaders(args, dataset_name, random_seed):
    """데이터셋을 로드하고 train/val/test로 분할한 뒤 DataLoader로 변환"""
    
    # --------------------------------------------------------------------------------
    # 1) 재현성을 위한 설정
    # --------------------------------------------------------------------------------
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

    # --------------------------------------------------------------------------------
    # 2) (X, y) 만들기: heart 데이터셋이면 preprocessing_heart_datasets, 아니면 preprocessing
    # --------------------------------------------------------------------------------
    DATASETS = {}
    DATASETS[dataset_name] = get_dataset(args, dataset_name, random_seed)
    
    # 데이터 전처리
    if dataset_name in ['cleveland', 'hungarian', 'switzerland', 'heart_statlog']:
        X = DATASETS[dataset_name][0].drop('target_binary', axis=1)
        y = (DATASETS[dataset_name][0]['target_binary'] == 'yes').astype(int)
    else:
        X, y = preprocessing(DATASETS, dataset_name)

    # Train/Val/Test Split (60/20/20)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_seed
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=random_seed
    )

    maker = Table2EmbeddingTransformer(args, args.source_dataset_name)
    maker.fit(X_train, y_train)
    train_dataset = maker.transform(X_train, y_train)
    val_dataset = maker.transform(X_val)
    test_dataset = maker.transform(X_test)

    for emb, label in zip(val_dataset, y_val):
        emb['y'] = torch.tensor([label], dtype = torch.long)
    for emb, label in zip(test_dataset, y_test):
        emb['y'] = torch.tensor([label], dtype = torch.long)
    

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
    )
    
    return {
        'data': (X_train, X_val, X_test, y_train, y_val, y_test),
        'loaders': (train_loader, val_loader, test_loader),
        'num_classes': len(np.unique(y))
    }

def prepare_few_shot_dataloaders(X_train_few, y_train_few, val_loader, test_loader, args):
    """Few-shot 데이터에 대한 DataLoader 생성"""
    
    # Table2EmbeddingTransformer 인스턴스 생성 및 학습
    maker = Table2EmbeddingTransformer(args, args.source_dataset_name)
    maker.fit(X_train_few, y_train_few)
    
    # Few-shot 학습 데이터 변환
    train_dataset_few = maker.transform(X_train_few, y_train_few)
    
    # Few-shot 학습 데이터용 DataLoader 생성
    train_loader_few = DataLoader(
        train_dataset_few,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    # validation과 test loader는 그대로 사용
    return {
        'train': train_loader_few,
        'val': val_loader,
        'test': test_loader
    }







def prepare_embedding_dataloaders(args, dataset_name):
    """저장된 embedding 데이터를 로드하고 train/val/test로 분할한 뒤 DataLoader로 변환"""
    
    '''
        최대 성능을 위해 주석처리함. 
        2025.07.24. 주석처리 된 코드를 재활성화하면, 
        재현성이 확실히 보장됨.
    '''
    # 재현성을 위한 설정
    # os.environ['PYTHONHASHSEED'] = str(args.random_seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = True
    # 파일 경로 설정
    base_path = "/storage/personal/eungyeop/dataset/embedding"
    if args.embed_type == "_":
        sub_dir = sub_dir = f"tabular_embeddings_/{args.llm_model}"
    else:
        sub_dir = f"tabular_embeddings_{args.embed_type}/{args.llm_model}"
    emb_name =  f"embedding_{dataset_name}.pkl"
    file_path = os.path.join(base_path, sub_dir, emb_name)
    
    print(f"Loading embeddings from: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # 데이터 로드
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    embeddings = data['embeddings']
    labels = [emb['y'].item() for emb in embeddings]
    num_classes = data['num_classes']
    
    # Train/Val/Test Split (60/20/20)
    indices = list(range(len(embeddings)))
    train_val_idx, test_idx = train_test_split(
        indices, test_size=0.2,
        stratify=labels,
        random_state=args.random_seed
    )
    
    train_val_embeddings = [embeddings[i] for i in train_val_idx]
    train_val_labels = [labels[i] for i in train_val_idx]
    
    train_idx, val_idx = train_test_split(
        list(range(len(train_val_embeddings))),
        test_size=0.25,
        stratify=train_val_labels,
        random_state=args.random_seed
    )
    
    # Split datasets
    train_dataset = [train_val_embeddings[i] for i in train_idx]
    val_dataset = [train_val_embeddings[i] for i in val_idx]
    test_dataset = [embeddings[i] for i in test_idx]
    
    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    print(f"Training data size: {len(train_dataset)}")
    print(f"Validation data size: {len(val_dataset)}")
    print(f"Test data size: {len(test_dataset)}")
    
    return {
        'loaders': (train_loader, val_loader, test_loader),
        'num_classes': num_classes
    }


# def get_few_shot_embedding_samples(train_loader, args):
#     """train_loader에서 embedding data의 few-shot 샘플링을 수행"""
#     np.random.seed(args.random_seed)
#     dataset = train_loader.dataset
#     labels = [data['y'].item() for data in dataset]
#     num_classes = len(set(labels))
    
#     shot = args.few_shot
#     base_samples_per_class = shot // num_classes
#     remainder = shot % num_classes
    
#     # 남은 샘플을 랜덤하게 분배
#     extra_samples = random.sample(range(num_classes), remainder)
    
#     support_data = []
#     for cls in range(num_classes):
#         # embedding data는 dictionary 형태이므로 y key로 label 접근
#         cls_data = [data for data in dataset if data['y'].item() == cls]
#         sample_num = base_samples_per_class + (1 if cls in extra_samples else 0)
        
#         if len(cls_data) < sample_num:
#             warnings.warn(f"Class {cls} has fewer samples ({len(cls_data)}) than required ({sample_num}). Using replacement sampling.")
#             selected = random.choices(cls_data, k=sample_num)
#         else:
#             selected = random.sample(cls_data, k=sample_num)
        
#         support_data.extend(selected)
    
#     print(f"Few-shot training data size: {len(support_data)}")
#     class_dist = Counter([data['y'].item() for data in support_data])
#     print(f"Class distribution in few-shot data: {dict(class_dist)}")
    
#     return DataLoader(support_data, batch_size=args.batch_size, shuffle=True)

def get_few_shot_embedding_samples(train_loader, args):
   """train_loader에서 embedding data의 few-shot 샘플링을 수행 (표준 K-shot)"""
   np.random.seed(args.random_seed)
   dataset = train_loader.dataset
   labels = [data['y'].item() for data in dataset]
   num_classes = len(set(labels))
   
   shot_per_class = args.few_shot  # 각 클래스당 shot 개수
   
   support_data = []
   for cls in range(num_classes):
       # embedding data는 dictionary 형태이므로 y key로 label 접근
       cls_data = [data for data in dataset if data['y'].item() == cls]
       
       if len(cls_data) < shot_per_class:
           warnings.warn(f"Class {cls} has fewer samples ({len(cls_data)}) than required ({shot_per_class}). Using replacement sampling.")
           selected = random.choices(cls_data, k=shot_per_class)
       else:
           selected = random.sample(cls_data, k=shot_per_class)
       
       support_data.extend(selected)
   
   total_samples = num_classes * shot_per_class
   print(f"Few-shot training data size: {len(support_data)} (={num_classes} classes × {shot_per_class} shots)")
   class_dist = Counter([data['y'].item() for data in support_data])
   print(f"Class distribution in few-shot data: {dict(class_dist)}")
   
   return DataLoader(support_data, batch_size=args.batch_size, shuffle=True)







def load_tabular_and_split(args, DATASETS, dataset_name, few_shot=False):
    """
    기존 DATASETS(딕셔너리 형태: DATASETS[dataset_name] = (DataFrame, ...))에서
    찾고자 하는 dataset_name의 데이터를 꺼내어, (X, y) 형태로 만든 뒤,
    train / val / test로 나눠주는 함수.
    
    - preprocessing(DATASETS, dataset_name)를 통해 (X, y) 얻기
    - heart_datasets는 별도 함수( preprocessing_heart_datasets )에서 처리
    - few_shot=True일 경우, train 데이터의 클래스를 shot 만큼만 추출
    """

    # --------------------------------------------------------------------------------
    # 1) 재현성을 위한 설정
    # --------------------------------------------------------------------------------
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)

    # --------------------------------------------------------------------------------
    # 2) (X, y) 만들기: heart 데이터셋이면 preprocessing_heart_datasets, 아니면 preprocessing
    # --------------------------------------------------------------------------------
    from .data_dataloaders import preprocessing  # 본인 실제 경로에 따라 조정
    X, y = preprocessing(DATASETS, dataset_name)

    # --------------------------------------------------------------------------------
    # 3) 클래스로부터 학습/검증/테스트 split
    # --------------------------------------------------------------------------------
    class_labels = y.values
    num_classes = len(set(class_labels))

    # 8:2 (train_val:test) split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        stratify=class_labels, 
        random_state=args.random_seed
    )

    # train_val에서 25%를 val로 나누어 최종 train:val:test = 60%:20%:20%
    class_labels_train_val = y_train_val.values
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=0.25,
        stratify=class_labels_train_val,
        random_state=args.random_seed
    )

    # --------------------------------------------------------------------------------
    # 4) Few-shot이면, train 데이터를 클래스별 shot만큼 뽑아 subset으로 만든다
    # --------------------------------------------------------------------------------
    if few_shot:
        print("Preparing Few-shot Dataset...")
        shot = args.few_shot
        base_samples_per_class = shot // num_classes
        remainder = shot % num_classes
        extra_samples = random.sample(range(num_classes), remainder)

        X_train_few = []
        y_train_few = []
        for cls in range(num_classes):
            cls_indices = np.where(y_train == cls)[0]
            sample_num = base_samples_per_class + (1 if cls in extra_samples else 0)

            if len(cls_indices) < sample_num:
                warnings.warn(f"Class {cls} has fewer samples ({len(cls_indices)}) than required ({sample_num}). Using replacement sampling.")
                selected_indices = random.choices(cls_indices, k=sample_num)
            else:
                selected_indices = random.sample(list(cls_indices), k=sample_num)

            X_train_few.append(X_train.iloc[selected_indices])
            y_train_few.append(y_train.iloc[selected_indices])

        X_train = pd.concat(X_train_few, axis=0)
        y_train = pd.concat(y_train_few, axis=0)

        print(f"Few-shot training data size: {len(X_train)}")
        class_dist = Counter(y_train)
        print(f"Class distribution in few-shot data: {dict(class_dist)}")
    else:
        print("Using full training dataset.")
        print(f"Training data size: {len(X_train)}")

    # --------------------------------------------------------------------------------
    # 5) PyTorch Dataset 정의
    # --------------------------------------------------------------------------------
    class TabularDataset(Dataset):
        def __init__(self, X_df, y_df):
            self.X = X_df.values
            self.y = y_df.values
        def __len__(self):
            return len(self.X)
        def __getitem__(self, idx):
            # float32로 변환
            X_item = torch.tensor(self.X[idx], dtype=torch.float32)
            y_item = torch.tensor(self.y[idx], dtype=torch.long)
            return X_item, y_item

    # 데이터셋 구성
    train_dataset = TabularDataset(X_train, y_train)
    val_dataset   = TabularDataset(X_val,   y_val)
    test_dataset  = TabularDataset(X_test,  y_test)

    # --------------------------------------------------------------------------------
    # 6) DataLoader 생성
    # --------------------------------------------------------------------------------
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False)

    pdb.set_trace()
    return train_loader, val_loader, test_loader, num_classes













'''

    Newly added code : 2024.12.18.

'''


'''
    for few shot settings
'''

def prepare_fewshot_source_dataset(args, X_train_full, y_train_full, X_test, y_test, random_state=42):
    np.random.seed(random_state)
    num_classes = len(np.unique(y_train_full))
    
    shot = args.dataset_shot
    samples_per_class = shot // num_classes
    remainder = shot % num_classes

    support_X, support_y = [], []
    for cls in range(num_classes):
        cls_indices = np.where(y_train_full == cls)[0]
        sample_num = samples_per_class + (1 if remainder > 0 else 0)
        if remainder > 0:
            remainder -= 1
        
        if len(cls_indices) < sample_num:
            selected_indices = np.random.choice(cls_indices, sample_num, replace=True)
        else:
            selected_indices = np.random.choice(cls_indices, sample_num, replace=False)
        
        support_X.append(X_train_full.iloc[selected_indices])
        support_y.extend([cls] * sample_num)

    X_train_few = pd.concat(support_X, ignore_index=True)
    y_train_few = np.array(support_y)

    maker = Table2GraphTransformer(include_edge_attr=True, lm_model="gpt2", n_components=768, n_jobs=1)
    maker.fit(X_train_few, y_train_few)
    train_graphs = maker.transform(X_train_few, y_train_few)
    test_graphs = maker.transform(X_test, y_test)

    for graph, label in zip(train_graphs + test_graphs, list(y_train_few) + list(y_test)):
        graph.y = torch.tensor([label], dtype=torch.long)

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)

    return (X_train_few, X_test, y_train_few, y_test), train_loader, test_loader, num_classes



def prepare_fewshot_target_dataset(args, X_train_full, y_train_full, X_test, y_test, random_state=42):
    np.random.seed(random_state)
    num_classes = len(np.unique(y_train_full))
    
    shot = args.dataset_shot
    samples_per_class = shot // num_classes
    remainder = shot % num_classes

    support_X, support_y = [], []
    for cls in range(num_classes):
        cls_indices = np.where(y_train_full == cls)[0]
        sample_num = samples_per_class + (1 if remainder > 0 else 0)
        if remainder > 0:
            remainder -= 1
        
        if len(cls_indices) < sample_num:
            selected_indices = np.random.choice(cls_indices, sample_num, replace=True)
        else:
            selected_indices = np.random.choice(cls_indices, sample_num, replace=False)
        
        support_X.append(X_train_full.iloc[selected_indices])
        support_y.extend([cls] * sample_num)

    X_train_few = pd.concat(support_X, ignore_index=True)
    y_train_few = np.array(support_y)

    maker = Table2GraphTransformer(include_edge_attr=True, lm_model="gpt2", n_components=768, n_jobs=1)
    maker.fit(X_train_few, y_train_few)
    train_graphs = maker.transform(X_train_few, y_train_few)
    test_graphs = maker.transform(X_test, y_test)

    for graph, label in zip(train_graphs + test_graphs, list(y_train_few) + list(y_test)):
        graph.y = torch.tensor([label], dtype=torch.long)

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)

    return (X_train_few, X_test, y_train_few, y_test), train_loader, test_loader, num_classes


'''
    Old ones 2024.12.05
'''
def prepare_full_dataset(args):
    DATASETS = {} 
    DATASETS[args.dataset_name] = get_dataset_ml(args.dataset_name, args.dataset_shot, args.dataset_seed)
    maker = Table2GraphTransformer(include_edge_attr=True, lm_model="gpt2", n_components=768, n_jobs=1)
    X,y = preprocessing(DATASETS,args.dataset_name)
    le = LabelEncoder()
    y = le.fit_transform(y)
    num_classes = len(le.classes_)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    maker = Table2GraphTransformer(include_edge_attr=True, lm_model="gpt2", n_components=768, n_jobs=1)
    maker.fit(X_train,y_train)
    train_graphs = maker.transform(X_train,y_train)
    test_graphs = maker.transform(X_test, y_test)
    
    for graph, label in zip(test_graphs, y_test):
        graph.y = torch.tensor([label],dtype = torch.long)
    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)
    return (X_train, X_test, y_train, y_test), train_loader, test_loader , num_classes





def prepare_fewshot_dataset(args, X_train_full, y_train_full, X_test, y_test, random_state=42):
    np.random.seed(random_state)
    num_classes = len(np.unique(y_train_full))
    
    shot = args.dataset_shot
    samples_per_class = shot // num_classes
    remainder = shot % num_classes

    support_X, support_y = [], []
    for cls in range(num_classes):
        cls_indices = np.where(y_train_full == cls)[0]
        sample_num = samples_per_class + (1 if remainder > 0 else 0)
        if remainder > 0:
            remainder -= 1
        
        if len(cls_indices) < sample_num:
            selected_indices = np.random.choice(cls_indices, sample_num, replace=True)
        else:
            selected_indices = np.random.choice(cls_indices, sample_num, replace=False)
        
        support_X.append(X_train_full.iloc[selected_indices])
        support_y.extend([cls] * sample_num)

    X_train_few = pd.concat(support_X, ignore_index=True)
    y_train_few = np.array(support_y)

    maker = Table2GraphTransformer(include_edge_attr=True, lm_model="gpt2", n_components=768, n_jobs=1)
    maker.fit(X_train_few, y_train_few)
    train_graphs = maker.transform(X_train_few, y_train_few)
    test_graphs = maker.transform(X_test, y_test)

    for graph, label in zip(train_graphs + test_graphs, list(y_train_few) + list(y_test)):
        graph.y = torch.tensor([label], dtype=torch.long)

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)

    return (X_train_few, X_test, y_train_few, y_test), train_loader, test_loader, num_classes

def prepare_full_target_dataset(args):
    DATASETS = {} 
    DATASETS[args.target_dataset_name] = get_dataset(args.target_dataset_name, args.dataset_shot, args.dataset_seed)
    maker = Table2GraphTransformer(include_edge_attr=True, lm_model="gpt2", n_components=768, n_jobs=1)
    X,y = preprocessing(DATASETS,args.target_dataset_name)
    le = LabelEncoder()
    y = le.fit_transform(y)
    num_classes = len(le.classes_)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    maker = Table2GraphTransformer(include_edge_attr=True, lm_model="gpt2", n_components=768, n_jobs=1)
    maker.fit(X_train,y_train)
    train_graphs = maker.transform(X_train,y_train)
    test_graphs = maker.transform(X_test, y_test)
    
    for graph, label in zip(test_graphs, y_test):
        graph.y = torch.tensor([label],dtype = torch.long)
    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)
    return (X_train, X_test, y_train, y_test) , train_loader, test_loader , num_classes
def prepare_multiple_datasets_old(args):
    source_dataset_names = args.source_datasets
    target_dataset_name = args.target_dataset 
    

    all_loaders = {}
    num_classes = {}
    original_tabular = {}
    feature_names = {}
    models = {}
    optimizers = {}
    criterions = {}


    '''
       1. Source dataset Preparation
    '''

    for source_dataset_name in source_dataset_names:
        #args.source_dataset_name = dataset
        (X_train, X_test, y_train, y_test), train_loader, test_loader, num_class = prepare_dataloaders(args, source_dataset_name)
        all_loaders[source_dataset_name] = {
            'train':train_loader,
            'test':test_loader
        }
        num_classes[source_dataset_name] = num_class
        original_tabular[source_dataset_name] = {
            'X_train' : X_train,
            'X_test' : X_test, 
            'y_train' : y_train, 
            'y_test' : y_test
        }
        feature_names[source_dataset_name] = X_train.columns.tolist()

        is_binary = num_class == 2 
        output_dim = 1 if is_binary else num_class

        device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
        model = select_model(args, device, output_dim)
        models[source_dataset_name] = model

        criterion = nn.BCEWithLogitsLoss() if is_binary else nn.CrossEntropyLoss()
        criterions[source_dataset_name] = criterion

        optimizer = optim.Adam(model.parameters(), lr=args.source_lr, weight_decay=1e-5)
        optimizers[source_dataset_name] = optimizer

    '''
        2. Target dataset Preparation
    '''

    (X_train, X_test, y_train, y_test), train_loader, test_loader, num_class = prepare_dataloaders(args, target_dataset_name)
    all_loaders[target_dataset_name] = {
        'train':train_loader,
        'test':test_loader
    }
    num_classes[target_dataset_name] = num_class
    original_tabular[target_dataset_name] = {
        'X_train' : X_train,
        'X_test' : X_test,
        'y_train' : y_train,
        'y_test' : y_test
    }
    feature_names[target_dataset_name] = X_train.columns.tolist()
    is_binary = num_classes = 2
    output_dim = 1 if is_binary else num_class

    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    model = select_model(args, device, output_dim)
    models[target_dataset_name] = model
    optimizer = optim.Adam(model.parameters(), lr=args.target_lr, weight_decay=1e-5)
    optimizers[target_dataset_name] = optimizer

    return source_dataset_names, target_dataset_name, all_loaders, num_classes, original_tabular, feature_names, models, optimizers
def prepare_multiple_datasets(args):
    source_dataset_names = args.source_datasets
    target_dataset_name = args.target_dataset 

    meta_info = {
        "all_loaders": {},
        "num_classes": {},
        "original_tabular": {},
        "feature_names": {},
        "models": {},
        "optimizers": {},
        "criterions": {}
    }

    '''
       1. Source dataset Preparation
    '''
    for source_dataset_name in source_dataset_names:
        (X_train, X_test, y_train, y_test), train_loader, test_loader, num_class = prepare_dataloaders(args, source_dataset_name)
        
        meta_info["all_loaders"][source_dataset_name] = {
            'train': train_loader,
            'test': test_loader
        }
        meta_info["num_classes"][source_dataset_name] = num_class
        meta_info["original_tabular"][source_dataset_name] = {
            'X_train': X_train,
            'X_test': X_test, 
            'y_train': y_train, 
            'y_test': y_test
        }
        meta_info["feature_names"][source_dataset_name] = X_train.columns.tolist()

        is_binary = num_class == 2 
        output_dim = 1 if is_binary else num_class
        device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
        model = select_model(args, device, output_dim)
        meta_info["models"][source_dataset_name] = model
        meta_info["criterions"][source_dataset_name] = nn.BCEWithLogitsLoss() if is_binary else nn.CrossEntropyLoss()
        meta_info["optimizers"][source_dataset_name] = optim.Adam(model.parameters(), lr=args.source_lr, weight_decay=1e-5)

    '''
        2. Target dataset Preparation
    '''
    (X_train, X_test, y_train, y_test), train_loader, test_loader, num_class = prepare_dataloaders(args, target_dataset_name)
    
    meta_info["all_loaders"][target_dataset_name] = {
        'train': train_loader,
        'test': test_loader
    }
    meta_info["num_classes"][target_dataset_name] = num_class
    meta_info["original_tabular"][target_dataset_name] = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    meta_info["feature_names"][target_dataset_name] = X_train.columns.tolist()

    is_binary = num_class == 2
    output_dim = 1 if is_binary else num_class
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    model = select_model(args, device, output_dim)
    meta_info["models"][target_dataset_name] = model
    meta_info["criterions"][target_dataset_name] = nn.BCEWithLogitsLoss() if is_binary else nn.CrossEntropyLoss()
    meta_info["optimizers"][target_dataset_name] = optim.Adam(model.parameters(), lr=args.target_lr, weight_decay=1e-5)

    return meta_info
