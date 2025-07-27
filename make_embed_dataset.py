import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple
import os
import sys
import pickle, torch, random
import argparse, pdb
#from LLM.tabular.ProtoLLM.utils.table_to_embedding_ours import Table2EmbeddingTransformer

class TabularToEmbeddingDataset:
    def __init__(self, args, base_path: str = "/storage/personal/eungyeop/dataset/table/"):
        self.base_path = base_path
        self.args = args
        self.data_source = "label_table" if args.label else "origin_table"
        self.dataset_and_class = {
            "heart": ['target_binary', ['no','yes']],
            "heart_target_1" : ['target_binary', ['no','yes']],
            "heart_target_2" : ['target_binary', ['no','yes']],
            "heart_target_3" : ['target_binary', ['no','yes']],
            "heart_target_4" : ['target_binary', ['no','yes']],
            "cleveland": ['target_binary', ['no','yes']],
            "credit-g" :['target_binary', ['no','yes']],
            "hungarian": ['target_binary', ['no','yes']],
            "switzerland": ['target_binary', ['no','yes']],
            "heart_statlog": ['target_binary', ['no','yes']],
            "adult": ['target_binary', ['no','yes']],
            "diabetes": ['target_binary', ['no','yes']],
            "breast" : ['target_binary', [0,1]],
            "higgs" : ['target_binary', [0,1]],
            'bank' : ['target_binary', ['no','yes']],
            'higgs_sampled' : ["target_binary", [0.0, 1.0]],
            'magic_telescope' : ["target_binary", ['g','h']],
            "car": ['target_multiclass', ['unacceptable', 'acceptable', 'very good', 'good']],
            'forest_covertype_sampled': ['target_multiclass', [1,2,3,4,5,6,7]],
            "communities": ['target_multiclass', ['medium', 'high', 'low']],
        }
        self.transformer_class = self._get_transformer_class(args.embed_type)

    def _get_transformer_class(self, embed_type: str):
        if embed_type == 'carte':
            from table_to_embedding_carte import Table2EmbeddingTransformer
            print(f"Using CarTE transformer for embed_type : {embed_type}")
        elif embed_type == 'carte_desc':
            from table_to_embedding_carte_desc import Table2EmbeddingTransformer
            print(f"Using CarTE_desc transformer for embed_type : {embed_type}")
        elif embed_type == 'ours':
            from table_to_embedding_ours import Table2EmbeddingTransformer
            print(f"Using Ours transformer for embed_type : {embed_type}")
        elif embed_type == 'ours2':
            from table_to_embedding_ours2 import Table2EmbeddingTransformer
            print(f"Using Ours2 transformer for embed_type : {embed_type}")
        return Table2EmbeddingTransformer

    def preprocessing(self, DATASETS: pd.DataFrame, data_name: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """데이터셋 전처리"""
        assert data_name in self.dataset_and_class, f"{data_name} is not a valid dataset name"
        
        class_name = self.dataset_and_class[data_name][0] 
        class_values = self.dataset_and_class[data_name][1]

        if class_name == 'target_binary':
            # 데이터셋별 원래 label 이름을 target_binary로 변경
            if data_name == 'adult' and 'income' in DATASETS.columns:
                DATASETS['target_binary'] = DATASETS['income']
                DATASETS = DATASETS.drop('income', axis=1)
            elif data_name == 'diabetes' and 'Outcome' in DATASETS.columns:
                DATASETS['target_binary'] = DATASETS['Outcome']
                DATASETS = DATASETS.drop('Outcome', axis=1)
            elif data_name =='breast' and 'target' in DATASETS.columns:
                DATASETS['target_binary'] = DATASETS['target']
                DATASETS = DATASETS.drop('target', axis = 1)
            elif data_name == 'bank' and 'Class' in DATASETS.columns:
                DATASETS['target_binary'] = DATASETS['Class']
                DATASETS = DATASETS.drop('Class', axis = 1)
            elif data_name == 'higgs_sampled' and 'target' in DATASETS.columns:
                DATASETS['target_binary'] = DATASETS['target']
                DATASETS = DATASETS.drop('target', axis = 1)
            elif data_name == 'magic_telescope' and 'Class' in DATASETS.columns:
                DATASETS['target_binary'] = DATASETS['Class']
                DATASETS = DATASETS.drop('Class', axis = 1)
            elif data_name == 'credit-g' and 'class' in DATASETS.columns:
                DATASETS['target_binary'] = DATASETS['class']
                DATASETS = DATASETS.drop('class', axis = 1)

            
            class_mapping = {label: idx for idx, label in enumerate(class_values)}
            
        elif class_name == 'target_multiclass':
            # 데이터셋별 원래 label 이름을 target_multiclass로 변경
            if data_name == 'car' and 'class' in DATASETS.columns:
                DATASETS['target_multiclass'] = DATASETS['class']
                DATASETS = DATASETS.drop('class', axis=1)
            elif data_name == 'forest_covertype_sampled' and 'Cover_Type' in DATASETS.columns:
                DATASETS['target_multiclass'] = DATASETS['Cover_Type']
                DATASETS = DATASETS.drop('Cover_Type', axis=1)
            elif data_name == 'communities' and 'ViolentCrimesPerPop' in DATASETS.columns:
                DATASETS['target_multiclass'] = DATASETS['ViolentCrimesPerPop']
                DATASETS = DATASETS.drop('ViolentCrimesPerPop', axis=1)
            class_mapping = {label: idx for idx, label in enumerate(class_values)}
        
        X = DATASETS.drop(class_name, axis=1)
        X = X.reset_index(drop=True)
        
        y = DATASETS[class_name]
        y = y.map(class_mapping).astype(int)
        y = y.reset_index(drop=True)
        
        return X, y

    def convert_to_embedding(self, data_name: str) -> None:
        """테이블 데이터를 임베딩으로 변환"""
        # 데이터 로드
        file_path = os.path.join(self.base_path, self.data_source, f"{data_name}.csv")
        df = pd.read_csv(file_path)
        
        # 전처리
        X, y = self.preprocessing(df, data_name)
        
        
        # 임베딩 변환

        maker = self.transformer_class(
            args = self.args,
            source_dataset_name = data_name
        )
        #pdb.set_trace()
        data = maker.fit_transform(X, y)
        
        # 저장 경로 설정
        base_save_path = "/storage/personal/eungyeop/dataset/embedding"
        sub_dir = f"tabular_embeddings_{self.args.embed_type}/{self.args.llm_model}"
        
        save_dir = os.path.join(base_save_path, sub_dir)
        os.makedirs(save_dir, exist_ok=True)
        
        # 파일명 생성
        emb_name = f"embedding_{data_name}.pkl"
        if self.args.label:
            emb_name = f"embedding_label_{data_name}.pkl"
        
        # 임베딩 저장
        emb_path = os.path.join(save_dir, emb_name)
        with open(emb_path, 'wb') as f:
            pickle.dump({
                'embeddings': data,
                'feature_names': list(X.columns),
                'random_seed': self.args.random_seed,
                'num_classes': len(set(y))
            }, f)
        
        print(f"Completed {data_name}")
        #print(f"- Train samples: {len(data)}")
        print(f"- Num classes: {len(set(y))}")
        print(f"- Saved to: {emb_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--input_dim', type=int, default=768)
    parser.add_argument('--label', action='store_true',
                       help='If True, uses label_table. If False, uses origin_table.')
    parser.add_argument('--cpu', type = int, default = 30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--scaler_type', type=str, 
                       default='pow',
                       choices=['std', 'pow'],
                       help='Type of scaler to use for numerical features.')
    parser.add_argument('--llm_model', type = str, default='gpt2_mean', choices=['gpt2_mean','gpt2_auto','sentence-bert', 'bio-bert', 'bio-clinical-bert', 'LLAMA_mean', 'LLAMA_auto'],
                        help='Name of the language model to use')
    parser.add_argument('--embed_type', default = 'carte', choices = ['carte', 'carte_desc','ours','ours2'])
    args = parser.parse_args()
    
    # 재현성을 위한 설정
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #torch.use_deterministic_algorithms(True)
    import psutil
    p = psutil.Process()
    p.cpu_affinity(range(args.cpu, args.cpu + 3))
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
    
    converter = TabularToEmbeddingDataset(args)
    datasets_to_process = [
        #'credit-g',
        #"communities",
        #"forest_covertype_sampled",
        #'magic_telescope',
        #'higgs_sampled',
        #'car',
        #'bank',
        #"breast"
        #"heart",
        #"heart_target_1",
        #"heart_target_2",
        #"heart_target_3",
        "heart_target_4",
        #"diabetes",
        #"adult"
    ]
    
    for dataset_name in datasets_to_process:
        print(f"\nProcessing {dataset_name}")
        converter.convert_to_embedding(data_name=dataset_name)