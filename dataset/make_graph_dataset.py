import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple
import os
import sys
sys.path.append('..')
from utils.table_to_graph_attention import Table2GraphTransformer
import pickle
import argparse

class TabularToGraphDataset:
    def __init__(self, args, base_path: str = "/mnt/storage/personal/eungyeop/dataset/table/"):
        self.base_path = base_path
        self.args = args
        # 데이터 소스 경로 설정
        self.data_source = "label_table" if args.label else "origin_table"
        self.dataset_and_class = {
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

    def preprocessing(self, DATASETS: pd.DataFrame, data_name: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """데이터셋 전처리"""
        assert data_name in self.dataset_and_class, f"{data_name} is not a valid dataset name"

        class_name = self.dataset_and_class[data_name][0]
        class_values = self.dataset_and_class[data_name][1]
        
        class_mapping = {label: idx for idx, label in enumerate(class_values)}
        
        X = DATASETS.drop(class_name, axis=1)
        X = X.reset_index(drop=True)

        y = DATASETS[class_name]
        y = y.map(class_mapping).astype(int)
        y = y.reset_index(drop=True)
        
        return X, y

    def convert_to_graph(self, data_name: str) -> None:
        """테이블 데이터를 그래프로 변환"""
        # 데이터 로드 (label 여부에 따른 경로 설정)
        file_path = os.path.join(self.base_path, self.data_source, f"{data_name}.csv")
        df = pd.read_csv(file_path)
        
        # 전처리
        X, y = self.preprocessing(df, data_name)
        
        # use_leaf_edges를 graph_type에 따라 설정
        use_leaf_edges = (self.args.graph_type == 'full')
        
        # 그래프 변환
        transformer = Table2GraphTransformer(
            include_edge_attr=True, 
            graph_type=self.args.graph_type,
            lm_model="gpt2", 
            n_components=768, 
            n_jobs=1,
            use_attention_init=self.args.use_attention_init,
            use_hypergraph=self.args.use_hypergraph,
            corr_threshold=0.5,
            use_FD=self.args.FD,
            dataset_name=data_name  # 데이터셋 이름 전달
        )
        graphs = transformer.fit_transform(X, y)
        
        # 저장 경로 설정
        base_save_path = "/mnt/storage/personal/eungyeop/dataset/graph"
        graph_type = "fully_connected_graph" if use_leaf_edges else "star_graph"
        
        # 데이터 타입 결정 (origin/label + FD 여부)
        if self.args.label:
            data_type = "label_FD" if self.args.FD else "label"
        else:
            data_type = "origin_FD" if self.args.FD else "origin"
        
        save_dir = os.path.join(base_save_path, f"{data_type}_graph", graph_type)
        os.makedirs(save_dir, exist_ok=True)
        
        # 파일명 생성
        graph_structure = "fully_connected" if use_leaf_edges else "star"
        if self.args.FD:
            graph_name = f"{graph_structure}_{data_type}_{data_name}.pkl"
        else:
            graph_name = f"{graph_structure}_{data_type}_{data_name}.pkl"
        
        # 그래프 저장
        graph_path = os.path.join(save_dir, graph_name)
        with open(graph_path, 'wb') as f:
            pickle.dump(graphs, f)
        
        print(f"Completed {data_name}")
        print(f"- Total graphs: {len(graphs)}")
        print(f"- Num classes: {len(set(y))}")
        print(f"- Saved to: {graph_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--graph_type', type=str, choices=['star', 'full'],
                      default='star', help='Type of graph structure to create')
    parser.add_argument('--use_attention_init', action='store_true',
                      help='If True, initializes attention weights.')
    parser.add_argument('--use_hypergraph', type=str, default='none',
                      choices=['none', 'basic', 'correlation'],
                      help='Type of hypergraph to use.')
    parser.add_argument('--label', action='store_true',
                      help='If True, uses label_table. If False, uses origin_table.')
    parser.add_argument('--FD', action='store_true',
                      help='If True, uses feature descriptions in node embeddings.')
    args = parser.parse_args()
    
    converter = TabularToGraphDataset(args)
    datasets_to_process = [
         "cleveland", "hungarian", 
        "switzerland", "heart_statlog", "heart"
    ]
    
    for dataset_name in datasets_to_process:
        print(f"\nProcessing {dataset_name}")
        converter.convert_to_graph(data_name=dataset_name)