from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
import numpy as np 
import pandas as pd 
import torch 
import torch.nn as nn
from scipy.io import arff
import os
import random
import openai
import time
import torch
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import networkx as nx
from typing import Union
from torch_geometric.data import Data
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import remove_isolated_nodes, to_dense_adj
from torch_geometric.data import Data
from torch_geometric.utils import scatter
from sklearn.pipeline import make_pipeline

def _create_edge_index(
        num_nodes : int, 
        edge_attr : torch.tensor, 
        undirected : bool = False, 
        self_loop : bool = True,
):
    edge_index_ = torch.combinations(torch.arange(num_nodes),2).transpose(0,1)
    edge_index_ = edge_index_[:,(edge_index_[0] == 0)]
    edge_index = edge_index_.clone() 
    edge_attr_ = edge_attr.clone()
    
    #undirected 
    if undirected:
        edge_index = torch.hstack((edge_index, torch.flipud(edge_index)))
        edge_attr_ = torch.vstack((edge_attr_,edge_attr_))
    
    #self-loop 
    if self_loop:
        edge_index_self_loop = torch.vstack(
            (edge_index_[1].unique(), edge_index_[1].unique())
        )
        edge_index = torch.hstack((edge_index, edge_index_self_loop))
        edge_attr_ = torch.vstack( 
            (edge_attr_,torch.ones(num_nodes - 1 , edge_attr_.size(1)))
        )
    return edge_index, edge_attr_

def _create_full_edge_index(
        num_nodes: int,
        edge_attr: torch.tensor,
        undirected: bool = False,
        self_loop: bool = True,
):
    """Create edge_index including both star-graph structure and leaf-to-leaf connections"""
    # Center-to-leaf edges 
    center_edges = torch.vstack((
        torch.zeros(num_nodes - 1 , dtype = torch.long),
        torch.arange(1, num_nodes)
    ))
    # Leaf-to-leaf edges
    leaf_edges = torch.combinations(torch.arange(1, num_nodes), 2).t()

    # Combine edges
    edge_index = torch.cat([center_edges, leaf_edges], dim = 1)

    center_attr = edge_attr[:num_nodes - 1]
    leaf_attr = torch.ones((leaf_edges.size(1), edge_attr.size(1)), dtype = torch.float32)
    edge_attr = torch.vstack([center_attr, leaf_attr])

    if undirected:
        edge_index = torch.cat([edge_index, torch.flip(edge_index, [0])], dim=1)
        edge_attr = torch.vstack([edge_attr, edge_attr])
    
    if self_loop:
        self_edges = torch.arange(num_nodes).unsqueeze(0).repeat(2, 1)
        edge_index = torch.cat([edge_index, self_edges], dim=1)
        self_attr = torch.ones((num_nodes, edge_attr.size(1)), dtype=torch.float32)
        edge_attr = torch.vstack([edge_attr, self_attr])
    
    return edge_index, edge_attr

def create_attention_based_hypergraph(attention_weights: torch.tensor, threshold: float = 0.5, min_group_size: int = 2):
    """Attention score 기반으로 hypernode 식별"""
    n_features = attention_weights.size(0)
    attention_pairs = torch.nonzero(attention_weights > threshold)

    G = nx.Graph()
    for i, j in attention_pairs:
        G.add_edge(i.item(), j.item())
    feature_groups = [] 
    for component in nx.connected_components(G):
        if len(component) >= min_group_size:
            feature_groups.append(list(component))
    H = torch.zeros((n_features, len(feature_groups)))
    for i, group in enumerate(feature_groups):
        H[group, i] = 1 
    return H, feature_groups

def create_attention_based_hypergraph_geometric(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    attention_weights: torch.Tensor,
    threshold: float = 0.5,
    min_group_size: int = 2
):
    """torch_geometric 기반의 attention score hypergraph"""
    # Dense adjacency matrix with attention weights
    adj_matrix = to_dense_adj(
        edge_index, 
        edge_attr=attention_weights
    ).squeeze(0)
    
    # Find connected components based on attention threshold
    mask = adj_matrix > threshold
    connected_indices = torch.nonzero(mask)
    
    # Group nodes into hypernodes
    node_groups = {}
    for i, j in connected_indices:
        i, j = i.item(), j.item()
        if i not in node_groups:
            node_groups[i] = {i}
        if j not in node_groups:
            node_groups[j] = {j}
        node_groups[i].update(node_groups[j])
        node_groups[j].update(node_groups[i])
    
    # Filter groups by size
    feature_groups = [
        list(group) for group in set(map(frozenset, node_groups.values()))
        if len(group) >= min_group_size
    ]
    
    # Create hypergraph incidence matrix
    H = torch.zeros((x.size(0), len(feature_groups)))
    for i, group in enumerate(feature_groups):
        H[group, i] = 1.0
        
    return H, feature_groups

def create_message_based_hypergraph(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    n_layers: int = 2,
    importance_threshold: float = 0.5
):
    """Message magnitude를 통한 feature importance 기반 hypergraph"""
    importance_scores = []
    h = x
    
    # Compute message magnitudes across layers
    for _ in range(n_layers):
        # Compute messages
        row, col = edge_index
        msg = h[col]  # messages from source nodes
        
        # Aggregate messages for each target node using scatter
        msg_magnitude = scatter(
            msg, 
            row, 
            dim=0, 
            dim_size=x.size(0),
            reduce='mean'
        )
        
        # Update importance scores
        node_importance = torch.norm(msg_magnitude, p=2, dim=1)
        importance_scores.append(node_importance)
        
        # Update node features for next layer
        h = msg_magnitude
    
    # Average importance across layers
    avg_importance = torch.stack(importance_scores).mean(0)
    
    # Create hypernodes based on importance
    important_nodes = torch.where(avg_importance > importance_threshold)[0]
    
    # Create hypergraph incidence matrix
    H = torch.zeros((x.size(0), len(important_nodes)))
    H[important_nodes, torch.arange(len(important_nodes))] = 1.0
    
    return H, important_nodes.tolist()

def compute_attention_weights(edge_attr: torch.tensor) -> torch.tensor:
    "Compute initial attention weighs based on edge attribute sim"
    norm = torch.norm(edge_attr, dim = 1, keepdim = True)
    normalized = edge_attr / (norm + 1e-8)
    similarity = torch.mm(normalized, normalized.t())
    attention_weights = torch.nn.functional.softmax(similarity, dim = 1)
    return attention_weights

def create_hypergraph_laplacian(
        num_nodes: int, 
        edge_index: torch.tensor, 
        edge_attr: torch.tensor, 
        H: torch.tensor = None
):
    """Create hypergraph Laplacian matrix"""
    if H is None:
        H = torch.zeros((num_nodes, edge_index.size(1)), dtype = torch.float32)
        H[edge_index[0], torch.arange(edge_index.size(1))] = 1 
        H[edge_index[1], torch.arange(edge_index.size(1))] = 1

    D_v = torch.diag(torch.sum(H, dim = 1))
    D_e = torch.diag(torch.sum(H, dim = 0))

    D_v_sqrt_inv = torch.diag(1.0 / torch.sqrt(torch.sum(H, dim = 1) + 1e-8))
    D_e_inv = torch.diag(1.0 / (torch.sum(H, dim = 0) + 1e-8))

    L = torch.eye(num_nodes) - torch.mm(torch.mm(torch.mm(torch.mm(D_v_sqrt_inv,H),D_e_inv),H.t()),D_v_sqrt_inv)
    return L

def create_correlation_based_hypergraph(
    X_numerical: pd.DataFrame, 
    edge_index: torch.tensor,
    threshold: float = 0.5
):
    """
    numerical features에 대해서만 hypergraph를 생성
    """
    # numerical features 간의 correlation만 계산
    corr_matrix = X_numerical.corr().abs()
    n_num_features = len(corr_matrix)
    
    # Create hyperedges for correlated numerical features
    hyperedges = []
    for i in range(n_num_features):
        correlated = corr_matrix.index[corr_matrix.iloc[i] > threshold].tolist()
        if len(correlated) > 1:
            hyperedges.append(correlated)
    
    # H matrix는 numerical features의 수에 맞춤
    H = torch.zeros((n_num_features, len(hyperedges)))
    for j, edge in enumerate(hyperedges):
        for node in edge:
            node_idx = X_numerical.columns.get_loc(node)
            H[node_idx, j] = 1
    
    return H

class Table2GraphTransformer(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        *,
        include_edge_attr: bool = False,
        lm_model: str = "gpt2",
        n_components: float = 768,
        n_jobs: int = 1,
        graph_type: str = 'star',
        use_attention_init: bool = False, 
        use_hypergraph: str = 'None',
        corr_threshold: float = 0.5,
        use_FD: bool = False,
        dataset_name: str = None,  # 데이터셋 이름 추가
    ):
        super(Table2GraphTransformer, self).__init__()
        self.include_edge_attr = include_edge_attr
        self.lm_model = lm_model
        self.n_components = n_components
        self.n_jobs = n_jobs
        self.scalers = {}  # feature별 scaler 저장

        self.use_leaf_edges = (graph_type == 'full')  # graph_type에 따라 use_leaf_edges 설정
        self.use_attention_init = use_attention_init
        self.use_hypergraph = use_hypergraph 
        self.corr_threshold = corr_threshold 
        self.use_FD = use_FD
        self.dataset_name = dataset_name

    def fit(self, X, y=None):
        self.y_ = y
        self.is_fitted_ = False

        # Load language_model
        if not hasattr(self, "lm_model_"):
            self._load_lm_model()

        # Relations
        cat_col_names = X.select_dtypes(include="object").columns
        cat_col_names = cat_col_names.str.replace("\n", " ", regex=True).str.lower()
        self.cat_col_names = list(cat_col_names)
        num_col_names = X.select_dtypes(exclude="object").columns
        num_col_names = num_col_names.str.replace("\n", " ", regex=True).str.lower()
        self.num_col_names = list(num_col_names)
        self.col_names = self.cat_col_names + self.num_col_names
        print(f"self.cat_col_names : {self.cat_col_names}")
        print(f"self.num_col_names : {self.num_col_names}")
        # Numerical columns standardization setup


        self.num_transformer_ = PowerTransformer().set_output(transform="pandas")
        return self

    def transform(self, X, y=None):
        # Preprocess the features
        X_ = X.copy()
        X_ = X_.replace("\n", " ", regex=True)
        num_data = X_.shape[0]

        y_ = None
        if self.y_ is not None:
            y_ = np.array(self.y_)
            y_ = torch.tensor(y_).reshape((num_data, 1))

        # Separate categorical and numerical columns
        X_categorical = X_.select_dtypes(include="object").copy()
        X_categorical.columns = self.cat_col_names
        X_numerical = X_.select_dtypes(exclude="object").copy()
        X_numerical.columns = self.num_col_names

        # Features for names
        cat_names = pd.melt(X_categorical)["value"]
        cat_names = cat_names.dropna()
        cat_names = cat_names.astype(str)
        cat_names = cat_names.str.replace("\n", " ", regex=True).str.lower()
        cat_names = cat_names.unique()
        names_total = np.hstack([self.col_names, cat_names])
        names_total = np.unique(names_total)
        name_dict = {names_total[i]: i for i in range(names_total.shape[0])}

        # preprocess values
        name_attr_total = self._transform_names(names_total)
        if len(self.num_col_names) != 0:
            X_numerical = self._transform_numerical(X_numerical)
        if not self.is_fitted_:
            self.is_fitted_ = True

        data_graph = [
            self._graph_construct(
                X_categorical,
                X_numerical,
                name_attr_total,
                name_dict,
                y_,
                idx=i,
            )
            for i in range(num_data)
        ]

        if self.y_ is not None:
            self.y_ = None

        return data_graph

    def _load_lm_model(self):
        if self.lm_model == "gpt2":
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = GPT2Model.from_pretrained("gpt2")

    # def _transform_numerical(self, X):
    #     X_num = X.copy()
    #     for feature in X_num.columns:
    #         if feature in self.scalers:
    #             X_num[feature] = self.scalers[feature].transform(X_num[feature].values.reshape(-1, 1))
    #     return X_num
    def _transform_numerical(self, X):
        """Transform numerical columns using power transformer"""
        X_num = X.copy()
        if not self.is_fitted_:
            X_num = self.num_transformer_.fit_transform(X_num)
        else:
            X_num = self.num_transformer_.transform(X_num)
        return X_num

    def _transform_names(self, names_total):
        if self.lm_model == "gpt2":
            # Feature name embeddings
            inputs = self.tokenizer(names_total.tolist(), return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            name_embeddings = outputs.last_hidden_state.mean(dim=1).numpy().astype(np.float32)  # (N, 768)
            
            if self.use_FD:
                # Feature description embeddings
                descriptions = [self._get_feature_description(name) for name in names_total]
                desc_inputs = self.tokenizer(descriptions, return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    desc_outputs = self.model(**desc_inputs)
                desc_embeddings = desc_outputs.last_hidden_state.mean(dim=1).numpy().astype(np.float32)  # (N, 768)
                
                # Combine embeddings by taking mean
                combined_embeddings = (name_embeddings + desc_embeddings) / 2  # (N, 768) 유지
                return combined_embeddings
            
            return name_embeddings

    def _get_feature_description(self, feature_name: str) -> str:
        """Feature name에 대한 description을 반환"""
        if self.dataset_name is None:
            return feature_name
            
        feature_desc_path = f"/mnt/storage/personal/eungyeop/dataset/feature_description/{self.dataset_name}/{self.dataset_name}-metadata.json"
        
        if not hasattr(self, 'feature_descriptions'):
            with open(feature_desc_path, 'r') as f:
                self.feature_descriptions = json.load(f)
        
        return self.feature_descriptions.get(feature_name, feature_name)

    def _graph_construct(
        self,
        X_categorical,
        X_numerical,
        name_attr_total,
        name_dict,
        y,
        idx,
    ):
        # Obtain the data for a 'idx'-th row
        data_cat = X_categorical.iloc[idx]
        data_cat = data_cat.dropna()
        num_cat = len(data_cat)
        if num_cat != 0:
            data_cat = data_cat.str.replace("\n", " ", regex=True).str.lower()
        data_num = X_numerical.iloc[idx]
        data_num = data_num.dropna()
        num_num = len(data_num)

        # edge_attributes
        if self.include_edge_attr:
            edge_attr_cat = [name_attr_total[name_dict[x]] for x in data_cat.index]
            edge_attr_cat = np.array(edge_attr_cat).astype(np.float32)
            edge_attr_num = [name_attr_total[name_dict[x]] for x in data_num.index]
            edge_attr_num = np.array(edge_attr_num).astype(np.float32)
        else:
            edge_attr_cat = np.ones((num_cat, self.n_components)).astype(np.float32)
            edge_attr_num = np.ones((num_num, self.n_components)).astype(np.float32)

        # node_attributes
        x_cat = [name_attr_total[name_dict[x]] for x in data_cat]
        x_cat = np.array(x_cat).astype(np.float32)
        x_cat = torch.tensor(x_cat)
        if x_cat.size(0) == 0:
            x_cat = x_cat.reshape(0, self.n_components)
            edge_attr_cat = edge_attr_cat.reshape(0, self.n_components)

        x_num_ = np.array(data_num).astype("float32")
        x_num = x_num_.reshape(-1, 1) * edge_attr_num
        x_num = torch.tensor(x_num)
        if x_num.size(0) == 0:
            x_num = x_num.reshape(0, self.n_components)
            edge_attr_num = edge_attr_num.reshape(0, self.n_components)

        # combined node/edge attributes
        x = torch.vstack((x_cat, x_num))
        x = torch.vstack((torch.ones((1, x.size(1))), x))
        edge_attr = np.vstack((edge_attr_cat, edge_attr_num))
        edge_attr = torch.tensor(edge_attr)

        # edge_index
        num_nodes = num_cat + num_num + 1


        if self.use_leaf_edges:
            edge_index, edge_attr = _create_full_edge_index(num_nodes, edge_attr, False, True)
        else:
            edge_index, edge_attr = _create_edge_index(num_nodes, edge_attr, False, True)

        if self.use_attention_init:
            attention_weights = compute_attention_weights(edge_attr)
        else:
            attention_weights = None 
        if self.use_hypergraph == 'basic':
            laplacian = create_hypergraph_laplacian(num_nodes, edge_index, edge_attr)
        elif self.use_hypergraph == 'correlation':
            H = create_correlation_based_hypergraph(X_numerical, edge_index, threshold= 0.5)
            laplacian = create_hypergraph_laplacian(len(X_numerical.columns), edge_index, edge_attr, H)
        else:
            laplacian = None 
        

        # Set the center node
        Z = torch.mul(edge_attr, x[edge_index[1]])
        x[0, :] = Z[(edge_index[0] == 0), :].mean(dim=0)

        # Target
        if y is not None:
            y_ = y[idx].clone()
        else:
            y_ = torch.tensor([])

        # graph index (g_idx)
        g_idx = idx
            
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            attention_weights = attention_weights,
            laplacian = laplacian,
            y=y_,
            g_idx=g_idx,
        )

        return data