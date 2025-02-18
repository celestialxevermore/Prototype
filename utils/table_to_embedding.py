from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
import numpy as np 
import pandas as pd 
import pdb
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
from sklearn.pipeline import make_pipeline

class Table2EmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(
            self, args, source_dataset_name):
        super(Table2EmbeddingTransformer, self).__init__()
        
        self.args = args
        self.input_dim = args.input_dim
        self.is_fitted_ = False
        self.scaler_type = args.scaler_type
        self.llm_model_name = args.llm_model
        self.source_dataset_name = source_dataset_name
        self._load_lm_model()
        #self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.label_description = self._get_label_desc()
        self.label_embedding = self._transform_label()
        self.n_components = args.input_dim
        n_jobs: int = 1
    def fit(self, X, y=None):
        """Fit function used for the Table2GraphTransformer

        Parameters
        ----------
        X : pandas DataFrame (n_samples, n_features)
            The input data used to transform to graphs.

        y : None
            Ignored.

        Returns
        -------
        self : object
            Fitted transformer.
        """
        self.y_ = y
        self.is_fitted_ = False

        if not hasattr(self, "lm_model_"):
            self._load_lm_model()
        
        # Relations
        cat_col_names = X.select_dtypes(include="object").columns 
        cat_col_names = cat_col_names.str.replace("\n", " ", regex=True)
        self.cat_col_names = list(cat_col_names)
        num_col_names = X.select_dtypes(exclude="object").columns 
        num_col_names = num_col_names.str.replace("\n", " ", regex=True)
        self.num_col_names = list(num_col_names)
        self.col_names = self.cat_col_names + self.num_col_names # n_i  = \{n_1, n_2, ... n_d\}
        self.cat_name_to_description = {name: self._get_feature_description(name) for name in self.cat_col_names}
        self.num_name_to_description = {name: self._get_feature_description(name) for name in self.num_col_names}

        self.num_transformer = PowerTransformer().set_output(transform="pandas")

        return self


    def _load_lm_model(self):
        if self.llm_model_name == "gpt2":
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')
            self.gpt2_config.num_hidden_layers = 12
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True

            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    config=self.gpt2_config,
                )
            #self.llm_model = self.llm_model
            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                )
            except EnvironmentError:  # downloads tokenizer from HF if not already done
                print("Local tokenizer files not found. Attempting to download...")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                )
            #self.tokenizer.pad_token = self.tokenizer.eos_token
            #self.model = GPT2Model.from_pretrained("gpt2")
            for param in self.llm_model.parameters():
                param.requires_grad = False
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token


    def _get_label_desc(self) -> str:
        if self.source_dataset_name is None:
            raise ValueError("source_dataset_name is not set")
        label_desc_path = f"/storage/personal/eungyeop/dataset/feature_description/{self.source_dataset_name}/{self.source_dataset_name}-metadata.json"
    
        with open(label_desc_path, 'r') as f:
            metadata = json.load(f)
            # target_binary 키에 해당하는 description을 가져옴
            label_description = metadata.get('target_binary')
            #pdb.set_trace()
            if label_description is None:
                raise ValueError("target_binary description not found in metadata")
        return label_description
    
    def _transform_label(self):
        """
        레이블 설명을 임베딩으로 변환
        """
        label_input = self.tokenizer(self.label_description, return_tensors="pt", padding=True, truncation=True)
        label_input = {k: v for k, v in label_input.items()}  # GPU로 이동
        
        with torch.no_grad():
            label_embedding = self.llm_model(**label_input).last_hidden_state.mean(dim=1)  # (1, embed_dim)
        return label_embedding
        
    def _get_feature_description(self, feature_name: str) -> str:
        if self.source_dataset_name is None:
            ValueError("source_dataset_name is not set")
        feature_desc_path = f"/storage/personal/eungyeop/dataset/feature_description/{self.source_dataset_name}/{self.source_dataset_name}-metadata.json"
        
        if not hasattr(self, 'feature_descriptions'):
            with open(feature_desc_path, 'r') as f:
                metadata = json.load(f)
                # target_binary를 제외한 feature descriptions만 저장
                self.feature_descriptions = {k: v for k, v in metadata.items() if k != 'target_binary'}
        return self.feature_descriptions.get(feature_name, feature_name)

    def transform(self, X, y=None):
        """Apply Table2GraphTransformer to each row of the data

        Parameters
        ----------
        X : Pandas DataFrame. (n_samples, n_features)
            The input data used to transform to embeddings.
            Heterogeneous features are separated into categorical and numerical features.

        y : None
            Ignored.

        Returns
        -------
        Embedding Data : list of size (n_samples).
            The list of transformed embedding data.
        """
        X_ = X.copy()
        X_ = X_.replace("\n", " ", regex=True)
        num_data = X_.shape[0]

        
        if y is not None:
            if isinstance(y, pd.Series):
                y = y.values
            y_ = torch.tensor(y).reshape((num_data, 1))
        else:
            y_ = None

        # Separate categorical and numerical columns
        X_categorical = X_.select_dtypes(include="object").copy()
        X_categorical.columns = self.cat_col_names
        X_numerical = X_.select_dtypes(exclude="object").copy()
        X_numerical.columns = self.num_col_names
        X_numerical_raw = X_numerical.copy()
        # Features for names
        cat_col_names = X.select_dtypes(include="object").columns 
        cat_col_names = cat_col_names.str.replace("\n", " ", regex=True)
        self.cat_col_names = list(cat_col_names)
        num_col_names = X.select_dtypes(exclude="object").columns 
        num_col_names = num_col_names.str.replace("\n", " ", regex=True)
        self.num_col_names = list(num_col_names)
        self.col_names = self.cat_col_names + self.num_col_names # n_i  = \{n_1, n_2, ... n_d\}
        self.cat_name_to_description = {name: self._get_feature_description(name) for name in self.cat_col_names}
        self.num_name_to_description = {name: self._get_feature_description(name) for name in self.num_col_names}
    
        if self.scaler_type == 'pow':
            self.num_transformer = PowerTransformer().set_output(transform="pandas")
            X_num = X[self.num_col_names]
            self.num_transformer.fit(X_num)
        self.is_fitted_ = True

        if len(self.num_col_names) != 0:
            X_numerical = self._transform_num(X_numerical)
        if not self.is_fitted_:
            self.is_fitted = True 
        

        embedding_data = [
            self._get_embeddings(
                X_categorical,
                X_numerical,
                X_numerical_raw,
                self.cat_name_to_description,
                self.num_name_to_description,
                y_,
                idx=i,
            )
            for i in range(num_data)
        ]

        return embedding_data




    def _transform_cat(self, X_categorical, cat_name_to_description):
        #pdb.set_trace()
        name_embeddings = []
        desc_embeddings = [] 
        value_embeddings = [] 
        for feature_name in X_categorical.columns:
            # Name embedding
            name_input = self.tokenizer(feature_name, return_tensors="pt", padding=True, truncation=True)
            
            with torch.no_grad():
                name_emb = self.llm_model(**name_input).last_hidden_state.mean(dim=1)
                name_embeddings.append(name_emb)
            # Description embedding
            desc = cat_name_to_description[feature_name]
            desc_input = self.tokenizer(desc, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                desc_emb = self.llm_model(**desc_input).last_hidden_state.mean(dim=1)
                desc_embeddings.append(desc_emb)
            # Value embeddings
            values = X_categorical[feature_name].values[0]
            
            value_input = self.tokenizer(values, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                value_emb = self.llm_model(**value_input).last_hidden_state.mean(dim=1)
                value_embeddings.append(value_emb)
            
        name_embeddings = torch.stack(name_embeddings, dim=0)
        desc_embeddings = torch.stack(desc_embeddings, dim=0)
        value_embeddings = torch.stack(value_embeddings, dim=0)
        return name_embeddings, desc_embeddings, value_embeddings
    


    def _transform_num(self, X):
        X_num = X.copy()
        if not self.is_fitted_:
            X_num = self.num_transformer.fit_transform(X_num)
        else:
            X_num = self.num_transformer.transform(X_num)
        return X_num 
    
    def _transform_num_raw(self, x_num_raw, num_name_to_desriptions):
        """
            수치형 데이터의 임베딩을 생성 (단일 샘플)
            Args:
                x_num: torch.Tensor - PowerTransformer 통과한 값들 (n_num_features,)
                x_num_raw: torch.Tensor - 원본 numerical 값들 (n_num_features,)
                descriptions: dict - {feature_name: feature_description}
            Returns:
                name_embeddings: (n_num_features, embed_dim)
                desc_embeddings: (n_num_features, embed_dim)
                prompt_embeddings: (n_num_features, embed_dim)
        """
        name_embeddings = []
        desc_embeddings = []
        prompt_embeddings = [] 
        for i, col_name in enumerate(self.num_col_names):
            # Name embedding
            name_input = self.tokenizer(col_name, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                name_emb = self.llm_model(**name_input).last_hidden_state.mean(dim=1)
                name_embeddings.append(name_emb)
            # Description embedding
            desc = num_name_to_desriptions.get(col_name, col_name)
            desc_input = self.tokenizer(desc, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                desc_emb = self.llm_model(**desc_input).last_hidden_state.mean(dim=1)
                desc_embeddings.append(desc_emb)

            col_values = x_num_raw[col_name].values
            min_val = col_values.min().item()
            max_val = col_values.max().item()
            mean_val = col_values.mean().item()
            std_val = col_values.std().item()

            prompt = (
                f"<|start_prompt|>"
                f"Column Name: {col_name}. "
                f"Statistics: Min = {min_val:.2f}, Max = {max_val:.2f}, "
                f"Mean = {mean_val:.2f}, Std = {std_val:.2f}. "
                "<|end_prompt|>"
            )

            prompt_input = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                prompt_emb = self.llm_model(**prompt_input).last_hidden_state.mean(dim=1)
                prompt_embeddings.append(prompt_emb)
            #pdb.set_trace()
        name_embeddings = torch.stack(name_embeddings, dim=0)
        desc_embeddings = torch.stack(desc_embeddings, dim=0)
        prompt_embeddings = torch.stack(prompt_embeddings, dim=0)

        return name_embeddings, desc_embeddings, prompt_embeddings
    

    def _get_embeddings(self, X_categorical, X_numerical, X_numerical_raw, cat_name_to_description, num_name_to_description, y, idx):
        """Transform to graph objects.

        Parameters
        ----------
        X_categorical : Pandas DataFrame of shape (n_samples, n_categorical_features)
            The input pandas DataFrame containing only the categorical features.
        X_numerical : Pandas DataFrame of shape (n_samples, n_numerical_features)
            The input pandas DataFrame containing only the numerical features.
        y : array-like of shape (n_samples,)
            The target variable to try to predict.
        idx: int
            The index of a particular data point used to transform into graphs

        Returns
        -------
        Embedding
        """
        data_cat = X_categorical.iloc[idx]
        data_cat = data_cat.dropna()
        num_cat = len(data_cat)
        if num_cat != 0:
            data_cat = data_cat.str.replace("\n", " ", regex=True).str.lower()
        
        data_num = X_numerical.iloc[idx]
        data_num = data_num.dropna()
        num_num = len(data_num)
        
        data_num_raw = X_numerical_raw.iloc[idx]
        data_num_raw = data_num_raw.dropna()
        
        cat_name_embeddings, cat_desc_embeddings, cat_value_embeddings = self._transform_cat(
            pd.DataFrame(data_cat).T,
            cat_name_to_description)
        
        num_name_embeddings, num_desc_embeddings, num_prompt_embeddings = self._transform_num_raw(
            pd.DataFrame(data_num_raw).T,
            num_name_to_description
        )
        
        x_num_ = torch.tensor(np.array(data_num).astype("float32"))
        num_prompt_embeddings = x_num_.view(-1, 1, 1) * num_prompt_embeddings

        num_prompt_embeddings = num_prompt_embeddings.clone().detach()
        if num_prompt_embeddings.size(0) == 0:
            num_prompt_embeddings = num_prompt_embeddings.reshape(0, self.n_components)

        if y is not None:
            y_ = y[idx].clone()
        else:
            y_ = torch.tensor([])
        s_idx = idx 

        
        data = {
        'cat_name_embeddings': cat_name_embeddings,
        'cat_desc_embeddings': cat_desc_embeddings,
        'cat_value_embeddings': cat_value_embeddings,
        'num_name_embeddings': num_name_embeddings,
        'num_desc_embeddings': num_desc_embeddings,
        'num_prompt_embeddings': num_prompt_embeddings,
        'y': y_,
        's_idx': s_idx
        }
        
        return data
