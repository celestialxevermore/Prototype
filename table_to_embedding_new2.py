from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, BertModel, BertTokenizer

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
        # GPT2 모델
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
            except EnvironmentError:
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    config=self.gpt2_config,
                )
            
            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                )
            except EnvironmentError:
                print("Local tokenizer files not found. Attempting to download...")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                )
            
            for param in self.llm_model.parameters():
                param.requires_grad = False
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        # Sentence-BERT 모델
        elif self.llm_model_name == 'sentence-bert':
            try:
                from sentence_transformers import SentenceTransformer
                self.llm_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device = 'cpu')
                self.tokenizer = None 
                print("Sentence-BERT 모델 로딩 완료")
            except ImportError:
                raise ImportError("sentence-transformers 패키지를 설치해주세요: pip install sentence-transformers")
        elif self.llm_model_name == 'LLAMA':
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.output_attentions = True 
            self.llama_config.output_hidden_states = True 
                
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config)
            except EnvironmentError:
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config)
            self.tokenizer = LlamaTokenizer.from_pretrained('huggyllama/llama-7b', trust_remote_code=True,local_files_only=False)
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # BioBERT 모델
        elif self.llm_model_name == 'bio-bert':
            try:
                self.llm_model = BertModel.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
                self.tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
                print("BioBERT 모델 로딩 완료")
            except EnvironmentError:
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
                self.tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
                
        # Bio_ClinicalBERT 모델
        elif self.llm_model_name == "bio-clinical-bert":
            try:
                self.llm_model = BertModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
                self.tokenizer = BertTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
                print("Bio_ClinicalBERT 모델 로딩 완료")
            except EnvironmentError:
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
                self.tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
                
        # LLaMA 계열 모델
        elif self.llm_model_name == "bio-llama":
            try:
                self.llm_model = LlamaModel.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
                self.tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
                print("Bio-Medical Llama 모델 로딩 완료")
            except EnvironmentError:
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained("ContactDoctor/Bio-Medical-Llama-3-8B")
                self.tokenizer = LlamaTokenizer.from_pretrained("ContactDoctor/Bio-Medical-Llama-3-8B")
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        else:
            raise ValueError(f"지원하지 않는 LLM 모델: {self.llm_model_name}")
            
        # 모든 모델 파라미터 고정
        if hasattr(self, 'llm_model') and self.llm_model is not None:
            for param in self.llm_model.parameters():
                param.requires_grad = False
                
        


    def _get_label_desc(self) -> str:
        if self.source_dataset_name is None:
            raise ValueError("source_dataset_name is not set")
        label_desc_path = f"/storage/personal/eungyeop/dataset/feature_description/{self.source_dataset_name}/{self.source_dataset_name}-metadata.json"
    
        with open(label_desc_path, 'r') as f:
            metadata = json.load(f)
            # target_binary 키에 해당하는 description을 가져옴
            label_description = metadata.get('target_binary')
            
            if label_description is None:
                raise ValueError("target_binary description not found in metadata")
        return label_description
    
    def _transform_label(self):
        """
        레이블 설명을 임베딩으로 변환
        """
        with torch.no_grad():
            # 1. Sentence-BERT 모델 처리 (tokenizer 없음)
            if self.llm_model_name == "sentence-bert":
                label_embedding = self.llm_model.encode(self.label_description, convert_to_tensor=True)
                label_embedding = label_embedding.unsqueeze(0)
            
            # 2. 다른 모델들 (BERT 계열, GPT2, LLaMA)
            else:
                label_input = self.tokenizer(self.label_description, return_tensors="pt", padding=True, truncation=True)
                outputs = self.llm_model(**label_input)
                
                # BERT 계열 모델은 [CLS] 토큰 사용
                if self.llm_model_name in ["bio-bert", "bio-clinical-bert"]:
                    label_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] 토큰 (1, embed_dim)
                # GPT2, LLaMA는 mean pooling
                else:
                    label_embedding = outputs.last_hidden_state.mean(dim=1)  # (1, embed_dim)
        
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


        y_ = None 
        if self.y_ is not None:
            y_ = np.array(self.y_)
            y_ = torch.tensor(y_).reshape((num_data, 1))
        

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
                y_[i] if y_ is not None else None,
                idx=i,
            )
            for i in range(num_data)
        ]

        return embedding_data

    def _transform_cat(self, X_categorical, X_categorical_total, cat_name_to_description):
        
        cat_name_value_embeddings = [] 
        desc_embeddings = []
        desc_texts = [] 
        for feature_name in X_categorical.columns:
            
            # Description embedding
            desc = cat_name_to_description[feature_name]
            desc_texts.append(desc)
            if self.llm_model_name == 'sentence-bert':
                with torch.no_grad():
                    desc_emb = self.llm_model.encode(desc, convert_to_tensor=True)
                    
                    current_value = X_categorical[feature_name].values[0]
                    unique_values = X_categorical_total[feature_name].unique() 
                    name_value_text = (
                        f"{feature_name} can have these values: {','.join(unique_values)} "
                        f"Current value is {current_value}."
                    )
                    name_value_emb = self.llm_model.encode(name_value_text, convert_to_tensor=True)
                    cat_name_value_embeddings.append(name_value_emb)
                    desc_embeddings.append(desc_emb)
            else:
                desc_input = self.tokenizer(desc, return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    outputs = self.llm_model(**desc_input)
                    if self.llm_model_name in ["bio-bert", 'bio-clinical-bert']:
                        desc_emb = outputs.last_hidden_state[:,0,:].squeeze(0)
                    else:
                        desc_emb = outputs.last_hidden_state.mean(dim=1).squeeze(0)
                desc_embeddings.append(desc_emb)
                
                # Value embeddings
                current_value = X_categorical[feature_name].values[0]
                unique_values = X_categorical_total[feature_name].unique() 
                name_value_text = (
                    f"{feature_name} can have these values: {','.join(unique_values)}. "
                    f"Current value is {current_value}. "
                )
                name_value_input = self.tokenizer(name_value_text, return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    name_value_emb = self.llm_model(**name_value_input).last_hidden_state.mean(dim=1).squeeze(0)
                    cat_name_value_embeddings.append(name_value_emb)
                    

        cat_name_value_embeddings = torch.stack(cat_name_value_embeddings, dim=0)
        desc_embeddings = torch.stack(desc_embeddings, dim=0)
        return cat_name_value_embeddings, desc_embeddings, desc_texts
    


    def _transform_num(self, X):
        X_num = X.copy()
        if not self.is_fitted_:
            X_num = self.num_transformer.fit_transform(X_num)
        else:
            X_num = self.num_transformer.transform(X_num)
        return X_num 
    
    def _transform_num_raw(self, x_num_raw, X_numerical_raw_total, num_name_to_desriptions):
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
        desc_embeddings = []
        prompt_embeddings = [] 
        desc_texts = [] 
        for i, col_name in enumerate(self.num_col_names):
            # Name embedding
            
            desc = num_name_to_desriptions.get(col_name, col_name)
            desc_texts.append(desc)

            if self.llm_model_name == 'sentence-bert':
                with torch.no_grad():
                    desc_emb = self.llm_model.encode(desc, convert_to_tensor = True)
                    desc_embeddings.append(desc_emb)
            else:
                desc_input = self.tokenizer(desc, return_tensors="pt", padding=True, truncation=True)

                with torch.no_grad():
                    outputs = self.llm_model(**desc_input)
                    if self.llm_model_name in ["bio-bert", "bio-clinical-bert"]:
                        desc_emb = outputs.last_hidden_state[:,0,:].squeeze(0)
                    else:
                        desc_emb = outputs.last_hidden_state.mean(dim=1).squeeze(0)
                desc_embeddings.append(desc_emb)

            col_values = x_num_raw[col_name].values
            value = col_values[0]

            min_val = X_numerical_raw_total[col_name].min()
            max_val = X_numerical_raw_total[col_name].max()
            mean_val = X_numerical_raw_total[col_name].mean() 
            std_val = X_numerical_raw_total[col_name].std() 

            relative_position = ""
            if abs(value - mean_val) < 0.1:
                relative_position = "equal to"
            elif value > mean_val:
                std_diff = (value - mean_val) / std_val
                relative_position = f"{std_diff:.2f} standard deviations above"
            else:
                std_diff = (mean_val - value) / std_val
                relative_position = f"{std_diff:.2f} standard deviations below"
            
            bin_category = ""
            z_score = (value - mean_val) / std_val if std_val > 0 else 0 
            
            if abs(z_score) <= 0.5:
                bin_category = "Within normal range (close to mean)"
            elif z_score >0.5 and z_score < 1.5:
                bin_category = "Moderately above mean"
            elif z_score >=1.5:
                bin_category = "Significantly above mean"
            elif z_score <=-0.5 and z_score > -1.5:
                bin_category = "Moderately below mean"
            else:
                bin_category = "Significantly below mean"
            
            prompt = (
                f"The distribution of the column {col_name} is as follows: "
                f"Column Name: {col_name}."
                f"Min = {min_val:.2f}, Max = {max_val:.2f}, "
                f"Mean = {mean_val:.2f}, Std = {std_val:.2f}. "
                f"Current value: {col_name} = {value:.2f}, which is {relative_position}."
                f"Relative position: This value {value:.2f} falls in the '{bin_category}' of the distribution."
            )
            
            # 모델별 처리 방식 분기 - 프롬프트 임베딩
            if self.llm_model_name == 'sentence-bert':
                with torch.no_grad():
                    prompt_emb = self.llm_model.encode(prompt, convert_to_tensor=True)
            else:
                prompt_input = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    outputs = self.llm_model(**prompt_input)
                    if self.llm_model_name in ["bio-bert", "bio-clinical-bert"]:
                        prompt_emb = outputs.last_hidden_state[:,0,:].squeeze(0)  # [CLS] 토큰
                    else:
                        prompt_emb = outputs.last_hidden_state.mean(dim=1).squeeze(0)  # mean pooling

            prompt_embeddings.append(prompt_emb)
        desc_embeddings = torch.stack(desc_embeddings, dim=0)
        prompt_embeddings = torch.stack(prompt_embeddings, dim=0)   
        return prompt_embeddings, desc_embeddings, desc_texts
    

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
        if y is not None:
            y_ = y.clone()
        else:
            y_ = torch.tensor([])
        s_idx = idx 

        data = {
            'label_description_embeddings': self.label_embedding,
            'y': y_,
            's_idx': s_idx
        }

        # Categorical features exist
        if len(X_categorical.columns) > 0:
            data_cat = X_categorical.iloc[idx]
            data_cat = data_cat.dropna()
            
            num_cat = len(data_cat)
            if num_cat != 0:
                data_cat = data_cat.str.replace("\n", " ", regex=True)

                cat_name_value_embeddings, cat_desc_embeddings, cat_desc_texts = self._transform_cat(
                    pd.DataFrame(data_cat).T,
                    X_categorical,
                    cat_name_to_description)
                
                data.update({
                    'cat_name_value_embeddings': cat_name_value_embeddings,
                    'cat_desc_embeddings': cat_desc_embeddings,
                    'cat_desc_texts': cat_desc_texts
                })

        # Numerical features exist
        if len(X_numerical.columns) > 0:
            data_num = X_numerical.iloc[idx]
            data_num = data_num.dropna()
            num_num = len(data_num)
            
            data_num_raw = X_numerical_raw.iloc[idx]
            data_num_raw = data_num_raw.dropna()
            num_prompt_embeddings , num_desc_embeddings, num_desc_texts = self._transform_num_raw(
                pd.DataFrame(data_num_raw).T,
                X_numerical_raw,
                num_name_to_description
            )
            x_num_ = torch.tensor(np.array(data_num).astype("float32"))
            num_prompt_embeddings = x_num_.view(-1, 1) * num_prompt_embeddings

            num_prompt_embeddings = num_prompt_embeddings.clone().detach()
            if num_prompt_embeddings.size(0) == 0:
                num_prompt_embeddings = num_prompt_embeddings.reshape(0, self.n_components)
            
            data.update({
                'num_prompt_embeddings': num_prompt_embeddings,
                'num_desc_embeddings': num_desc_embeddings,
                'num_desc_texts': num_desc_texts
            })
        # print(f"label_description_embeddings: {self.label_embedding.shape}")
        # print(f"y: {y_.shape}") 
        # print(f"s_idx: {s_idx}")
        # print(f"cat_name_value_embeddings: {data['cat_name_value_embeddings'].shape}")
        # print(f"cat_desc_embeddings: {data['cat_desc_embeddings'].shape}")
        # print(f"cat_desc_texts: {data['cat_desc_texts']}")
        # print(f"num_prompt_embeddings: {data['num_prompt_embeddings'].shape}")
        # print(f"num_desc_embeddings: {data['num_desc_embeddings'].shape}")
        # print(f"num_desc_texts: {data['num_desc_texts']}")
        # pdb.set_trace()
        return data
