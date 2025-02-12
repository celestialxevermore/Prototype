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
from sklearn.pipeline import make_pipeline
#from configs.directory import config_directory
#from skrub import MinHashEncoder  # change to skrub

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

class Table2GraphTransformer(TransformerMixin, BaseEstimator):
    """Transformer from tables to a list of graphs.

    The list of graphs are generated in a row-wise fashion.

    Parameters
    ----------
    include_edge_attr : bool, default = True
        Indicates whether to include the edge features or not.
    lm_model : {'gpt2'}, default = 'gpt2'
        The lm_model used to initialize the features of nodes and edges.
    n_components : int, default = 768
        The number of components for the GPT2 encoder.
    n_jobs : : int, default=1
        Number of jobs to run in parallel for minhash encoder.
    """

    def __init__(
        self,
        *,
        include_edge_attr: bool = False,
        lm_model: str = "gpt2",
        n_components: float = 768,
        n_jobs: int = 1,
    ):
        super(Table2GraphTransformer, self).__init__()

        self.include_edge_attr = include_edge_attr
        self.lm_model = lm_model
        self.n_components = n_components
        self.n_jobs = n_jobs

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

        # Numerical transformer - Powertransformer
        self.num_transformer_ = PowerTransformer().set_output(transform="pandas")

        return self

    def transform(self, X, y=None):
        """Apply Table2GraphTransformer to each row of the data

        Parameters
        ----------
        X : Pandas DataFrame. (n_samples, n_features)
            The input data used to transform to graphs.

        y : None
            Ignored.

        Returns
        -------
        Graph Data : list of size (n_samples).
            The list of transformed graph data.
        """

        # Preprocess the features
        X_ = X.copy()
        X_ = X_.replace("\n", " ", regex=True)
        num_data = X_.shape[0]

        # Preprocess the target
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
        """Load the language model for features of nodes and edges."""

        if self.lm_model == "gpt2":
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = GPT2Model.from_pretrained("gpt2")

    def _transform_numerical(self, X):
        """Transform numerical columns using powertransformer"""

        X_num = X.copy()
        if not self.is_fitted_:
            X_num = self.num_transformer_.fit_transform(X_num)
        else:
            X_num = self.num_transformer_.transform(X_num)
        return X_num

    def _transform_names(self, names_total):
        """Obtain the feature for a given list of string values"""

        if self.lm_model == "gpt2":
            inputs = self.tokenizer(names_total.tolist(), return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            name_attr_total = outputs.last_hidden_state.mean(dim=1).numpy().astype(np.float32)
        return name_attr_total

    def _graph_construct(
        self,
        X_categorical,
        X_numerical,
        name_attr_total,
        name_dict,
        y,
        idx,
    ):
        """Transform to graph objects.

        Parameters
        ----------
        X_categorical : Pandas DataFrame of shape (n_samples, n_categorical_features)
            The input pandas DataFrame containing only the categorical features.
        X_numerical : Pandas DataFrame of shape (n_samples, n_numerical_features)
            The input pandas DataFrame containing only the numerical features.
        name_attr_total : Numpy array of shape (n_words, n_dim_fasttext)
            The features of each word (or sentence) in the name_dict.
        name_dict : List of shape (n_words,)
            Total list of words (or sentences) that the data contains.
        y : array-like of shape (n_samples,)
            The target variable to try to predict.
        idx: int
            The index of a particular data point used to transform into graphs

        Returns
        -------
        Graph : Graph object
            The graph object from torch_geometric
        """

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
        edge_index, edge_attr = _create_edge_index(num_nodes, edge_attr, False, True)

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
            y=y_,
            g_idx=g_idx,
        )

        return data
    

# def FC_create_edge_index(
#         num_nodes : int, 
#         edge_attr : torch.tensor, 
#         undirected : bool = False, 
#         self_loop : bool = True,
# ):
#     """
#     Create a fully-connected (complete) graph among all nodes,
#     optionally add undirected edges and self-loops.
#     (Modified from original star-graph logic by removing the line
#      that filters for (edge_index_[0] == 0).)
#     """
#     # Generate all combinations of node pairs (i < j).
#     edge_index_ = torch.combinations(torch.arange(num_nodes), 2).transpose(0, 1)
#     # -- Removed the star-graph filter:
#     # edge_index_ = edge_index_[:, (edge_index_[0] == 0)]

#     # Copy for usage
#     edge_index = edge_index_.clone()
#     edge_attr_ = edge_attr.clone()

#     # If undirected, add reverse edges
#     if undirected:
#         edge_index = torch.hstack((edge_index, torch.flipud(edge_index)))
#         edge_attr_ = torch.vstack((edge_attr_, edge_attr_))

#     # If self_loop is True, add loops for all nodes
#     if self_loop:
#         # Create self-loop edges
#         self_loop_index = torch.vstack(
#             (torch.arange(num_nodes), torch.arange(num_nodes))
#         )
#         edge_index = torch.hstack((edge_index, self_loop_index))

#         # For self-loops, we can add ones (or zeros) as a placeholder
#         self_loop_attr = torch.ones((num_nodes, edge_attr_.size(1)))
#         edge_attr_ = torch.vstack((edge_attr_, self_loop_attr))

#     return edge_index, edge_attr_


# class FCTable2GraphTransformer(TransformerMixin, BaseEstimator):
#     """Transformer from tables to a list of graphs.

#     The list of graphs are generated in a row-wise fashion.

#     Parameters
#     ----------
#     include_edge_attr : bool, default = True
#         Indicates whether to include the edge features or not.
#     lm_model : {'gpt2'}, default = 'gpt2'
#         The lm_model used to initialize the features of nodes and edges.
#     n_components : int, default = 768
#         The number of components for the GPT2 encoder.
#     n_jobs : int, default=1
#         Number of jobs to run in parallel for minhash encoder.
#     """

#     def __init__(
#         self,
#         * ,
#         include_edge_attr: bool = False,
#         lm_model: str = "gpt2",
#         n_components: float = 768,
#         n_jobs: int = 1,
#     ):
#         super(FCTable2GraphTransformer, self).__init__()
#         self.include_edge_attr = include_edge_attr
#         self.lm_model = lm_model
#         self.n_components = n_components
#         self.n_jobs = n_jobs

#     def fit(self, X, y=None):
#         self.y_ = y
#         self.is_fitted_ = False

#         # Load language_model
#         if not hasattr(self, "lm_model_"):
#             self._load_lm_model()

#         # Identify categorical and numerical columns
#         cat_col_names = X.select_dtypes(include="object").columns
#         cat_col_names = cat_col_names.str.replace("\n", " ", regex=True).str.lower()
#         self.cat_col_names = list(cat_col_names)

#         num_col_names = X.select_dtypes(exclude="object").columns
#         num_col_names = num_col_names.str.replace("\n", " ", regex=True).str.lower()
#         self.num_col_names = list(num_col_names)

#         self.col_names = self.cat_col_names + self.num_col_names

#         # Numerical transformer
#         self.num_transformer_ = PowerTransformer().set_output(transform="pandas")

#         return self

#     def transform(self, X, y=None):
#         # Preprocess the features
#         X_ = X.copy()
#         X_ = X_.replace("\n", " ", regex=True)
#         num_data = X_.shape[0]

#         # Target
#         y_ = None
#         if self.y_ is not None:
#             y_ = np.array(self.y_)
#             y_ = torch.tensor(y_).reshape((num_data, 1))

#         # Separate categorical / numerical
#         X_categorical = X_.select_dtypes(include="object").copy()
#         X_categorical.columns = self.cat_col_names

#         X_numerical = X_.select_dtypes(exclude="object").copy()
#         X_numerical.columns = self.num_col_names

#         # For LM embeddings of names
#         cat_names = pd.melt(X_categorical)["value"]
#         cat_names = cat_names.dropna().astype(str)
#         cat_names = cat_names.str.replace("\n", " ", regex=True).str.lower()
#         cat_names = cat_names.unique()

#         names_total = np.hstack([self.col_names, cat_names])
#         names_total = np.unique(names_total)
#         name_dict = {names_total[i]: i for i in range(names_total.shape[0])}

#         # LM transformation of names
#         name_attr_total = self._transform_names(names_total)

#         # transform numerical columns
#         if len(self.num_col_names) != 0:
#             X_numerical = self._transform_numerical(X_numerical)

#         if not self.is_fitted_:
#             self.is_fitted_ = True

#         data_graph = [
#             self._graph_construct(
#                 X_categorical,
#                 X_numerical,
#                 name_attr_total,
#                 name_dict,
#                 y_,
#                 idx=i,
#             )
#             for i in range(num_data)
#         ]

#         if self.y_ is not None:
#             self.y_ = None

#         return data_graph

#     def _load_lm_model(self):
#         if self.lm_model == "gpt2":
#             self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#             self.tokenizer.pad_token = self.tokenizer.eos_token
#             self.model = GPT2Model.from_pretrained("gpt2")

#     def _transform_numerical(self, X):
#         X_num = X.copy()
#         if not self.is_fitted_:
#             X_num = self.num_transformer_.fit_transform(X_num)
#         else:
#             X_num = self.num_transformer_.transform(X_num)
#         return X_num

#     def _transform_names(self, names_total):
#         if self.lm_model == "gpt2":
#             inputs = self.tokenizer(names_total.tolist(), return_tensors="pt", padding=True, truncation=True)
#             with torch.no_grad():
#                 outputs = self.model(**inputs)
#             name_attr_total = outputs.last_hidden_state.mean(dim=1).numpy().astype(np.float32)
#         else:
#             # fallback if needed
#             name_attr_total = np.random.randn(names_total.shape[0], self.n_components).astype(np.float32)
#         return name_attr_total

#     def _graph_construct(
#         self,
#         X_categorical,
#         X_numerical,
#         name_attr_total,
#         name_dict,
#         y,
#         idx,
#     ):
#         # Obtain data for row idx
#         data_cat = X_categorical.iloc[idx].dropna()
#         num_cat = len(data_cat)
#         if num_cat != 0:
#             data_cat = data_cat.str.replace("\n", " ", regex=True).str.lower()

#         data_num = X_numerical.iloc[idx].dropna()
#         num_num = len(data_num)

#         # Build edge_attr
#         if self.include_edge_attr:
#             edge_attr_cat = [name_attr_total[name_dict[x]] for x in data_cat.index]
#             edge_attr_cat = np.array(edge_attr_cat).astype(np.float32)
#             edge_attr_num = [name_attr_total[name_dict[x]] for x in data_num.index]
#             edge_attr_num = np.array(edge_attr_num).astype(np.float32)
#         else:
#             edge_attr_cat = np.ones((num_cat, self.n_components)).astype(np.float32)
#             edge_attr_num = np.ones((num_num, self.n_components)).astype(np.float32)

#         # Node attributes
#         x_cat = [name_attr_total[name_dict[x]] for x in data_cat]
#         x_cat = np.array(x_cat).astype(np.float32)
#         x_cat = torch.tensor(x_cat)
#         if x_cat.size(0) == 0:
#             x_cat = x_cat.reshape(0, self.n_components)
#             edge_attr_cat = edge_attr_cat.reshape(0, self.n_components)

#         x_num_ = np.array(data_num).astype("float32")
#         x_num = x_num_.reshape(-1, 1) * edge_attr_num
#         x_num = torch.tensor(x_num)
#         if x_num.size(0) == 0:
#             x_num = x_num.reshape(0, self.n_components)
#             edge_attr_num = edge_attr_num.reshape(0, self.n_components)

#         # Combined node attributes: 
#         #   cat features stacked, then numeric features
#         x = torch.vstack((x_cat, x_num))
#         # Prepend center node = ones((1, n_components)) -> in original code
#         x = torch.vstack((torch.ones((1, x.size(1))), x))

#         # Combined edge_attr
#         edge_attr = np.vstack((edge_attr_cat, edge_attr_num))
#         edge_attr = torch.tensor(edge_attr)

#         # Create edge_index & edge_attr with fully connected 
#         num_nodes = num_cat + num_num + 1
#         edge_index, edge_attr = FC_create_edge_index(num_nodes, edge_attr, undirected=False, self_loop=True)

#         # set center node features = mean of connected nodes
#         Z = torch.mul(edge_attr, x[edge_index[1]])
#         x[0, :] = Z[(edge_index[0] == 0), :].mean(dim=0)

#         if y is not None:
#             y_ = y[idx].clone()
#         else:
#             y_ = torch.tensor([])

#         g_idx = idx

#         data = Data(
#             x=x,
#             edge_index=edge_index,
#             edge_attr=edge_attr,
#             y=y_,
#             g_idx=g_idx,
#         )

#         return data
