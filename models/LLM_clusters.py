import numpy as np 
import torch 
import torch.nn as nn 
import torch.optim as optim 
import random
import typing as tp
#from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence 
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from layers.StandardNorm import Normalize
from sklearn.cluster import KMeans
from torch_geometric.nn import GCNConv, global_mean_pool 



def log(x):
    return torch.log(x + 1e-8)

def div(x,y):
    return x / (y + 1e-8)

def get_seq_length(sequence):
    used = torch.sign(torch.max(torch.abs(sequence), 2)[0])
    length = torch.sum(used, 1)
    length = length.int()
    return length



'''
    Newly Added Code 2024.07.08
    Scenario 1. Graph Embedding into Frozen LLM. 

'''
class Vanilla_GNN(torch.nn.Module):
    def __init__(self):
        super(Vanilla_GNN,self).__init__(self.input_dim, self.output_dim)
        self.conv1 = GCNConv(self.input_dim, 128)
        self.conv2 = GCNConv(128, self.output_dim)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars 
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)
    
    def forward(self,x):
        x = self.dropout(self.linear(self.flatten(x)))
        return x 


''' 
    The Basic Tabular Prediction Head
'''
class TabularHead(nn.Module):
    def __init__(self, X):
        super(TabularHead, self).__init__()
        self.weights = nn.ParameterList([
            nn.Parameter(torch.ones(x_each.shape[1],1) / x_each.shape[1]) for x_each in X
        ])
    
    def forward(self,x):
        x_total_score = [] 
        for idx, x_each in enumerate(x):
            x_score = x_each @ torch.clamp(self.weights[idx],min=0)
            x_total_score.append(x_score)
        x_total_score = torch.cat(x_total_score,dim=-1)
        return x_total_score 



''' 
    LLM
'''
class Model(nn.Module):

    def __init__(self, configs, a, b): 
        super(Model, self).__init__()
        self.task_name = configs.task_name 
        self.pred_len = configs.pred_len 
        self.seq_len = configs.seq_len 
        self.d_ff = configs.d_ff # 4096 
        self.tok_k = 5 
        self.d_llm = configs.llm_dim 
        #self.patch_len = configs.patch_len 
        #self.stride = configs.stride 
        '''
            LARGE LANGUAGE MODEL IMPORT GPT2 Only
        '''
        if configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')

            self.gpt2_config.num_hidden_layers = configs.llm_layers 
            self.gpt2_config.output_attentions = True 
            self.gpt2_config.output_hidden_states = True 
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code = True,
                    local_files_only = True,
                    config = self.gpt2_config,
                )
            except EnvironmentError:
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code = True,
                    local_files_only = False,
                    config = self.gpt2_config,
                )
        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise Exception('LLM model is not defined')
        
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        ### FROZEN ###
        for param in self.llm_model.parameters():
            param.requires_grad = False 
        if configs.prompt_domain:
            self.description = configs.content 
        else:
            self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'

        '''
            LARGE LANGUAGE MODEL WORD EMBEDDINGS AND OTHER CONFIGURATIONS
        '''
        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.dropout = nn.Dropout(configs.dropout)
        self.normalize_layers = Normalize(configs.enc_in, affine=False)
        

        ### TIME-LLM ###
        # self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        # self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        # self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        # self.head_nf = self.d_ff * self.patch_nums

        # if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
        #     self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
        #                                          head_dropout=configs.dropout)
        # else:
        #     raise NotImplementedError

        '''
            Tabular Prediction. 
            X : enc_x 
            Y : label 
        
        '''
        def forward(self):
            if self.task_name == 'Tabular_prediction':
                dec_out = self.tabular_projection(x_enc)
                return dec_out 

        

        def tabular_projection(self, x_enc):
            x_enc - self.normalize_layers(x_enc,'norm')






        '''
            코드 설명.
            start_col은 str일수도 있고, None일수도 있음.
            start_col_dist는 Dict일수도 List일수도 있음. (여기까지는 Union) 그런데, Optional이기 떄문에, None일수도 있음이 추가됨.
        
        '''
        def _get_start_sampler(self, start_col: tp.Optional[str],
                                start_col_dist: tp.Optional[tp.Union[tp.Dict,tp.List]]) -> TaptapStart:
            if start_col and start_col_dist is None:
                pass