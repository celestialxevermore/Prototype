import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

class FeatureRelationModule(nn.Module):
    def __init__(self, feature_dim = 768, hidden_dim = 256):
        super().__init__()
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, feature_embeddings):
        n_features = feature_embeddings.size(0)
        relations = []
        for i in range(n_features):
            for j in range(i+1, n_features):
                pair = torch.cat([feature_embeddings[i], feature_embeddings[j]], dim = -1)
                relation_score = self.feature_encoder(pair)
                relations.append((i,j, relation_score))
        return relations
    
class CoreSubgraphGenerator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.feature_dim = self.args.feature_dim
        self.similarity_threshold = self.args.similarity_threshold
    
    def forward(self, feature_name_list):
        all_features = torch.cat(feature_name_list, dim = 0)
        normalized_features = F.normalize(all_features, p = 2, dim = 1)

        similarity_matrix = torch.mm(normalized_features, normalized_features.t())

        source_feature_counts = [feat.size(0) for feat in feature_name_list]

        mask = similarity_matrix > self.similarity_threshold
        core_patterns = [] 

        
