import torch
import torch.nn as nn
import torch.nn.functional as F

from .pivot import PivotGCN




class GraphClf(nn.Module):
    '''
    A wrapper for GNN.
    '''
    def __init__(self, config):
        super(GraphClf, self).__init__()
        self.config = config
        self.name = 'GraphClf'
        self.device = config['device']
        nfeat = config['num_feat'] #feature维度
        nclass = config['num_class'] #标签数量
        hidden_size = config['hidden_size']
        self.dropout = config['dropout']

        self.feature_extractor = nn.Linear(
            in_features=nfeat, out_features=hidden_size)

        gcn_module = PivotGCN
        self.encoder = gcn_module(nfeat=nfeat,
                                  nhid=hidden_size,
                                  nclass=nclass,
                                  graph_hops=config.get('graph_hops', 2),
                                  dropout=self.dropout,
                                  batch_norm=config.get('batch_norm', False))
        

        
        
        

    def forward(self, node_features, adj):
        print("adj:", adj.shape, adj)
        node_vec = self.encoder(node_features, adj) 
        print("node_vec of GraphClf in graph_clf", node_vec.shape, node_vec)
  
        
        return node_vec

        
        
        
        
    