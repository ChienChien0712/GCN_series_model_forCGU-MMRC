import torch
import torch.nn as nn
import torch.nn.functional as F

# NN layers and models
class GraphConv(nn.Module):
    def __init__(self,
                in_features,
                out_features,
                activation=None,
                dropout_gcn=0,
                gcn_bias=None):
        super(GraphConv, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features,bias=gcn_bias)
        self.activation = activation
        self.drop = nn.Dropout(p=dropout_gcn) if dropout_gcn > 0.0 else nn.Identity()
    def forward(self, data):
        x, A = data[:2]
        x = self.drop(x)
        x = self.fc(x)
        x = torch.spmm(A, x)
        x = self.activation(x) 
        return (x, A)
    
    
class GCN(nn.Module):
    def __init__(self,
                 in_features,
                 out_features, #n_class
                 filters_gcn=None,
                 dropout_gcn=0,
                 gcn_bias=True,
                 gcn_activation=None):
        super(GCN, self).__init__()
        self.hidden = filters_gcn.copy()
        self.hidden.append(out_features)
        # GCN
        self.gconv = nn.Sequential(*([GraphConv(in_features=in_features if layer == 0 else self.hidden[layer - 1], 
                                                out_features=f, 
                                                activation=gcn_activation if layer != len(self.hidden)-1 else nn.Identity(),
                                                dropout_gcn=dropout_gcn,
                                                gcn_bias=gcn_bias) for layer, f in enumerate(self.hidden)]))       
    def forward(self, data):
        x = self.gconv(data)[0]
        x = F.log_softmax(x, dim=1)
        return x  


