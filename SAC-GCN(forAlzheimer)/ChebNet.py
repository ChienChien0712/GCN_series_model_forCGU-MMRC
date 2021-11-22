import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class ChebConv(nn.Module):
    def __init__(self, K, in_features, out_features, enable_bias, act_func, droprate):
        super(ChebConv, self).__init__()
        self.K = K
        self.in_features = in_features
        self.out_features = out_features
        self.enable_bias = enable_bias
        self.act_func = act_func
        self.weight = nn.Parameter(torch.FloatTensor((K+1) * in_features, out_features))
        self.drop = nn.Dropout(p=droprate) if droprate > 0.0 else nn.Identity()
        if self.enable_bias == True:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.initialize_parameters()

    def initialize_parameters(self):
        init.xavier_uniform_(self.weight)

        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, data):
        x, L = data[:2]
        n_vertex = L.shape[0]
        x = self.drop(x)
        x_0 = x
        x_1 = torch.mm(L, x)
        if self.K < 0:
            raise ValueError(f'ERROR: K has to be a positive integer, but received {self.K}.')  
        elif self.K == 0:
            x_list = [x_0]
        elif self.K == 1:
            x_list = [x_0, x_1]
        elif self.K >= 2:
            x_list = [x_0, x_1]
            for k in range(2, self.K + 1):
                x_list.append(torch.mm(2 * L, x_list[k - 1]) - x_list[k - 2])
        x_tensor = torch.stack(x_list, dim=2)

        x_mul = torch.mm(x_tensor.reshape(n_vertex, -1), self.weight)

        if self.bias is not None:
            x_chebconv = x_mul + self.bias
        else:
            x_chebconv = x_mul
        x_chebconv = self.act_func(x_chebconv)
        
        return (x_chebconv, L)

class ChebyNet(nn.Module):
    def __init__(self, K, in_features, out_features, filters_gcn=None, enable_bias=True, act_func=None, droprate=0):
        super(ChebyNet, self).__init__()
        self.hidden = filters_gcn.copy()
        self.hidden.append(out_features)
        # GCN
        self.gconv = nn.Sequential(*([ChebConv(K=K,
                                               in_features=in_features if layer == 0 else self.hidden[layer - 1], 
                                               out_features=f, 
                                               act_func=act_func if layer != len(self.hidden)-1 else nn.Identity(),
                                               droprate=droprate,
                                               enable_bias=enable_bias) for layer, f in enumerate(self.hidden)]))             
        
    def forward(self, data):
        x = self.gconv(data)[0]
        x = F.log_softmax(x, dim=1)
        return x        
