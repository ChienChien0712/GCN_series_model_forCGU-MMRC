import matplotlib.pyplot as plt
import numpy as np
import os
import copy
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.parameter import Parameter
from os.path import join as pjoin
import pandas as pd
import sys
plt.rcParams['figure.figsize'] = (15.0, 9.0)
plt.rcParams['figure.dpi'] = 100

# Hyper-parameters
dataset = sys.argv[1]
n_folds = int(sys.argv[2])
training_size_p = float(sys.argv[3]) 
balance = eval(sys.argv[4])
batch_size = int(sys.argv[5])
epochs = int(sys.argv[6])
lr = float(sys.argv[7])
wdecay = float(sys.argv[8])
#Graph Convolution Layer
model_name = sys.argv[9] #'GCN'
filters_gcn = [int(i.strip()) for i in sys.argv[10].split(',')]
graph_kernel = sys.argv[11]  #'Normalized-Laplacian' #Normalization #Cluster-GCN
connection = sys.argv[12] #'residual connection' #dense connection #'None'
gcn_activation = eval('nn.'+sys.argv[13]+'(inplace=True)')#'ELU' #'ReLU' 
dropout_gcn = float(sys.argv[14])
gcn_bias = eval(sys.argv[15])
#information pooling
pooling_method = sys.argv[16] #max,sum,mean
#fc
n_hidden_fc = sys.argv[17] #'None' or '32,16'
if n_hidden_fc != 'None':
    n_hidden_fc = list(map(int,n_hidden_fc.strip().split(',')))
fc_activation = eval('nn.'+sys.argv[18]+'(inplace=True)')#'ELU' #'ReLU' #'Identity'    
dropout_fc = float(sys.argv[19])
fc_bias = eval(sys.argv[20])

#device
device = sys.argv[21]  # 'cuda', 'cpu'
seed = sys.argv[22]
threads = int(sys.argv[23]) #線程數目
log_interval = int(sys.argv[24])

#output folder
output_folder = sys.argv[25]


# NN layers and models
class GraphConv(nn.Module):
    def __init__(self,
                in_features,
                out_features,
                activation=None,
                dropout_gcn=0,
                gcn_bias=True,
                gconv_order=None):
        super(GraphConv, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features,bias=gcn_bias)
        self.activation = activation
        self.drop = nn.Dropout(p=dropout_gcn) if dropout_gcn > 0.0 else nn.Identity()
        self.gconv_order = gconv_order
        
    def L_batch(self, A):
        batch, N = A.shape[:2]
        I = torch.eye(N).unsqueeze(0).to(device)
        A_hat = A + I 
        if graph_kernel == 'Normalized-Laplacian':
            D_hat = (torch.sum(A_hat, 1)) ** (-0.5)
            L = D_hat.view(batch, N, 1) * A_hat * D_hat.view(batch, 1, N)
        elif graph_kernel == 'Normalization':
            D_hat = (torch.sum(A_hat, 1)) ** (-1)
            L = D_hat.view(batch, N, 1) * A_hat 
        elif graph_kernel == 'Cluster-GCN':
            D_hat = torch.sum(A_hat, 1)**(-1)
            A_hat = D_hat.view(batch, N, 1) * A_hat
            for i in range(len(A_hat)):
                A_hat[i] = A_hat[i]+torch.diagflat(torch.diag(A_hat[i]))
            L = A_hat
        return L

    def forward(self, data):
        x, A = data[:2]
        x_res = data[0]
        x = self.drop(x)
        x = self.fc(torch.bmm(self.L_batch(A), x))
        x = self.activation(x) 
        if (connection == 'residual-connection')&(self.gconv_order != 0):
            assert x.shape == x_res.shape, "'residual connection' was set," +\
                                           "hyper-parameter 'filters_gcn' should be set in same dimension.\n" +\
                                           "filters_gcn you set is %s."%filters_gcn +\
                                           "try to use %s."%' or '.join([str([i]*len(filters_gcn)) for i in set(filters_gcn)])
            x = x + x_res
        if (connection == 'dense-connection')&(self.gconv_order != 0):
            x = torch.cat([x,x_res],2)
        return (x, A)


class GCN(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 filters_gcn=[64,64,64],
                 dropout_gcn=0,
                 n_hidden_fc=[32,16,8],
                 dropout_fc=0.2,
                 gcn_bias=True,
                 gcn_activation=nn.ReLU(inplace=True),
                 fc_bias=True,
                 fc_activation=None):
        super(GCN, self).__init__()
        
        # GCN
        if (connection == 'dense-connection'):
            filters_gcn_dense = [sum(filters_gcn[:i+1]) for i in range(len(filters_gcn))]
            self.gconv = nn.Sequential(*([GraphConv(in_features=in_features if layer == 0 else filters_gcn_dense[layer - 1], 
                                                    out_features=f, 
                                                    activation=gcn_activation,
                                                    dropout_gcn=dropout_gcn,
                                                    gcn_bias=gcn_bias,
                                                    gconv_order=layer) for layer, f in enumerate(filters_gcn)]))
        else:
            self.gconv = nn.Sequential(*([GraphConv(in_features=in_features if layer == 0 else filters_gcn[layer - 1], 
                                                    out_features=f, 
                                                    activation=gcn_activation,
                                                    dropout_gcn=dropout_gcn,
                                                    gcn_bias=gcn_bias,
                                                    gconv_order=layer) for layer, f in enumerate(filters_gcn)]))
        
        # Fully connected layers
        fc = []
        pooling_noumber = len(pooling_method.split(','))
        if n_hidden_fc != 'None':
            for layer, f in enumerate(n_hidden_fc):
                if dropout_fc > 0:
                    fc.append(nn.Dropout(p=dropout_fc))
                else:
                    fc.append(nn.Identity())
                if layer == 0:
                    if (connection == 'dense-connection'):
                        fc.append(nn.Linear(filters_gcn_dense[-1]*pooling_noumber, n_hidden_fc[layer], bias=fc_bias)) 
                    else:
                        fc.append(nn.Linear(filters_gcn[-1]*pooling_noumber, n_hidden_fc[layer], bias=fc_bias)) 
                    fc.append(fc_activation)
                else:
                    fc.append(nn.Linear(n_hidden_fc[layer-1], n_hidden_fc[layer], bias=fc_bias))   
                    fc.append(fc_activation)
            n_last = n_hidden_fc[-1]
        else:
            if (connection == 'dense-connection'):
                n_last = filters_gcn_dense[-1]*pooling_noumber
            else:
                n_last = filters_gcn[-1]*pooling_noumber
        #last layer
        if dropout_fc > 0:
            fc.append(nn.Dropout(p=dropout_fc))
        else:
            fc.append(nn.Identity())            
        fc.append(nn.Linear(n_last, out_features, bias=fc_bias))
        self.fc = nn.Sequential(*fc) 
        
    def forward(self, data):
        mask = data[2].clone()
        N_nodes = torch.sum(mask, dim=1).reshape(len(torch.sum(mask, dim=1)),1)
        
        x = self.gconv(data)[0]
        
        pooling_ls = []
        if 'max' in pooling_method:
            max_pooling = torch.max(x, 1)[0]
            pooling_ls.append(max_pooling)
        if 'sum' in pooling_method:
            sum_pooling = torch.sum(x, 1)
            pooling_ls.append(sum_pooling)
        if 'mean' in pooling_method:
            mean_pooling = torch.sum(x, 1)/N_nodes
            pooling_ls.append(mean_pooling)
        x = torch.cat(pooling_ls,1)  
        
        x = self.fc(x) 
        x = F.log_softmax(x, dim=1)
        return x  
    
# Data loader and reader
class GraphData(torch.utils.data.Dataset):
    def __init__(self,datareader,fold_id,split):
        self.fold_id = fold_id #預設0，只執行一次
        self.split = split #"train" or "test"
        self.rnd_state = datareader.rnd_state
        self.set_fold(datareader.data, fold_id) #利用方法，建立屬性。set_fold()在下面
        
        
    def set_fold(self, data, fold_id):
        self.total = len(data['labels']) #graph數目
        self.N_nodes_max = data['N_nodes_max'] #最多node的graph之node數目
        self.n_classes = data['n_classes'] #graph分類的種類數目
        self.features_dim = data['features_dim'] #node的feature數目
        self.idx = data['splits'][fold_id][self.split]#train或test的index
        self.labels = copy.deepcopy([data['labels'][i] for i in self.idx])#特定index(train or test)下的graph labels
        self.adj_list = copy.deepcopy([data['adj_list'][i] for i in self.idx])#特定index(train or test)下的A矩陣
        self.features_onehot = copy.deepcopy([data['features_onehot'][i] for i in self.idx])#特定index(train or test)下的node feature
        print('%s: %d/%d' % (self.split.upper(), len(self.labels), len(data['labels'])))
        self.indices = np.arange(len(self.idx))  # sample indices for this epoch(for這次epoch，index從新編碼)
        self.label_to_target = data['label_to_target']
        self.node_idx_to_id = data['node_idx_to_id']
        self.targets = data['targets']
        
    def pad(self, mtx, desired_dim1, desired_dim2=None, value=0):
        sz = mtx.shape
        assert len(sz) == 2, ('only 2d arrays are supported', sz)
        if desired_dim2 is not None:
            mtx = np.pad(mtx, ((0, desired_dim1 - sz[0]), (0, desired_dim2 - sz[1])), 'constant', constant_values=value)
        else:
            mtx = np.pad(mtx, ((0, desired_dim1 - sz[0]), (0, 0)), 'constant', constant_values=value)
        return mtx
    
    def nested_list_to_torch(self, data):
        if isinstance(data, dict):
            keys = list(data.keys())           
        for i in range(len(data)):
            if isinstance(data, dict):
                i = keys[i]
            if isinstance(data[i], np.ndarray):
                data[i] = torch.from_numpy(data[i]).float()
            elif isinstance(data[i], list):
                data[i] = list_to_torch(data[i])
        return data
        
    def __len__(self): #__len__:未來可以len(類別)，呼叫下面code
        return len(self.labels)

    def __getitem__(self, index):#__getitem__:未來這個類別可以使用[]索引，來完成下面code
        index = self.indices[index]
        N_nodes_max = self.N_nodes_max
        N_nodes = self.adj_list[index].shape[0]
        graph_support = np.zeros(self.N_nodes_max)
        graph_support[:N_nodes] = 1
        #1.把features捕到620,預設補0
        #2.把adj補到620*620,預設補0
        #3.graph_support: mask
        #4.每個圖的真正nodes數
        return self.nested_list_to_torch([self.pad(self.features_onehot[index].copy(), self.N_nodes_max),  # node_features
                                          self.pad(self.adj_list[index], self.N_nodes_max, self.N_nodes_max),  # adjacency matrix
                                          graph_support,  # mask with values of 0 for dummy (zero padded) nodes, otherwise 1 
                                          N_nodes,
                                          int(self.labels[index]),
                                          self.idx[index]])  # convert to torch

class DataReader():
    def __init__(self,
                 data_dir, 
                 rnd_state=None, 
                 training_size_p=None,
                 folds=None,
                 balance=None):
        self.data_dir = data_dir
        self.rnd_state = np.random.RandomState() if rnd_state == 'Random' else np.random.RandomState(int(rnd_state))
        
        files = os.listdir(self.data_dir)
        
        #data starage!
        data = {}
        #1. nodes:為dict，{node_id:graph_id}
        #2. graphs:為dict,{graph_id:np.array([node_id 1,node_id 2,...])}
        nodes, graphs = self.read_graph_nodes_relations(list(filter(lambda f: f.find('graph_indicator') >= 0, files))[0])
        #3. data['node_id_to_idx']
        node_id_to_idx, node_idx_to_id= self.read_node_ID(list(filter(lambda f: f.find('node_features') >= 0, files))[0])
        data['node_id_to_idx'] = node_id_to_idx
        data['node_idx_to_id'] = node_idx_to_id
        #4. data['features_onehot']
        data['features_onehot'] = self.read_node_features(list(filter(lambda f: f.find('node_features') >= 0, files))[0], nodes, graphs)  
        #data['adj_list']
        data['adj_list'] = self.read_graph_adj(list(filter(lambda f: f.find('_A') >= 0, files))[0], nodes, graphs,node_id_to_idx) 
        #data['labels'] 0開始
        target_to_label = {}
        label_to_target = {}
        targets = np.array(self.parse_txt_file(list(filter(lambda f: f.find('graph_labels') >= 0, files))[0], 
                                      line_parse_fn=lambda s: s.strip()))
        data['targets'] = targets
        target_category = sorted(list(set(targets)))
        for l, t in enumerate(target_category): 
            target_to_label[t] = l
            label_to_target[l] = t
        data['labels'] = np.array([target_to_label[t] for t in targets])
        data['target_to_label'] = target_to_label
        data['label_to_target'] = label_to_target
        n_edges, degrees = [], []
        for sample_id, adj in enumerate(data['adj_list']):
            N = len(adj)  # number of nodes
            n = np.sum(adj)  # total sum of edges
            n_edges.append(int(n/2))  # undirected edges, so need to divide by 2
            degrees.extend(list(np.sum(adj, 1)))
        features_dim = len(data['features_onehot'][0][0])
        shapes = [len(adj) for adj in data['adj_list']]
        N_nodes_max = np.max(shapes)
        classes = target_category
        n_classes = len(target_category)

        print('N nodes avg/std/min/max: \t%.2f/%.2f/%d/%d' % (np.mean(shapes), np.std(shapes), np.min(shapes), np.max(shapes)))
        print('N edges avg/std/min/max: \t%.2f/%.2f/%d/%d' % (np.mean(n_edges), np.std(n_edges), np.min(n_edges), np.max(n_edges)))
        print('Node degree avg/std/min/max: \t%.2f/%.2f/%d/%d' % (np.mean(degrees), np.std(degrees), np.min(degrees), np.max(degrees)))
        print('Node features dim: \t\t%d' % features_dim)
        print('N classes: \t\t\t%d' % n_classes)
        print('Classes: \t\t\t%s' %(', '.join(classes)))
        for lbl in classes:
            print('Class %s: \t\t\t%s samples' % (lbl, np.sum(targets == lbl)))
        #判斷每個資料中，graph數量是否相等
        N_graphs = len(data['labels']) 
        assert N_graphs == len(data['adj_list']) == len(data['features_onehot']), 'invalid data'

        train_ids, test_ids = self.split_ids(data['labels'], rnd_state=self.rnd_state, training_size_p=training_size_p,
                                             folds=n_folds, balance=balance)
        splits = [] #塞入dict('train':[index...],'test':[index...])
        for fold in range(folds):
            splits.append({'train': train_ids[fold],
                           'test': test_ids[fold]})
        
        data['splits'] = splits #folds份的train和test之index
        data['N_nodes_max'] = N_nodes_max
        data['features_dim'] = features_dim
        data['n_classes'] = n_classes #graph label種類數目
        
        self.data = data # data為一個dict()

    def split_ids(self, labels_all, rnd_state=None,folds=1, training_size_p=None, balance=False):
        if folds == 1:
            if balance == True:
                classes = list(set(labels_all))
                classes_dict = dict()
                for i in classes:
                    classes_dict[i] = []
                for idx,l in enumerate(labels_all):
                    classes_dict[l].append(idx)
                min_classes_n = len(labels_all)
                for i in classes:
                    if len(classes_dict[i]) < min_classes_n:
                        min_classes_n = len(classes_dict[i])
                training_size_per_class = int(np.round(min_classes_n*training_size_p))
                ids_all = np.arange(len(labels_all))
                ids = ids_all[rnd_state.permutation(len(ids_all))]
                train_ids = []
                for i in classes:
                    class_ls = np.array(classes_dict[i])
                    sampling = class_ls[rnd_state.permutation(len(class_ls))][0:training_size_per_class]

                    train_ids.extend(sampling)
                test_ids = [np.array([e for e in ids if e not in train_ids])]    
                train_ids = [np.array(train_ids)]
            else:
                ids_all = np.arange(len(labels_all))
                n = len(ids_all) #n:graph的數目
                ids = ids_all[rnd_state.permutation(n)]
                testing_size = int(np.round(n*(1-training_size_p)))
                test_ids = ids[0:testing_size] # 包著np.array()
                train_ids = [np.array([e for e in ids if e not in test_ids])] # 包著np.array()
                test_ids = [test_ids]
        elif folds > 1:
            ids_all = np.arange(len(labels_all))
            n = len(ids_all)
            ids = ids_all[rnd_state.permutation(n)]
            stride = int(np.ceil(n / float(folds)))
            test_ids = [ids[i: i + stride] for i in range(0, n, stride)]
            assert np.all(np.unique(np.concatenate(test_ids)) == sorted(ids_all)), 'some graphs are missing in the test sets'
            assert len(test_ids) == folds, 'invalid test sets'
            train_ids = []
            for fold in range(folds):
                train_ids.append(np.array([e for e in ids if e not in test_ids[fold]]))
                assert len(train_ids[fold]) + len(test_ids[fold]) == len(np.unique(list(train_ids[fold]) + list(test_ids[fold]))) == n, 'invalid splits'

        return train_ids, test_ids

    def parse_txt_file(self, fpath, line_parse_fn=None):
        #pjoin=os.path.join:路徑拼接
        #os.path.join([PATH_1], [PATH_2], [PATH_3], ...)-->return:[PATH_1]/[PATH_2]/[PATH_3]
        with open(pjoin(self.data_dir, fpath), 'r') as f:
            lines = f.readlines()
        #if line_parse_fn is not None else s:代表如果有處理字串函數就執行，否則就保留原本的樣子
        data = [line_parse_fn(s) if line_parse_fn is not None else s for s in lines]
        return data
    
    def read_graph_adj(self, fpath, nodes, graphs, node_id_to_idx):
        def fn_read_graph_adj(s):
            if ',' in s:
                return s.strip().split(',')
            else:
                return s.strip().split()
        edges = self.parse_txt_file(fpath, line_parse_fn=fn_read_graph_adj)
        adj_dict = {}
        for edge in edges:
            node1 = node_id_to_idx[edge[0].strip()]
            node2 = node_id_to_idx[edge[1].strip()]
            graph_id = nodes[node1]
            assert graph_id == nodes[node2], ('invalid data', graph_id, nodes[node2])
            
            if graph_id not in adj_dict:
                n = len(graphs[graph_id])
                adj_dict[graph_id] = np.zeros((n, n))
            ind1 = np.where(graphs[graph_id] == node1)[0]
            ind2 = np.where(graphs[graph_id] == node2)[0]
            assert len(ind1) == len(ind2) == 1, (ind1, ind2)
            adj_dict[graph_id][ind1, ind2] = 1
            adj_dict[graph_id][ind2, ind1] = 1
        adj_list = [adj_dict[graph_id] for graph_id in sorted(list(graphs.keys()))]        
        return adj_list
        
    #graph_indicator
    def read_graph_nodes_relations(self, fpath):
        #node從0開始
        #graph沒限定，但要是整數
        graph_ids = self.parse_txt_file(fpath, line_parse_fn=lambda s: int(s.rstrip()))
        nodes, graphs = {}, {}
        for node_id, graph_id in enumerate(graph_ids):
            if graph_id not in graphs:
                graphs[graph_id] = []
            graphs[graph_id].append(node_id)
            nodes[node_id] = graph_id
        graph_ids = np.unique(list(graphs.keys()))
        for graph_id in graphs:
            graphs[graph_id] = np.array(graphs[graph_id])
        return nodes, graphs

    def read_node_features(self, fpath, nodes, graphs):
        def fn_read_node_features(s):
            if ',' in s:
                return list(map(float,(s.strip().split(',')[1:])))
            else:
                return list(map(float,(s.strip().split()[1:])))
        node_features_all = self.parse_txt_file(fpath, line_parse_fn=fn_read_node_features)
        node_features = {}
        #node_features:資料格式和graphs相似
        for node_id, x in enumerate(node_features_all):
            graph_id = nodes[node_id]
            if graph_id not in node_features:
                node_features[graph_id] = [ None ] * len(graphs[graph_id])
            ind = np.where(graphs[graph_id] == node_id)[0]
            #assert 判斷式, 如果有誤回傳的內容
            assert len(ind) == 1, ind
            assert node_features[graph_id][ind[0]] is None, node_features[graph_id][ind[0]]
            node_features[graph_id][ind[0]] = x
        node_features_lst = [np.array(node_features[graph_id]) for graph_id in sorted(list(graphs.keys()))]
        return node_features_lst
    
    def read_node_ID(self, fpath):
        def fn_read_node_ID(s):
            if ',' in s:
                return s.strip().split(',')[0]
            else:
                return s.strip().split()[0]
        node_ID_all = self.parse_txt_file(fpath, line_parse_fn=fn_read_node_ID)
        assert len(node_ID_all) == len(set(node_ID_all))
        
        node_id_to_idx = {}#str:int
        node_idx_to_id = {}
        for node_idx, node_id in enumerate(node_ID_all):
            node_id_to_idx[node_id] = node_idx
            node_idx_to_id[node_idx] = node_id
        return node_id_to_idx, node_idx_to_id


print('Loading data')

datareader = DataReader(data_dir=dataset, 
                        rnd_state=seed,training_size_p=training_size_p,folds=n_folds,balance=balance)

train_acc_folds = []
test_acc_folds = []
for fold_id in range(n_folds):
    print('\nFOLD', fold_id+1)
    loaders = []
    for split in ['train', 'test']:
        #製作"train"或"test" graph data
        gdata = GraphData(fold_id=fold_id, datareader=datareader, split=split)
        loader = torch.utils.data.DataLoader(gdata, 
                                             batch_size=batch_size,
                                             shuffle=split.find('train') >= 0,
                                             num_workers=threads)
        loaders.append(loader)
        if split == 'train':
            training_size = len(gdata.idx)
    if model_name == 'GCN':
        model = GCN(in_features=loaders[0].dataset.features_dim, 
                    out_features=loaders[0].dataset.n_classes,
                    filters_gcn=filters_gcn,
                    gcn_bias=gcn_bias,
                    gcn_activation=gcn_activation,
                    dropout_gcn=dropout_gcn,
                    fc_bias=fc_bias,
                    n_hidden_fc=n_hidden_fc,
                    dropout_fc=dropout_fc,
                    fc_activation=fc_activation).to(device)    

    print('\nInitialize model')
    print(model)
    c = 0
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        c += p.numel()
    print('N trainable parameters:', c)

    optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr,
                weight_decay=wdecay,
                betas=(0.5, 0.999))
    scheduler = lr_scheduler.MultiStepLR(optimizer, [20, 30], gamma=0.1)


    def train(train_loader):
        scheduler.step()#每個batch就會改變學習率
        model.train()
        start = time.time()
        train_loss, correct, n_samples = 0, 0, 0
        train_loss_batch_ls = []
        train_acc_batch_ls = []
        for batch_idx, data in enumerate(train_loader):
            for i in range(len(data)):
                data[i] = data[i].to(device)
            optimizer.zero_grad()
            output = model(data)     
            loss = loss_fn(output, data[4])
            loss.backward()
            optimizer.step()
            time_iter = time.time() - start
            train_loss += loss.item() * len(output)
            n_samples += len(output)
            pred = output.detach().cpu().max(1, keepdim=True)[1]
            correct += pred.eq(data[4].detach().cpu().view_as(pred)).sum().item()
            acc = 100. * correct / n_samples
            train_loss_batch_ls.append(train_loss/n_samples)
            train_acc_batch_ls.append(acc/100)
            if batch_idx % log_interval == 0 or batch_idx == len(train_loader) - 1:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}(avg: {:.4f})\tAcc: {:.2f}%({}/{}) \tsec/iter: {:.4f}'.format(
                    epoch, n_samples, len(train_loader.dataset),
                    100. * (batch_idx + 1) / len(train_loader), loss.item(), train_loss / n_samples, 
                    acc, correct, n_samples, time_iter / (batch_idx + 1) ))    
        return train_loss_batch_ls, train_acc_batch_ls
    def test(test_loader):
        model.eval()
        start = time.time()
        test_loss, correct, n_samples = 0, 0, 0
        for batch_idx, data in enumerate(test_loader):
            for i in range(len(data)):
                data[i] = data[i].to(device)
            output = model(data)
            loss = loss_fn(output, data[4], reduction='sum')
            test_loss += loss.item()
            n_samples += len(output)
            pred = output.detach().cpu().max(1, keepdim=True)[1]
            correct += pred.eq(data[4].detach().cpu().view_as(pred)).sum().item()
        time_iter = time.time() - start
        test_loss /= n_samples
        acc = 100. * correct / n_samples
        print('Test set (epoch {}): Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(epoch+1, 
                                                                                              test_loss, 
                                                                                              correct, 
                                                                                              n_samples, acc))
        return test_loss,acc/100

    def predict(loader_full):
        idx_ls = []
        pred_ls = [] 
        label_ls = []
        length_ls = []
        output_ls = []
        print('[Trained Model]')
        for i in [0,1]:
            model.eval()     
            pred_tmp = []
            label_tmp = []
            for batch_idx, data in enumerate(loader_full[i]):
                for j in range(len(data)):
                    data[j] = data[j].to(device)
                output = model(data)
                idx_ls.extend(data[5].tolist())
                pred = output.detach().cpu().max(1, keepdim=True)[1]
                pred_ls.extend(pred.reshape(pred.shape[0]).tolist())
                label_ls.extend(data[4].tolist())
                output_ls.extend(output)
                pred_tmp.extend(pred.reshape(pred.shape[0]).tolist())
                label_tmp.extend(data[4].tolist())
            total = len(pred_tmp)
            c = sum(np.array(pred_tmp)==np.array(label_tmp)) 
            if i==0:
                print('Training Set: Accuracy=%.2f%%(%s/%s)'%(c*100/total,c,total))
                train_acc_folds.append(c*100/total)
            elif i==1:
                print('Testing Set: Accuracy=%.2f%%(%s/%s)'%(c*100/total,c,total))  
                test_acc_folds.append(c*100/total)
            length_ls.append(total)
        return idx_ls, pred_ls, label_ls, length_ls, output_ls

    train_loss_ls = []
    train_acc_ls = []
    test_loss_ls = []
    test_acc_ls = []
    loss_fn = F.nll_loss
    for epoch in range(epochs):
        train_loss, train_acc = train(loaders[0])
        test_loss, test_acc = test(loaders[1])
        train_loss_ls.extend(train_loss)
        train_acc_ls.extend(train_acc)
        test_loss_ls.append(test_loss)
        test_acc_ls.append(test_acc)   

    idx_ls, pred_ls, label_ls, length_ls, output_ls = predict(loaders)

    #plot
    length_train = range(len(train_loss_ls))
    length_test = range(int(np.ceil(training_size/batch_size))-1,len(train_loss_ls),int(np.ceil(training_size/batch_size)))
    plt.plot(length_train,train_acc_ls,label='training accuracy')
    plt.plot(length_test,test_acc_ls,label='validation accuracy')
    x_ticks = [0]+list(length_test)
    plt.xticks(x_ticks,list(range(0,epochs+1)))
    plt.xlabel('epoch',fontsize=18)
    plt.ylabel('accuracy',fontsize=18)
    plt.legend(fontsize=12)
    plt.grid(linestyle='--')
    plt.savefig(output_folder+'/acc_fold%s.png'%(fold_id+1))
    plt.clf()

    plt.plot(length_train,train_loss_ls,label='training loss')
    plt.plot(length_test,test_loss_ls,label='validation loss')
    plt.xticks(x_ticks,list(range(0,epochs+1)))
    plt.xlabel('epoch',fontsize=18)
    plt.ylabel('loss',fontsize=18)
    plt.legend(fontsize=12)
    plt.grid(linestyle='--')
    plt.savefig(output_folder+'/loss_fold%s.png'%(fold_id+1))
    plt.clf()    
    
    #xlsx
    writer = pd.ExcelWriter(output_folder+'/result_fold%s.xlsx'%(fold_id+1), engine = 'xlsxwriter')
    ################################################
    train_loss_acc_df = pd.DataFrame(zip(range(1,epochs+1),np.array(train_loss_ls)[length_test],np.array(train_acc_ls)[length_test]))
    train_loss_acc_df.columns = ['epoch','training loss','training accuracy']
    train_loss_acc_df.to_excel(writer,sheet_name='training',header=True,index=False) 
    test_loss_acc_df = pd.DataFrame(zip(range(1,epochs+1),test_loss_ls,test_acc_ls))
    test_loss_acc_df.columns = ['epoch','testing loss','testing accuracy']
    test_loss_acc_df.to_excel(writer,sheet_name='validation',header=True,index=False) 
    ################################################
    for i in range(len(model.gconv)):
        f_GCN_weight = pd.DataFrame(np.matrix(model.gconv[i].fc.weight.tolist()).T)
        f_GCN_weight.to_excel(writer,sheet_name='GCN%s_weight'%i,header=False,index=False)
        if gcn_bias:
            f_GCN_bias = pd.DataFrame(np.matrix(model.gconv[i].fc.bias.tolist()).T)
            f_GCN_bias.to_excel(writer,sheet_name='GCN%s_bias'%i,header=False,index=False)    
            
    fc_layer = 0
    for i in range(len(model.fc)):
        if i % 3 == 1:
            f_fc_weight = pd.DataFrame(np.matrix(model.fc[i].weight.tolist()).T)
            f_fc_weight.to_excel(writer,sheet_name='FC%s_weight'%fc_layer,header=False,index=False)    
            if fc_bias:
                f_fc_bias = pd.DataFrame(np.matrix(model.fc[i].bias.tolist()).T)
                f_fc_bias.to_excel(writer,sheet_name='FC%s_bias'%fc_layer,header=False,index=False)
            fc_layer += 1
    #################################################################
    label_target_ls = [loaders[0].dataset.label_to_target[i] for i in label_ls]
    pred_target_ls = [loaders[0].dataset.label_to_target[i] for i in pred_ls]
    id_ls = [loaders[0].dataset.node_idx_to_id[i] for i in idx_ls]
    splits = ['training']*length_ls[0]+['validation']*length_ls[1]
    prediction = pd.DataFrame(zip(idx_ls,id_ls,pred_target_ls,label_target_ls,splits))
    prediction.columns = ['IDX','ID','prediction','label','splits']
    
    class_prob = pd.DataFrame([np.exp(i.tolist())*100 for i in output_ls])
    columns = []
    for c in class_prob.columns:
        class_prob[c] = class_prob[c].map('{:,.2f}%'.format)
        columns.append('p('+loaders[0].dataset.label_to_target[c]+')')
    class_prob.columns = columns    
    prediction = pd.concat([prediction,class_prob], axis=1)
    
    prediction = prediction.sort_values(by=['IDX'])
    prediction = prediction.loc[:,'ID':]
    corrects = []
    for i in prediction['prediction'] == prediction['label']:
        if i:
            corrects.append(1)
        else:
            corrects.append(0)
    prediction['correct'] = corrects
    prediction.to_excel(writer,sheet_name='Prediction',header=True,index=False)  
    #######################################################################################
    #sampling info.
    class_number_train = []
    class_proportion_train = []
    class_number_val = []
    class_proportion_val = []    
    
    label_set = sorted(list(set(prediction['label'])))
    for i in label_set:
        train_n = len(prediction[(prediction['label']==i)&(prediction['splits']=='training')])
        train_total = len(prediction[(prediction['splits']=='training')])
        val_n = len(prediction[(prediction['label']==i)&(prediction['splits']=='validation')])
        val_total = len(prediction[(prediction['splits']=='validation')])
        class_number_train.append(train_n)
        class_proportion_train.append('%.1f%%'%(train_n*100/train_total))
        class_number_val.append(val_n)
        class_proportion_val.append('%.1f%%'%(val_n*100/val_total))  
    sampling_info = pd.DataFrame(zip(label_set,class_number_train,class_proportion_train,class_number_val,class_proportion_val))
    sampling_info.columns = ['label','training set','proportion of training set','validation set','proportion of validation set']
    sampling_info.to_excel(writer, sheet_name = 'sampling_info',index=False)    
    ############################################################################################
    #class_acc
    Number_Train = []
    Correct_Train = []
    Number_Val = []
    Correct_Val = []
    Values = list(loaders[0].dataset.label_to_target.values())
    for i in Values:
        Correct_Train.append(len(prediction[(prediction['correct']==1)&(prediction['label']==i)&(prediction['splits']=='training')]))
        Number_Train.append(len(prediction[(prediction['label']==i)&(prediction['splits']=='training')]))
        Correct_Val.append(len(prediction[(prediction['correct']==1)&(prediction['label']==i)&(prediction['splits']=='validation')]))
        Number_Val.append(len(prediction[(prediction['label']==i)&(prediction['splits']=='validation')]))
        Rate_Train = np.array(Correct_Train)*100/np.array(Number_Train)
        Rate_Val = np.array(Correct_Val)*100/np.array(Number_Val)
    Class_Prob = pd.DataFrame(zip(Values,Correct_Train,Number_Train,Rate_Train,Correct_Val,Number_Val,Rate_Val))
    Class_Prob.columns = ['class','train_correct','train_number','train_accuracy','validation_correct','validation_number','validation_accuracy']
    Class_Prob['train_accuracy'] = Class_Prob['train_accuracy'].map('{:,.2f}%'.format)
    Class_Prob['validation_accuracy'] = Class_Prob['validation_accuracy'].map('{:,.2f}%'.format)
    Class_Prob.to_excel(writer, sheet_name = 'class_acc',index=False)          
    #################################################################################################
    #label_pred_mx_train
    #label_pred_mx_val
    class_to_onehot = dict(zip([loaders[0].dataset.label_to_target[i] for i in range(len(loaders[0].dataset.label_to_target))],range(len(loaders[0].dataset.label_to_target))))
    label_pred_mx_train = np.zeros([len(class_to_onehot),len(class_to_onehot)])
    label_pred_mx_val = np.zeros([len(class_to_onehot),len(class_to_onehot)])
    for i in range(len(prediction)):
        if prediction.at[i,'splits'] == 'training':
            label_pred_mx_train[class_to_onehot[prediction.at[i,'label']]][class_to_onehot[prediction.at[i,'prediction']]] += 1
        elif prediction.at[i,'splits'] == 'validation':
            label_pred_mx_val[class_to_onehot[prediction.at[i,'label']]][class_to_onehot[prediction.at[i,'prediction']]] += 1
    label_pred_mx_train = pd.DataFrame(label_pred_mx_train,dtype=int)
    label_pred_mx_val = pd.DataFrame(label_pred_mx_val,dtype=int)        
    label_pred_mx_train.columns = [loaders[0].dataset.label_to_target[i] for i in range(len(loaders[0].dataset.label_to_target))]
    label_pred_mx_val.columns = [loaders[0].dataset.label_to_target[i] for i in range(len(loaders[0].dataset.label_to_target))]       
    label_pred_mx_train['label\pred'] = [loaders[0].dataset.label_to_target[i] for i in range(len(loaders[0].dataset.label_to_target))]
    label_pred_mx_val['label\pred'] = [loaders[0].dataset.label_to_target[i] for i in range(len(loaders[0].dataset.label_to_target))]        
    label_pred_mx_train = pd.concat([label_pred_mx_train['label\pred'],label_pred_mx_train[[loaders[0].dataset.label_to_target[i] for i in range(len(loaders[0].dataset.label_to_target))]]],axis=1)
    label_pred_mx_val = pd.concat([label_pred_mx_val['label\pred'],label_pred_mx_val[[loaders[0].dataset.label_to_target[i] for i in range(len(loaders[0].dataset.label_to_target))]]],axis=1)
    label_pred_mx_train.to_excel(writer, sheet_name = 'train_label_pred',index=False)         
    label_pred_mx_val.to_excel(writer, sheet_name = 'val_label_pred',index=False)    
    ############################################################################################
    writer.save()





summary = []
summary.append('traing set:')
for i in range(len(train_acc_folds)):
    summary.append('accuracy of fold #%2d: %.2f%%'%(i+1,train_acc_folds[i]))
summary.append('%s-folds accuracy: %.2f%% (std=%.2f%%)'%(n_folds,np.mean(train_acc_folds),np.std(train_acc_folds)))
summary.append('testing set:')
for i in range(len(test_acc_folds)):
    summary.append('accuracy of fold #%2d: %.2f%%'%(i+1,test_acc_folds[i]))
summary.append('%s-folds accuracy: %.2f%% (std=%.2f%%)'%(n_folds,np.mean(test_acc_folds),np.std(test_acc_folds)))
f1 = open(output_folder+'/summary_acc.txt','w')
f1.write('\n'.join(summary))
f1.close()    







if graph_kernel == 'Cluster-GCN':
    tilde_A = '(D+I)^(-1)(A+I)'
else:
    tilde_A = 'A+I'

if graph_kernel == 'Normalized-Laplacian':
    kernel_description = 'tilde(D)^(-1/2)tilde(A)tilde(D)^(-1/2)'
elif graph_kernel == 'Normalization':
    kernel_description = 'tilde(D)^(-1)tilde(A)'
elif graph_kernel == 'Cluster-GCN':
    kernel_description = 'tilde(A)+lambda*diag(tilde(A)); default lambda=1'
model_structure_full = []

for j in range(len(model.gconv)):
    model_structure = str(model.gconv[j]).split('\n')
    for i in range(len(model_structure)):
        if i != len(model_structure)-1:
            model_structure_full.append(model_structure[i])
            
    model_structure_full.append('  (adjacency matrix): tilde(A) = %s'%(tilde_A))
    model_structure_full.append('  (graph kernel): '+kernel_description)
    model_structure_full.append('  (connection): '+connection)
    model_structure_full.append(')')


layer_info = len(filters_gcn)-1
pooling_description = 'concatenate(%s pooling) over nodes by GCN layer%s'%(pooling_method,layer_info)

    
model_structure_full.append(' '*int(len(pooling_description)/2) +'|||')
model_structure_full.append(' '*int(len(pooling_description)/2) +'vvv')
model_structure_full.append(pooling_description)  
model_structure_full.append(' '*int(len(pooling_description)/2) +'|||')
model_structure_full.append(' '*int(len(pooling_description)/2) +'vvv')


fully_connected = []
for i in range(len(model.fc)):
    if i%3==1:
        fully_connected.append('FullyConnected(')
        fully_connected.append('  (fc): %s'%model.fc[i])
        fully_connected.append('  (dropout): %s'%model.fc[i-1])
        if i != len(model.fc)-1:
            fully_connected.append('  (activation): %s'%model.fc[i+1])
        else:
            fully_connected.append('  (activation): softmax')
        fully_connected.append(')')
model_structure_full.extend(fully_connected)


f1 = open(output_folder+'/model.txt','w')
f1.write('\n'.join(model_structure_full))
f1.close()