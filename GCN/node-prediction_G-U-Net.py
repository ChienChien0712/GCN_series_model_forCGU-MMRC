import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import os
import random
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import sys

plt.rcParams['figure.figsize'] = (15.0, 9.0)
plt.rcParams['figure.dpi'] = 100

path = sys.argv[1]
n_folds = int(sys.argv[2])
training_size = float(sys.argv[3])
epochs = int(sys.argv[4])
lr = float(sys.argv[5])
wdecay = float(sys.argv[6])
model_name = sys.argv[7]
#downGCN
filters_gcn = [int(i.strip()) for i in sys.argv[8].split(',')]
A_normalized_method = sys.argv[9] #'Normalized-Laplacian' #'Normalization'
gcn_activation = eval('nn.'+sys.argv[10]+'(inplace=True)') #ReLU #ELU #Identity
dropout_gcn = float(sys.argv[11])
gcn_bias = eval(sys.argv[12])
#gPool
pooling_ratios = [float(i.strip()) for i in sys.argv[13].split(',')]
assert len(filters_gcn)==(len(pooling_ratios)+1), "len(filters_gcn) should equal len(pooling_ratios)-1"
activation_gPool= eval('torch.'+sys.argv[14])#tanh #sigmoid
dropout_gPool=float(sys.argv[15])
bias_gPool=eval(sys.argv[16])
#gUnpool+upGCN
filters_up_gcn = sys.argv[17] #same as gPool Layer 

if filters_up_gcn == 'same_as_gPool_layer':
    filters_up_gcn = filters_gcn[:-1]
    filters_up_gcn.reverse()
else:
    filters_up_gcn = [int(i.strip()) for i in filters_up_gcn.split(',')]
#finalGCN
final_gcn = eval(sys.argv[18])

seed = sys.argv[19]
if seed in ['Random','random'] :
    seed = random.randrange(0, 10000, 1)
else:
    seed = int(seed)
log_interval = int(sys.argv[20])
  
output_url = sys.argv[21]

######################################################################################################################
def encode_onehot(labels):
    classes = sorted(list(set(labels)))
    if '9999' in classes:
        classes.remove('9999')
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    onehot_to_class = {np.where(np.identity(len(classes))[i, :])[0][0]:c for i, c in enumerate(classes)}
    classes_dict['9999'] = [0]+[0]*(len(classes)-1)
    labels_onehot = np.array(list(map(classes_dict.get, labels)),dtype=np.int32)
    return labels_onehot, onehot_to_class
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    #csr matrix轉回coo matrix
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    #indices:儲存row和col
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    #values:儲存非0數值
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
def normalize(mx):
    """Row-normalize sparse matrix"""
    if mx.toarray()[0][0] == 0:
        mx = mx + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
def laplacian(mx):
    """compute L=D^-0.5 * (mx) * D^-0.5"""
    if mx.toarray()[0][0] == 0:
        mx = mx + sp.eye(mx.shape[0])    
    degree = np.array(mx.sum(1))
    d_hat = sp.diags(np.power(degree, -0.5).flatten())
    laplacian_mx = d_hat.dot(mx).dot(d_hat)
    return laplacian_mx
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct*100 / len(labels)
######################################################################################################################
def load_data(path="",n_folds=None,training_size=None,A_normalized_method=None):
    print('loading dataset...')
    files = os.listdir(path)
    #idx_features
    idx_features_url = path+'/'+list(filter(lambda f: f.find('_node_features') >= 0, files))[0]
    f1 = open(idx_features_url)
    if ',' in f1.readline():
        idx_features = np.genfromtxt(idx_features_url,delimiter=',',dtype=np.dtype(str))
    else:
        idx_features = np.genfromtxt(idx_features_url,dtype=np.dtype(str))

    #features (normalized)
    features = sp.csr_matrix(idx_features[:, 1:], dtype=np.float32)
    features = features

    #labels (one-hot)
    labels_url = path+'/'+list(filter(lambda f: f.find('_node_label') >= 0, files))[0]
    labels, onehot_to_class = encode_onehot(np.genfromtxt(labels_url,dtype=np.dtype(str)))
    labels_name = np.genfromtxt(labels_url,dtype=np.dtype(str))
    #ID
    ID = idx_features[:, 0]
    ID_to_idx = {j: i for i, j in enumerate(ID)}
    idx_to_ID = {i: j for i, j in enumerate(ID)}

    #edges
    files = os.listdir(path)
    edges_url = path+'/'+list(filter(lambda f: f.find('_A') >= 0, files))[0]
    f1 = open(edges_url)
    if ',' in f1.readline():
        edges_unordered = np.genfromtxt(edges_url,delimiter=',',dtype=np.dtype(str))
    else:
        edges_unordered = np.genfromtxt(edges_url,dtype=np.dtype(str))
    f1.close()
    edges = np.array(list(map(ID_to_idx.get, edges_unordered.flatten())),dtype=np.int32).reshape(edges_unordered.shape)

    #A (symmetric)
    edges = edges[edges[:,0]!=edges[:,1]]
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    
    
    #setting training set & val set
    labels_df = pd.DataFrame(labels)
    gp = set(labels_df[labels_df.sum(1)!=0].index)
    idx_unknown = list(labels_df[labels_df.sum(1)==0].index)

    assert n_folds >= 1, "'n_folds' should >= 1"
    if n_folds == 1:
        #size
        known_size = len(gp)
        train_size = round(known_size*training_size)
        #sampling
        idx_train = set(random.sample(gp,k=train_size))
        idx_val = gp - idx_train
        #tolist
        idx_train = torch.LongTensor(sorted(list(idx_train)))
        idx_val = torch.LongTensor(sorted(list(idx_val)))
        #to Multiple list
        idx_train_ls = [idx_train]
        idx_val_ls = [idx_val]
    else:
        #size
        gp_full = set(labels_df[labels_df.sum(1)!=0].index)
        known_size = len(gp)
        val_size = round(known_size*(1/n_folds))

        idx_val_ls = []
        idx_train_ls = []
        for i in range(n_folds-1):
            #sampling
            idx_val = set(random.sample(gp,k=val_size))
            idx_train = gp_full - idx_val
            idx_val_ls.append(torch.LongTensor(sorted(list(idx_val))))
            idx_train_ls.append(torch.LongTensor(sorted(list(idx_train))))
            gp = gp - idx_val
        idx_val_ls.append(torch.LongTensor(sorted(list(gp))))
        idx_train_ls.append(torch.LongTensor(sorted(list(gp_full-gp))))

    #轉Tensor
    features = torch.FloatTensor(np.array(features.todense()))
    labels_location = []
    for i in labels:
        if sum(i) == 0:
            labels_location.append(9999)
        else:
            labels_location.append(np.where(i)[0][0])
    labels = torch.LongTensor(np.array(labels_location))
    #adj = sparse_mx_to_torch_sparse_tensor(adj)
    
    #n_class
    n_class = len(onehot_to_class)

    return adj, features, labels, idx_train_ls, idx_val_ls, ID, labels_name, n_class, onehot_to_class
######################################################################################################################
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
        if A_normalized_method == 'Normalization':
            self.kernel = normalize
        elif A_normalized_method == 'Normalized-Laplacian':  
            self.kernel = laplacian
            
    def forward(self, data):#要coo_matrix才能用self.kernel
        x, A = data[:2]
        x = self.drop(x)
        x = self.fc(x)
        x = torch.spmm(sparse_mx_to_torch_sparse_tensor(self.kernel(A)),x)
        x = self.activation(x) 
        return (x, A)
    
class gPool(nn.Module):
    def __init__(self,
                in_features,
                pooling_ratio=None,
                dropout_gPool=0,
                bias_gPool=None,
                activation_gPool=None):
        super(gPool,self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=1,bias=bias_gPool)
        self.activation = activation_gPool
        self.drop = nn.Dropout(p=dropout_gPool) if dropout_gPool > 0.0 else nn.Identity()
        self.pooling_ratio = pooling_ratio
        
    def forward(self, data):
        x, A = data[:2]
        score = self.drop(x)
        score = self.fc(score)
        score = score / torch.sum(score**2).view(1, 1)**0.5
        score = score.squeeze()
        score = self.activation(score)
        values, idx = torch.topk(score, max(2,int(self.pooling_ratio*x.shape[0])))
        new_x = x[idx, :]
        values = torch.unsqueeze(values, -1)
        new_x = torch.mul(new_x, values)
        new_A = A.todense()[idx,:]
        new_A = new_A[:,idx]
        new_A = sp.coo_matrix(new_A)
        return (new_x, new_A), idx
class gUnpool(nn.Module):
    def __init__(self, *args):
        super(gUnpool,self).__init__()
        
    def forward(self, data, up_A, idx):
        x, A = data
        new_x = x.new_zeros(up_A.shape[0], x.shape[1])
        new_x[idx] = x
        return (new_x, up_A)
        
class GraphUnet(nn.Module):
    def __init__(self,
                 in_features,
                 out_features, #n_class
                 filters_gcn=None,
                 dropout_gcn=0,
                 gcn_bias=True,
                 gcn_activation=None,
                 pooling_ratios=None,
                 dropout_gPool=0,
                 bias_gPool=None,
                 activation_gPool=None,
                 filters_up_gcn=None):
        super(GraphUnet, self).__init__()
        self.hidden = filters_gcn

        # down GCN
        self.down_gconv = nn.ModuleList([GraphConv(in_features=in_features if layer == 0 else self.hidden[layer - 1], 
                                                   out_features=f, 
                                                   activation=gcn_activation,
                                                   dropout_gcn=dropout_gcn,
                                                   gcn_bias=gcn_bias) for layer, f in enumerate(self.hidden)])    
        #gPool
        self.gPool = nn.ModuleList([gPool(in_features=self.hidden[i],
                                          pooling_ratio=pooling_ratios[i],
                                          dropout_gPool=dropout_gPool,
                                          bias_gPool=bias_gPool,
                                          activation_gPool=activation_gPool) for i in range(len(self.hidden)-1)])
        #gUnpool
        self.gUnpool = nn.ModuleList([gUnpool() for layer, f in enumerate(filters_up_gcn)])
        
        # up GCN
        if final_gcn:
            self.up_gconv = nn.ModuleList([GraphConv(in_features=filters_gcn[-1] if layer == 0 else filters_up_gcn[layer - 1], 
                                                     out_features=f, 
                                                     activation=gcn_activation,
                                                     dropout_gcn=dropout_gcn,
                                                     gcn_bias=gcn_bias) for layer, f in enumerate(filters_up_gcn)])  
            # final GCN
            self.final_gconv = nn.ModuleList([GraphConv(in_features=filters_up_gcn[-1],
                                                        out_features=out_features,
                                                        activation=nn.Identity(),
                                                        dropout_gcn=dropout_gcn,
                                                        gcn_bias=gcn_bias)])
        else:
            self.up_gconv = nn.ModuleList([GraphConv(in_features=filters_gcn[-1] if layer == 0 else filters_up_gcn[layer - 1], 
                                                     out_features=f if layer < len(filters_up_gcn)-1 else out_features, 
                                                     activation=gcn_activation if layer < len(filters_up_gcn)-1 else nn.Identity(),
                                                     dropout_gcn=dropout_gcn,
                                                     gcn_bias=gcn_bias) for layer, f in enumerate(filters_up_gcn)])         
    def forward(self, data):
        #GCN-->gPool-->GCN-->gPool-->.......-->GCN
        up_idx = []
        up_As = [data[1]]
        for i in range(len(self.down_gconv)):
            data = self.down_gconv[i](data)
            if i != len(self.gPool):
                data, idx = self.gPool[i](data)
                up_idx.append(idx)
                up_As.append(data[1])
        up_idx.reverse()
        up_As = up_As[:-1]
        up_As.reverse()
        #gUnpool-->GCN-->......-->gUnpool-->GCN
        for i in range(len(self.up_gconv)):
            data = self.gUnpool[i](data,up_As[i],up_idx[i])
            data = self.up_gconv[i](data)
        #final GCN
        if final_gcn:
            for layer, gconv in enumerate(self.final_gconv):
                x = gconv(data)[0]
        else:
            x = data[0]
        x = F.log_softmax(x, dim=1)
        return x  
######################################################################################################################
adj, features, labels, idx_train_ls, idx_val_ls, ID, labels_name, n_class, onehot_to_class = load_data(path=path, n_folds=n_folds, training_size=training_size,A_normalized_method=A_normalized_method)
######################################################################################################################
train_acc_folds = []
val_acc_folds = []
for fold_id in range(n_folds):
    model = GraphUnet(in_features=features.shape[1],
                out_features=n_class,
                filters_gcn=filters_gcn,
                dropout_gcn=dropout_gcn,
                gcn_bias=gcn_bias,
                gcn_activation=gcn_activation,
                pooling_ratios=pooling_ratios,
                dropout_gPool=dropout_gPool,
                bias_gPool=bias_gPool,
                activation_gPool=activation_gPool,
                filters_up_gcn=filters_up_gcn)  
    optimizer = optim.Adam(model.parameters(),lr=lr, weight_decay=wdecay)    
    #if fold_id == 0:
    print(model)
    print('\nFOLD', fold_id+1)
    print('training:%s/%s'%(len(idx_train_ls[fold_id]),len(idx_train_ls[fold_id])+len(idx_val_ls[fold_id])))
    print('validation:%s/%s'%(len(idx_val_ls[fold_id]),len(idx_train_ls[fold_id])+len(idx_val_ls[fold_id])))
    Loss_Train = []
    Acc_Train = []
    Loss_Val = []
    Acc_Val = []
    def train(epoch):
        t = time.time()
        #training
        model.train()
        optimizer.zero_grad()
        data = (features,adj)
        output = model(data)
        loss_train = F.nll_loss(output[idx_train_ls[fold_id]], labels[idx_train_ls[fold_id]]) 
        acc_train = accuracy(output[idx_train_ls[fold_id]], labels[idx_train_ls[fold_id]])
        Loss_Train.append(loss_train.tolist())
        Acc_Train.append(acc_train.tolist())
        loss_train.backward()
        optimizer.step()

        #validation
        model.eval()
        data = (features,adj)
        output = model(data)
        loss_val = F.nll_loss(output[idx_val_ls[fold_id]], labels[idx_val_ls[fold_id]]) 
        acc_val = accuracy(output[idx_val_ls[fold_id]], labels[idx_val_ls[fold_id]])
        Loss_Val.append(loss_val.tolist())
        Acc_Val.append(acc_val.tolist())
        if ((epoch+1) % log_interval == 0) or (epoch+1==epochs)or (epoch==0):
            print('Epoch: {:04d}'.format(epoch+1),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'acc_train: {:.2f}%'.format(acc_train.item()),
                  'loss_val: {:.4f}'.format(loss_val.item()),
                  'acc_val: {:.2f}%'.format(acc_val.item()),
                  'time: {:.4f}s'.format(time.time() - t))

    prediction = []
    prob = []
    def test():
        model.eval()
        data = (features,adj)
        output = model(data)
        prob.append(output)
        loss_val = F.nll_loss(output[idx_val_ls[fold_id]], labels[idx_val_ls[fold_id]])
        acc_train = accuracy(output[idx_train_ls[fold_id]], labels[idx_train_ls[fold_id]])
        acc_val = accuracy(output[idx_val_ls[fold_id]], labels[idx_val_ls[fold_id]])
        train_acc_folds.append(acc_train)
        val_acc_folds.append(acc_val)
        preds = output.max(1)[1].type_as(labels)
        preds = preds.tolist()
        preds = [onehot_to_class[i] for i in preds]        
        prediction.extend(preds)
        print("Validation set results:",
              "loss= {:.4f}".format(loss_val.item()),
              "accuracy= {:.2f}%".format(acc_val.item()))


    # Train model
    t_total = time.time()
    for epoch in range(epochs):
        train(epoch)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    test()
    
    #########################################################################
    writer = pd.ExcelWriter(output_url+'/result_fold%s.xlsx'%(fold_id+1), engine = 'xlsxwriter')
    training_df = pd.DataFrame(zip(list(range(1,epochs+1)),Loss_Train,Acc_Train,Loss_Val,Acc_Val))
    training_df.columns = ['epoch','training loss','training accuracy','validation loss','validation accuracy']
    training_df.to_excel(writer, sheet_name = 'training&validation',index=False)
    #########################################################################
    for i in range(len(model.down_gconv)):
        f_down_GCN_weight = pd.DataFrame(np.matrix(model.down_gconv[i].fc.weight.tolist()).T)
        f_down_GCN_weight.to_excel(writer,sheet_name='downGCN%s_weight'%i,header=False,index=False)
        if gcn_bias:
            f_down_GCN_bias = pd.DataFrame(np.matrix(model.down_gconv[i].fc.bias.tolist()).T)
            f_down_GCN_bias.to_excel(writer,sheet_name='downGCN%s_bias'%i,header=False,index=False) 
        if i != len(model.gPool):
            f_gPool_weight = pd.DataFrame(np.matrix(model.gPool[i].fc.weight.tolist()).T)
            f_gPool_weight.to_excel(writer,sheet_name='gPool%s_weight'%i,header=False,index=False)
            if bias_gPool:
                f_gPool_bias = pd.DataFrame(np.matrix(model.gPool[i].fc.bias.tolist()).T)
                f_gPool_bias.to_excel(writer,sheet_name='gPool%s_bias'%i,header=False,index=False) 
    for i in range(len(model.up_gconv)):
        f_up_gconv_weight = pd.DataFrame(np.matrix(model.up_gconv[i].fc.weight.tolist()).T) 
        f_up_gconv_weight.to_excel(writer,sheet_name='upGCN%s_weight'%i,header=False,index=False)
        if gcn_bias:
            f_up_GCN_bias = pd.DataFrame(np.matrix(model.up_gconv[i].fc.bias.tolist()).T)
            f_up_GCN_bias.to_excel(writer,sheet_name='upGCN%s_bias'%i,header=False,index=False) 
    if final_gcn:
        for i in range(len(model.final_gconv)):
            f_final_gconv_weight = pd.DataFrame(np.matrix(model.final_gconv[i].fc.weight.tolist()).T) 
            f_final_gconv_weight.to_excel(writer,sheet_name='GCN%s_weight'%i,header=False,index=False)       
            if gcn_bias:
                f_final_GCN_bias = pd.DataFrame(np.matrix(model.final_gconv[i].fc.bias.tolist()).T)
                f_final_GCN_bias.to_excel(writer,sheet_name='GCN%s_bias'%i,header=False,index=False) 
    ################################################################################
    split = np.empty((len(labels_name)),dtype=np.object)
    for i in idx_train_ls[fold_id].tolist():
        split[i] = 'training'
    for i in idx_val_ls[fold_id].tolist():
        split[i] = 'validation'
    corrects = []
    for i,j in zip(prediction,labels_name):
        if j == '9999':
            corrects.append('')
        elif i == j:
            corrects.append(1)
        elif i != j:
            corrects.append(0)
    preds_df = pd.DataFrame(zip(ID,prediction,labels_name,split,corrects))
    preds_df.columns = ['ID','prediction','label','splits','correct']
    
    class_prob = pd.DataFrame(np.exp(prob[0].tolist())*100)
    columns = []
    for c in class_prob.columns:
        class_prob[c] = class_prob[c].map('{:,.2f}%'.format)
        columns.append('p('+onehot_to_class[c]+')')
    class_prob.columns = columns
    preds_df = pd.concat([preds_df,class_prob], axis=1)
    
    preds_df.to_excel(writer, sheet_name = 'prediction',index=False)
    ################################################################################
    #sampling info.
    class_number_train = []
    class_proportion_train = []
    class_number_val = []
    class_proportion_val = []    
    
    label_set = sorted(list(set(labels_name)))
    if '9999' in label_set:
        label_set.remove('9999')
    for i in label_set:
        class_number_train.append(sum(labels_name[idx_train_ls[fold_id]]== i))
        class_proportion_train.append('%.1f%%'%(sum(labels_name[idx_train_ls[fold_id]]== i)*100/len(idx_train_ls[fold_id])))
        class_number_val.append(sum(labels_name[idx_val_ls[fold_id]]== i))
        class_proportion_val.append('%.1f%%'%(sum(labels_name[idx_val_ls[fold_id]]== i)*100/len(idx_val_ls[fold_id])))  
    sampling_info = pd.DataFrame(zip(label_set,class_number_train,class_proportion_train,class_number_val,class_proportion_val))
    sampling_info.columns = ['label','training set','proportion of training set','validation set','proportion of validation set']
    sampling_info.to_excel(writer, sheet_name = 'sampling_info',index=False)    
    #################################################################################################
    #class_acc
    Number_Train = []
    Correct_Train = []
    Number_Val = []
    Correct_Val = []
    Values = onehot_to_class.values()
    for i in Values:
        Correct_Train.append(len(preds_df[(preds_df['correct']==1)&(preds_df['label']==i)&(preds_df['splits']=='training')]))
        Number_Train.append(len(preds_df[(preds_df['label']==i)&(preds_df['splits']=='training')]))
        Correct_Val.append(len(preds_df[(preds_df['correct']==1)&(preds_df['label']==i)&(preds_df['splits']=='validation')]))
        Number_Val.append(len(preds_df[(preds_df['label']==i)&(preds_df['splits']=='validation')]))
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
    class_to_onehot = dict(zip([onehot_to_class[i] for i in range(len(onehot_to_class))],range(len(onehot_to_class))))
    label_pred_mx_train = np.zeros([len(onehot_to_class),len(onehot_to_class)])
    label_pred_mx_val = np.zeros([len(onehot_to_class),len(onehot_to_class)])
    for i in range(len(preds_df)):
        if preds_df.at[i,'label'] == '9999':
            continue
        if preds_df.at[i,'splits'] == 'training':
            label_pred_mx_train[class_to_onehot[preds_df.at[i,'label']]][class_to_onehot[preds_df.at[i,'prediction']]] += 1
        elif preds_df.at[i,'splits'] == 'validation':
            label_pred_mx_val[class_to_onehot[preds_df.at[i,'label']]][class_to_onehot[preds_df.at[i,'prediction']]] += 1
    label_pred_mx_train = pd.DataFrame(label_pred_mx_train,dtype=int)
    label_pred_mx_val = pd.DataFrame(label_pred_mx_val,dtype=int)        
    label_pred_mx_train.columns = [onehot_to_class[i] for i in range(len(onehot_to_class))]
    label_pred_mx_val.columns = [onehot_to_class[i] for i in range(len(onehot_to_class))]       
    label_pred_mx_train['label\pred'] = [onehot_to_class[i] for i in range(len(onehot_to_class))]
    label_pred_mx_val['label\pred'] = [onehot_to_class[i] for i in range(len(onehot_to_class))]        
    label_pred_mx_train = pd.concat([label_pred_mx_train['label\pred'],label_pred_mx_train[[onehot_to_class[i] for i in range(len(onehot_to_class))]]],axis=1)
    label_pred_mx_val = pd.concat([label_pred_mx_val['label\pred'],label_pred_mx_val[[onehot_to_class[i] for i in range(len(onehot_to_class))]]],axis=1)
    label_pred_mx_train.to_excel(writer, sheet_name = 'train_label_pred',index=False)         
    label_pred_mx_val.to_excel(writer, sheet_name = 'val_label_pred',index=False)          
    writer.save()

    Epoch = range(1,epochs+1)
    plt.plot(Epoch,Loss_Train,label='training loss')
    plt.plot(Epoch,Loss_Val,label='validation loss')
    plt.xlabel('epoch',fontsize=18)
    plt.ylabel('loss',fontsize=18)    
    plt.legend(fontsize=18)
    plt.grid(linestyle='--')
    plt.savefig(output_url+'/loss_fold%s.png'%(fold_id+1))
    plt.clf()
    
    plt.plot(Epoch,np.array(Acc_Train)/100,label='training accuracy')
    plt.plot(Epoch,np.array(Acc_Val)/100,label='validation accuracy')
    plt.xlabel('epoch',fontsize=18)
    plt.ylabel('accuracy',fontsize=18)
    plt.legend(fontsize=18)
    plt.grid(linestyle='--')
    plt.savefig(output_url+'/acc_fold%s.png'%(fold_id+1))
    plt.clf()
######################################################################################################################
summary = []
summary.append('traing set:')
for i in range(len(train_acc_folds)):
    summary.append('accuracy of fold #%2d: %.2f%%'%(i+1,train_acc_folds[i]))
summary.append('%s-folds accuracy: %.2f%% (std=%.2f%%)'%(n_folds,np.mean(train_acc_folds),np.std(train_acc_folds)))
summary.append('validation set:')
for i in range(len(val_acc_folds)):
    summary.append('accuracy of fold #%2d: %.2f%%'%(i+1,val_acc_folds[i]))
summary.append('%s-folds accuracy: %.2f%% (std=%.2f%%)'%(n_folds,np.mean(val_acc_folds),np.std(val_acc_folds)))
f1 = open(output_url+'/summary_acc.txt','w')
f1.write('\n'.join(summary))
f1.close()    
########################################################################
model_structure = []
model_structure.append('gPooling(')
for i in range(len(model.down_gconv)):
    model_structure.append('  '+'\n  '.join(('(%s): '%i+str(model.down_gconv[i])).split('\n')))
    if i != len(model.gPool):
        tmp = ('(%s): '%i+str(model.gPool[i])).split('\n')[:-1]
        tmp.append('  (top k): %s'%pooling_ratios[i])
        tmp.append('  (activation): %s'%activation_gPool.__name__)
        tmp.append(')')
        model_structure.append('  '+'\n  '.join(tmp))
model_structure.append(')')     

model_structure.append('gUnpooling(')
for i in range(len(model.up_gconv)):
    model_structure.append('  '+'\n  '.join(('(%s): '%i+str(model.gUnpool[i])).split('\n')))
    model_structure.append('  '+'\n  '.join(('(%s): '%i+str(model.up_gconv[i])).split('\n')))   
model_structure.append(')')        

if final_gcn:
    model_structure.append('GCN(')
    for i in range(len(model.final_gconv)):
        model_structure.append('  '+'\n  '.join(('(%s): '%i+str(model.final_gconv[i])).split('\n')))
    model_structure.append(')')      
tmp = model_structure[-2].split('\n')
tmp[-3]='    (activation): Softmax()'
model_structure[-2] = '\n'.join(tmp)
f1 = open(output_url+'/model.txt','w')
f1.write('\n'.join(model_structure))
f1.close()