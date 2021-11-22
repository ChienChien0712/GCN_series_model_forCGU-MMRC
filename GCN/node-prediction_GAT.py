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
#GAT
n_hidden = int(sys.argv[8])
dropout_gat = float(sys.argv[9])
alpha_leakyReLU = float(sys.argv[10])
n_att = int(sys.argv[11])
seed = sys.argv[12]
if seed in ['Random','random']:
    seed = random.randrange(0, 1000000, 1)
else:
    seed = int(seed)
log_interval = int(sys.argv[13])
output_url = sys.argv[14]


######################################################################################################################################################
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
def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)
def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
#################################################################################################################################################################################################
def load_data(path="",n_folds=None,training_size=None):
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
    #features = normalize_features(features)

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
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    
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
    adj = torch.FloatTensor(np.array(adj.todense()))
    #adj = sparse_mx_to_torch_sparse_tensor(adj)
    
    #n_class
    n_class = len(onehot_to_class)

    return adj, features, labels, idx_train_ls, idx_val_ls, ID, labels_name, n_class, onehot_to_class
################################################################################################################################################################################################
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Linear(in_features=in_features, out_features=out_features,bias=False)
        self.a = nn.Linear(in_features=2*out_features, out_features=1,bias=False)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.concatenate = torch.cat
        self.elu = nn.ELU(inplace=True)
    def forward(self, inp, adj):
        h = self.W(inp)
        N = h.size()[0] #node size
        a_inp = self.concatenate([h.repeat(1,N).view(N * N, -1), h.repeat(N,1)], dim=1).view(N, -1, 2*self.out_features)
        e = self.leakyrelu(self.a(a_inp).squeeze(2)) #shape=[node size, node size]
        
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj>0, e, zero_vec) #沒鄰接:負很大，有鄰接:e
        attention = F.softmax(attention, dim=1) #負很大->softmax->0
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        if self.concat:
            return self.elu(h_prime)
        else:
            return h_prime        
    def extra_repr(self):
        lines = []
        lines.append('(hidden_features): Linear(in_features=%s, out_features=%s, bias=False)'%(self.in_features,self.out_features))
        lines.append('(attetion): Attetion(')
        lines.append('  (concat_ij): Concat(in_features=%s, out_features=%s*2)'%(self.out_features,self.out_features))
        lines.append('  (a): Linear(in_features=%s, out_features=1, bias=False)'%(self.out_features*2))
        lines.append('  (leakyrelu): LeakyReLU(negative_slope=%s)'%(self.alpha))
        lines.append('  (concat_edges): Concat(in_features=1, out_features=1-hop edges)')
        lines.append('  (softmax): Softmax(in_features=1-hop edges, out_features=1-hop edges)')
        lines.append('  (dropout_attetion): Dropout(p=%s)'%self.dropout)
        lines.append(')')
        lines.append('(weighted_hidden_features): Matmul(attention, hidden_features)')
        if self.concat:
            lines.append('(elu): ELU(alpha=1.0)')        
        lines = '\n'.join(lines)
        return lines
        

    
class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.nhid = nhid
        self.nheads = nheads
        self.nclass = nclass
        
        self.attentions = [GraphAttentionLayer(in_features=nfeat,
                                               out_features=nhid, 
                                               dropout=dropout, 
                                               alpha=alpha, 
                                               concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)      
    def extra_repr(self):
        lines = []
        lines.append('GAT(')
        lines.append('  (dropout): Dropout(p=%s)'%self.dropout)
        lines.append('->')
        for i, att in enumerate(self.attentions):
            lines.append('  (attention_%d): GraphAttentionLayer('%(i))
            for j in att.extra_repr().split('\n'):
                lines.append('    '+j)
            lines.append('  )')
        lines.append('->')             
        lines.append('  (concat_attention_0-%s): Concat(in_features=%s, out_features=%s*%slayers)'%(self.nheads-1,
                                                                                    self.nhid, self.nhid, self.nheads))
        lines.append('->')               
        lines.append('  (dropout): Dropout(p=%s)'%self.dropout)
        lines.append('->')  
        lines.append('  (attention_out): GraphAttentionLayer(')
        for j in self.out_att.extra_repr().split('\n'):
            lines.append('    '+j)
        lines.append('  )')
        lines.append('->')
        lines.append('  (elu): ELU(alpha=1.0)')          
        lines.append('->')
        lines.append('  (softmax): Softmax(in_features=%s, out_features=%s)'%(self.nclass,self.nclass))
        lines.append(')')
        lines='\n'.join(lines)
        
        return lines
################################################################################################################################################################################################
print("Prepare to load data.")
adj, features, labels, idx_train_ls, idx_val_ls, ID, labels_name, n_class, onehot_to_class = load_data(path=path, n_folds=n_folds, training_size=training_size)
print("data loaded");
################################################################################################################################################################################################
train_acc_folds = []
val_acc_folds = []
for fold_id in range(n_folds):
    model = GAT(nfeat=features.shape[1],
                nhid=n_hidden,
                nclass=n_class,
                dropout=dropout_gat,
                alpha=alpha_leakyReLU,
                nheads=n_att)
    optimizer = optim.Adam(model.parameters(),lr=lr, weight_decay=wdecay)    
    #if fold_id == 0:
    print(model.extra_repr())
    print('seed:',seed)
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
        output = model(features,adj)
        loss_train = F.nll_loss(output[idx_train_ls[fold_id]], labels[idx_train_ls[fold_id]]) 
        acc_train = accuracy(output[idx_train_ls[fold_id]], labels[idx_train_ls[fold_id]])
        Loss_Train.append(loss_train.tolist())
        Acc_Train.append(acc_train.tolist())
        loss_train.backward()
        optimizer.step()

        #validation
        model.eval()
        output = model(features,adj)
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
        output = model(features,adj)
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
    for i in range(len(model.attentions)):
        W_weight = pd.DataFrame(np.matrix(model.attentions[i].W.weight.tolist()).T)
        W_weight.to_excel(writer,sheet_name='W%s_weight'%i,header=False,index=False)  
        a_weight = pd.DataFrame(np.matrix(model.attentions[i].a.weight.tolist()).T)
        a_weight.to_excel(writer,sheet_name='a%s_weight'%i,header=False,index=False)    
        
    W_weight = pd.DataFrame(np.matrix(model.out_att.W.weight.tolist()).T)
    W_weight.to_excel(writer,sheet_name='W_out_weight',header=False,index=False)  
    a_weight = pd.DataFrame(np.matrix(model.out_att.a.weight.tolist()).T)
    a_weight.to_excel(writer,sheet_name='a_out_weight',header=False,index=False)   
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
        

################################################################################################################################################################################################
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


f1 = open(output_url+'/model.txt','w')
f1.write(model.extra_repr())
f1.close()