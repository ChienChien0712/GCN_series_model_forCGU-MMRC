import pandas as pd
import numpy as np
from scipy.spatial import distance
from sklearn.svm import SVC
from sklearn.feature_selection import RFE

from models.ChebNet import *
from utils.utility import *
from utils.dataloader_SamplingbyTrainRate import *
import os
import re

import torch
import scipy.sparse as sp
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys

#hyperparameter
path = sys.argv[1] 
repeat = int(sys.argv[2])
training_rate = float(sys.argv[3])
Top_nFeatures = int(sys.argv[4])

epochs = int(sys.argv[5])
lr = float(sys.argv[6])
wdecay = float(sys.argv[7])
K_ChebNet = int(sys.argv[8])
bias = eval(sys.argv[9])
droprate = float(sys.argv[10])
mid_layer_dim = [int(i.strip()) for i in sys.argv[11].split(',')]
w_score1 = float(sys.argv[12])
#控制S A C
GCN_model = sys.argv[13]

out_url = sys.argv[14]

#similarity_aware_receptive_field
    #True:Similarity-aware Receptive Field, False: Similarity only
if GCN_model == 'GCN':
    similarity_aware_receptive_field = False #True:Similarity-aware Receptive Field, False: Similarity only
    adaptive_mechanism = False
    calibration_mechanism = False
elif GCN_model == 'S-GCN':
    similarity_aware_receptive_field = True
    adaptive_mechanism = False
    calibration_mechanism = False
elif GCN_model == 'SA-GCN':
    similarity_aware_receptive_field = True 
    adaptive_mechanism = True
    calibration_mechanism = False
elif GCN_model == 'SAC-GCN':
    similarity_aware_receptive_field = True
    adaptive_mechanism = True
    calibration_mechanism = True

###############################################################################################################  
def train1(epoch):
    Loss_Train, Acc_Train, Loss_Val, Acc_Val = [], [], [], []
    #training
    model1.train()
    optimizer1.zero_grad()
    data = (X1,L1)
    output = model1(data)
    loss_train = F.nll_loss(output[:n_train], Y[:n_train]) 
    acc_train = accuracy(output[:n_train], Y[:n_train])
    loss_train.backward()
    optimizer1.step()
    #validation (先用test)
    model1.eval()
    data = (X1,L1)
    output = model1(data)
    loss_val = F.nll_loss(output[n_train:], Y[n_train:])  
    acc_val = accuracy(output[n_train:], Y[n_train:]) 
    return loss_train.tolist(), acc_train.tolist(), loss_val.tolist(), acc_val.tolist()
def predict1():
    model1.eval()
    data = (X1,L1)
    output = model1(data)
    return output
def train2(epoch):
    #training
    model2.train()
    optimizer2.zero_grad()
    data = (X2,L2)
    output = model2(data)
    loss_train = F.nll_loss(output[:n_train], Y[:n_train]) 
    acc_train = accuracy(output[:n_train], Y[:n_train])
    loss_train.backward()
    optimizer2.step()
    #validation (先用test)
    model2.eval()
    data = (X2,L2)
    output = model2(data)
    loss_val = F.nll_loss(output[n_train:], Y[n_train:])  
    acc_val = accuracy(output[n_train:], Y[n_train:]) 
    return loss_train.tolist(), acc_train.tolist(), loss_val.tolist(), acc_val.tolist()  
def predict2():
    model2.eval()
    data = (X2,L2)
    output = model2(data)
    return output    
def train_12(epoch):
    Loss_Train, Acc_Train, Loss_Val, Acc_Val = [], [], [], []
    #training
    model1.train()
    model2.train()
    optimizer1.zero_grad()
    optimizer2.zero_grad()
    data1 = (X1,L1)
    data2 = (X2,L2)
    output1 = model1(data1)
    output2 = model2(data2)
    output = torch.log(w_score1*torch.exp(output1)+(1-w_score1)*torch.exp(output2))
    loss_train = F.nll_loss(output[:n_train], Y[:n_train]) 
    acc_train = accuracy(output[:n_train], Y[:n_train])
    loss_train.backward()
    optimizer1.step()
    optimizer2.step()
    #validation (先用test)
    model1.eval()
    model2.eval()
    data1 = (X1,L1)
    data2 = (X2,L2)
    output1 = model1(data1)
    output2 = model2(data2)
    output = torch.log(w_score1*torch.exp(output1)+(1-w_score1)*torch.exp(output2))   
    loss_val = F.nll_loss(output[n_train:], Y[n_train:])  
    acc_val = accuracy(output[n_train:], Y[n_train:])       
    return loss_train.tolist(), acc_train.tolist(), loss_val.tolist(), acc_val.tolist()
def predict_12():
    model1.eval()
    model2.eval()    
    data1 = (X1,L1)
    data2 = (X2,L2)
    output1 = model1(data1)
    output2 = model2(data2)
    output = torch.log(w_score1*torch.exp(output1)+(1-w_score1)*torch.exp(output2))  
    return output        
def train_sa(epoch):
    Loss_Train, Acc_Train, Loss_Val, Acc_Val = [], [], [], []
    #training
    model1.train()
    model2.train()
    optimizer1.zero_grad()
    optimizer2.zero_grad()
    data1 = (X1,L1_sa)
    data2 = (X2,L2_sa)
    output1 = model1(data1)
    output2 = model2(data2)
    output = torch.log(w_score1*torch.exp(output1)+(1-w_score1)*torch.exp(output2))
    loss_train = F.nll_loss(output[:n_train], Y[:n_train]) 
    acc_train = accuracy(output[:n_train], Y[:n_train])
    loss_train.backward()
    optimizer1.step()
    optimizer2.step()
    #validation (先用test)
    model1.eval()
    model2.eval()
    data1 = (X1,L1_sa)
    data2 = (X2,L2_sa)
    output1 = model1(data1)
    output2 = model2(data2)
    output = torch.log(w_score1*torch.exp(output1)+(1-w_score1)*torch.exp(output2))   
    loss_val = F.nll_loss(output[n_train:], Y[n_train:])  
    acc_val = accuracy(output[n_train:], Y[n_train:])       
    return loss_train.tolist(), acc_train.tolist(), loss_val.tolist(), acc_val.tolist()
def predict_sa():
    model1.eval()
    model2.eval()    
    data1 = (X1,L1_sa)
    data2 = (X2,L2_sa)
    output1 = model1(data1)
    output2 = model2(data2)
    output = torch.log(w_score1*torch.exp(output1)+(1-w_score1)*torch.exp(output2))  
    return output        
def train_sac(epoch):
    Loss_Train, Acc_Train, Loss_Val, Acc_Val = [], [], [], []
    #training
    model1.train()
    model2.train()
    optimizer1.zero_grad()
    optimizer2.zero_grad()
    data1 = (X1,L_sac)
    data2 = (X2,L_sac)
    output1 = model1(data1)
    output2 = model2(data2)
    output = torch.log(w_score1*torch.exp(output1)+(1-w_score1)*torch.exp(output2))
    loss_train = F.nll_loss(output[:n_train], Y[:n_train]) 
    acc_train = accuracy(output[:n_train], Y[:n_train])
    loss_train.backward()
    optimizer1.step()
    optimizer2.step()
    #validation (先用test)
    model1.eval()
    model2.eval()
    data1 = (X1,L_sac)
    data2 = (X2,L_sac)
    output1 = model1(data1)
    output2 = model2(data2)
    output = torch.log(w_score1*torch.exp(output1)+(1-w_score1)*torch.exp(output2))   
    loss_val = F.nll_loss(output[n_train:], Y[n_train:])  
    acc_val = accuracy(output[n_train:], Y[n_train:])       
    return loss_train.tolist(), acc_train.tolist(), loss_val.tolist(), acc_val.tolist()
def predict_sac():
    model1.eval()
    model2.eval()    
    data1 = (X1,L_sac)
    data2 = (X2,L_sac)
    output1 = model1(data1)
    output2 = model2(data2)
    output = torch.log(w_score1*torch.exp(output1)+(1-w_score1)*torch.exp(output2))  
    return output        
############################################################################################################################    
ROW_NAME = []
SEN = []
SPE = []
FN = []
FP = []
ACC = []
PROB_train0 = []
PROB_train1 = []
PROB_test0 = []
PROB_test1 = []
writer = pd.ExcelWriter(out_url+'/'+'(%s)%srepeat_result.xlsx'%(GCN_model,repeat), engine = 'xlsxwriter')
for sampling in range(repeat):
    print('repeat %s'%sampling)
    Features1, Features2, Subject_data, train_ids, test_ids = load_data_TrainRate(path=path,training_rate=training_rate)
    ###>>>Process Adjacency Matrix (A1,A2)<<<###
    y = [1 if i=='AD' else 0 for i in Subject_data[0][train_ids]]

    #SVM-RFE (by training set) Features1
    Features1_train = np.array(pd.DataFrame(Features1).loc[train_ids])
    svc = SVC(kernel="linear", C=1)
    rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
    rfe.fit(Features1_train, y)
    idx = []
    for i in range(1,Top_nFeatures+1): #Find Top n Features
        idx.append(list(rfe.ranking_).index(i))
    Features1_low_dim = pd.DataFrame(Features1)[idx]
    #SVM-RFE (by training set) Features2
    Features2_train = np.array(pd.DataFrame(Features2).loc[train_ids])
    svc = SVC(kernel="linear", C=1)
    rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
    rfe.fit(Features2_train, y)
    idx = []
    for i in range(1,Top_nFeatures+1): #Find Top n Features
        idx.append(list(rfe.ranking_).index(i))
    Features2_low_dim = pd.DataFrame(Features2)[idx]

    #Graph Edge Connection
    Subject_data_train = Subject_data.loc[train_ids]
    idx_train_NC = Subject_data_train[Subject_data_train[0]=='MCI'].index #'''Negative Control'''
    idx_train_Pt = Subject_data_train[Subject_data_train[0]=='AD'].index #'''Patient'''
    idx_test = test_ids
    #Adjacency Matrix (A) only connection: NC-->Pt-->test
    n = len(Subject_data)
    A = np.zeros([n,n])
    for i in range(len(idx_train_NC)):
        for j in range(len(idx_train_NC)):
            A[i,j] = 1
    for i in range(len(idx_train_NC),len(idx_train_NC)+len(idx_train_Pt)):
         for j in range(len(idx_train_NC),len(idx_train_NC)+len(idx_train_Pt)):
            A[i,j] = 1   
    for i in range(len(idx_train_NC)+len(idx_train_Pt),n):
         for j in range(n):
            A[i,j] = 1  
    #################################################
    '''
    A = sp.coo_matrix(A)
    A = A + A.T.multiply(A.T > A) - A.multiply(A.T > A)
    A = np.array(A.todense())
    '''
    #################################################
    #Edge Weight Initialization
    idx_order = idx_train_NC.append(idx_train_Pt).append(idx_test)
    #Similarity (Features1)
    Features1_low_dim = Features1_low_dim.loc[idx_order]
    corr_dist = distance.pdist(np.array(Features1_low_dim),metric='correlation')
    corr_dist = distance.squareform(corr_dist)
    sigma = np.mean(corr_dist)
    Features1_simirality = np.exp(-corr_dist**2/(2*sigma**2))
    #Similarity (Features2)
    Features2_low_dim = Features2_low_dim.loc[idx_order]
    corr_dist = distance.pdist(np.array(Features2_low_dim),metric='correlation')
    corr_dist = distance.squareform(corr_dist)
    sigma = np.mean(corr_dist)
    Features2_simirality = np.exp(-corr_dist**2/(2*sigma**2))
    #Phenotypic Information
    n = len(Subject_data)
    R = np.zeros([n,n])
    for i in Subject_data.columns[1:]:
        a = list(Subject_data.loc[idx_order][i])
        b = list(Subject_data.loc[idx_order][i])
        for j in range(len(a)):
            for k in range(len(b)):
                if a[j]==b[k]:
                    R[j,k]+=1        
    #################################################
    #Initialized A
    if similarity_aware_receptive_field:
        A1 = A*Features1_simirality*R
        A2 = A*Features2_simirality*R
    else:
        A1 = A*Features1_simirality
        A2 = A*Features2_simirality
    A1 = sp.coo_matrix(A1)
    A2 = sp.coo_matrix(A2)
    #################################################
    L1 = torch.FloatTensor(calc_sym_norm_lap(A1))
    L2 = torch.FloatTensor(calc_sym_norm_lap(A2))
    #################################################
    #Feature matrix X
    X1 = torch.FloatTensor(np.array(pd.DataFrame(Features1_low_dim).loc[idx_order]))
    X2 = torch.FloatTensor(np.array(pd.DataFrame(Features2_low_dim).loc[idx_order]))
    #Y
    Y = torch.LongTensor([1 if i=='AD' else 0 for i in Subject_data[0][idx_order]])
    out_feature = len(set(Y.tolist()))
    #################################################
    #Create Model
    #ChebNet
    model1 = ChebyNet(K=K_ChebNet,
                      in_features=X1.shape[1],
                      out_features=out_feature,
                      filters_gcn=mid_layer_dim,
                      enable_bias=bias,
                      droprate=droprate,
                      act_func=nn.ReLU(inplace=True))
    model2 = ChebyNet(K=K_ChebNet,
                      in_features=X2.shape[1],
                      out_features=out_feature,
                      filters_gcn=mid_layer_dim,
                      enable_bias=bias,
                      droprate=droprate,
                      act_func=nn.ReLU(inplace=True))
    #info
    n_train = len(idx_train_NC.append(idx_train_Pt))

    #Optimizer
    optimizer1 = optim.Adam(model1.parameters(),lr=lr, weight_decay=wdecay)
    optimizer2 = optim.Adam(model2.parameters(),lr=lr, weight_decay=wdecay)
    #################################################
    

    ############# Train model #####################
    #GCN or S-GCN
    if adaptive_mechanism == False:
        #training
        Loss_Train, Acc_Train, Loss_Val, Acc_Val = [], [], [], []
        for epoch in range(epochs):
            loss_t, acc_t, loss_v, acc_v = train_12(epoch)
            Loss_Train.append(loss_t)
            Acc_Train.append(acc_t)
            Loss_Val.append(loss_v)
            Acc_Val.append(acc_v)    
        #get validation results
        pred_val = predict_12().max(1)[1].type_as(Y)[n_train:].tolist()
        Y_val = Y[n_train:].tolist()               
    #SA-GCN or SAC-GCN    
    elif adaptive_mechanism == True:
        #training
        Loss_Train1, Acc_Train1, Loss_Val1, Acc_Val1 = [], [], [], []
        for epoch in range(epochs):
            loss_t, acc_t, loss_v, acc_v = train1(epoch)
            Loss_Train1.append(loss_t)
            Acc_Train1.append(acc_t)
            Loss_Val1.append(loss_v)
            Acc_Val1.append(acc_v)       
        Loss_Train2, Acc_Train2, Loss_Val2, Acc_Val2 = [], [], [], []
        for epoch in range(epochs):
            loss_t, acc_t, loss_v, acc_v = train2(epoch)
            Loss_Train2.append(loss_t)
            Acc_Train2.append(acc_t)
            Loss_Val2.append(loss_v)
            Acc_Val2.append(acc_v) 
        ###########################################################################################    
        # Get Score (Probability)
        score1 = np.exp(predict1().tolist()).T[1]
        score2 = np.exp(predict2().tolist()).T[1]            
        #Adaptive Mechanism (Similarity of Score)
        score1_dist = distance.squareform(distance.pdist(score1.reshape(-1,1), metric='euclidean'))
        score2_dist = distance.squareform(distance.pdist(score2.reshape(-1,1), metric='euclidean'))
        sigma1 = np.mean(score1_dist)
        sigma2 = np.mean(score2_dist)
        score1_simirality = np.exp(-score1_dist**2/(2*sigma1**2))
        score2_simirality = np.exp(-score2_dist**2/(2*sigma2**2))
        A1_sa = A*score1_simirality*R
        A2_sa = A*score2_simirality*R            
            
        #SA-GCN
        if calibration_mechanism == False:
            A1_sa = sp.coo_matrix(A1_sa)
            A2_sa = sp.coo_matrix(A2_sa)
            L1_sa = torch.FloatTensor(calc_sym_norm_lap(A1_sa))
            L2_sa = torch.FloatTensor(calc_sym_norm_lap(A2_sa))
            #Train Model
            #initialize Parameters
            for i in range(len(model1.gconv)):
                model1.gconv[i].initialize_parameters()
                model2.gconv[i].initialize_parameters()
            Loss_Train, Acc_Train, Loss_Val, Acc_Val = [], [], [], []
            for epoch in range(epochs):
                loss_t, acc_t, loss_v, acc_v = train_sa(epoch)
                Loss_Train.append(loss_t)
                Acc_Train.append(acc_t)
                Loss_Val.append(loss_v)
                Acc_Val.append(acc_v)   
            #get validation results    
            pred_val = predict_sa().max(1)[1].type_as(Y)[n_train:].tolist()
            Y_val = Y[n_train:].tolist()  
            
        #SAC-GCN
        elif calibration_mechanism == True:
            #Calibration Mechanism 
            A_sac = A1_sa*A2_sa 
            A_sac = (A_sac.T/A_sac.sum(axis=1)).T #normalization by row sum
            A_sac = sp.coo_matrix(A_sac)     
            L_sac = torch.FloatTensor(calc_sym_norm_lap(A_sac))
    
            #Train Model
            #initialize Parameters
            for i in range(len(model1.gconv)):
                model1.gconv[i].initialize_parameters()
                model2.gconv[i].initialize_parameters()
            Loss_Train, Acc_Train, Loss_Val, Acc_Val = [], [], [], []
            for epoch in range(epochs):
                loss_t, acc_t, loss_v, acc_v = train_sac(epoch)
                Loss_Train.append(loss_t)
                Acc_Train.append(acc_t)
                Loss_Val.append(loss_v)
                Acc_Val.append(acc_v)   
            #get validation results    
            pred_val = predict_sac().max(1)[1].type_as(Y)[n_train:].tolist()
            Y_val = Y[n_train:].tolist()


    a,b,c,d = 0,0,0,0
    for i in range(len(pred_val)):
        if (pred_val[i]==1)&(Y_val[i]==1):
            a+=1
        elif (pred_val[i]==1)&(Y_val[i]==0): 
            b+=1
        elif (pred_val[i]==0)&(Y_val[i]==1): 
            c+=1
        elif (pred_val[i]==0)&(Y_val[i]==0): 
            d+=1
    ROW_NAME.append('repeat %s'%(sampling))
    SEN.append(a/(a+c))
    SPE.append(d/(b+d))
    FN.append(c/(a+c))
    FP.append(b/(b+d))
    ACC.append((a+d)/(a+b+c+d))
    ####################################################
    #機率儲存
    '''
    prob = np.round(np.exp(predict_sac().T.tolist()[1]),4)
    PROB_train0.append(prob[:len(idx_train_NC)])
    PROB_train1.append(prob[len(idx_train_NC):len(idx_train_NC)+len(idx_train_Pt)])
    prob_test0 = []
    prob_test1 = []
    Y_test = Y[len(idx_train_NC)+len(idx_train_Pt):].tolist()
    for i in range(len(Y_test)):
        if Y_test[i] == 0:
            prob_test0.append(prob[len(idx_train_NC)+len(idx_train_Pt):][i])
        elif Y_test[i] == 1:
            prob_test1.append(prob[len(idx_train_NC)+len(idx_train_Pt):][i])
    PROB_test0.append(prob_test0)
    PROB_test1.append(prob_test1)
    '''
'''    
PROB_train0 = pd.DataFrame(PROB_train0)
PROB_train1 = pd.DataFrame(PROB_train1)
PROB_test0 = pd.DataFrame(PROB_test0)   
PROB_test1 = pd.DataFrame(PROB_test1)   

PROB_train0.index = ROW_NAME
PROB_train1.index = ROW_NAME 
PROB_test0.index = ROW_NAME    
PROB_test1.index = ROW_NAME    
PROB_train0.to_csv(path+'/'+'PROB_train0.csv',header=False)
PROB_train1.to_csv(path+'/'+'PROB_train1.csv',header=False)
PROB_test0.to_csv(path+'/'+'PROB_test0.csv',header=False)
PROB_test1.to_csv(path+'/'+'PROB_test1.csv',header=False)
'''
ROW_NAME.append('mean')
SEN.append(np.mean(SEN))
SPE.append(np.mean(SPE))
FN.append(np.mean(FN))
FP.append(np.mean(FP))
ACC.append(np.mean(ACC))
ROW_NAME.append('std')
SEN.append(np.std(SEN))
SPE.append(np.std(SPE))
FN.append(np.std(FN))
FP.append(np.std(FP))
ACC.append(np.std(ACC))
df = pd.DataFrame([ROW_NAME,SEN,SPE,FN,FP,ACC]).T
df.columns = ['run','sen','spe','fn','fp','acc']
df.to_excel(writer, sheet_name='validation', index=False)

#hyperparameter
files = os.listdir(path)    
r = re.compile(".*_network.xlsx")
Feature_mx = list(filter(r.match, files))  
hyperparameters = [repeat,training_rate, 1-training_rate, Top_nFeatures, epochs, lr, wdecay, K_ChebNet, bias, droprate, \
                   Top_nFeatures, '/'.join(list(map(str,mid_layer_dim))), out_feature, \
                   Feature_mx[0], Feature_mx[1], w_score1, 1-w_score1, \
                   GCN_model, similarity_aware_receptive_field, adaptive_mechanism, calibration_mechanism] 
hyperparameters_index = ['repeat','training proportion','validation proportion','#feature selected by SVM-RFE', 'epoch', 'learning rate', \
                         'w_decay of L2-regularization','K of ChebNet','bias','dropout rate','input features', \
                         'hidden layer', 'output features', 'feature1','feature2','w_feature1','w_feature2', \
                         'Model','similarity aware receptive field', 'adaptive mechanism', 'calibration_mechanism']
hyperparameters = pd.DataFrame(hyperparameters)
hyperparameters.index = hyperparameters_index
hyperparameters.to_excel(writer, sheet_name='hyperparameters', header=False)

writer.save()
