import numpy as np
import os
import re
import pandas as pd

'''
Need following files in the assigned path:
1. xxx_network.xlsx
2. xxx_network.xlsx
3. xxxScores.xlsx
'''
def load_data_TrainRate(path="",training_rate=None,seed=None):
    #print('loading dataset...')
    
    files = os.listdir(path)    
    r = re.compile(".*_network.xlsx")
    Feature_mx = list(filter(r.match, files))    
    #process Features 1
    Features1_excel = pd.ExcelFile(path+'/'+Feature_mx[0])
    #print('Features1:%s'%Feature_mx[0])
    Features1= []
    for i in Features1_excel.sheet_names:
        Features1_corr = Features1_excel.parse(i,header=None)
        indices = np.where(np.triu(np.ones(Features1_corr.shape), k=1).astype(bool))
        Features1.append(list(np.array(Features1_corr)[indices])) 
    #print('\tsubject number:%s'%len(Features1))
    #print('\tlength of Features1:%s'%len(Features1[0]))     
    #process Features 2
    Features2_excel = pd.ExcelFile(path+'/'+Feature_mx[1])
    #print('Features2:%s'%Feature_mx[1])
    Features2= []
    for i in Features2_excel.sheet_names:
        Features2_corr = Features2_excel.parse(i,header=None)
        indices = np.where(np.triu(np.ones(Features2_corr.shape), k=1).astype(bool))
        Features2.append(list(np.array(Features2_corr)[indices])) 
    #print('\tsubject number:%s'%len(Features2))
    #print('\tlength of Features1:%s'%len(Features2[0]))  
    #print()
    
    #Clinical Data
    #print('loading clinical data...')
    files = os.listdir(path)    
    r = re.compile(".*Scores.xlsx")
    clinical_data = list(filter(r.match, files)) 
    Subject_data = pd.read_excel(path+'/'+clinical_data[0],header=None) 
    '''
    for i in Subject_data.columns:
        print('Clinical Feature %s'%i)
        for j in np.unique(Subject_data[i]):
            print('%s:%d'%(j,sum(Subject_data[i]==j)))
        print('-------------------')    
    '''    
    #training&testing index
    rnd_state = np.random.RandomState() if seed==None else np.random.RandomState(seed)
    n = len(Subject_data)
    ids_all = Subject_data.index
    ids = ids_all[rnd_state.permutation(n)]
    stride = int(n*training_rate)
    train_ids = ids[0:stride]
    test_ids = ids[stride:]
    return Features1, Features2, Subject_data, train_ids, test_ids