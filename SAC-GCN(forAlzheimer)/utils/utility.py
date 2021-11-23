import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import torch

def laplacian(sparse_mx):
    """compute L=D^-0.5 * (mx) * D^-0.5"""
    degree = np.array(sparse_mx.sum(1))
    d_hat = sp.diags(np.power(degree, -0.5).flatten())
    laplacian_mx = d_hat.dot(sparse_mx).dot(d_hat)
    return laplacian_mx


def encode_onehot(labels):
    classes = sorted(list(set(labels)))
    if '9999' in classes:
        classes.remove('9999')
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    onehot_to_class = {np.where(np.identity(len(classes))[i, :])[0][0]:c for i, c in enumerate(classes)}
    classes_dict['9999'] = [0]+[0]*(len(classes)-1)
    labels_onehot = np.array(list(map(classes_dict.get, labels)),dtype=np.int32)
    return labels_onehot, onehot_to_class


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct*100 / len(labels)

def calc_sym_norm_lap(sp_sys_adj):
    '''
    symmetric normalized Laplacian
    '''
    n_vertex = sp_sys_adj.shape[0]
    I = sp.csc_matrix(sp.identity(n_vertex))
    sym_norm_lap = I - laplacian(sp_sys_adj)
    count = 0
    while True:
        try:
            count +=1    
            ev_max = max(eigsh(sym_norm_lap, which='LM', return_eigenvectors=False,ncv=30))
            break
        except:
            #print('NoConvergenceError: try again %s'%count)
            if count == 100:
                #print('NoConvergenceError!')
                break                
    wid_sym_norm_lap = 2 / ev_max * sym_norm_lap - I
    wid_sym_norm_lap = np.array(wid_sym_norm_lap.toarray(), dtype=np.float32)
    return wid_sym_norm_lap