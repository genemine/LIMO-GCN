import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
import random
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.preprocessing import MinMaxScaler

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def cal_auc_prc(output,labels):
    output_test=output
    output_test=output_test.cpu()
    output_test=output_test.detach().numpy()
    output_test=output_test[:,1]
    labels_test=labels.cpu().numpy()
    AUROC = roc_auc_score(labels_test, output_test)
    AUPRC = average_precision_score(labels_test, output_test)
    return round(AUROC,4),round(AUPRC,4)

def get_loc(genex,genes):
    locs=[]
    for genei in genex:
        if genei in genes:
            locs.append(np.where(genes==genei)[0][0])
    return np.array(locs)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def load_adj(data):
    adj_mat_dense=data.values
    adj=sp.coo_matrix(adj_mat_dense)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = torch.FloatTensor(np.array(adj.todense()))
    return adj

def load_data(path="data/",path_feature="data/feature.txt"):
    """Load functional network as an adjacency matrix"""
    dat=pd.read_csv(path+"FGN.txt",sep='\t',header=0,index_col=0)
    genes=dat.index
    dat=dat.reindex(genes)
    dat=dat.fillna(0)
    adj=load_adj(dat)
    """Load feature matrix """
    features=pd.read_csv(path_feature,sep='\t',header=0,index_col=0)
    features=features.reindex(genes)
    features=features.fillna(0)
    for genei in features.columns:
        if genei in features.index:
            features.loc[genei,genei]=1
    #scaler = MinMaxScaler()
    #features=scaler.fit_transform(features)
    features=torch.FloatTensor(np.array(features))
    genes=np.array(genes)
    """Load labels """
    inp_pos=path+'pos.txt'
    f=open(inp_pos)
    pos=[line.strip() for line in f]
    inp_neg=path+'neg.txt'
    f=open(inp_neg)
    neg=[line.strip() for line in f]
    labels=pd.Series([0]*len(dat))
    index=genes
    labels.index=index
    labels.loc[set(pos).intersection(index)]=1
    labels=np.array(labels)
    labels=torch.from_numpy(labels)
    y=len(pos)*[1]+len(neg)*[0]
    y=pd.Series(y)
    y.index=pos+neg
    y=y.loc[y.index.intersection(dat.index)]
    genex=list(y.index)
    genex=random.sample(genex,len(genex))
    y=y[genex]
    return adj, features, genes,labels,y


