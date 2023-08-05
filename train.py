
from __future__ import division
from __future__ import print_function
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import scipy.sparse as sp
from utils import load_data, cal_auc_prc,get_loc

from models import LIMOGCN
from sklearn.metrics import roc_auc_score,average_precision_score
import random
import sys,os
from  sklearn.model_selection import KFold



'''
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    torch.cuda.manual_seed(args.seed)
'''
'''
if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
'''

random.seed(123)

def crossvalidation(a,num,lam):
    kf=KFold(n_splits=5)
    genex=[]
    y_pred=torch.tensor([])
    y_new=torch.tensor([])
    y=pd.Series(np.zeros(len(genes)))
    y.index=genes
    index=genes
    for trainx,val in kf.split(yk):
        gene_train=yk.index[trainx]
        gene_val=yk.index[val]
        idx_train=get_loc(gene_train,index)
        idx_val=get_loc(gene_val,index)
        genex=genex+list(genes[idx_val])
        # Training settings
        parser = argparse.ArgumentParser()
        parser.add_argument('--no-cuda', action='store_true', default=False,help='Disables CUDA training.')
        parser.add_argument('--fastmode', action='store_true', default=False,help='Validate during training pass.')
        parser.add_argument('--seed', type=int, default=42, help='Random seed.')
        parser.add_argument('--epochs', type=int, default=50,help='Number of epochs to train.')
        parser.add_argument('--lr', type=float, default=0.0001,help='Initial learning rate.')
        parser.add_argument('--weight_decay', type=float, default=0.0001,help='Weight decay.')
        parser.add_argument('--hidden', type=int, default=128,help='Number of hidden units.')
        parser.add_argument('--dropout', type=float, default=0.8,help='Dropout rate (1 - keep probability).')
        args = parser.parse_args()
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        # Model and optimizer
        model = LIMOGCN(nfeat=features.shape[1],nhid=args.hidden,nclass=2,dropout=args.dropout)
        optimizer = optim.Adam(model.parameters(),lr=args.lr, weight_decay=args.weight_decay)
        # Train model
        def train(idx_trainx,idx_val,epoch,a,lam,adj,features,labels):
            model.train()
            optimizer.zero_grad()
            outputx = model(features, adj, a)
            print(outputx.shape)
            output=outputx.to(torch.float32)
            labelsx=labels.to(torch.float32)
            print(len(output))
            print(len(idx_trainx))
            print(len(labelsx))
            loss_train = (1-lam)*F.l1_loss(output[idx_trainx,1], labelsx[idx_trainx])+lam*F.nll_loss(outputx[idx_trainx], labels[idx_trainx])
            loss_train.backward()
            optimizer.step()
            if not args.fastmode:
                # Evaluate validation set performance separately,
                # deactivates dropout during validation run.
                model.eval()
            outputx = model(features, adj, a)
            auc_train,auprc_train=cal_auc_prc(output[idx_trainx], labels[idx_trainx])
            auc_val,auprc_val=cal_auc_prc(output[idx_val], labels[idx_val])
            loss_val = F.l1_loss(output[idx_val,1], labelsx[idx_val])+F.nll_loss(outputx[idx_trainx], labels[idx_trainx])
            print('Epoch: {:04d}'.format(epoch+1),'loss_val: {:.4f}'.format(loss_val.item()),"auc_val= "+str(auc_val),"auprc_val= "+str(auprc_val))
            return auc_val,auprc_val,output
        for epoch in range(num):
            print(str(epoch))
            auc_val,auprc_val,output=train(idx_train,idx_val,epoch,a,lam,adj,features,labels)
        y_new=torch.cat((y_new,labels[idx_val]))
        y_pred=torch.cat((y_pred,output[idx_val]))
    pred=pd.DataFrame(np.exp(y_pred.cpu().detach().numpy())[:,1])
    pred['tag']=y_new
    pred.index=genex
    pred.columns=['score','tag']
    auc,auprc=cal_auc_prc(y_pred, y_new)
    print('AUROC:'+str(round(auc,4)))
    print('AUPRC:'+str(round(auprc,4)))
    return auc,auprc
# Load data
adj, features, genes,labels,yk = load_data(path="data/",path_feature="data/toyfeature.txt")


lam=0.6
a=0.9
num=100
auc,auprc=crossvalidation(a,num,lam)




