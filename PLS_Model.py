# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 01:04:07 2018

@author: Vishesh
"""
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

#Data import
#%%
f=open("Data@10.txt","r")
Lines=f.readlines()
X=[]
Y=[]
for line in Lines:
    line=line.split("\n")[0]
    Data=line.split("\t")
    X.append([int(Data[0]),int(Data[1])])
    Y.append(int(Data[2]))
    
Xpred=X[0]
#Data filteration according to Euclidean distance
#%%
Xtrain=[]
Ytrain=[]
Xdist=euclidean_distances(X,[Xpred])
    
#Partial Least Square Regression
#%%
pls2 = PLSRegression(n_components=2)
pls2.fit(X, Y)
Ypred = pls2.predict(X,copy=True)