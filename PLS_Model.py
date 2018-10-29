# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 01:04:07 2018

@author: Vishesh
"""
from sklearn.cross_decomposition import PLSRegression

#Data import
#%%
f=open("Data@10.txt","r")
Lines=f.readlines()
X=[]
Y=[]
for line in Lines:
    line=line.split("\n")[0]
    Data=line.split("\t")
    X.append(Data[:2])
    Y.append(Data[2])
    
#Partial Least Square Regression
#%%
pls2 = PLSRegression(n_components=2)
pls2.fit(X, Y)
Ypred = pls2.predict(X,copy=True)