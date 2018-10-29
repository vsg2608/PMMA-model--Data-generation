# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 01:04:07 2018

@author: Vishesh
"""
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics.pairwise import euclidean_distances

def normalize(X,Xnorm):
    for i in range(len(X)):
        for j in range(len(Xnorm)):
            X[i][j]=X[i][j]/Xnorm[j]
    return X

#Data import
#%%
f=open("Data@10.txt","r")
Lines=f.readlines()
X=[]
Y=[]
for line in Lines:
    line=line.split("\n")[0]
    Data=line.split("\t")
    X.append([float(Data[0]),float(Data[1])])
    Y.append(float(Data[2]))
    
#Data normalisation and filteration according to Euclidean distance
#%%
Xnorm=[323.15,1000]
X=normalize(X,Xnorm)

#Filteration by euclidean distance    
#Training- Partial Least Square Regression
#%%
euclideanThreshold=0.01
Xpred=X[4]
Xdist=euclidean_distances(X,[Xpred])
Xtrain=[]
Ytrain=[]
for i in range(len(X)):
    if(Xdist[i]<euclideanThreshold):
        Xtrain.append(X[i])
        Ytrain.append(Y[i])
pls2 = PLSRegression(n_components=2)
pls2.fit(Xtrain, Ytrain)
Ypred = pls2.predict([Xpred],copy=True)
print(Ypred)
print(Y[0])
print("length of Xtrain=",len(Xtrain))