# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 01:04:07 2018

@author: Vishesh
"""
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

def normalize(X,Xnorm):
    for i in range(len(X)):
        for j in range(len(Xnorm)):
            X[i][j]=X[i][j]/Xnorm[j]
    return X

#Data import
#%%
f=open("Data@10.txt","r")
f2=open("Data@test.txt","r")
Lines=f.readlines()
X=[]
Y=[]
for line in Lines:
    line=line.split("\n")[0]
    Data=line.split("\t")
    X.append([float(Data[0]),float(Data[1])])
    Y.append(float(Data[2]))
Xact=[]
Yact=[]
Lines=f2.readlines()
for line in Lines:
    line=line.split("\n")[0]
    Data=line.split("\t")
    Xact.append([float(Data[0]),float(Data[1])])
    Yact.append(float(Data[2]))
    
#Data normalisation and filteration according to Euclidean distance
#%%
Xnorm=[323.15,1000]
Xact=normalize(Xact,Xnorm)
X=normalize(X,Xnorm)

#Filteration by euclidean distance    
#Training- Partial Least Square Regression
#%%
euclideanThreshold=0.01
Ypred=[]
count=0
for Xpred in Xact:
    Xdist=euclidean_distances(X,[Xpred])
    Xtrain=[]
    Ytrain=[]
    for i in range(len(X)):
        if(Xdist[i]<euclideanThreshold):
            Xtrain.append(X[i])
            Ytrain.append(Y[i])
    if(len(Xtrain)==0):
        Ypred.append(-.25)
        count+=1
        continue
    pls2 = PLSRegression(n_components=2)
    pls2.fit(Xtrain, Ytrain)
    Ypredict = pls2.predict([Xpred],copy=True)
    Ypred.append(Ypredict[0][0])

#Plots
#%%
plt.style.use('seaborn')
plt.scatter(Yact,Ypred)
plt.plot([-.1,1,0,0,0,1],[-.1,1,0,1,0,0],color='k',linewidth=1)
plt.xlabel("Yact")
plt.xlabel("Ypred")
plt.show()
print(count)