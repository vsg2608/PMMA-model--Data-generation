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
T=[]
M=[]
for x in X:
    T.append(x[0])
    M.append(x[1])
t=[]
m=[]
for x in Xact:
    t.append(x[0])
    m.append(x[1])
To=[]
Mo=[]

#Filteration by euclidean distance    
#Training- Partial Least Square Regression
#%%
euclideanThreshold=0.001
Ypred=[]
count=0
Lengths=[]
for Xpred in Xact:
    Xdist=euclidean_distances(X,[Xpred])
    Xtrain=[]
    Ytrain=[]
    for i in range(len(X)):
        if(Xdist[i]<euclideanThreshold):
            Xtrain.append(X[i])
            Ytrain.append(Y[i])
    Lengths.append(len(Xtrain))
    if(len(Xtrain)==0):
        Ypred.append(-0.2)
        count+=1
        To.append(Xpred[0])
        Mo.append(Xpred[1])
        continue
    pls2 = PLSRegression(n_components=2)
    pls2.fit(Xtrain, Ytrain)
    Ypredict = pls2.predict([Xpred],copy=True)
    Ypred.append(Ypredict[0][0])

#Plots
#%%
#plt.style.use('fivethirtyeight')
plt.style.use("default")
plt.suptitle("Ypred vs Yact | EuclideanThreshold= "+str(euclideanThreshold)+" | Outliers= "+str(count))
grid=plt.GridSpec(1, 9, wspace=0.4, hspace=0.3)
#plt.suptitle("Ypred vs Yact | EuclideanThreshold= "+str(euclideanThreshold)+" | Outliers= "+str(count),pad=10)
plt.figure(1)
plt.subplot(grid[0:,0:4])
plt.plot([-0.2,1,0,0,0,0,1,-0.2],[-0.2,1,0,1,-0.2,0,0,0],color='k',linewidth=1)
plt.scatter(Yact,Ypred, color="red",marker="o",linewidth=0.1,alpha=0.7)
plt.xlabel("Yactual")
plt.ylabel("Ypredicted")

plt.subplot(grid[0:,5:])
plt.plot([1,1,1,0.93,1.1],[-0.5,2.2,1,1,1],color='k',linewidth=1)
plt.scatter(T,M, linewidth=0.001,alpha=0.1)
plt.scatter(t,m, linewidth=0.001,alpha=0.5,color='r')
plt.scatter(To,Mo, linewidth=0.001,alpha=0.7,color='y')
plt.xlabel("Temprature")
plt.ylabel("Initial moles of monomer")
plt.savefig("results/7"+str(euclideanThreshold)+".png",dp=1000)
plt.show()