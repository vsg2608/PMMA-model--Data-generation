import matlab.engine
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import numpy as np
from random import gauss
import time
import math
import random
import pprint
eng = matlab.engine.start_matlab()
eng.addpath(r'C:\Users\Vishesh\Desktop\Workspace\BTP',nargout=0)
#%%
Xnorm=[323.15,1000]
I_0= 0.00258               #Initial moles of initiator
M_0= 1E6*1.0                   #Initial moles of monomer
T=323.45
R_lm =0.0
Tf=10.0
euclideanThreshold=0.05
manhattanThreshold=0.05
cosineThreshold=0.999999
minDataPoints=20
#Adative Similarty measure
localizationParameter=0.05 #psi
tolerance=0.001
tuningParameter=0.5 #alpha

def normalize(X,Xnorm):
    for i in range(len(X)):
        for j in range(len(Xnorm)):
            X[i][j]=X[i][j]/Xnorm[j]
    return X

#Read data from the txt file
def readData(s):
    f=open(s,"r")
    X=[]
    Y=[]
    Lines= f.readlines()
    for line in Lines:
        line=line.split("\n")[0]
        Data=line.split("\t")
        X.append([float(Data[0]),float(Data[1])])
        Y.append(float(Data[2]))
    f.close()
    return [X,Y]

#Write data to the txt file
def writeData(s,X,Y,Xnorm):
    f=open(s,'a')
    f.write(str(X[0]*Xnorm[0])+"\t"+str(X[1]*Xnorm[1])+"\t"+str(Y)+"\n")
    f.close()

 
def square_rooted(x):
    return round(math.sqrt(sum([a*a for a in x])),3)
 
def cosine_similarity(x,y):
    numerator = sum(a*b for a,b in zip(x,y))
    denominator = square_rooted(x)*square_rooted(y)
    return round(numerator/float(denominator),3)

def manhattan_distance(x,y):
    return sum(abs(a-b) for a,b in zip(x,y))
  
def euclideanPoints(X,Y,Xpred):
    Xdist=euclidean_distances(X,[Xpred])
    Xtrain=[]
    Ytrain=[]
    for k in range(10):
        for i in range(len(X)):
            if(Xdist[i]<euclideanThreshold*(2**k)):
                Xtrain.append(X[i])
                Ytrain.append(Y[i])
        if(len(Xtrain)>minDataPoints):
            return [Xtrain,Ytrain]
        
def manhattanPoints(X,Y,Xpred):
    Xtrain=[]
    Ytrain=[]
    for k in range(10):
        for i in range(len(X)):
            if(manhattan_distance(X[i],Xpred)<manhattanThreshold*(2**k)):
                Xtrain.append(X[i])
                Ytrain.append(Y[i])
        if(len(Xtrain)>minDataPoints):
            return [Xtrain,Ytrain]

def cosineSimilarityPoints(X,Y,Xpred):
    Xtrain=[]
    Ytrain=[]
    for k in range(10):
        print(k)
        for i in range(len(X)):
            similarity=cosine_similarity(X[i],Xpred)
            if(similarity>cosineThreshold/(2**k)):
                Xtrain.append(X[i])
                Ytrain.append(Y[i])
        if(len(Xtrain)>minDataPoints):
            print(len(Xtrain))
            return [Xtrain,Ytrain]
        
def jaccard_similarity(x,y):
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality/float(union_cardinality)

def jaccardSimilarityPoints(X,Y,Xpred):
    Xtrain=[]
    Ytrain=[]
    for k in range(10):
        print(k)
        for i in range(len(X)):
            similarity=jaccard_similarity(X[i],Xpred)
            if(similarity>cosineThreshold/(2**k)):
                Xtrain.append(X[i])
                Ytrain.append(Y[i])
        if(len(Xtrain)>minDataPoints):
            return [Xtrain,Ytrain]
    
#Just in Time function with partial linear regression as local model
def predictConversion(Xpred):
    [X,Y]=readData("Data@100.txt")
    [Xpred]=normalize([Xpred],Xnorm)
    X=normalize(X,Xnorm)
    [Xtrain,Ytrain]=euclideanPoints(X,Y,Xpred)
    #[Xtrain,Ytrain]=manhattanPoints(X,Y,Xpred)
    #[Xtrain,Ytrain]=cosineSimilarityPoints(X,Y,Xpred)
    #[Xtrain,Ytrain]=jaccardSimilarityPoints(X,Y,Xpred)
    pls2 = PLSRegression(n_components=2)
    pls2.fit(Xtrain, Ytrain)
    Ypredict = pls2.predict([Xpred],copy=True)
    Ypred=Ypredict[0][0]
    return Ypred

def rmse(predictions, targets):
    Sum=0
    for i in range(len(predictions)):
        Sum+=((predictions[i]-targets[i])**2)
    return Sum/len(predictions)

def weightedPls(Xquery):
    dn=[] 
    for x in X:
        xdef=x-Xquery
        temp=xdef.dot(theta)
        temp=temp.dot(xdef)
        dn.append(math.sqrt(temp))
    sigmaD=np.var(dn)
    
    Xtrain=[]
    Ytrain=[]
    for I in range(len(X)):
        Wn=math.exp(-dn[I]/(sigmaD*localizationParameter))
        Xtrain.append(X[I]*Wn)
        Ytrain.append(Y[I]*Wn)
    
    pls2 = PLSRegression(n_components=2)
    pls2.fit(Xtrain, Ytrain)
    Ypredict = pls2.predict([Xquery],copy=True)
    Ypred=Ypredict[0][0]
    return Ypred

def weightedPLS(Xquery):
    dn=[] 
    for x in X:
        xdef=x-Xquery
        temp=xdef.dot(theta)
        temp=temp.dot(xdef)
        dn.append(math.sqrt(temp))
    sigmaD=np.var(dn)
    
    Xtrain=[]
    Ytrain=[]
    for I in range(len(X)):
        Wn=math.exp(-dn[I]/(sigmaD*localizationParameter))
        Xtrain.append(X[I]*Wn)
        Ytrain.append(Y[I]*Wn)
    
    pls2 = PLSRegression(n_components=2)
    pls2.fit(Xtrain, Ytrain)
    return pls2.coef_

#%% 
n=1000
Tempratures=[gauss(323.15,10)for i in range(n)]
R_lms= [gauss(1000,300)for i in range(n)]
#%%
Yactuals=[]
Ypredicts=[]
Time=[]
i=0

currentTime=time.time()
Xpred=[Tempratures[i],R_lms[i]]
[X,Y]=readData("Data@1000.txt")
X=normalize(X,Xnorm)
X = np.array(X, dtype=np.float)
Y = np.array(Y, dtype=np.float)

thetaNew=np.ones((2,2))
thetaNew[0][1]=thetaNew[1][0]=0
thetas1=[]
thetas2=[]
iterator=[]
[Xpred]=normalize([Xpred],Xnorm)
randA=random.sample(range(1, len(X)), 100)
for _ in range(5):
    theta=thetaNew.copy()
    pprint.pprint(theta)
    thetas1.append(theta[0][0])
    thetas2.append(theta[1][1])
    iterator.append(_)
    coeff1=[]
    coeff2=[]
    for q in randA:
        Coefficient=weightedPLS(X[q])
        coeff1.append(Coefficient[0][0])
        coeff2.append(Coefficient[1][0])
    
    V=[] #Variance
    V.append(np.var(coeff1))
    V.append(np.var(coeff2))
    thetaNew[0][0]=math.pow(V[0],tuningParameter)
    thetaNew[1][1]=math.pow(V[1],tuningParameter)


plt.style.use("default")  
plt.plot(iterator[1:],thetas1[1:])
plt.plot(iterator[1:],thetas2[1:])
plt.show()

#%%

Ypred=weightedPls(Xpred)
Ypredicts.append(Ypred)
T=Xpred[0]*Xnorm[0]
R_lm=Xpred[1]*Xnorm[1]
Yactual=eng.MMA_Simulation(I_0, M_0, T, R_lm, Tf    )   #Actual value from ODE
writeData("Data@1000.txt",Xpred,Yactual,Xnorm)        #writes data back to txt file
Yactuals.append(Yactual)
Time.append(time.time()-currentTime)

RMSE= rmse(Ypredicts,Yactuals)
Correlation= np.corrcoef(Ypredicts,Yactuals)[1][0]
averageTime=np.mean(Time)

    #%% PLots
plt.style.use("default")
plt.suptitle("Ypred vs Yact | Adaptive Similarity\n RMSE= "+str(RMSE)+"\nCorr= "+str(Correlation)+"\nAverageTime= "+str(averageTime))
plt.plot([-0.2,1,0,0,0,0,1,-0.2],[-0.2,1,0,1,-0.2,0,0,0],color='k',linewidth=1)
plt.scatter(Yactuals,Ypredicts, color="red",marker="o",linewidth=0.1,alpha=0.7)
plt.xlabel("Yactual")
plt.ylabel("Ypredicted")
plt.savefig("results/AS_1.png",dp=1000)
plt.show()

print(RMSE)
print(Correlation)
print(averageTime)
''''''
 