import matlab.engine
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import numpy as np
from random import gauss
import time
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
minDataPoints=20

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
            if(manhattan_distance(X[i],Xpred)<euclideanThreshold*(2**k)):
                Xtrain.append(X[i])
                Ytrain.append(Y[i])
        if(len(Xtrain)>minDataPoints):
            return [Xtrain,Ytrain]
    
#Just in Time function with partial linear regression as local model
def predictConversion(Xpred):
    [X,Y]=readData("Data@10.txt")
    [Xpred]=normalize([Xpred],Xnorm)
    X=normalize(X,Xnorm)
    
    [Xtrain,Ytrain]=euclideanPoints(X,Y,Xpred)
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

    

#%% 
n=100
Tempratures=[gauss(323.15,10)for i in range(n)]
R_lms= [gauss(1000,300)for i in range(n)]
#%%
Yactuals=[]
Ypredicts=[]
Time=[]
for i in range(10):
    currentTime=time.time()
    Xpred=[Tempratures[i],R_lms[i]]
    Ypred=predictConversion(Xpred)  #Prediction using JIT model
    Ypredicts.append(Ypred)
    T=Xpred[0]*Xnorm[0]
    R_lm=Xpred[1]*Xnorm[1]
    Yactual=eng.MMA_Simulation(I_0, M_0, T, R_lm, Tf)   #Actual value from ODE
    writeData("Data@10.txt",Xpred,Yactual,Xnorm)        #writes data back to txt file
    Yactuals.append(Yactual)
    Time.append(time.time()-currentTime)

RMSE= rmse(Ypredicts,Yactuals)
Correlation= np.corrcoef(Ypredicts,Yactuals)[1][0]
averageTime=np.mean(Time)

#%% PLots
plt.style.use("default")
plt.suptitle("Ypred vs Yact | Euclidean Distance\n RMSE= "+str(RMSE)+"\nCorr= "+str(Correlation)+"\nAverageTime= "+str(averageTime))
plt.plot([-0.2,1,0,0,0,0,1,-0.2],[-0.2,1,0,1,-0.2,0,0,0],color='k',linewidth=1)
plt.scatter(Yactuals,Ypredicts, color="red",marker="o",linewidth=0.1,alpha=0.7)
plt.xlabel("Yactual")
plt.ylabel("Ypredicted")
plt.show()


