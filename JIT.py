import matlab.engine
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
#eng = matlab.engine.start_matlab()

Xnorm=[323.15,1000]
I_0= 0.00258               #Initial moles of initiator
M_0= 1E6*1.0                   #Initial moles of monomer
T=323.45
R_lm =0.0
Tf=10.0
euclideanThreshold=0.05

def normalize(X,Xnorm):
    for i in range(len(X)):
        for j in range(len(Xnorm)):
            X[i][j]=X[i][j]/Xnorm[j]
    return X

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

def writeData(s,X,Y,Xnorm):
    f=open(s,'w')
    for i in range(len(X)):
        f.write(str(X[i][0]*Xnorm[0])+"\t"+str(X[i][1]*Xnorm[1])+"\t"+str(Y[i])+"\n")
    f.close()
   

[X,Y]=readData("Data@10.txt")



f2=open("Data@test.txt","r")
Xact=[]
Yact=[]
Lines=f2.readlines()
for line in Lines:
    line=line.split("\n")[0]
    Data=line.split("\t")
    Xact.append([float(Data[0]),float(Data[1])])
    Yact.append(float(Data[2]))

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

Ypred=[]
count=0
Lengths=[]

Xpred=Xact[0]
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
pls2 = PLSRegression(n_components=2)
pls2.fit(Xtrain, Ytrain)
Ypredict = pls2.predict([Xpred],copy=True)
Ypred.append(Ypredict[0][0])

    











