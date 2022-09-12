import numpy as np
import pandas as pd
from numpy import genfromtxt
import random
def acc(pre,y):
    s=0
    for i in range(len(y)):
        if pre[i]==y[i]:
            s+=1
    return s/len(y)
def recall(pre,y):
    su=0
    makh=0
    for i in range(len(y)):
        if pre[i]==1 and y[i]==1:
            su+=1
            makh+=1
        elif pre[i]==0 and y[i]==1:
            makh+=1
    return su/makh
def precision(pre,y):
    su=0
    makh=0
    for i in range(len(y)):
        if pre[i]==1 and y[i]==1:
            su+=1
            makh+=1
        elif pre[i]==1 and y[i]==0:
            makh+=1
    return su/makh          
data = genfromtxt('breast_data.csv', delimiter=',')
y = genfromtxt('breast_truth.csv', delimiter=',')
centers = genfromtxt('mu_init.csv', delimiter=',')
#c1=centers[:,0]
#c2=centers[:,1]
testdata=data[round(0.8*len(data)):]
testy=y[round(0.8*len(y)):]
data=data[0:round(0.8*len(data))]
y=y[0:round(0.8*len(y))]
cluster=np.zeros(len(data))
data = data / data.max(axis=0)
c1=random.choice(data)
c2=random.choice(data)
cluster=np.zeros(len(data))
for l in range (100):
    sc1=np.zeros(30)
    sc2=np.zeros(30)
    f1=0
    f2=0
    for i in range(len(data)):
        dis1=sum(np.power(data[i,:]-c1,2))
        dis2=sum(np.power(data[i,:]-c2,2))
        if dis1>dis2:
            cluster[i]=2
        else:
            cluster[i]=1
    for i in range(len(data)):
        
        if cluster[i]==1:
            sc1+=data[i,:]
            f1+=1
        else:
            sc2+=data[i,:]
            f2+=1
    c1=sc1/f1
    c2=sc2/f2
cluster=cluster-1
clustert=np.zeros(len(testy))
for i in range (len(testy)):
    dis1=sum(np.power(data[i,:]-c1,2))
    dis2=sum(np.power(data[i,:]-c2,2))
    if dis1>dis2:
        clustert[i]=2
    else:
        clustert[i]=1
clustert=clustert-1
if acc(cluster,y)>0.5:
    print(acc(cluster,y))
    print(recall(cluster,y))
    print(precision(cluster,y))
else:
    cluster=np.power(0,cluster)
    print(acc(cluster,y))
    print(recall(cluster,y))
    print(precision(cluster,y))
    