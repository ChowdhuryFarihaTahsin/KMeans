import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as rd

df = pd.read_csv('data.txt ', sep= " " , header = None,  dtype = None)
#print(df)
points=df.values
x1=points[:,0]
x2=points[:,1]

print('---------------- TASK 1 ----------------')
color=['purple']
plt.scatter(x1,x2,c=color,label='unclustered data')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Plot of Unclustered data points')
plt.show()

r=points.shape[0]
c=points.shape[1] 
iterationNo=100
K=2

Centroids=np.array([]).reshape(c,0) 

for i in range(K):
    randomPoints=rd.randint(0,r-1)
    print(randomPoints)
    Centroids=np.c_[Centroids,points[randomPoints]]
    print(Centroids)
 
finalResult={}

EuclidianDistance=np.array([]).reshape(r,0)
for k in range(K):
       tempDist=np.sum((points-Centroids[:,k])**2,axis=1)
       EuclidianDistance=np.c_[EuclidianDistance,tempDist]
C=np.argmin(EuclidianDistance,axis=1)+1
print(C)
TempResult={}
for k in range(K):
    TempResult[k+1]=np.array([]).reshape(2,0)
for i in range(r):
    TempResult[C[i]]=np.c_[TempResult[C[i]],points[i]]     
for k in range(K):
    TempResult[k+1]=TempResult[k+1].T    
for k in range(K):
     Centroids[:,k]=np.mean(TempResult[k+1],axis=0) 
     
for i in range(iterationNo): 
      EuclidianDistance=np.array([]).reshape(r,0)
      for k in range(K):
          tempDist=np.sum((points-Centroids[:,k])**2,axis=1)
          EuclidianDistance=np.c_[EuclidianDistance,tempDist]
      C=np.argmin(EuclidianDistance,axis=1)+1
      
      TempResult={}
      for k in range(K):
          TempResult[k+1]=np.array([]).reshape(2,0)
      for i in range(r):
          TempResult[C[i]]=np.c_[TempResult[C[i]],points[i]]
      for k in range(K):
          TempResult[k+1]=TempResult[k+1].T
      for k in range(K):
          Centroids[:,k]=np.mean(TempResult[k+1],axis=0)
      finalResult=TempResult
      
print(finalResult)
print('---------------- TASK 2, 3 ----------------')
color=['green','red']
labels=['cluster1','cluster2']
for k in range(K):
    plt.scatter(finalResult[k+1][:,0],finalResult[k+1][:,1],c=color[k],label=labels[k])
plt.scatter(Centroids[0,:],Centroids[1,:],c='blue',label='Centroids')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Clustered data points Using K-Means')
plt.legend()
plt.show()