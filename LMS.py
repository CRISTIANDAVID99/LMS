from ast import For
from calendar import c
import csv
from tkinter import Y
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

#carga de base de datos 
bd=pd.read_csv("iris.csv",header=None)
#head=['sepal_l','sepal_w','petal_l','petal_w','class']
#bd.columns=head
bd=bd[(bd[4]=="Iris-versicolor")|(bd[4]=="Iris-setosa")]

#tratamiento de datos

bdr=bd.iloc[np.random.permutation(bd.index)].reset_index(drop=True)
x0=pd.DataFrame(np.ones(100))

x=bdr.iloc[:,[0,1]]
x=pd.concat([x0,x], axis=1)
x.columns=range(x.shape[1])

y=bdr.iloc[:,4]
yk=y.astype("category").cat.codes
xt=x.iloc[:79,:]
xv=x.iloc[80:99,:]
yt=y.iloc[:79]
yv=y.iloc[80:99]

#Algoritmo lms
W=pd.DataFrame(np.random.rand(1,3))
G=np.array([])
for i in range(1000):
    ind=np.random.randint(79)
    yi=1 if yt.iloc[ind]=="Iris-setosa" else -1
    u=0.1
    #print("----------------------------------------------------------------")
    #print("Praton de X:")
    #print(x.iloc[ind,:])
    #print("salida:")
    #print(yi)
    #print("pesos W:")
    #print(W)
    g=W.iloc[0,0]+W.iloc[0,1]*xt.iloc[ind,1]+W.iloc[0,2]*xt.iloc[ind,2]
    #print("Clasificacion:")
    #print(g)
    #print("Actualizacion:")
    #print(g*yi)
    G=np.append(G,g-yi)
    W=W if (g*yi)>1 else W-(u*(g-yi)*xt.iloc[ind,:])
    #print("new W:")
    #print(W)


xs=np.linspace(0,10)
f=(-W.iloc[0,0]-W.iloc[0,1]*xs)/W.iloc[0,2]
plot1=plt.figure(1)
plt.plot(xs,f)
plt.scatter(bd.iloc[0:49,0],bd.iloc[0:49,1],c='r')
plt.scatter(bd.iloc[50:99,0],bd.iloc[50:99,1],c='b')   
plt.title("iris")
plt.xlabel("length(cm)")
plt.ylabel("width(cm)")
plot2=plt.figure(2)
plt.plot(G)
plt.title("Error Vs Iteracion u=1")
plt.xlabel("Error estimado")
plt.ylabel("Iteración")
Wr=pd.concat([W]*100).reset_index(drop=True)
yk=yk.replace({0:-1})
yf=Wr.dot(x.transpose())
yf=yf/abs(yf)
print(pd.crosstab(yk,yf.iloc[1,:]))
plt.show()
