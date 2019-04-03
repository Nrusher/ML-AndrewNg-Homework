#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
        os.chdir(os.path.join(os.getcwd(), 'ANG\homework\ex3\python'))
        print(os.getcwd())
except:
	pass

#%% 载入包
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.io as sio
from sklearn.metrics import classification_report

#%% 读取文件并查看文件信息
path = 'ex3data1.mat'
print(sio.whosmat('ex3data1.mat'))
data = sio.loadmat(path)
X = data["X"]
X = np.insert(arr=X,obj=0,values = np.ones(X.shape[0]),axis= 1)
print(X.shape)
Y = data["y"]
Y = Y.reshape(Y.shape[0])
print(Y.shape)

path = 'ex3weights.mat'
print(sio.whosmat('ex3weights.mat'))
data = sio.loadmat(path)
Theta1 = data["Theta1"]
print(Theta1.shape)
Theta2 = data["Theta2"]
print(Theta2.shape)

#%%
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

#%%
def predict(Theta1,Theta2,X):
        Z1 = sigmoid(Theta1@X.T)
        # print(Z1.shape)
        Z1 = np.insert(arr=Z1,obj=0,values = np.ones(Z1.shape[1]),axis=0)
        # print(Z1.shape)
        Z2 = Theta2@Z1
        Z2 = Z2.T
        # print(Z2.shape)

        ans = np.zeros(X.shape[0])

        for i in range(X.shape[0]):
            ans[i] = np.argmax(Z2[i]) + 1

        return ans

predictions = predict(Theta1,Theta2,X)
correct = [1 if int(a)==b else 0 for (a,b) in zip(predictions,Y)]
accuracy = (sum(correct) / len(correct))
print("accuracy",accuracy,"\r\n")
print(classification_report(predictions, Y))

#%%显示图
plt.show()
