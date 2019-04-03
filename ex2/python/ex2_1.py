#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
        os.chdir(os.path.join(os.getcwd(), 'ANG\homework\ex2'))
        print(os.getcwd())
except:
	pass

#%% 载入包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.metrics import classification_report

#%% 读取文件并查看文件信息
path = 'ex2data1.txt'
data = pd.read_csv(path,header=None,names=['ex1','ex2','y/n'])
print(data.describe())
print(data.head())

#%% 读取文件并查看文件信息
positive = data[data['y/n'].isin([1])]
negative = data[data['y/n'].isin([0])]

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive['ex1'], positive['ex2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['ex1'], negative['ex2'], s=50, c='r', marker='x', label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')

#%% 实现sigmoid函数并画图
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

nums = np.arange(-10, 10, step=0.5)
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(nums, sigmoid(nums), 'r')

#%% 定义损失函数
def J(theta,X,Y):
    theta = np.array(theta)
    X = np.array(X)
    Y = np.array(Y)

    prediction = np.dot(X,theta)
    prediction = sigmoid(prediction)
    cost = np.dot(0-Y.T,np.log(prediction)) - np.dot(1-Y.T,np.log(1-prediction))
    cost = cost/X.shape[0]
    return cost[0][0]

#%% 定义梯度
def gradient(theta,X,Y):
        theta = np.array(theta)
        X = np.array(X)
        Y = np.array(Y)

        ans = sigmoid(np.dot(X,theta))-Y
        ans = np.dot(X.T,ans)
        ans = (1.0/X.shape[0])*ans

        ans = ans.reshape(3,1)

        return ans
#%% [markdown]
# 这里的数据特征归一化很重要，直接影响到后面的训练效果，当然在实际预测时依然要对原始数据进行归一化处理。
#%% 处理原始数据
# data.insert(0, 'Ones', 1)

cols = data.shape[1]
X = data.iloc[:,0:cols-1]
X = (X - X.mean())/X.std()
X.insert(0,'Ones',1)
Y = data.iloc[:,cols-1:cols]

X = X.values
X = np.array(X)
X = X.reshape(-1,3)
Y = Y.values
Y = np.array(Y)
Y = Y.reshape(-1,1)
theta = np.zeros((3,1))

#%% 测试J和gradient函数
print("cost",J(theta,X,Y))
print("gradient",gradient(theta,X,Y))

#%% 定义梯度下降法
def gradient_descent(theta,X,Y,alpha,iters):
    cost = np.zeros(iters)

    for i in range(iters):
        cost[i] = J(theta,X,Y)
        # print(cost[i])
        # update = sigmoid(np.dot(X,theta))-Y
        # update = np.dot(X.T,update)
        # update = (alpha/X.shape[0])*update
        update = alpha*gradient(theta,X,Y)
        theta = theta-update

    return theta,cost

#%%使用自定义梯度下降训练,画代价函数
alpha = 0.1
iters = 10000
theta,cost = gradient_descent(theta,X,Y,alpha,iters)

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')

#%%定义验证函数，进行预测和准确率计算
def predict(X,theta):
    theta = np.array(theta)
    X = np.array(X)

    prediction = np.dot(X,theta)
    prediction = sigmoid(prediction)

    return [1 if x >= 0.5 else 0 for x in prediction]

predictions = predict(X,theta)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, Y)]
accuracy = (sum(map(int, correct)) / len(correct))
print("accuracy",accuracy)
print(classification_report(predictions, Y))
#%%显示图
plt.show()
