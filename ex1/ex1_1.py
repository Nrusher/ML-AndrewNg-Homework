import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#读取数据
path = 'ex1data1.txt'
data = pd.read_csv(path,header=None,names=['Population','Profit'])

# 查看数据描述
print(data.describe())
print(data.head())

# 画数据散点图
data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))

# 处理数据令X_0为1
data.insert(0, 'Ones', 1)

# 分离XY数据
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
Y = data.iloc[:,cols-1]

# 转化数据格式
X = X.values
Y = Y.values
X.reshape(-1,2)
Y = Y.reshape(-1,1)
X = np.matrix(X)
Y = np.matrix(Y)
print(X.shape)
print(Y.shape)

# 定义代价函数
def J(theta,X,Y):
    j_theta = np.dot(X,theta)-Y
    j_theta = np.dot(j_theta.T,j_theta)
    j_theta = j_theta/(2*X.shape[0])
    return j_theta

# 初始化theta及迭代次数等信息
theta = np.zeros((2,1),np.float64)
theta = np.matrix(theta)
iters = 1500
alpha = 0.01

# task1 打印theta=[0,0]时的代价
print(J(theta,X,Y))

# 定义梯度下降法
def gradient_descent(theta,X,Y,alpha,iters):
    cost = np.zeros(iters)

    for i in range(iters):
        cost[i] = J(theta,X,Y)
        print(cost[i])
        update = np.dot(X,theta)-Y
        update = np.dot(X.T,update)
        update = (alpha/X.shape[0])*update
        theta = theta-update

    return theta,cost

# 利用梯度下降法进行估计
theta,cost=gradient_descent(theta,X,Y,alpha,iters)
print(theta)

# 画出代价函数曲线
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')

# 画出估计结果直线
x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = theta[0, 0] + (theta[1, 0] * x)
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()