import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#读取数据
path = 'ex1data2.txt'
data = pd.read_csv(path,header=None,names=['size','number of bedrooms','price'])

# 特征归一化处理
data = (data - data.mean())/data.std()

# 查看数据描述
print(data.describe())
print(data.head())

# 处理数据令X_0为1
data.insert(0, 'Ones', 1)

# 分离XY数据
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
Y = data.iloc[:,cols-1]

# 转化数据格式
X = X.values
Y = Y.values
X.reshape(-1,3)
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
theta = np.zeros((3,1),np.float64)
theta = np.matrix(theta)

iters = 1500
alpha = 0.1
theta,cost=gradient_descent(theta,X,Y,alpha,iters)

# 画出代价函数曲线
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')

# 使用正规方程
theta_1 = np.dot(np.dot(np.linalg.pinv(np.dot(X.T,X)),X.T),Y)
print(theta_1)

# 画图
plt.show()