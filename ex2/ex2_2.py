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
from sklearn.metrics import classification_report#这个包是评价报告

#%% 读取文件并查看文件信息
path = 'ex2data2.txt'
data = pd.read_csv(path,header=None,names=['test1','test2','y/n'])
print(data.describe())
print(data.head())

#%% 读取文件并查看文件信息
positive = data[data['y/n'].isin([1])]
negative = data[data['y/n'].isin([0])]

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive['test1'], positive['test2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['test1'], negative['test2'], s=50, c='r', marker='x', label='Not Admitted')
ax.legend()
ax.set_xlabel('test1')
ax.set_ylabel('test2')

#%% 实现sigmoid函数并画图
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

nums = np.arange(-10, 10, step=0.5)
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(nums, sigmoid(nums), 'r')

#%% 拓展特征
degree = 6
x1 = data['test1']
x2 = data['test2']

for i in range(0, degree+1):
    for j in range(0, i+1):
        data['(X1^' + str(i-j) + ')' + '(X2^'+str(j)+ ')'] = np.power(x1, i-j) * np.power(x2, j)

data.drop('test1', axis=1, inplace=True)
data.drop('test2', axis=1, inplace=True)

yn = data['y/n']
data.drop(labels=['y/n'], axis=1, inplace=True)
data.insert(data.columns.size, 'y/n', yn)

data.head()

#%%处理数据
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
Y = data.iloc[:,cols-1:cols]

X = X.values
X = np.array(X)
X = X.reshape(-1,cols-1)
Y = Y.values
Y = np.array(Y)
Y = Y.reshape((Y.shape[0],))

print(X.shape)
print(Y.shape)

#%% 定义损失函数并测试
def J(theta,X,Y,l=1):
    theta = np.array(theta)
    X = np.array(X)
    Y = np.array(Y)

    prediction = sigmoid(X@theta)
    cost = -Y@np.log(prediction) - (1-Y)@np.log(1-prediction)
    cost = cost/X.shape[0]

    r_cost = theta[1:]@theta[1:]
    r_cost = (l/(2*X.shape[0]))*r_cost

    cost = cost + r_cost
    
    return cost

theta = np.zeros(cols-1)
l = 1

print("cost:",J(theta,X,Y,1))

#%% [markdown]
#  在python函数编写中应当注意其参数的传递方式，可变参数如列表，字典等，在函数内对参数进行操作会影响原本参数的值！相当于传址，若不希望改变原值，可以进行深度拷贝解决这一问题，列表提供copy方法只解决外层拷贝，即嵌套列表的最外层是完全重新复制一份，而内层还是共用的。对于一般的数值，则表现为值传递。下面的例子说明了以上问题
# ```python
# def test1(theta):
#         theta[0] = 0

# theta = np.array([1,2,3])
# test1(theta)
# print(theta)

# theta = [[1],2,[3]]
# theta1 = theta.copy()
# theta1[0][0] = 0
# theta1[1] = 0
# print(theta)
# print(theta1)

# def test2(theta):
#         theta = 0

# theta = 1
# test2(theta)
# print(theta)
# ```

#%% 定义梯度
def gradient(theta,X,Y,l = 1):
        theta = np.array(theta)
        X = np.array(X)
        Y = np.array(Y)

        ans = sigmoid(X@theta)-Y
        ans = (1.0/X.shape[0])*(X.T@ans)

        rg = (l/X.shape[0])*theta

        ans = ans + rg
        ans[0] = ans[0] - rg[0]

        return ans

print("gradient",gradient(theta,X,Y,1))

#%% 定义梯度下降法
def gradient_descent(theta,X,Y,alpha,l,iters):
    cost = np.zeros(iters)
    show  = iters//100

    for i in range(iters):
        cost[i] = J(theta,X,Y,1)
        if i % show == 0:
                print(i,':',cost[i])
        update = gradient(theta,X,Y,l)
        theta = theta-alpha*update

    return theta,cost

#%%使用自定义梯度下降训练,画代价函数
lambda_value = 1
alpha = 1
iters = 1000
theta = np.zeros(cols-1)
theta,cost = gradient_descent(theta,X,Y,alpha,lambda_value,iters)

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')

#%%定义验证函数，进行预测和准确率计算
def predict(X,theta):
    theta = np.array(theta)
    X = np.array(X)

    prediction = sigmoid(X@theta)

    return [1 if x >= 0.5 else 0 for x in prediction]

predictions = predict(X,theta)
correct = [1 if (a == b) else 0 for (a, b) in zip(predictions, Y)]
accuracy = (sum(map(int, correct)) / len(correct))
print("accuracy",accuracy)

#%%使用scipy的优化器直接优化
ans = opt.minimize(fun=J,x0 = theta, args=(X,Y,l),method = "TNC",jac = gradient)
theta = ans.x

predictions = predict(X,theta)
correct = [1 if (a == b) else 0 for (a, b) in zip(predictions, Y)]
accuracy = (sum(map(int, correct)) / len(correct))
print("accuracy",accuracy)

print(classification_report(predictions,Y))

#%%显示图
plt.show()
