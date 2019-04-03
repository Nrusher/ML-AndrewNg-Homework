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

def load_data(path, transpose=True):
    data = sio.loadmat(path)
    y = data.get('y')  # (5000,1)
    y = y.reshape(y.shape[0])  # make it back to column vector

    X = data.get('X')  # (5000,400)

    if transpose:
        # for this dataset, you need a transpose to get the orientation right
        X = np.array([im.reshape((20, 20)).T for im in X])

        # and I flat the image again to preserve the vector presentation
        X = np.array([im.reshape(400) for im in X])

    return X, y

X, Y = load_data('ex3data1.mat')

print(X.shape)
print(Y.shape)

#%%画图
def plot_an_image(image):
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.matshow(image.reshape((20, 20)), cmap=matplotlib.cm.binary)
    plt.xticks(np.array([]))  # just get rid of ticks
    plt.yticks(np.array([]))

pick_one = np.random.randint(0, 5000)
plot_an_image(X[pick_one, :])
plt.show()
print('this should be {}'.format(Y[pick_one]))

#%%画100张图
def plot_100_image(X):
    """ sample 100 image and show them
    assume the image is square

    X : (5000, 400)
    """
    size = int(np.sqrt(X.shape[1]))

    # sample 100 image, reshape, reorg it
    sample_idx = np.random.choice(np.arange(X.shape[0]), 100)  # 100*400
    sample_images = X[sample_idx, :]

    fig, ax_array = plt.subplots(nrows=10, ncols=10, sharey=True, sharex=True, figsize=(8, 8))

    for r in range(10):
        for c in range(10):
            ax_array[r, c].matshow(sample_images[10 * r + c].reshape((size, size)),
                                   cmap=matplotlib.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))  
            #绘图函数，画100张图片

plot_100_image(X)

#%%数据预处理
X = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)
print(X.shape)
theta = np.zeros(X.shape[1])
print(theta.shape)

#%%
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

#%% [markdown]
# 在numpy中运算符号@可以代表矩阵相乘,numpy中一维的array类型的数据既可以是行向量，亦可以是列向量例如
# ``` python
# a = np.array([[1,2],[3,4]])
# b = np.array([1,1])
# print(a@b)
# print(b@a)
# ```
# 这两种运算都不会出现错误，第一个print将b当作列向量，而第二个则当作行向量，运算结果依然是一个一维的array，当两个向量相乘时则两种运算的结果时相同的

#%% 定义损失函数
def J(theta,X,Y):
    theta = np.array(theta)
    X = np.array(X)
    Y = np.array(Y)

    prediction = sigmoid(X@theta)
    cost = np.dot(0-Y,np.log(prediction)) - np.dot(1-Y,np.log(1-prediction))
    cost = cost/X.shape[0]
    return cost

print('J_test:',J(theta,X,Y))

#%% 定义梯度函数
def gradient(theta,X,Y):
        theta = np.array(theta)
        X = np.array(X)
        Y = np.array(Y)

        ans = sigmoid(np.dot(X,theta))-Y
        ans = np.dot(X.T,ans)
        ans = (1.0/X.shape[0])*ans
        return ans

print('gradient_test:',gradient(theta,X,Y))

#%% [markdown]
# numpy中不同计算方式可能有不同的计算精度
# ``` python
# theta = np.ones(10000)*0.1
# print(np.power(theta, 2).sum()-theta@theta)
# print(np.power(theta, 2).sum()-np.dot(theta,theta))
# ```
# 避免在原始数据上执行修改等操作，应进行拷贝后进行，避免对数据修改后，对其它调用该数据的函数产生影响。勤加括号保平安。
#%% 定义正则化损失函数
def J_reg(theta,X,Y,l=1):
    theta = np.array(theta)
    X = np.array(X)
    Y = np.array(Y)

    cost = J(theta,X,Y)

    reg_term = (l/(2.0*len(X)))*(theta[1:]@theta[1:])
    cost = cost + reg_term
    return cost

print('J_reg_test:',J_reg(theta,X,Y))

#%% 定义正则化梯度函数
def gradient_reg(theta,X,Y,l=1):
        theta = np.array(theta)
        X = np.array(X)
        Y = np.array(Y)

        ans = gradient(theta,X,Y)

        reg_term = (l/len(X))*theta
        reg_term[0]=0
        ans = ans + reg_term
        return ans

print('gradient_reg_test:',gradient_reg(theta,X,Y))

#%%
def one_vs_all(theta,X,Y,l=1):
        Theta = np.zeros((10,X.shape[1]))
        for i in range(1,11,1):
                theta = np.zeros(X.shape[1])
                Y_train = [1 if x==i else 0 for x in Y]
                ans = opt.minimize(fun=J_reg,x0=theta,args=(X,Y_train),method='TNC',jac=gradient_reg)
                Theta[i-1] = ans.x
                print(i,':',ans.success)     
        return Theta

Theta = one_vs_all(theta,X,Y)
#%%
print(Theta.shape)
print(Y.shape)
print(X.shape)


#%%
def predict(X,theta):
    theta = np.array(theta)
    X = np.array(X)

    prediction = np.dot(X,theta.T)
    prediction = sigmoid(prediction)

    ans = np.zeros(X.shape[0])

    for i in range(X.shape[0]):
            ans[i] = np.argmax(prediction[i]) + 1

    return ans

predictions = predict(X,Theta)
correct = [1 if int(a)==b else 0 for (a,b) in zip(predictions,Y)]
accuracy = (sum(correct) / len(correct))
print("accuracy",accuracy,"\r\n")
print(classification_report(predictions, Y))

#%%保存训练结果
sio.savemat("Theta.mat",{"Theta":Theta})

#%%显示图
plt.show()
