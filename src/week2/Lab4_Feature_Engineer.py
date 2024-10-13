import numpy as np
import matplotlib.pyplot as plt
from lab_utils_multi import zscore_normalize_features, run_gradient_descent_feng
np.set_printoptions(precision=2)  # reduced display precision on numpy arrays
# 本节将探索特征工程和多项式回归，它允许您使用线性回归的机制来拟合非常复杂甚至非常非线性的函数。

# create target data 下面将以y = 1 + x^2作为真实数据作为例子
x = np.arange(0, 20, 1) # 特征为从0-19的数
y = 1 + x**2 # 目标为1+x^2的结果
X = x.reshape(-1, 1) # 大X就是经过变换以后的特征，但是这里其实和原特征一样

model_w,model_b = run_gradient_descent_feng(X,y,iterations=1000, alpha = 1e-2)

# 可以看到，此时用一次的线性回归不太合适
plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("no feature engineering")
plt.plot(x,X@model_w + model_b, label="Predicted Value");  plt.xlabel("X"); plt.ylabel("y"); plt.legend();
# plt.show()
plt.close()

# create target data
x = np.arange(0, 20, 1)
y = 1 + x**2

# Engineer features
X = x**2      #<-- added engineered feature 这里对特征进行了变换，将特征的量级上升了一次

X = X.reshape(-1, 1)  #X should be a 2-D Matrix
model_w,model_b = run_gradient_descent_feng(X, y, iterations=10000, alpha = 1e-5)

plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("Added x**2 feature")
plt.plot(x, np.dot(X,model_w) + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend();
# plt.show()
plt.close()

# create target data
x = np.arange(0, 20, 1)
y = x**2

# engineer features .
#通过列拼接的方式，将原始的 x、x 的平方 x**2 和 x 的立方 x**3 组合成一个新的二维数组 X，这一步是特征工程，创建了新的特征。
# 在 NumPy 中，np.c_[]主要用于按列连接两个或多个数组。
'''
比如
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

c = np.c_[a, b]
print(c)
输出：
array([[1, 4],
       [2, 5],
       [3, 6]])
'''
X = np.c_[x, x**2, x**3]

model_w,model_b = run_gradient_descent_feng(X, y, iterations=10000, alpha=1e-7)

plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("x, x**2, x**3 features")
plt.plot(x, X@model_w + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend();
# plt.show()
plt.close()

#上面，多项式特征是根据它们与目标数据的匹配程度来选择的。
# 下面用了另一种思路，就是用变换以后的特征值的结果直接与y进行映射，如果是直线，那就说明狠合适
# create target data
x = np.arange(0, 20, 1)
y = x**2

# engineer features .
X = np.c_[x, x**2, x**3]   #<-- added engineered feature
X_features = ['x','x^2','x^3']

fig,ax=plt.subplots(1, 3, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X[:,i],y)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("y")
# 这里的第一个图就是原图像，第二个图会发现图像变成直线的了，第三个图上凸了。
# plt.show()
plt.close()


 # Scaling features
 # create target data
x = np.arange(0,20,1)
X = np.c_[x, x**2, x**3]
# 在 NumPy 中，np.ptp()是 “peak to peak” 的缩写，用于计算数组中沿指定轴的取值范围（最大值与最小值之差）。
'''
b = np.array([[2, 4, 6], [8, 10, 12]])
# 计算整个数组的取值范围
range_all = np.ptp(b)
print(range_all)

# 计算每列的取值范围
range_per_column = np.ptp(b, axis=0)
print(range_per_column)

# 计算每行的取值范围
range_per_row = np.ptp(b, axis=1)
print(range_per_row)

输出
10
array([6, 6, 6])
array([4, 2])
'''
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X,axis=0)}") # 计算并打印原始 X 数组中每列的峰峰值（最大值与最小值之差）

# add mean_normalization
X = zscore_normalize_features(X)
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X,axis=0)}") # 进行特征缩放，之后的结果的峰值差距显著变小

# 下面采用了更加激进的学习率，通过特征缩放使梯度下降的收敛速度加快
x = np.arange(0,20,1)
y = x**2

X = np.c_[x, x**2, x**3]
X = zscore_normalize_features(X)

model_w, model_b = run_gradient_descent_feng(X, y, iterations=100000, alpha=1e-1)

plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("Normalized x x**2, x**3 feature")
# plt.plot(x,X@model_w + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()
plt.close()

# 通过特征工程，甚至可以对非常复杂的功能进行建模：
x = np.arange(0,20,1)
y = np.cos(x/2)

X = np.c_[x, x**2, x**3,x**4, x**5, x**6, x**7, x**8, x**9, x**10, x**11, x**12, x**13]
X = zscore_normalize_features(X)

model_w,model_b = run_gradient_descent_feng(X, y, iterations=1000000, alpha = 1e-1)

plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("Normalized x x**2, x**3 feature")
plt.plot(x,X@model_w + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()


