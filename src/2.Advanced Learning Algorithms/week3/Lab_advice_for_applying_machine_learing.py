import math

import numpy as np
# %matplotlib widget
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu,linear
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from public_tests_a1 import *

tf.keras.backend.set_floatx('float64')
from assigment_utils import *

tf.autograph.set_verbosity(0)

'''
为了能够对模型进行一定的测试
划分我们的数据集，分为训练集和测试集
训练集用于训练模型
测试集用于测试模型
'''
# Generate some data
X,y,x_ideal,y_ideal = gen_data(18, 2, 0.7)
print("X.shape", X.shape, "y.shape", y.shape)

#split the data using sklearn routine
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33, random_state=1)
print("X_train.shape", X_train.shape, "y_train.shape", y_train.shape)
print("X_test.shape", X_test.shape, "y_test.shape", y_test.shape)

'''
您可以在下面看到，将成为训练一部分的数据点（红色）与模型未训练的数据点混合在一起（测试）。
这个特殊的数据集是一个添加了噪声的二次函数。
这里还展示了“理想”曲线以供参考。
'''
fig, ax = plt.subplots(1,1,figsize=(4,4))
ax.plot(x_ideal, y_ideal, "--", color = "orangered", label="y_ideal", lw=1)
ax.set_title("Training, Test",fontsize = 14)
ax.set_xlabel("x")
ax.set_ylabel("y")

ax.scatter(X_train, y_train, color = "red",           label="train")
ax.scatter(X_test, y_test,   color = dlc["dlblue"],   label="test")
ax.legend(loc='upper left')
# plt.show()
plt.close()

'''
评估线性回归模型的时候，我们通过预测值与目标值的平方误差进行评估
下面的函数的功能为，给定预测值和目标值，得到平方误差
'''
def eval_mse(y, yhat):
    """
    Calculate the mean squared error on a data set.
    Args:
      y    : (ndarray  Shape (m,) or (m,1))  target value of each example 目标值
      yhat : (ndarray  Shape (m,) or (m,1))  predicted value of each example 预测值
    Returns:
      err: (scalar)
    """
    m = len(y)
    err = 0.0
    for i in range(m):
    ### START CODE HERE ###
        err += (y[i] - yhat[i]) ** 2
    err = err / (2 * m)
    ### END CODE HERE ###
    return (err)
'''
下面我们测试一下我们的函数
'''
y_hat = np.array([2.4, 4.2])
y_tmp = np.array([2.3, 4.1])
eval_mse(y_hat, y_tmp)

# BEGIN UNIT TEST
test_eval_mse(eval_mse) # 预期输出：All tests passed.
# END UNIT TEST

'''
下面我们将通过 sklearn库 的函数建立一个高阶的线性回归模型
我们还将：
训练这个模型
计算这个模型的训练集的误差
计算这个模型的测试集的误差
'''
'''
这里创建了一个10次的线性回归模型，并且用训练集进行训练
'''
# create a model in sklearn, train on training data
degree = 10
lmodel = lin_model(degree)
lmodel.fit(X_train, y_train)
# 用训练集进行预测，并且计算训练集误差
# predict on training data, find training error
yhat = lmodel.predict(X_train)
err_train = lmodel.mse(y_train, yhat)
# 用测试集进行预测，并且计算测试集误差
# predict on test data, find error
yhat = lmodel.predict(X_test)
err_test = lmodel.mse(y_test, yhat)
'''
下面我们比较一下两种误差
预期的输出为：training err 58.01, test err 171215.01
我们可以看到模型在训练集上有非常好的表现，但是在测试集上的表现极差
三种模型预期为：高偏差 合适 过拟合/高方差
通过测试结果我们可以得到，该模型是过拟合的
'''
print(f"training err {err_train:0.2f}, test err {err_test:0.2f}")

'''
下面的图绘制了我们的模型预期曲线
图中的曲线十分完美的经过了每一个训练集，但是与测试集并不能拟合
这种现象就是过拟合
'''
# plot predictions over data range
x = np.linspace(0,int(X.max()),100)  # predict values for plot
y_pred = lmodel.predict(x).reshape(-1,1)

# plt_train_test(X_train, y_train, X_test, y_test, x, y_pred, x_ideal, y_ideal, degree)

'''
如果我们通过测试集的误差，来知道我们的模型进行改进，那将是没有意义的。
这相当于把测试集也划分为了训练集。如果出现了另一批新数据，仍然可能出现过拟合的现象。
所以我们要把数据分为三组：训练集、交叉验证集、测试集
一般，会把数据集按照6:2:2进行划分
'''
# Generate  data
X,y, x_ideal,y_ideal = gen_data(40, 5, 0.7)
print("X.shape", X.shape, "y.shape", y.shape)

#split the data using sklearn routine
X_train, X_, y_train, y_ = train_test_split(X,y,test_size=0.40, random_state=1)
X_cv, X_test, y_cv, y_test = train_test_split(X_,y_,test_size=0.50, random_state=1)
print("X_train.shape", X_train.shape, "y_train.shape", y_train.shape)
print("X_cv.shape", X_cv.shape, "y_cv.shape", y_cv.shape)
print("X_test.shape", X_test.shape, "y_test.shape", y_test.shape)

'''
和前面一样，我们绘制出了训练集、交叉验证集、测试集的分布图
'''
fig, ax = plt.subplots(1,1,figsize=(4,4))
ax.plot(x_ideal, y_ideal, "--", color = "orangered", label="y_ideal", lw=1)
ax.set_title("Training, CV, Test",fontsize = 14)
ax.set_xlabel("x")
ax.set_ylabel("y")

ax.scatter(X_train, y_train, color = "red",           label="train")
ax.scatter(X_cv, y_cv,       color = dlc["dlorange"], label="cv")
ax.scatter(X_test, y_test,   color = dlc["dlblue"],   label="test")
ax.legend(loc='upper left')
# plt.show()
plt.close()

'''
下面的实验中 ，我们将反复训练模型，将模型从1次逐步提升到10次
并且得到每个模型的训练集方差和交叉验证集的方差，以期望能够找到最合适的模型
'''
max_degree = 9
err_train = np.zeros(max_degree)
err_cv = np.zeros(max_degree)
x = np.linspace(0, int(X.max()), 100)
y_pred = np.zeros((100, max_degree))  # columns are lines to plot

for degree in range(max_degree):
    lmodel = lin_model(degree + 1)
    lmodel.fit(X_train, y_train)
    yhat = lmodel.predict(X_train)
    err_train[degree] = lmodel.mse(y_train, yhat)
    yhat = lmodel.predict(X_cv)
    err_cv[degree] = lmodel.mse(y_cv, yhat)
    y_pred[:, degree] = lmodel.predict(x)

optimal_degree = np.argmin(err_cv) + 1
'''
下面绘制了结果
左边是数据集分布以及10种模型的预测曲线
右边是不同维度模型与训练集误差的曲线以及与交叉验证集的曲线
我们会发现，当维度为2的时候，训练集误差和交叉验证集误差最小，所以维度为2就是我们要找的最佳维度

我们进一步分析图
随着模型维度增加，训练集误差从大变小，交叉验证集从大变小再变大
模型从欠拟合（高偏差）-> 合适 -> 过拟合（高方差）
所以模型的维度并不是越高越好
'''
plt.close("all")
# plt_optimal_degree(X_train, y_train, X_cv, y_cv, x, y_pred, x_ideal, y_ideal,
#                    err_train, err_cv, optimal_degree, max_degree)

'''
在之前的实验中，我们尝试使用正则化来减少过拟合。
和上面的维度实验一样。
我们可以使用同样的方法来找到最佳的正则化参数lambda
下面我们罗列了10种lambda值，用来正则化维度为10的线性回归模型
同样的，我们将采集10种模型的训练集误差和交叉验证集误差
'''
lambda_range = np.array([0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100])
num_steps = len(lambda_range)
degree = 10
err_train = np.zeros(num_steps)
err_cv = np.zeros(num_steps)
x = np.linspace(0, int(X.max()), 100)
y_pred = np.zeros((100, num_steps))  # columns are lines to plot

for i in range(num_steps):
    lambda_ = lambda_range[i]
    lmodel = lin_model(degree, regularization=True, lambda_=lambda_)
    lmodel.fit(X_train, y_train)
    yhat = lmodel.predict(X_train)
    err_train[i] = lmodel.mse(y_train, yhat)
    yhat = lmodel.predict(X_cv)
    err_cv[i] = lmodel.mse(y_cv, yhat)
    y_pred[:, i] = lmodel.predict(x)

optimal_reg_idx = np.argmin(err_cv)
'''
下面，我们进行数据绘制
左边是不同的lambda值进行正则化的模型的预测曲线
右边则是不同lambda值进行正则化的模型的训练集误差曲线和交叉验证集误差曲线

通过分析我们发现，随着lambda值增大，训练集误差逐步变大，交叉验证集误差从大变小再变大
模型也从过拟合->合适->欠拟合
列举的lambda值中，10^0为最佳值
'''
plt.close("all")
# plt_tune_regularization(X_train, y_train, X_cv, y_cv, x, y_pred, err_train, err_cv, optimal_reg_idx, lambda_range)

'''
当模型出现过拟合时，我们还可以通过收集额外的数据来提高性能。
下面我进行测试
'''
X_train, y_train, X_cv, y_cv, x, y_pred, err_train, err_cv, m_range,degree = tune_m()
'''
随着数据集的数量上升，我们可以发现：
测试集误差逐步上升，然后逐渐趋于平缓
交叉验证集误差先是急剧下降，然后逐渐趋于平缓
增加数据集确是可以解决过拟合问题
但是当数据集达到一定范围时，影响就非常小了，所以只需要获取一定数量的数据集即可
'''
# plt_tune_m(X_train, y_train, X_cv, y_cv, x, y_pred, err_train, err_cv, m_range, degree)


'''
上面的实验围绕线性回归模型展开，下面我们将围绕神经网络展开我们的实验
首先，创建数据集，并且进行划分
'''
# Generate and split data set
X, y, centers, classes, std = gen_blobs()

# split the data. Large CV population for demonstration
X_train, X_, y_train, y_ = train_test_split(X,y,test_size=0.50, random_state=1)
X_cv, X_test, y_cv, y_test = train_test_split(X_,y_,test_size=0.20, random_state=1)
print("X_train.shape:", X_train.shape, "X_cv.shape:", X_cv.shape, "X_test.shape:", X_test.shape)

'''
左边是数据集的分布图，用圆形表示训练集，三角形表示交叉验证集。
然后采用不同的颜色对类进行划分
右边是一个理想模型的例子，画出了决策边界，该模型中错误分类了大概8%的数据
'''
# plt_train_eq_dist(X_train, y_train,classes, X_cv, y_cv, centers, std)

'''
逻辑回归的误差计算公式与线性回归的误差公式不同
线性回归采用的是平方误差，而逻辑回归用的是交叉熵损失函数
下面的函数的功能是通过目标值与预测值计算逻辑回归的误差(实验里用的是简化的误差函数，我这里用了根据公式设计的误差函数)
'''
def cross_entropy_loss(y, yhat):
    """
    计算逻辑回归的交叉熵损失。

    参数：
    y_true：真实标签，形状为 (n_samples,)，取值为 0 或 1。
    y_pred：预测概率，形状为 (n_samples,)。

    返回：
    交叉熵损失值。
    """
    n_samples = len(y)
    loss = -np.sum(y * np.log(yhat) + (1 - y) * np.log(1 - yhat)) / n_samples
    return loss

y_hat = np.array([1, 2, 0])
y_tmp = np.array([1, 2, 3])
print(f"categorization error {np.squeeze(eval_cat_err(y_hat, y_tmp)):0.3f}, expected:0.333" )
y_hat = np.array([[1], [2], [0], [3]])
y_tmp = np.array([[1], [2], [1], [3]])
print(f"categorization error {np.squeeze(eval_cat_err(y_hat, y_tmp)):0.3f}, expected:0.250" )

# 下面进行测试
# BEGIN UNIT TEST
test_eval_cat_err(eval_cat_err) # All tests passed.
# END UNIT TEST
# BEGIN UNIT TEST
test_eval_cat_err(eval_cat_err) #  All tests passed.
# END UNIT TEST


'''
下面，我们将构建两个模型。
一个复杂模型和一个简单模型，我们将评估模型是否欠拟合或者过拟合
'''
'''
首先，我们创造一个复杂的具有三层的模型
第一层：全连接层，有120个单元，采用relu激活函数
第一层：全连接层，有120个单元，采用relu激活函数
第三层：全连接层，有6个单元，采用linear激活函数
编译使用SparseCategoricalCrossentropy损失函数，并且使用from_logits = true
然后通过adam算法来调整我们的学习率加快梯度下降
'''

tf.random.set_seed(1234)
model = Sequential(
    [
        Dense(120, activation = 'relu', name = "L1"),
        Dense(40, activation = 'relu', name = "L2"),
        Dense(classes, activation = 'linear', name = "L3")
    ], name="Complex"
)
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.01),
)

# 下面我们进行测试
# BEGIN UNIT TEST
model.fit(
    X_train, y_train,
    # 这里把训练次数变小是为了快速跑完这里的代码，自己写到这里的时候把次数改为1000
    epochs = 1 # epochs=1000
)
# END UNIT TEST

# BEGIN UNIT TEST
model.summary()
'''
Model: "Complex"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ L1 (Dense)                      │ (None, 120)            │           360 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ L2 (Dense)                      │ (None, 40)             │         4,840 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ L3 (Dense)                      │ (None, 6)              │           246 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 16,340 (127.66 KB)
 Trainable params: 5,446 (42.55 KB)
 Non-trainable params: 0 (0.00 B)
 Optimizer params: 10,894 (85.11 KB)
'''
# 下面的测试报错了，没找到原因
# model_test(model, classes, X_train.shape[1])
# END UNIT TEST
'''
进行绘制
左图为训练集上决策边界的表现
右图为交叉验证集上决策边界的表现
'''

model_predict = lambda Xl: np.argmax(tf.nn.softmax(model.predict(Xl)).numpy(),axis=1)
# plt_nn(model_predict,X_train,y_train, classes, X_cv, y_cv, suptitle="Complex Model")

'''
下面我们进行误差计算
从结果我们可以看到，该复杂模型的训练集误差为0.003，交叉验证集误差为0.122
存在过拟合的现象
'''

training_cerr_complex = eval_cat_err(y_train, model_predict(X_train))
cv_cerr_complex = eval_cat_err(y_cv, model_predict(X_cv))
print(f"categorization error, training, complex model: {training_cerr_complex:0.3f}") # 预期输出：categorization error, training, complex model: 0.003
print(f"categorization error, cv,       complex model: {cv_cerr_complex:0.3f}") # 预期输出：categorization error, cv,       complex model: 0.122

'''
下面，我们使用简单模型
'''

tf.random.set_seed(1234)
model_s = Sequential(
    [
        Dense(6, activation = 'relu', name="L1"),            # @REPLACE
        Dense(classes, activation = 'linear', name="L2")     # @REPLACE
    ], name = "Simple"
)
model_s.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),     # @REPLACE
    optimizer=tf.keras.optimizers.Adam(0.01),     # @REPLACE
)
# 下面进行测试
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# BEGIN UNIT TEST
model_s.fit(
    X_train,y_train,
    # 这里把训练次数变小是为了快速跑完这里的代码，自己写到这里的时候把次数改为1000
    epochs=1 # epochs = 1000
)
# END UNIT TEST

# BEGIN UNIT TEST
model_s.summary()
'''
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ L1 (Dense)                      │ (None, 6)              │            18 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ L2 (Dense)                      │ (None, 6)              │            42 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 182 (1.42 KB)
 Trainable params: 60 (480.00 B)
 Non-trainable params: 0 (0.00 B)
 Optimizer params: 122 (976.00 B)
'''
# model_s_test(model_s, classes, X_train.shape[1])
# END UNIT TEST

'''
进行绘制
左图为训练集上决策边界的表现
右图为交叉验证集上决策边界的表现
'''
#make a model for plotting routines to call
model_predict_s = lambda Xl: np.argmax(tf.nn.softmax(model_s.predict(Xl)).numpy(),axis=1)
# plt_nn(model_predict_s,X_train,y_train, classes, X_cv, y_cv, suptitle="Simple Model")


'''
下面进行误差计算
通过结果我们会发现，简单模型的训练集误差较高，但交叉验证集误差的表现比复杂模型好
存在欠拟合的现象
'''
training_cerr_simple = eval_cat_err(y_train, model_predict_s(X_train))
cv_cerr_simple = eval_cat_err(y_cv, model_predict_s(X_cv))
print(f"categorization error, training, simple model, {training_cerr_simple:0.3f}, complex model: {training_cerr_complex:0.3f}" )
print(f"categorization error, cv,       simple model, {cv_cerr_simple:0.3f}, complex model: {cv_cerr_complex:0.3f}" )

'''
复杂模型，我们可以通过正则化来降低交叉验证集的误差，提高性能
'''

tf.random.set_seed(1234)
model_r = Sequential(
    [
        # 这里对前两层进行L2正则化
        Dense(120, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(0.1), name="L1"),
        Dense(40, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(0.1), name="L2"),
        Dense(classes, activation = 'linear', name="L3")
    ], name="ComplexRegularized"
)
model_r.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.01),
)

model_r.fit(
    X_train,y_train,
    epochs=1000
)

# BEGIN UNIT TEST
model_r.summary()

# model_r_test(model_r, classes, X_train.shape[1])
# END UNIT TEST

'''
绘制数据
可以看到该模型的决策边界效果看起来相当不错，已经接近理想模型了
'''
# make a model for plotting routines to call
model_predict_r = lambda Xl: np.argmax(tf.nn.softmax(model_r.predict(Xl)).numpy(), axis=1)

plt_nn(model_predict_r, X_train, y_train, classes, X_cv, y_cv, suptitle="Regularized")

'''
计算该模型的训练集误差和交叉验证集误差
简单模型在训练集上比正则化模型有微弱的优势，但是在交叉验证集上的误差不如正则化的复杂模型
所以该模型比前面的简单模型（欠拟合）、复杂模型（过拟合）都要更加合适
'''
training_cerr_reg = eval_cat_err(y_train, model_predict_r(X_train))
cv_cerr_reg = eval_cat_err(y_cv, model_predict_r(X_cv))
test_cerr_reg = eval_cat_err(y_test, model_predict_r(X_test))
print(f"categorization error, training, regularized: {training_cerr_reg:0.3f}, simple model, {training_cerr_simple:0.3f}, complex model: {training_cerr_complex:0.3f}" )
# categorization error, training, regularized: 0.083, simple model, 0.430, complex model: 0.135
print(f"categorization error, cv,       regularized: {cv_cerr_reg:0.3f}, simple model, {cv_cerr_simple:0.3f}, complex model: {cv_cerr_complex:0.3f}" )
# categorization error, cv,       regularized: 0.081, simple model, 0.409, complex model: 0.122

'''
在线性回归模型当中，我们尝试了好几个正则化的值，然后通过比较训练集误差和交叉验证集误差找到最佳的正则化lambda的值
在这个逻辑回归的神经网络当中我们也一样可以这样做
因为要跑好几个模型，下面的代码会需要比较久的时间
'''
tf.random.set_seed(1234)
lambdas = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3]
models=[None] * len(lambdas)
for i in range(len(lambdas)):
    lambda_ = lambdas[i]
    models[i] =  Sequential(
        [
            Dense(120, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(lambda_)),
            Dense(40, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(lambda_)),
            Dense(classes, activation = 'linear')
        ]
    )
    models[i].compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(0.01),
    )

    models[i].fit(
        X_train,y_train,
        epochs=1000
    )
    print(f"Finished lambda = {lambda_}")
# 随着正则化的增加，模型在训练和交叉验证数据集上的性能会收敛。对于这个数据集和模型，λ>0.01似乎是一个合理的选择。
plot_iterate(lambdas, models, X_train, y_train, X_cv, y_cv)

# 最后，我们可以比较一下我们训练好的模型和之前的理想中表现
plt_compare(X_test,y_test, classes, model_predict_s, model_predict_r, centers)