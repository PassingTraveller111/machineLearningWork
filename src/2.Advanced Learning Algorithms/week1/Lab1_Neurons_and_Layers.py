'''
在这个实验室里，我们将探索神经元/单位和层的内部运作。
特别是，实验室将与你在课程1中掌握的模型、回归/线性模型和逻辑模型进行比较。
该实验室将介绍Tensorflow，并演示如何在该框架中实现这些模型。
'''

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# 下面的导入如果报红了，不必关心，可能是pycharm的bug，代码可以正常运行
from tensorflow.keras.layers import Dense, Input #  Dense（全连接层）和 Input（输入层）。这些层用于构建神经网络的架构。
from tensorflow.keras import Sequential # Sequential 是一种用于构建顺序模型的方式，允许将层依次添加到模型中。
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy # 导入 MeanSquaredError（均方误差损失函数）和 BinaryCrossentropy（二分类交叉熵损失函数）。这些损失函数用于衡量模型预测与真实值之间的差异。
from tensorflow.keras.activations import sigmoid # 导入 sigmoid 激活函数。激活函数用于在神经网络中引入非线性，增强模型的表达能力。
from lab_utils_common import dlc
from lab_neurons_utils import plt_prob_1d, sigmoidnp, plt_linear, plt_logistic
plt.style.use('./deeplearning.mplstyle')
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0) # 设置 tf.autograph 的日志级别为 0，可能是为了控制 TensorFlow 的自动图转换的日志输出。


# 下面是本次实验的数据集，房价预测的例子
X_train = np.array([[1.0], [2.0]], dtype=np.float32)           #(size in 1000 square feet)
Y_train = np.array([[300.0], [500.0]], dtype=np.float32)       #(price in 1000s of dollars)

fig, ax = plt.subplots(1,1)
ax.scatter(X_train, Y_train, marker='x', c='r', label="Data Points")
ax.legend( fontsize='xx-large')
ax.set_ylabel('Price (in 1000s of dollars)', fontsize='xx-large')
ax.set_xlabel('Size (1000 sqft)', fontsize='xx-large')
# plt.show()
plt.close()
# 创建了一个全连接层（Dense层），隐藏层的一种
# units=1：表示这个全连接层的输出维度为 1。即该层只有一个神经元。
# activation='linear'：指定激活函数为线性激活函数。线性激活函数就是不做任何非线性变换，直接输出输入的加权和，即 。在某些情况下，如线性回归问题中，可能会使用线性激活函数。
linear_layer = Dense(units=1, activation = 'linear', )
# 获得该层的权重w和偏置b
# 由于权重尚未实例化，因此没有权重。让我们在X_train中的一个例子中尝试这个模型。这将触发权重的实例化。
# 请注意，层的输入必须是二维的，因此我们将对其进行重塑。
linear_layer.get_weights()
# 用我们的数据集训练模型
a1 = linear_layer(X_train[0].reshape(1,1)) # tf.Tensor([[-1.07]], shape=(1, 1), dtype=float32)
print(a1)
# 打印现在的权重w和偏置b
w, b= linear_layer.get_weights()
print(f"w = {w}, b={b}") # w = [[-1.07]], b=[0.]

# 下面是手动设置权重w和偏置b的例子
set_w = np.array([[200]])
set_b = np.array([100])

# set_weights takes a list of numpy arrays
linear_layer.set_weights([set_w, set_b])
print(linear_layer.get_weights())


# 用之前训练的线性回归模型的第一个样本进行预测
a1 = linear_layer(X_train[0].reshape(1,1))
print(a1)
# 用Numpy的点乘公式同样进行预测
alin = np.dot(set_w,X_train[0].reshape(1,1)) + set_b
print(alin)
# 下面进行批量的预测，并且绘制图表进行对比
prediction_tf = linear_layer(X_train)
prediction_np = np.dot( X_train, set_w) + set_b
# plt_linear(X_train, Y_train, prediction_tf, prediction_np)

# 下面是之前做过的逻辑回归的例子
X_train = np.array([0., 1, 2, 3, 4, 5], dtype=np.float32).reshape(-1,1)  # 2-D Matrix
Y_train = np.array([0,  0, 0, 1, 1, 1], dtype=np.float32).reshape(-1,1)  # 2-D Matrix

pos = Y_train == 1
neg = Y_train == 0

fig,ax = plt.subplots(1,1,figsize=(4,3))
ax.scatter(X_train[pos], Y_train[pos], marker='x', s=80, c = 'red', label="y=1")
ax.scatter(X_train[neg], Y_train[neg], marker='o', s=100, label="y=0", facecolors='none',
              edgecolors=dlc["dlblue"],lw=3)

ax.set_ylim(-0.08,1.1)
ax.set_ylabel('y', fontsize=12)
ax.set_xlabel('x', fontsize=12)
ax.set_title('one variable plot')
ax.legend(fontsize=12)
# plt.show()
plt.close()

# 用Sequential进行连接，不过这里只有一层、一个神经单元
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(1, input_dim=1,  activation = 'sigmoid', name='L1')
    ]
)
# model.summary（）显示了模型中的参数层数和数量。此模型中只有一个层，该层只有一个单元。该单元有两个参数，
model.summary()
'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
L1 (Dense)                   (None, 1)                 2         
=================================================================
Total params: 2
Trainable params: 2
Non-trainable params: 0
_________________________________________________________________
'''
# 下面，我们看看L1层的权重和偏置，此时还未进行训练
logistic_layer = model.get_layer('L1')
w,b = logistic_layer.get_weights()
print(w,b)
print(w.shape,b.shape)
# 手动设置权重和偏置值
set_w = np.array([[2]])
set_b = np.array([-4.5])
# set_weights takes a list of numpy arrays
logistic_layer.set_weights([set_w, set_b])
print(logistic_layer.get_weights())

# 下面我们通过预置了权重和偏置的模型进行预测
a1 = model.predict(X_train[0].reshape(1,1))
print(a1)
# 这个是传统的使用numpy写的模型进行预测
alog = sigmoidnp(np.dot(set_w,X_train[0].reshape(1,1)) + set_b)
print(alog)

plt_logistic(X_train, Y_train, model, set_w, set_b, pos, neg)