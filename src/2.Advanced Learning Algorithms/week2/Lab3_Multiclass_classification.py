import numpy as np
import matplotlib.pyplot as plt
# %matplotlib widget
from sklearn.datasets import make_blobs
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
np.set_printoptions(precision=2)
from lab_utils_multiclass_TF import *
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

# 在本实验室中，将看到一个在Tensorflow中构建多类网络的示例。然后，我们将看看神经网络是如何做出预测的。

# 我们将使用Scikit Learn make_blobs函数制作一个包含4个类别的训练数据集，如下图所示。
# make 4-class dataset for classification
classes = 4
m = 100
centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
std = 1.0
X_train, y_train = make_blobs(n_samples=m, centers=centers, cluster_std=std,random_state=30)
# 绘制数据集，每种颜色代表了一类数据集，这里一共有四类
plt_mc(X_train,y_train,classes, centers, std=std)
# show classes in data set
print(f"unique classes {np.unique(y_train)}") # unique classes [0 1 2 3] y的值一共有四种，分别为0、1、2、3
# show how classes are represented
print(f"class representation {y_train[:10]}") # class representation [3 3 3 0 3 3 3 3 2 0]
# show shapes of our dataset
print(f"shape of X_train: {X_train.shape}, shape of y_train: {y_train.shape}") # shape of X_train: (100, 2), shape of y_train: (100,)

# 建立模型

'''
下面是一个如何在Tensorflow中构建这个网络的示例。
请注意，输出层使用线性而不是softmax激活。
虽然可以在输出层中包含softmax，但如果在训练期间将线性输出传递给损失函数，则数值上更稳定。
如果该模型用于预测概率，则可以在该点应用softmax。
'''
tf.random.set_seed(1234)  # applied to achieve consistent results
model = Sequential(
    [
        Dense(2, activation = 'relu',   name = "L1"),
        Dense(4, activation = 'linear', name = "L2")
    ]
)

model.compile(
    # 指定损失函数为稀疏类别交叉熵损失函数。
    # 设置 from_logits=True 表示模型的输出是未经过激活函数处理的原始预测值（logits），通常在最后一层没有激活函数时使用这个设置。
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    # 选择优化器为 Adam 优化器，并设置学习率为 0.01。
    # Adam 优化器是一种常用的优化算法，能够自适应地调整学习率，在很多深度学习任务中表现良好。
    optimizer=tf.keras.optimizers.Adam(0.01),
)
# 训练模型
model.fit(
    X_train,y_train,
    epochs=200
)
# 下面的函数将绘制数据集以及训练出来的决策边界
plt_cat_mc(X_train, y_train, model, classes)

# 获取第一层的训练后的参数
# gather the trained parameters from the first layer
l1 = model.get_layer("L1")
W1,b1 = l1.get_weights()

# plot the function of the first layer
# 绘制第一层隐藏层
plt_layer_relu(X_train, y_train.reshape(-1,), W1, b1, classes)

# gather the trained parameters from the output layer
l2 = model.get_layer("L2")
W2, b2 = l2.get_weights()
# create the 'new features', the training examples after L1 transformation
Xl2 = np.zeros_like(X_train)
Xl2 = np.maximum(0, np.dot(X_train,W1) + b1)

plt_output_layer_linear(Xl2, y_train.reshape(-1,), W2, b2, classes,
                        x0_rng = (-0.25,np.amax(Xl2[:,0])), x1_rng = (-0.25,np.amax(Xl2[:,1])))