import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from IPython.display import display, Markdown, Latex
from sklearn.datasets import make_blobs
# %matplotlib widget
from matplotlib.widgets import Slider
from lab_utils_common import dlc
from lab_utils_softmax import plt_softmax
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

# 在本实验中，我们将探索softmax函数。在解决多类分类问题时，该函数用于Softmax回归和神经网络。

# 下面是用于进行线性变换的softmax函数
def my_softmax(z):
    ez = np.exp(z)              #element-wise exponenial
    sm = ez/np.sum(ez)
    return(sm)

plt.close("all")
# softmax将多个类的输出经过线性变换以后变成概率，各个类别的概率之和为1
plt_softmax(my_softmax)

# make  dataset for example 为实验设置数据集
centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
X_train, y_train = make_blobs(n_samples=2000, centers=centers, cluster_std=1.0,random_state=30)

# 下面我们创建了一个用softmax作为输出层解决多类分类问题的神经网络例子
model = Sequential(
    [
        Dense(25, activation = 'relu'),
        Dense(15, activation = 'relu'),
        Dense(4, activation = 'softmax')    # 输出层用softmax作为激活函数
    ]
)
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001),
)

model.fit(
    X_train,y_train,
    epochs=10
)
# 因为以softmax作为输出层的激活函数，所以模型最后的输出是概率
# 下面进行预测
p_nonpreferred = model.predict(X_train)
print(p_nonpreferred [:2])
print("largest value", np.max(p_nonpreferred), "smallest value", np.min(p_nonpreferred))





preferred_model = Sequential(
    [
        Dense(25, activation = 'relu'),
        Dense(15, activation = 'relu'),
        # softmax作为激活函数放在输出层的话，会有精度问题
        # 所以用linear作为激活函数
        Dense(4, activation = 'linear')
    ]
)
preferred_model.compile(
    # loss函数有一个额外的参数：from_logits=True。这通知损失函数，softmax操作应包含在损失计算中。这允许优化实现。
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  #<-- Note
    optimizer=tf.keras.optimizers.Adam(0.001),
)

preferred_model.fit(
    X_train,y_train,
    epochs=10
)

# 这种优化以后的模型的输出并不是概率，编译中的那步操作只是让损失函数计算时包括softmax操作
p_preferred = preferred_model.predict(X_train)
print(f"two example output vectors:\n {p_preferred[:2]}")
print("largest value", np.max(p_preferred), "smallest value", np.min(p_preferred))

# 输出的不是概率，如果期望输出的是概率，那么需要用softmax函数处理一下，操作如下
'''
输出的是前两个测试样例的预测结果
每一行有四个元素，分别代表该样例下，四种类别的概率分别是多少
two example output vectors:
 [[2.40e-03 7.58e-03 9.68e-01 2.24e-02]
 [9.97e-01 2.28e-03 2.13e-04 1.31e-05]]
largest value 0.9999988 smallest value 1.6588557e-09
'''
sm_preferred = tf.nn.softmax(p_preferred).numpy()
print(f"two example output vectors:\n {sm_preferred[:2]}")
print("largest value", np.max(sm_preferred), "smallest value", np.min(sm_preferred))


# 下面我们以 （输出 类别）的形式打印数据
for i in range(5):
    print( f"{p_preferred[i]}, category: {np.argmax(p_preferred[i])}")