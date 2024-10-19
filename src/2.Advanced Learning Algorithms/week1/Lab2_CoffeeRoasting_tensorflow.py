# 本节实验将以咖啡烘培搭建一个简单的神经网络

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from lab_utils_common import dlc
from lab_coffee_utils import load_coffee_data, plt_roast, plt_prob, plt_layer, plt_network, plt_output_unit
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)


# 加载数据集
X,Y = load_coffee_data();
print(X.shape, Y.shape)
# 绘制数据集
plt_roast(X,Y)

# 归一化数据，加快梯度下降的速度
print(f"Temperature Max, Min pre normalization: {np.max(X[:,0]):0.2f}, {np.min(X[:,0]):0.2f}")
print(f"Duration    Max, Min pre normalization: {np.max(X[:,1]):0.2f}, {np.min(X[:,1]):0.2f}")
norm_l = tf.keras.layers.Normalization(axis=-1) # 创建一个Normalization层对象norm_l。axis=-1表示对最后一个维度（即每一行）进行归一化。
norm_l.adapt(X)  # 让这个归一化层学习数据X的均值和方差，以便后续对数据进行归一化。
Xn = norm_l(X) # 使用学习到的均值和方差对数据X进行归一化，得到归一化后的数据Xn。
print(f"Temperature Max, Min post normalization: {np.max(Xn[:,0]):0.2f}, {np.min(Xn[:,0]):0.2f}")
print(f"Duration    Max, Min post normalization: {np.max(Xn[:,1]):0.2f}, {np.min(Xn[:,1]):0.2f}")

Xt = np.tile(Xn,(1000,1))
Yt= np.tile(Y,(1000,1))
print(Xt.shape, Yt.shape)

# 构建神经网络
tf.random.set_seed(1234)  # applied to achieve consistent results
model = Sequential( # 创建一个顺序模型（Sequential model），它是一种线性堆叠的神经网络层结构
    [
        tf.keras.Input(shape=(2,)), # 定义模型的输入层，指定输入数据的形状为 2 维。这意味着输入数据的每个样本应该是一个长度为 2 的向量。
        Dense(3, activation='sigmoid', name = 'layer1'), # 添加一个全连接层（Dense layer）作为第一层，该层有 3 个神经元，使用sigmoid激活函数，并命名为layer1
        Dense(1, activation='sigmoid', name = 'layer2') # Dense(1, activation='sigmoid', name='layer2')：添加另一个全连接层作为第二层，该层有 1 个神经元，也使用sigmoid激活函数，并命名为layer2。
     ]
)

# 查看神经网络的结构
model.summary()
'''
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ layer1 (Dense)                  │ (None, 3)              │             9 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ layer2 (Dense)                  │ (None, 1)              │             4 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 13 (52.00 B)
 Trainable params: 13 (52.00 B)
 Non-trainable params: 0 (0.00 B)
'''
# 下面是我们计算的每一层的权重w与偏置值b的数量
# L1的输入为两个，有三个神经元，所以有2*3个w，3个b
# 因为L1有三个神经元，所以L1有三个输出
L1_num_params = 2 * 3 + 3   # W1 parameters  + b1 parameters
# L2有三个输入，一个神经元，所以有3*1个w，1个b
L2_num_params = 3 * 1 + 1   # W2 parameters  + b2 parameters
print("L1 params = ", L1_num_params, ", L2 params = ", L2_num_params  )

# 下面，我们通过get_weights()函数查看每一层的参数，验证一下是否和我们的预期对应
W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()
print(f"W1{W1.shape}:\n", W1, f"\nb1{b1.shape}:", b1)
print(f"W2{W2.shape}:\n", W2, f"\nb2{b2.shape}:", b2)

# model.compile语句定义了一个损失函数，并指定了编译优化。
# model.fit语句运行梯度下降，并将权重拟合到数据中（计算出每个神经的w和b）。
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
)

model.fit(
    Xt,Yt,
    epochs=10,
)

# 下面我们重新查看参数w和b，会发现被更新了
W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()
print("W1:\n", W1, "\nb1:", b1)
print("W2:\n", W2, "\nb2:", b2)

'''
接下来，我们将加载一些之前训练中保存的重量。这是为了使这个笔记本对Tensorflow随时间的变化保持鲁棒性。
不同的训练运行可能会产生不同的结果，下面的适用于特定的解决方案。
可以注释代码重新运行，查看不同
'''
# W1 = np.array([
#     [-8.94,  0.29, 12.89],
#     [-0.17, -7.34, 10.79]] )
# b1 = np.array([-9.87, -9.28,  1.01])
# W2 = np.array([
#     [-31.38],
#     [-27.86],
#     [-32.79]])
# b2 = np.array([15.54])
# model.get_layer("layer1").set_weights([W1,b1])
# model.get_layer("layer2").set_weights([W2,b2])

# 下面，我们对厕所数据进行预测
X_test = np.array([
    [200,13.9],  # postive example
    [200,17]])   # negative example
X_testn = norm_l(X_test)
predictions = model.predict(X_testn)
print("predictions = \n", predictions)
'''
predictions = 
 [[9.63e-01]
 [3.03e-08]]
 '''

# 为了将概率转换为决策（1或0），我们设置一个阈值，>=0.5为1，反之为0
yhat = np.zeros_like(predictions)
for i in range(len(predictions)):
    if predictions[i] >= 0.5:
        yhat[i] = 1
    else:
        yhat[i] = 0
print(f"decisions = \n{yhat}")
# 下面这种实现方式会更简洁
yhat = (predictions >= 0.5).astype(int)
print(f"decisions = \n{yhat}")
'''
让我们检查一下这些单元的功能(有三个神经元)，以确定它们在咖啡烘焙决策中的作用。
我们将绘制每个节点的所有输入值（持续时间、温度）的输出。
每个单位都是一个逻辑函数，其输出范围可以从零到一。图中的阴影表示输出值。
'''
plt_layer(X,Y.reshape(-1,),W1,b1,norm_l)
'''
我们可以创建一个三维图，计算三个输入的所有可能组合的输出。如下图所示。
上面，高输出值对应于“烤得不好”的区域。下面，最大输出位于三个输入都是与“烤得好”区域对应的小值的区域。
'''
plt_output_unit(W2,b2)
'''
最后的图表显示了整个网络的运行情况。
左图是由蓝色阴影表示的最后一层的原始输出。这覆盖在由X和O表示的训练数据上。
右图是决策阈值后网络的输出。这里的X和O对应于网络做出的决定。
以下内容需要一点时间才能运行
'''
netf= lambda x : model.predict(norm_l(x))
plt_network(X,Y,netf)