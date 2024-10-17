"""
 这节实验尝试根据数据集绘制散点图，并且通过模型预测函数，得到预测值，绘制预测曲线
"""
"""
NumPy，一个流行的科学计算库
Matplotlib，一个流行的数据绘图库
"""
import numpy as np
"""
pyplot 是 matplotlib 中的一个子模块，提供了一种类似于 MATLAB 的绘图方式，使用起来非常方便。
通过 as plt 将其重命名为 plt，以便在后续代码中更简洁地调用。
"""
import matplotlib.pyplot as plt
"""
plt.style.use() 函数用于设置绘图的样式。
'./deeplearning.mplstyle' 是一个路径，表示要使用的样式文件的位置。
这个样式文件可能包含了自定义的颜色、字体、线条样式等设置，以使得绘制的图形具有特定的外观风格。
"""
plt.style.use('deeplearning.mplstyle')
"""
x_train is the input variable (size in 1000 square feet) x是输入变量/特征
y_train is the target (price in 1000s of dollars) y是目标
下面这段代码在定义数据集
"""
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")
# Number of training examples m  m就是训练样本的数量
"""
x_train.shape 返回一个元组，表示 x_train 的维度信息。
例如，如果 x_train 是一个二维数组，形状可能是 (m, n)，其中 m 是样本数量，n 是每个样本的特征数量。
"""
print(f"x_train.shape: {x_train.shape}")
m = x_train.shape[0]
print(f"Number of training examples is: {m}")
"""
或者可以用下面的方式得到样本的总数
"""
m = len(x_train)
print(f"Number of training examples is: {m}")
"""
可以用下面几个方法绘制散点图
"""
# Plot the data points
"""
plt.scatter() 函数用于绘制散点图。
x_train 和 y_train 分别是房屋面积和价格的数据。
marker='x' 设置散点的标记为 x 形状。
c='r' 设置散点的颜色为红色（r 代表红色）。
这里绘制实际数据的散点图
"""
plt.scatter(x_train, y_train, marker='x', c='r')
# Set the title 写标题
plt.title("Housing Prices")
# Set the y-axis label y轴标题
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label x轴标题
plt.xlabel('Size (1000 sqft)')
# 展示绘制的图形,想看该图形，就解除下面的注释
# plt.show()
# 清除绘制数据
plt.close()
"""
下面定义一个模型输出函数，用于计算线性模型的预测值。它接受输入数据x、模型参数w和b，并返回对应的预测值y
"""
def compute_model_output(x, w, b):
    # 得到样本数量
    m = x.shape[0]
    # 创建一个长度为m，值全为0的数组，用于存储预测值
    f_wb = np.zeros(m)
    # 循环，根据线性函数 f=wx+b 得到不同x的预测值，存储到f_wb中
    for i in range(m):
        f_wb[i] = w * x[i] + b
    return f_wb

"""
下面我们将根据模型预测值绘制
"""
# 设置模型的参数w、b为100
w = 100
b = 100
# 使用模型获得预测值
tmp_f_wb = compute_model_output(x_train, w, b)

# 绘制模型预测曲线，颜色为蓝色，标签为Our Prediction
plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')

# 绘制数据集散点图
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')

# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
# 显示图例，就是解释哪条线是Our Prediction，哪条是Actual Values
plt.legend()
# 展示绘制的图形,想看该图形，就解除下面的注释
# plt.show()
plt.close()
"""
下面，我们尝试使用不同的w、b去绘制预测曲线
"""
w1 = 100
b1 = 100
w2 = 300
b2 = 50
w3 = 200
b3 = 100
tmp1_f_wb = compute_model_output(x_train, w1, b1)
tmp2_f_wb = compute_model_output(x_train, w2, b2)
tmp3_f_wb = compute_model_output(x_train, w3, b3)
plt.plot(x_train, tmp1_f_wb, c='b',label='Our Prediction1')
plt.plot(x_train, tmp2_f_wb, c='g',label='Our Prediction2')
plt.plot(x_train, tmp3_f_wb, c='c',label='Our Prediction3')

plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')
plt.title("Housing Prices")
plt.ylabel('Price (in 1000s of dollars)')
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()
"""
不同的w、b得到不同的预测曲线，曲线3的预测结果与真实值最接近
"""