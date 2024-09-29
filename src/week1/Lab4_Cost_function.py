"""
这节实验尝试通过代价函数，得到关于wb的代价曲线
"""
import numpy as np
# %matplotlib widget
import matplotlib.pyplot as plt
# 预先定义了一些函数
from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl
plt.style.use('./deeplearning.mplstyle')
# 定义数据集
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
# 代价函数是用来衡量模型使用不同的系数w、b等，在当前数据集下的准确度 J(w,b) 具体公式可以看思维导图

def compute_cost(x, y, w, b):
    """
    Computes the cost function for linear regression.
    用于计算线性回归的代价函数
    Args:
      x (ndarray (m,)): Data, m examples
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters

    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # number of training examples
    m = x.shape[0]

    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum = cost_sum + cost
    total_cost = (1 / (2 * m)) * cost_sum

    return total_cost

# 下面将用事先定义的函数绘制 关于w的该数据集的代价函数的线性图，b默认为100，想要绘制该图，就解除下面的注释
# plt_intuition(x_train, y_train)

# 只有一个系数时，代价函数的图像是线性图，但如果是关于w、b两个系数的代价函数的图，就将是三维的，需要用3D的形式呈现

x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480,  430,   630, 730,])
# 下面绘制了关于w、b的一个代价函数的3D图，在图的最低点，也就是代价最小的点，此时模型预测的代价/准确率最低
plt.close('all')
fig, ax, dyn_items = plt_stationary(x_train, y_train)
updater = plt_update_onclick(fig, ax, x_train, y_train, dyn_items)
soup_bowl()