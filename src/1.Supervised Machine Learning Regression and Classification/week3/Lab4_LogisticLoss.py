import numpy as np
# %matplotlib widget
import matplotlib.pyplot as plt
from plt_logistic_loss import  plt_logistic_cost, plt_two_logistic_loss_curves, plt_simple_example
from plt_logistic_loss import soup_bowl, plt_logistic_squared_error
plt.style.use('deeplearning.mplstyle')

# 本节实验将尝试使用线性回归的损失函数（平方误差损失 the squared error loss）来处理分类问题
# 探索为什么不适用于逻辑回归


#在之前的实验中，平方误差损失具有随着成本导数导致最小值的良好性质
soup_bowl()

# 下面我们将尝试使用平方误差损失来处理分类问题，得到其成本的曲面图
# 我们会发现逻辑回归的成本曲线并不像线性回归的那么光滑，会出现非常多的局部最小值
x_train = np.array([0., 1, 2, 3, 4, 5],dtype=np.longdouble)
y_train = np.array([0,  0, 0, 1, 1, 1],dtype=np.longdouble)
plt_simple_example(x_train, y_train)

plt.close('all')
plt_logistic_squared_error(x_train,y_train)
plt.show()

# 所以逻辑回归需要一个更适合其非线性特性的成本函数。下面将从loss函数（即L函数）说起
# loss函数中，当y=0以及y=1时的曲线是不同的。这样就使得预测准确时的成本较小，预测十分不准确时给到很高的成本
# 下图中，左边就是y=1时的Loss函数曲线，右边为y=0时的曲线
# 当y=1时，预测结果越靠近1，成本就越小。预测结果越靠近0，成本就越大。
# 当y=0时，预测结果越靠近1，成本就越大。预测结果越靠近0，成本就越小。
plt_two_logistic_loss_curves()


# 通过这个信心的逻辑损失函数，我们就可以生成一个成本函数。这个成本函数的曲面是光滑的，可以很容易的找到局部最小值，非常适合梯度下降，
plt.close('all')
cst = plt_logistic_cost(x_train,y_train)