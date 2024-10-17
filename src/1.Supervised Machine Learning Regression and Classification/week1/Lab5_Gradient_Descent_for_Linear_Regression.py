# 本节实验尝试使用梯度算法计算线性回归模型的代价函数的最低值，从而得到一个相对优秀的模型
import math, copy
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('deeplearning.mplstyle')
from lab_utils_uni import plt_house_x, plt_contour_wgrad, plt_divergence, plt_gradients

x_train = np.array([1.0, 2.0])   #features
y_train = np.array([300.0, 500.0])   #target value

# 计算代价
def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i]) ** 2
    total_cost = 1 / (2 * m) * cost

    return total_cost

# 计算梯度，即w以及b的偏导数
def compute_gradient(x, y, w, b):
    m = x.shape[0]
    # 初始化dj_dw（损失函数关于权重w的梯度）和dj_db（损失函数关于偏置b的梯度）为 0。
    dj_dw = 0
    dj_db = 0
    # 批量梯度算法，通过所有的数据集计算当前w、b的平均的梯度，利用了简化后的偏导公式
    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_db += dj_db_i
        dj_dw += dj_dw_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db

# 下面的函数根据当前数据集以及代价计算函数，画出代价函数曲线。根据梯度计算函数，计算出b从-200到200、w从-100到600这个区间的部分梯度
plt_gradients(x_train,y_train, compute_cost, compute_gradient)
# plt.show()
plt.close()


def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    """
    Performs gradient descent to fit w,b. Updates w,b by taking
    num_iters gradient steps with learning rate alpha

    Args:
      x (ndarray (m,))  : Data, m examples （数据集中的输入/特征）
      y (ndarray (m,))  : target values （数据集中的输出）
      w_in,b_in (scalar): initial values of model parameters （模型参数的初始值）
      alpha (float):     Learning rate （学习率）
      num_iters (int):   number of iterations to run gradient descent (运行梯度下降的迭代次数)
      cost_function:     function to call to produce cost
      gradient_function: function to call to produce gradient

    Returns:
      w (scalar): Updated value of parameter after running gradient descent
      b (scalar): Updated value of parameter after running gradient descent
      J_history (List): History of cost values （迭代过程中代价的值）
      p_history (list): History of parameters [w,b] （迭代过程中w、b的值）
      """

    w = copy.deepcopy(w_in)  # 避免修改全局范围上的变量w
    # 下面两个数组是为了存储计算过程中迭代的w和b以及代价的值，用于之后画图
    J_history = []
    p_history = []
    b = b_in
    w = w_in
    # 下面是计算的过程，由我们来定迭代的次数（num_iters）
    for i in range(num_iters):
        # 计算当前w、b的梯度
        dj_dw, dj_db = gradient_function(x, y, w, b)

        # 根据当前梯度、学习率，更新b和w的值，这里用的是一起更新，注意不要交叉更新
        b = b - alpha * dj_db
        w = w - alpha * dj_dw

        # 存储当前迭代的w、b的值以及代价
        if i < 100000:  # 防止内存泄漏
            J_history.append(cost_function(x, y, w, b))
            p_history.append([w, b])
        # 每隔10次迭代打印一次成本以及当前的梯度和w、b的值
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")

    return w, b, J_history, p_history  # return w and J,w history for graphing

# initialize parameters
w_init = 0
b_init = 0
# 设置迭代次数和学习率
iterations = 10000
tmp_alpha = 1.0e-2
# run gradient descent
w_final, b_final, J_hist, p_hist = gradient_descent(x_train ,y_train, w_init, b_init, tmp_alpha,
                                                    iterations, compute_cost, compute_gradient)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")

# 绘制代价的迭代过程，代价在这个迭代的过程中和预期的一样不断变小
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
ax1.plot(J_hist[:100])
ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
ax1.set_title("Cost vs. iteration(start)");  ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost')            ;  ax2.set_ylabel('Cost')
ax1.set_xlabel('iteration step')  ;  ax2.set_xlabel('iteration step')
# plt.show()
plt.close()
# 下面，我们就可以通过最终的w、b来进行房价预测
print(f"1000 sqft house prediction {w_final*1.0 + b_final:0.1f} Thousand dollars")
print(f"1200 sqft house prediction {w_final*1.2 + b_final:0.1f} Thousand dollars")
print(f"2000 sqft house prediction {w_final*2.0 + b_final:0.1f} Thousand dollars")

# 下面的例子是当学习率过大时，代价无法收敛，越来越大的情况
# initialize parameters
w_init = 0
b_init = 0
# set alpha to a large value
iterations = 10
tmp_alpha = 8.0e-1
# run gradient descent
w_final, b_final, J_hist, p_hist = gradient_descent(x_train ,y_train, w_init, b_init, tmp_alpha,
                                                    iterations, compute_cost, compute_gradient)
plt_divergence(p_hist, J_hist,x_train, y_train)
plt.show()