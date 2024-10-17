import numpy as np
# %matplotlib widget
import matplotlib.pyplot as plt
from plt_one_addpt_onclick import plt_one_addpt_onclick
from lab_utils_common import draw_vthresh
plt.style.use('deeplearning.mplstyle')

# Input is an array.
'''
np.exp 用于计算输入数组中每个元素，以e为底数，元素为指数的数值，即e^i,i为元素。
np.exp(arr)，其中arr是输入的数组或数值。
import numpy as np

arr = np.array([1, 2, 3])
print(np.exp(arr))  # [ 2.71828183  7.3890561  20.08553692]

single_value = 5
print(np.exp(single_value))  # 148.4131591025766
'''
input_array = np.array([1,2,3])
exp_array = np.exp(input_array)

print("Input to exp:", input_array)
print("Output of exp:", exp_array)

# Input is a single number
input_val = 1
exp_val = np.exp(input_val)

print("Input to exp:", input_val)
print("Output of exp:", exp_val)

# 下面是激活函数的实现
def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z

    """
    # 激活函数就需要依赖与np.exp的计算
    g = 1 / (1 + np.exp(-z))

    return g

# 生成从-10到10的数
# Generate an array of evenly spaced values between -10 and 10
z_tmp = np.arange(-10,11)
# 用激活函数依次生成从-10到10的激活函数的输出值
# Use the function implemented above to get the sigmoid values
y = sigmoid(z_tmp)

# Code for pretty printing the two arrays next to each other
np.set_printoptions(precision=3)
print("Input (z), Output (sigmoid(z))")
# 将z_temp和y按列拼接起来
print(np.c_[z_tmp, y])

# 下面绘制了激活函数g的值关于z的图像
# 当z变大为负值时，sigmoid函数趋近于0，当z增大为正值时，sigmoid函数趋近于1
# Plot z vs sigmoid(z)
fig,ax = plt.subplots(1,1,figsize=(5,3))
ax.plot(z_tmp, y, c="b")

ax.set_title("Sigmoid function")
ax.set_ylabel('sigmoid(z)')
ax.set_xlabel('z')
draw_vthresh(ax,0)
plt.show()
