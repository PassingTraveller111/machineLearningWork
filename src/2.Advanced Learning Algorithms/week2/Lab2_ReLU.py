import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.style.use('./deeplearning.mplstyle')
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.activations import linear, relu, sigmoid
# %matplotlib widget
from matplotlib.widgets import Slider
from lab_utils_common import dlc
from autils import plt_act_trio
from lab_utils_relu import *
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

# 下面是三种常见的激活函数的函数图像
plt_act_trio()
# ReLU提供了一个连续的线性关系。此外，它还有一个输出为零的“关闭”范围。“关闭”功能使ReLU成为非线性激活。
# 下面的图例解释了ReLU的优势（虽然我没太看懂....）
_ = plt_relu_ex()