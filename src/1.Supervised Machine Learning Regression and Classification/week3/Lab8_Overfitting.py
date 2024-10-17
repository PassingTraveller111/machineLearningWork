# 在本节实验将学习什么情况会发生过拟合，以及一些解决方案
# %matplotlib widget
import matplotlib.pyplot as plt
from IPython.core.display_functions import display
from ipywidgets import Output
from plt_overfit import overfit_example, output
plt.style.use('deeplearning.mplstyle')

plt.close("all")
display(output) # display是jupyter的一个包
ofit = overfit_example(False)