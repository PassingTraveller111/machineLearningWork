import numpy as np    # it is an unofficial standard to use np for numpy
import time
'''
本节主要学习对矢量以及矩阵的创建以及操作
'''
from prompt_toolkit.shortcuts import clear

# 3.3 Vector Creation
# NumPy routines which allocate memory and fill arrays with value
'''
numpy.zeros(shape, dtype=float, order='C')
shape：表示数组的形状，可以是一个整数，表示一维数组的长度；也可以是一个元组，表示多维数组的维度。例如，(2, 3)表示创建一个 2 行 3 列的二维数组。
dtype：可选参数，指定数组的数据类型，默认为float（浮点数）。可以是 NumPy 支持的任何数据类型，如int、float、bool等。
order：可选参数，指定数组在内存中的存储顺序，默认为 'C'（C 风格，行优先）。也可以是 'F'（Fortran 风格，列优先）。
'''
a = np.zeros(4);                print(f"np.zeros(4) :   a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.zeros((4,));             print(f"np.zeros(4,) :  a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
'''
numpy.random.random_sample(size=None)
size：可选参数，表示生成的随机数的形状。可以是一个整数，表示一维数组的长度；也可以是一个元组，表示多维数组的维度。如果不指定size，则返回一个随机浮点数。
'''
a = np.random.random_sample(4); print(f"np.random.random_sample(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

# NumPy routines which allocate memory and fill arrays with value but do not accept shape as input argument
'''
在 NumPy 中，np.arange()是一个用于创建数值序列的函数。
numpy.arange(start, stop, step, dtype=None)
start：起始值，默认为 0。包含在生成的序列中。
stop：终止值，生成的序列不包含该值。
step：步长，默认为 1。
dtype：指定输出数组的数据类型。如果未指定，则根据输入值推断数据类型。
'''
a = np.arange(4.);              print(f"np.arange(4.):     a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
'''
在 NumPy 中，np.random.rand()用于生成随机数。
numpy.random.rand(d0, d1,..., dn)
参数表示生成的随机数数组的维度。如果不提供参数，则生成一个介于 0 和 1 之间的随机浮点数。
如果提供一个整数参数d0，则生成一个一维数组，长度为d0。
如果提供多个整数参数d0, d1,..., dn，则生成一个n维数组，形状为(d0, d1,..., dn)。
'''
a = np.random.rand(4);          print(f"np.random.rand(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

# NumPy routines which allocate memory and fill with user specified values
'''
在 NumPy 中，np.array()是用于创建数组的函数。
numpy.array(object, dtype=None, copy=True, order='K', subok=False, ndmin=0)
object：可以是一个列表、元组、数组或者其他可迭代对象，用于初始化数组的值。
dtype：可选参数，指定数组的数据类型。如果未指定，则根据输入数据自动推断。
copy：可选参数，指定是否复制输入数据。默认为True，表示复制数据；如果为False，则可能会共享输入数据的内存。
order：可选参数，指定数组在内存中的存储顺序。可以是 'C'（C 风格，行优先）、'F'（Fortran 风格，列优先）或 'K'（保持输入数据的顺序，如果输入数据是 Fortran 连续的，则为 'F'，否则为 'C'）。
subok：可选参数，指定是否允许返回的数组是子类。默认为False，表示返回的数组是基类数组；如果为True，则可能返回子类数组。
ndmin：可选参数，指定数组的最小维度。如果输入数据的维度小于ndmin，则会在前面添加维度，使得数组的维度至少为ndmin。
'''
a = np.array([5,4,3,2]);  print(f"np.array([5,4,3,2]):  a = {a},     a shape = {a.shape}, a data type = {a.dtype}")
# 因为数组中存在浮点数，所以整个数组的数据类型被提升为浮点数类型。
a = np.array([5.,4,3,2]); print(f"np.array([5.,4,3,2]): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

# 3.4 Operations on Vectors

# 3.4.1 Indexing
#vector indexing operations on 1-D vectors
# 创建一个0-9的数组
a = np.arange(10)
print(a)

#access an element
# 访问元素
print(f"a[2].shape: {a[2].shape} a[2]  = {a[2]}, Accessing an element returns a scalar")

# access the last element, negative indexes count from the end
# 下标为负数的元素，会循环访问
print(f"a[-1] = {a[-1]}")

#indexs must be within the range of the vector or they will produce and error
# 越界访问会报错
try:
    c = a[10]
except Exception as e:
    print("The error message you'll see is:")
    print(e)

# 3.4.2 Slicing
#vector slicing operations
# 下面是对数组的切片操作 通过:的三个数可以快速对数组进行切片
a = np.arange(10)
print(f"a         = {a}")

#access 5 consecutive elements (start:stop:step)
# 分号的三个元素分别为（开始：结束：步长）
c = a[2:7:1];     print("a[2:7:1] = ", c)

# access 3 elements separated by two
c = a[2:7:2];     print("a[2:7:2] = ", c)

# access all elements index 3 and above
# 从三开始的所有元素，结束部分不写，默认为数组长度。step不写，默认为1
c = a[3:];        print("a[3:]    = ", c)

# access all elements below index 3
# 到3为止的所有元素，开始部分不写，默认为0
c = a[:3];        print("a[:3]    = ", c)

# access all elements
c = a[:];         print("a[:]     = ", c)


# 3.4.3 Single vector operations
# 对一个向量的一些操作
a = np.array([1,2,3,4])
print(f"a             : {a}")
# negate elements of a
# 对所有的元素取负数
b = -a
print(f"b = -a        : {b}")

# sum all elements of a, returns a scalar
# 算总和
b = np.sum(a)
print(f"b = np.sum(a) : {b}")
# 算平均数
b = np.mean(a)
print(f"b = np.mean(a): {b}")
# 算平方
b = a**2
print(f"b = a**2      : {b}")


# 3.4.4 Vector Vector element-wise operations
# 两个数组相加，其实就是向量相加的规则
a = np.array([ 1, 2, 3, 4])
b = np.array([-1,-2, 3, 4])
print(f"Binary operators work element wise: {a + b}")

#try a mismatched vector operation
# 两个向量的长度要匹配
c = np.array([1, 2])
try:
    d = a + c
except Exception as e:
    print("The error message you'll see is:")
    print(e)

# 3.4.5 Scalar Vector operations
a = np.array([1, 2, 3, 4])

# multiply a by a scalar
# 向量的乘法
b = 5 * a
print(f"b = 5 * a : {b}")


# 3.4.6 Vector Vector dot product
# 向量点积

# 方法一 用循环去计算点积
def my_dot(a, b):
    """
   Compute the dot product of two vectors

    Args:
      a (ndarray (n,)):  input vector
      b (ndarray (n,)):  input vector with same dimension as a

    Returns:
      x (scalar):
    """
    x = 0
    for i in range(a.shape[0]):
        x = x + a[i] * b[i]
    return x
# test 1-D
a = np.array([1, 2, 3, 4])
b = np.array([-1, 4, 3, 2])
# 该方法成功计算了a和b的点积
print(f"my_dot(a, b) = {my_dot(a, b)}")

# 方法二，我们可以使用np.dot计算几个向量的点积
# test 1-D
a = np.array([1, 2, 3, 4])
b = np.array([-1, 4, 3, 2])
c = np.dot(a, b)
print(f"NumPy 1-D np.dot(a, b) = {c}, np.dot(a, b).shape = {c.shape} ")
c = np.dot(b, a)
print(f"NumPy 1-D np.dot(b, a) = {c}, np.dot(a, b).shape = {c.shape} ")

# 3.4.7 The Need for Speed: vector vs for loop
# 下面，我们比较一下两种方法的介计算速度

np.random.seed(1)
a = np.random.rand(10000000)  # very large arrays
b = np.random.rand(10000000)

tic = time.time()  # capture start time 捕获开始时间
c = np.dot(a, b)
toc = time.time()  # capture end time 捕获结束时间

print(f"np.dot(a, b) =  {c:.4f}")
print(f"Vectorized version duration: {1000*(toc-tic):.4f} ms ")

tic = time.time()  # capture start time
c = my_dot(a,b)
toc = time.time()  # capture end time

print(f"my_dot(a, b) =  {c:.4f}")
print(f"loop version duration: {1000*(toc-tic):.4f} ms ")

del(a);del(b)  #remove these big arrays from memory

# 从比较结果我们可以看到，向量的点积算法远快于使用循环。
# 这是因为python是单线程，只能串行执行代码。
# 但是np.dot可以并行计算，所以大大提高了计算效率，所以如果在之后的实验中碰到大量数据集的计算操作，使用这个方法计算能够大大提高效率

# 4 Matrices 矩阵

# 4.3 Matrix Creation 创造矩阵

a = np.zeros((1, 5)) # 1维5个元素
print(f"a shape = {a.shape}, a = {a}")

a = np.zeros((2, 1)) # 2维，每维1个元素
print(f"a shape = {a.shape}, a = {a}")

a = np.random.random_sample((1, 1))
print(f"a shape = {a.shape}, a = {a}")

# NumPy routines which allocate memory and fill with user specified values
a = np.array([[5], [4], [3]]);   print(f" a shape = {a.shape}, np.array: a = {a}")
a = np.array([[5],   # One can also
              [4],   # separate values
              [3]]); #into separate rows
print(f" a shape = {a.shape}, np.array: a = {a}")

# 4.4 Operations on Matrices 矩阵的操作

# 4.4.1 Indexing 索引

#vector indexing operations on matrices
'''
np.arange(6)创建一个包含从 0 到 5 的整数序列的一维数组，即[0, 1, 2, 3, 4, 5]。
.reshape(-1, 2)将这个一维数组重新塑造成一个二维数组，其中第二个参数2表示新数组的列数为 2。
而参数-1表示让 NumPy 根据给定的列数自动计算行数。
'''
a = np.arange(6).reshape(-1, 2)   #reshape is a convenient way to create matrices
print(f"a.shape: {a.shape}, \na= {a}")

#access an element 访问元素
print(f"\na[2,0].shape:   {a[2, 0].shape}, a[2,0] = {a[2, 0]},     type(a[2,0]) = {type(a[2, 0])} Accessing an element returns a scalar\n")

#access a row 访问某行
print(f"a[2].shape:   {a[2].shape}, a[2]   = {a[2]}, type(a[2])   = {type(a[2])}")

# 4.4.2 Slicing 切片

#vector 2-D slicing operations
a = np.arange(20).reshape(-1, 10) # 1行10个，自动切成两行
print(f"a = \n{a}")

#access 5 consecutive elements (start:stop:step) 第一个元素表示对第几行进行切片，对第0行进行切片
print("a[0, 2:7:1] = ", a[0, 2:7:1], ",  a[0, 2:7:1].shape =", a[0, 2:7:1].shape, "a 1-D array")

#access 5 consecutive elements (start:stop:step) in two rows 第一个元素不填，表示对所有行进行切片
print("a[:, 2:7:1] = \n", a[:, 2:7:1], ",  a[:, 2:7:1].shape =", a[:, 2:7:1].shape, "a 2-D array")

# access all elements 访问所有元素
print("a[:,:] = \n", a[:,:], ",  a[:,:].shape =", a[:,:].shape)

# access all elements in one row (very common usage) 访问第一行的所有元素
print("a[1,:] = ", a[1,:], ",  a[1,:].shape =", a[1,:].shape, "a 1-D array")
# same as 等价于这个操作
print("a[1]   = ", a[1],   ",  a[1].shape   =", a[1].shape, "a 1-D array")