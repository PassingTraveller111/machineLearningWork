'''
您将从一个示例数据集开始，该数据集将帮助您直观地了解K-means算法的工作原理。
之后，您将使用K-means算法进行图像压缩，将图像中出现的颜色数量减少到该图像中最常见的颜色。
'''

import numpy as np
import matplotlib.pyplot as plt
from utils import *

'''
下面的函数的功能是根据一组样本 形状为(m,n) 有m个样本样本有n个特征
和一组质心 形状为（k,n）有k个质心 有n个特征
找到每个样本离哪个质心最近 返回一个形状为（m）的数组，idx[i]为第i个样本距离哪个质心最近
'''
def find_closest_centroids(X, centroids):
    """
    Computes the centroid memberships for every example

    Args:
        X (ndarray): (m, n) Input values
        centroids (ndarray): k centroids

    Returns:
        idx (array_like): (m,) closest centroids

    """

    # Set K

    # You need to return the following variables correctly
    idx = np.zeros(X.shape[0], dtype=int)
    ### START CODE HERE ###
    for i in range(X.shape[0]):
        # Array to hold distance between X[i] and each centroids[j]
        distance = []
        min_ij = 0
        for j in range(centroids.shape[0]):
            norm_ij = np.linalg.norm(X[i] - centroids[j]) # 计算样本到每个质心的欧几里德距离
            distance.append(norm_ij)
        idx[i] = np.argmin(distance) # 找到distance列表中距离最小的索引
    ### END CODE HERE ###

    return idx


'''
下面，我们将测试一下我们的函数
'''

# Load an example dataset that we will be using
X = load_data()

print("First five elements of X are:\n", X[:5])
print('The shape of X is:', X.shape)

# Select an initial set of centroids (3 Centroids)
initial_centroids = np.array([[3,3], [6,2], [8,5]])

# Find closest centroids using initial_centroids
idx = find_closest_centroids(X, initial_centroids)

# Print closest centroids for the first three elements
print("First three elements in idx are:", idx[:3])

# UNIT TEST
from public_tests import *

find_closest_centroids_test(find_closest_centroids)


'''
下面，我们将根据每个样本的位置，计算出每一类样本的新质心的每个特征的值
'''
# UNQ_C2
# GRADED FUNCTION: compute_centpods

def compute_centroids(X, idx, K):
    """
    Returns the new centroids by computing the means of the
    data points assigned to each centroid.

    Args:
        X (ndarray):   (m, n) Data points
        idx (ndarray): (m,) Array containing index of closest centroid for each
                       example in X. Concretely, idx[i] contains the index of
                       the centroid closest to example i
        K (int):       number of centroids

    Returns:
        centroids (ndarray): (K, n) New centroids computed
    """

    # Useful variables
    m, n = X.shape

    # You need to return the following variables correctly
    centroids = np.zeros((K, n))

    ### START CODE HERE ###
    for k in range(K):
        points =  points = X[idx == k] # 从样本中取出距离第k个质心的样本
        centroids[k] = centroids[k] = np.mean(points, axis = 0) # 计算新质心的位置
    ### END CODE HERE ##

    return centroids

K = 3
centroids = compute_centroids(X, idx, K)

print("The centroids are:", centroids)

# UNIT TEST
compute_centroids_test(compute_centroids)


'''
下面，我们将尝试实现k-mean算法的实现
'''
def run_kMeans(X, initial_centroids, max_iters=10, plot_progress=False):
    """
    Runs the K-Means algorithm on data matrix X, where each row of X
    is a single example
    """

    # Initialize values
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros(m)

    # Run K-Means
    for i in range(max_iters):

        # Output progress
        print("K-Means iteration %d/%d" % (i, max_iters - 1))

        # For each example in X, assign it to the closest centroid
        # 对每个样本进行归类
        idx = find_closest_centroids(X, centroids)

        # Optionally plot progress
        # 记录移动数据，便于之后绘制
        if plot_progress:
            plot_progress_kMeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids

        # 计算新的质心
        # Given the memberships, compute new centroids
        centroids = compute_centroids(X, idx, K)

    # 根据前面的绘制信息，绘制
    # plt.show()
    plt.close()
    return centroids, idx

'''
进行测试
'''
# Load an example dataset
X = load_data()

# Set initial centroids
initial_centroids = np.array([[3,3],[6,2],[8,5]])
K = 3

# Number of iterations
max_iters = 10

centroids, idx = run_kMeans(X, initial_centroids, max_iters, plot_progress=True)

'''
下面，我们将实现随机初始化质心的位置
将随机取某个样本的特征值作为质心的特征值
'''
def kMeans_init_centroids(X, K):
    """
    This function initializes K centroids that are to be
    used in K-Means on the dataset X

    Args:
        X (ndarray): Data points
        K (int):     number of centroids/clusters

    Returns:
        centroids (ndarray): Initialized centroids
    """

    # Randomly reorder the indices of examples
    # 随机打乱样本的顺序
    randidx = np.random.permutation(X.shape[0])

    # Take the first K examples as centroids
    # 取前k个样本特征作为质心的特征
    centroids = X[randidx[:K]]

    return centroids

'''
下面，我们将进行推广kmeans算法压缩图片的实验
图像的每个像素由三个8位二进制数来指定颜色，即rgb编码
这样每个像素就有2^(8*3)种颜色
我们将颜色压缩到只有16种颜色，这样一来就只需要4位二进制就可以满足
从而实现图片压缩

具体过程：
该图像的每个像素点都视为一个样本，每个样本有三个特征，分布在rgb三维空间当中
我们要通过聚类算法，将之进行分组，分成最佳的16组
最后得到16个质心，就是我们要的16种颜色
'''
# Load an image of a bird
original_img = plt.imread('./images/bird_small.png')

# Visualizing the image
plt.imshow(original_img)
'''
前两个索引标识像素的位置（该图像为横纵各128个像素）
第三个数字即为rgb的值 共有三个
比如：original_img[50, 33, 2] 的意思就是第51行33列的元素的第3个rgba值
'''
print("Shape of original_img is:", original_img.shape) # Shape of original_img is: (128, 128, 3)


'''
因为k-mean算法只能接受2维数组，即（索引，特征）
所以要将两个位置信息转变成像素数
'''
# Divide by 255 so that all values are in the range 0 - 1
# 把所有的值都从0-255 变成了 0-1的值
original_img = original_img / 255

# Reshape the image into an m x 3 matrix where m = number of pixels
# (in this case m = 128 x 128 = 16384)
# 将图像重塑为m x 3矩阵，其中m=像素数
#（在这种情况下，m=128x128=16384）
# Each row will contain the Red, Green and Blue pixel values
# This gives us our dataset matrix X_img that we will use K-Means on.

X_img = np.reshape(original_img, (original_img.shape[0] * original_img.shape[1], 3))

'''
下面，我们将通过k-mean算法找到16个质心，迭代10次
'''
# Run your K-Means algorithm on this data
# You should try different values of K and max_iters here
K = 16
max_iters = 10

# Using the function you have implemented above.
initial_centroids = kMeans_init_centroids(X_img, K)

# Run K-Means - this takes a couple of minutes
centroids, idx = run_kMeans(X_img, initial_centroids, max_iters)

print("Shape of idx:", idx.shape)
print("Closest centroid for the first five elements:", idx[:5])

'''
最后，我们压缩图片
'''

# Represent image in terms of indices
X_recovered = centroids[idx, :]

# Reshape recovered image into proper dimensions
# 把图像重新变成原来的数据结构
X_recovered = np.reshape(X_recovered, original_img.shape)

'''
绘制图片
'''

# Display original image
fig, ax = plt.subplots(1,2, figsize=(8,8))
plt.axis('off')

ax[0].imshow(original_img*255)
ax[0].set_title('Original')
ax[0].set_axis_off()


# Display compressed image
ax[1].imshow(X_recovered*255)
ax[1].set_title('Compressed with %d colours'%K)
ax[1].set_axis_off()