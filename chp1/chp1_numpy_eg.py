# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 16:11:29 2024

@author: lich5
Numpy 练习题
"""
import numpy as np
#%% 数组的创建与基本操作
# 创建一个长度为10的全零数组，并将第五个元素设为1。
array = np.zeros(10)
array[4] = 1

# 创建一个包含0到9的数组。
array = range(10)
array = np.arange(10)
# 创建一个3x3的单位矩阵。
array = np.eye(3)
# 创建一个3x3x3的随机数组。
array = np.random.rand(3,3,3)

# 创建一个10到50的数组，并反转数组顺序。
array = np.arange(10, 51)
array[::-1]

#%% 数组索引与切片
# 创建一个3x3矩阵，并提取其中的非零元素。
array = np.random.rand(3,3)
nonzero = np.where(array !=0)[0]

# 创建一个5x5的随机矩阵，并将矩阵的最大值设为1，最小值设为0。
array = np.random.rand(5,5)
max_array = np.max(array)
min_array = np.min(array)
array = np.where(array == max_array, 1, array)
array = np.where(array == min_array, 0, array)

# 创建一个0到99的10x10的矩阵，并提取其边界值。
array = np.arange(100).reshape(10,10)
# array = np.reshape(array, (10,10))
top_row = array[0,:]
bottom_row = array[-1,:]
left_col = array[:,0]
right_col = array[:,-1]
boundry = np.concatenate((top_row,
                          bottom_row,
                          left_col[1:-1],
                          right_col[1:-1])
                         )
# 创建一个长度为10的随机数组，并将其元素按升序排序。

#%% 统计计算
# 设置随机种子并生成一个长度为5的随机数组
np.random.seed(1235)
array = np.random.randn(10)
print(array)
# 创建一个100个随机数的数组，计算其平均值、中位数、方差、标准差。
array = np.random.randn(100)
mean = np.mean(array)
mean = array.mean()
median = np.median(array)
var = np.var(array)
std = np.std(array)

# 创建一个100个随机数的数组，计算其前5个元素的累积和。
cumulative_summary = np.cumsum(array)[:5]

# 创建一个10x10的随机矩阵，计算其列的最大值与最小值。
m10x10 = np.random.randn(10,10)
col_max = np.max(m10x10, axis = 1)

# 创建一个长度为50的随机数组，统计其中大于0的元素个数。

# 创建一个10x10的随机矩阵，统计其中大于0.5的元素个数。

# 使用numpy生成一个1000x1000的随机矩阵，并计算其行均值和列均值。

# 创建一个包含1-100数字的1000个随机整数的数组，计算每个数值的累计出现次数。

#%% 数学操作与线性代数
# 创建两个随机数组，分别计算它们的和、差、积、商。

# 生成两个1000x100的随机矩阵，计算它们的矩阵乘法。

# 创建一个5x5的随机矩阵，并计算其行列式。

# 创建一个3x3的随机矩阵，并计算其逆矩阵。

# 创建一个长度为5的随机数组，并计算其平方和。

# 创建两个随机向量，计算它们的点积。

# 创建一个20x20的随机矩阵，并计算其奇异值分解（SVD）。

#%% 广播与形状操作
# 创建一个形状为 (3, 1) 和 (1, 3) 的两个数组，并使用广播机制计算它们的和。

# 创建一个随机数组，将其形状重塑为 (2, -1)。

# 创建一个4x4的随机矩阵，旋转90度后，再进行转置

# 创建一个随机数组，水平和垂直堆叠该数组的副本。

# 创建一个长度为10的随机数组，并添加一个新的维度，使其形状变为 (10, 1)。

#%% 特殊数组生成
# 创建一个5x5的棋盘矩阵（即0和1交替排列的矩阵）。

# 创建一个长度为10的数组，元素值从0到1均匀分布。

# 创建一个3x3的随机矩阵，并将其对角线元素设为1。

# 创建一个3x3的随机矩阵，并将其下三角元素设为0。

# 创建一个4x4的随机矩阵，并将其上三角元素设为1。
