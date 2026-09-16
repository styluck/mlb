# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 16:32:34 2024

@author: lich5
"""
import numpy as np

#%% 综合练习

# 1. 创建一个10x10的随机矩阵，归一化其所有元素到0到1之间
# 使用rand直接生成0-1之间的随机数，或用其他随机函数生成后归一化
m = np.random.rand(10, 10)
max_ = np.max(m)
min_ = np.min(m)
m = m/(max_ - min_)

# 2. 创建一个5x5的随机矩阵，找到其最大值的索引
m5 = np.random.rand(5, 5)
max_5 = np.max(m5)
i = np.where(m5 == max_5)

# 3. 创建一个10x3的随机矩阵，找出每行的最大值及其索引
matrix10x3 = np.random.rand(10, 3)
row_max_values = matrix10x3.max(axis=1)
row_max_indices = matrix10x3.argmax(axis=1)
print("3. 10x3矩阵每行的最大值及其索引:")
for i in range(10):
    print(f"行{i}: 最大值={row_max_values[i]:.4f}, 索引={row_max_indices[i]}")
print()

# 4. 创建一个长度为20的随机数组，找出其第二大的元素
array20 = np.random.randn(20)
# 方法1: 排序后取倒数第二个
sorted_array = np.sort(array20)
second_largest1 = sorted_array[-2]
# 方法2: 更高效的方式，不完整排序
second_largest2 = np.partition(array20, -2)[-2]
print(f"4. 长度为20的数组中第二大的元素: {second_largest1:.4f}")
print("两种方法结果一致:", np.isclose(second_largest1, second_largest2), "\n")

# 5. 创建一个3x3的随机矩阵，将其转换为仅包含0和1的矩阵（根据某个自定义阈值）
matrix3x3 = np.random.rand(3, 3)
threshold = 0.5  # 自定义阈值
binary_matrix = (matrix3x3 > threshold).astype('int')
print(f"5. 3x3矩阵转换为0-1矩阵(阈值={threshold}):")
print("原始矩阵:")
print(matrix3x3)
print("转换后:")
print(binary_matrix, "\n")

# 6. 创建一个包含1000个元素的数组，将其中的偶数替换为-1
array1000 = np.random.randint(0, 100, 1000)  # 生成0-99的随机整数
array1000[array1000 % 2 == 0] = -1  # 偶数替换为-1
print(f"6. 1000个元素的数组中，替换后的前10个元素: {array1000[:10]}")
print("验证替换结果 - 前10个中是否有偶数:", np.any(array1000[:10] % 2 == 0), "\n")

# 7. 创建一个5x5的随机矩阵，并将其中的奇数行逆序排列
# 注意：Python索引从0开始，所以奇数行指的是索引为1,3...的行
matrix5x5_2 = np.random.rand(5, 5)
# 对奇数行(1,3)进行逆序
matrix5x5_2[0::2, :] = matrix5x5_2[0::2, ::-1]
print("7. 奇数行逆序后的5x5矩阵:")
print(matrix5x5_2, "\n")

# 8. 创建一个长度为10的数组，查找数组中连续大于0.5的元素段
array10 = np.random.rand(10)
# 找到大于0.5的位置
mask = array10 > 0.5
# 找到连续True的起始和结束索引
edges = np.diff(np.concatenate(([False], mask, [False])))
segments = np.where(edges)[0].reshape(-1, 2)
print(f"8. 长度为10的数组: {array10.round(4)}")
print("连续大于0.5的元素段(起始索引, 结束索引):")
for start, end in segments:
    print(f"({start}, {end-1}) 值: {array10[start:end].round(4)}")
print()

# 9. 使用numpy计算两个随机数组之间的欧氏距离
array_a = np.random.rand(10)
array_b = np.random.rand(10)
# 方法1: 使用线性代数模块
euclidean_dist1 = np.linalg.norm(array_a - array_b)
# 方法2: 手动计算
euclidean_dist2 = np.sqrt(np.sum((array_a - array_b) **2))
print(f"9. 两个数组之间的欧氏距离: {euclidean_dist1:.4f}")
print("两种方法结果一致:", np.isclose(euclidean_dist1, euclidean_dist2), "\n")

# 10. 生成一个10x10的随机矩阵，查找其局部最大值（即比周围八个元素都大的值）
matrix10x10_2 = np.random.rand(10, 10)
local_max = np.zeros_like(matrix10x10_2, dtype=bool)

# 检查每个非边界元素是否为局部最大值
for i in range(1, 9):
    for j in range(1, 9):
        # 取出当前元素及其周围8个元素
        neighbors = matrix10x10_2[i-1:i+2, j-1:j+2]
        # 如果中心元素大于所有邻居，则是局部最大值
        if matrix10x10_2[i, j] == np.max(neighbors):
            local_max[i, j] = True

# 获取局部最大值的坐标和值
max_coords = np.where(local_max)
print("10. 10x10矩阵中的局部最大值:")
for i, j in zip(max_coords[0], max_coords[1]):
    print(f"位置({i}, {j}): 值={matrix10x10_2[i, j]:.4f}")

