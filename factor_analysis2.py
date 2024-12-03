import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import FactorAnalysis
# plt.rcParams['font.family'] = 'SimHei' 
# 读取数据 
# Read the Excel file using pandas
file_path = 'examp4_1.xls'
data = pd.read_excel(file_path, header=None)

# Extract the data and variable names
X = data.iloc[4:, 2:].values  # Data matrix, excluding the first 4 columns
varname = data.iloc[3, 2:].values  # Variable names (from row 4, columns 3 and onward)
obsname = data.iloc[4:, 1].values  # Observation names (from row 5, column 2 onward)

#%% 调用FactorAnalysis函数根据原始观测数据作因子分析 
# 从原始数据（实质还是相关系数矩阵）出发，进行因子分析，公共因子数为4
fa_4_factors = FactorAnalysis(n_components=4)
fa_4_factors.fit(X)
lambda_4 = fa_4_factors.components_.T  # Factor loadings (lambda)
psi_4 = fa_4_factors.noise_variance_  # Uniqueness (psi)

# Calculate contribution of each factor
contribut_4 = 100 * np.sum(lambda_4 ** 2, axis=0) / 8
cum_cont_4 = np.cumsum(contribut_4)

print("Contribution (4 factors):", contribut_4)
print("Cumulative Contribution (4 factors):", cum_cont_4)

#%% 从原始数据（实质还是相关系数矩阵）出发，进行因子分析，公共因子数为2
fa_2_factors = FactorAnalysis(n_components=2)
fa_2_factors.fit(X)
lambda_2 = fa_2_factors.components_.T  # Factor loadings (lambda)
psi_2 = fa_2_factors.noise_variance_  # Uniqueness (psi)
factor_scores_2 = fa_2_factors.transform(X)  # Factor scores (F)

# Calculate contribution of each factor
contribut_2 = 100 * np.sum(lambda_2 ** 2, axis=0) / 8
cum_cont_2 = np.cumsum(contribut_2)

print("Contribution (2 factors):", contribut_2)
print("Cumulative Contribution (2 factors):", cum_cont_2)

# Display factor loadings and names
factor_loading_df = pd.DataFrame(np.column_stack([varname, lambda_2]), columns=['Variable', 'Factor 1', 'Factor 2'])
print(factor_loading_df)

#%% 将因子得分F分别按耐力因子得分和速度因子得分进行排序 
# Combine observation names and factor scores
obsF = np.column_stack([obsname, factor_scores_2])
# Sort by endurance factor (first factor)
F1_sorted = obsF[np.argsort(obsF[:, 1].astype(float)), :]
# Sort by speed factor (second factor)
F2_sorted = obsF[np.argsort(obsF[:, 2].astype(float)), :]

# Prepare the headers
head = ['国家/地区', '耐力因子', '速度因子']
result1 = np.vstack([head, F1_sorted])
result2 = np.vstack([head, F2_sorted])

# Display results
print("按耐力因子得分排序:")
print(result1)
print("按速度因子得分排序:")
print(result2)

#%% 绘制因子得分负值的散点图 
plt.rcParams['font.family'] = 'SimHei' # SimSun
plt.figure()
plt.scatter(-factor_scores_2[:, 0], -factor_scores_2[:, 1], color='black')

for i, txt in enumerate(obsname):
    plt.text(-factor_scores_2[i, 0], -factor_scores_2[i, 1], txt, fontsize=8)

plt.xlabel('耐力因子得分（负值）')
plt.ylabel('速度因子得分（负值）')

plt.show()
# [EOF]
