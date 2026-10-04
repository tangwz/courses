---
course: "evaluating-synthetic-data-quality"
chapter: "advanced-statistical-fidelity"
lesson: "practice-implementing-multivariate-tests"
sourceId: 5659
sourceUrl: "https://apxml.com/zh/courses/evaluating-synthetic-data-quality/chapter-2-advanced-statistical-fidelity/practice-implementing-multivariate-tests"
title: "动手实践：实现多变量检验"
description: "Python 中多变量分布比较检验的代码实现。"
order: 6
plots: ["plots/5659-0.json"]
sourceHash: "d37237784ddd61611192f4e4b38a3c7e72d7578cd1a24f9f395afea10197a4ae"
sourceCorrections: []
---

通过 Python 动手实现多种多变量检验，用以评估统计保真度。比较模拟的“真实”数据集和“合成”数据集，以演示这些检验的运作方式和结果解释。

回顾第 1 章，您应该已经准备好一个 Python 环境，包含 NumPy、SciPy、Pandas，以及可选的 Scikit-learn 和 Matplotlib/Seaborn 等库。

### 设置：生成样本数据

首先，导入所需的库并生成一些简单的多变量数据。我们将创建一个服从多变量正态分布的“真实”数据集，以及一个可能略有不同（例如，均值、协方差或分布类型不同）的“合成”数据集，以演示这些检验。

```python
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics.pairwise import rbf_kernel
import plotly.graph_objects as go
import json # 用于嵌入 Plotly JSON

# 设置随机种子以保证结果可重现
np.random.seed(42)

# 生成“真实”数据（例如，二维正态分布）
mean_real = [0, 0]
cov_real = [[1, 0.5], [0.5, 1]] # 变量之间存在相关性
n_samples_real = 200
real_data = np.random.multivariate_normal(mean_real, cov_real, n_samples_real)
real_df = pd.DataFrame(real_data, columns=['Feature1', 'Feature2'])

# 生成“合成”数据（均值和协方差略有不同）
mean_synth = [0.2, -0.1] # 均值略有偏移
cov_synth = [[1.1, 0.45], [0.45, 0.9]] # 协方差略有不同
n_samples_synth = 200
synthetic_data = np.random.multivariate_normal(mean_synth, cov_synth, n_samples_synth)
synthetic_df = pd.DataFrame(synthetic_data, columns=['Feature1', 'Feature2'])

# 生成另一组“合成”数据（不同分布类型 - 例如，均匀噪声）
# 优秀的检验应该能轻易区分这一点
synthetic_data_uniform = np.random.rand(n_samples_synth, 2) * 4 - 2 # 在 -2 和 2 之间均匀分布
synthetic_df_uniform = pd.DataFrame(synthetic_data_uniform, columns=['Feature1', 'Feature2'])

print("真实数据样本:")
print(real_df.head())
print("\n合成数据（正态）样本:")
print(synthetic_df.head())
print("\n合成数据（均匀）样本:")
print(synthetic_df_uniform.head())
```

现在我们有三个数据集：`real_df`、`synthetic_df`（分布相似）和 `synthetic_df_uniform`（分布差异很大）。让我们应用一些检验。

### 比较均值向量 (vector)：Hotelling's T² 检验

如前所述，逐特征比较均值会忽略相关性。Hotelling's T² 检验比较两个多变量样本的均值向量。它是 t 检验的多变量推广。零假设 ($H_0$) 是两个样本具有相同的均值向量。

该检验假设各组之间具有多变量正态性和相等的协方差矩阵，尽管它具有一定的健壮性，特别是在样本量较大时。`scipy` 等库没有直接实现此检验，但我们可以使用外部库或自己实现其核心思想。对于本次实践，更简单的方法通常可以在专门的统计软件包中找到，或者可以进行近似计算。

为演示目的，我们假设有一个函数 `hotelling_t2(X, Y)` 返回 T² 统计量和 p 值。

```python
# 函数调用（实现通常需要专用库
# 或基于逆协方差矩阵的手动计算）
# T2_statistic, p_value = hotelling_t2(real_data, synthetic_data)
# print(f"Hotelling T2 检验（真实 vs. 合成正态）：统计量={T2_statistic:.4f}，p 值={p_value:.4f}")

# T2_statistic_uniform, p_value_uniform = hotelling_t2(real_data, synthetic_data_uniform)
# print(f"Hotelling T2 检验（真实 vs. 合成均匀）：统计量={T2_statistic_uniform:.4f}，p 值={p_value_uniform:.4f}")

# 由于 scipy 没有直接提供，我们将跳过执行
# 但会描述其解释。

# 解释：
# 较小的 p 值（例如 < 0.05）表明拒绝 H0，这意味着
# 均值向量之间存在统计学上的显著差异。
# 对于我们的 'real_data' vs 'synthetic_data'，我们可能会预期 p 值 > 0.05
# 这是因为均值差异相对于方差较小。
# 对于 'real_data' vs 'synthetic_data_uniform'，我们预期 p 值会非常小
# 表明均值向量显著不同。
```

尽管 Hotelling's T² 检验在比较均值方面很有效，但它并未说明整体分布形状或高阶矩。

### 比较分布：最大均值差异 (MMD)

MMD 是一种非参数 (parameter)度量，它计算两个分布在再生核希尔伯特空间 (RKHS) 中的嵌入 (embedding)之间的距离。MMD 值越小，表示分布之间越相似。它特别有用，因为它不对底层数据分布做出严格假设。

我们可以使用核函数计算 MMD 的经验估计值，常用的是径向基函数 (RBF) 核：$k(x, y) = \exp(-\gamma ||x - y||^2)$。

```python
def mmd_rbf(X, Y, gamma=1.0):
    """
    计算样本 X 和 Y 之间的最大均值差异 (MMD)
    使用径向基函数 (RBF) 核。

    参数：
        X (np.ndarray): 第一个样本 (n_samples_x, n_features)
        Y (np.ndarray): 第二个样本 (n_samples_y, n_features)
        gamma (float): RBF 核参数。

    返回：
        float: 计算出的 MMD 统计量（平方）。
    """
    if X.shape[1] != Y.shape[1]:
        raise ValueError("样本 X 和 Y 必须具有相同数量的特征。")

    K_XX = rbf_kernel(X, X, gamma=gamma)
    K_YY = rbf_kernel(Y, Y, gamma=gamma)
    K_XY = rbf_kernel(X, Y, gamma=gamma)

    n = K_XX.shape[0]
    m = K_YY.shape[0]

    # 使用无偏估计量形式的 U 统计量
    # 第 1 项：K(xi, xj) 对于 i != j 的和
    term1 = (K_XX.sum() - np.diag(K_XX).sum()) / (n * (n - 1)) if n > 1 else 0
    # 第 2 项：K(yi, yj) 对于 i != j 的和
    term2 = (K_YY.sum() - np.diag(K_YY).sum()) / (m * (m - 1)) if m > 1 else 0
    # 第 3 项：K(xi, yj) 的和
    term3 = K_XY.sum() / (n * m) if n > 0 and m > 0 else 0

    mmd_squared = term1 + term2 - 2 * term3
    # MMD 统计量通常报告为平方值，为避免浮点错误，确保非负
    return max(0, mmd_squared)

# 选择 gamma 值。这可以是启发式方法，例如 1 / (中位数成对距离)
# 或者只是一个默认值。这里使用一个简单的默认值进行演示。
gamma_val = 1.0 / real_data.shape[1] # 经验法则：1 / n_features

mmd_val_normal = mmd_rbf(real_data, synthetic_data, gamma=gamma_val)
mmd_val_uniform = mmd_rbf(real_data, synthetic_data_uniform, gamma=gamma_val)

print(f"\nMMD（真实 vs. 合成正态）：{mmd_val_normal:.6f}")
print(f"MMD（真实 vs. 合成均匀）：{mmd_val_uniform:.6f}")

# 解释：
# MMD 值是距离；越小越好。0 表示（经验上）分布完全相同。
# 我们看到，由相似正态分布生成的合成数据的 MMD 值远小于
# 来自均匀分布的合成数据。
# 这符合我们的预期。为了评估显著性，通常使用置换检验
# （在 X 和 Y 之间打乱数据以构建 MMD 的零分布），但
# 此处的相对大小已具有参考意义。
```

MMD 能够很好地评估整体分布相似性，捕捉的差异不仅仅是均值。

### 比较协方差结构

保持变量*之间*的关系通常与匹配单个变量分布同等重要。我们可以比较真实数据集和合成数据集的协方差矩阵 ($\Sigma$)。

```python
# 计算协方差矩阵
cov_matrix_real = np.cov(real_data, rowvar=False)
cov_matrix_synth = np.cov(synthetic_data, rowvar=False)
cov_matrix_synth_uniform = np.cov(synthetic_data_uniform, rowvar=False)

print("\n协方差矩阵（真实）：")
print(cov_matrix_real)
print("\n协方差矩阵（合成正态）：")
print(cov_matrix_synth)
print("\n协方差矩阵（合成均匀）：")
print(cov_matrix_synth_uniform)

# 使用 Frobenius 范数计算差异
diff_norm_normal = np.linalg.norm(cov_matrix_real - cov_matrix_synth, 'fro')
diff_norm_uniform = np.linalg.norm(cov_matrix_real - cov_matrix_synth_uniform, 'fro')

print(f"\n协方差差异的 Frobenius 范数（真实 vs. 合成正态）：{diff_norm_normal:.4f}")
print(f"协方差差异的 Frobenius 范数（真实 vs. 合成均匀）：{diff_norm_uniform:.4f}")

# 解释：
# Frobenius 范数给出了一个数字，表示两个矩阵之间差异的大小。
# 范数越小，表示协方差结构越相似。
# 正如预期，与均匀数据相比，使用相似参数生成的合成数据的差异较小，
# 均匀数据的协方差结构从根本上是不同的
# （如果特征独立，均匀数据的协方差矩阵接近对角线，并且缩放比例不同）。
# 此度量证实，
# `synthetic_data` 中的关系（协方差）比 `synthetic_data_uniform` 中保持得更好。
```

存在更正式的检验，如 Box's M 检验，用于统计比较协方差矩阵，但它们对非正态性敏感。Frobenius 范数提供了一种实用、直接的差异度量。

### 可视化检查

有时，一张图胜过千言万语。将数据可视化可以即时了解合成数据捕捉多变量结构的程度。散点图对二维数据很有效。

```python
# 准备 Plotly 数据
trace_real = go.Scatter(
    x=real_df['Feature1'],
    y=real_df['Feature2'],
    mode='markers',
    name='真实数据',
    marker=dict(color='#1c7ed6', size=5, opacity=0.6) # 蓝色
)

trace_synth_normal = go.Scatter(
    x=synthetic_df['Feature1'],
    y=synthetic_df['Feature2'],
    mode='markers',
    name='合成（正态）',
    marker=dict(color='#f76707', size=5, opacity=0.6) # 橙色
)

trace_synth_uniform = go.Scatter(
    x=synthetic_df_uniform['Feature1'],
    y=synthetic_df_uniform['Feature2'],
    mode='markers',
    name='合成（均匀）',
    marker=dict(color='#37b24d', size=5, opacity=0.6) # 绿色
)

# 绘制真实 vs 合成（正态）
layout1 = go.Layout(
    title='真实 vs. 合成（正态）数据分布',
    xaxis=dict(title='特征 1'),
    yaxis=dict(title='特征 2'),
    width=600, height=500,
    legend=dict(x=0.01, y=0.99)
)
fig1 = go.Figure(data=[trace_real, trace_synth_normal], layout=layout1)

# 绘制真实 vs 合成（均匀）
layout2 = go.Layout(
    title='真实 vs. 合成（均匀）数据分布',
    xaxis=dict(title='特征 1'),
    yaxis=dict(title='特征 2'),
    width=600, height=500,
    legend=dict(x=0.01, y=0.99)
)
fig2 = go.Figure(data=[trace_real, trace_synth_uniform], layout=layout2)

# 显示用于嵌入的 JSON（为简洁起见仅显示一个）
# 在实际环境中，您会渲染这些图。
plotly_json_str = fig1.to_json()
# print(plotly_json_str) # 这将输出长 JSON 字符串

# 文档的 Plotly JSON 简洁示例：
```



![真实 vs. 合成（正态）数据分布](plots/5659-0.json)



> 比较生成的真实数据（蓝色）和来自相似多变量正态分布的合成数据（橙色）。它们的重叠度和总体形状看起来非常相似。

如果您绘制 `real_df` 与 `synthetic_df_uniform` 的对比图，差异将非常明显：真实数据形成一个椭圆形云团，而均匀分布的合成数据则形成一个方形。对于更高维度的数据，在绘图前可以考虑使用配对图 (`seaborn.pairplot`) 或降维技术（如 PCA 或 t-SNE）。

### 总结

本次实践练习说明了如何实现和解读用于保真度评估的几种多变量统计检验：

- **Hotelling's T²:** 检验均值向量 (vector)的差异（需要仔细实现或特定库）。
- **MMD:** 提供非参数 (parameter)的分布距离度量，对形状和高阶矩敏感。
- **协方差矩阵范数:** 量化 (quantization)变量之间线性关系的差异。
- **可视化:** 提供关于数据结构相似性的直接定性了解。

运行这些检验，能够比单独的单变量比较提供更全面的统计保真度情况。您可以看到，设计为相似的合成数据 (`synthetic_df`) 比刻意不同的均匀数据 (`synthetic_df_uniform`) 表现更好（MMD 更低，协方差差异更小）。在实践中，您会将这些检验应用于实际的合成数据，并与原始数据集的结果进行比较，以量化统计属性的保留程度。请记住要考虑上下文 (context)：所需统计保真度的水平在很大程度上取决于合成数据的预期用途。

## 参考资料

- [Applied Multivariate Statistical Analysis](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQF0erTNfvaTfq1mMKfGuTMIBVEFlb2PUUKhZDBezkGf2l0Pu_ouakumMrWvBiiZKH8U22UH28sn-WpoJ7TlEXJooB-Vw8e4YsogKRLTIogyeF5nFc1nrthwwdAdO9Cbj-qQBapeQQC3pQm1ncC0DV90OtNI7icoKjgrfHqaUYQYPGzqdRXCClQSJPX7ZfE-VjohdoAQ02dhGK4FBBZOQuhaTA6HPzDsFMdrIBvvRBGU72BViqejBWPLvl55cjFjQVnb) — Richard A. Johnson and Dean W. Wichern (2013)
  Publisher: Pearson
  全面涵盖多元统计方法（包括Hotelling T²检验和协方差结构分析）的教科书。
- [A Kernel Two-Sample Test](https://www.jmlr.org/papers/v13/gretton12a.html) — Arthur Gretton, Karsten M. Borgwardt, Malte J. Rasch, Bernhard SchÃ¶lkopf, Alexander Smola (2012)
  Journal: Journal of Machine Learning Research; Publisher: Journal of Machine Learning Research; Volume: 13; Pages: 723−773
  引入最大均值差异（MMD）检验作为比较两个概率分布的非参数方法的开创性论文。
- [On the Evaluation of Synthetic Data](https://ojs.aaai.org/index.php/AAAI/article/view/11267) — Peter Henderson, Florian Kerschbaum (2017)
  Journal: Proceedings of the AAAI Conference on Artificial Intelligence; Publisher: AAAI Press; Volume: 31; Pages: 11267-11275; DOI: [10.1609/aaai.v31i1.11267](https://doi.org/10.1609/aaai.v31i1.11267)
  提供了评估合成数据质量和实用性的综述和框架，包括统计保真度方面。
- [sklearn.metrics.pairwise.rbf_kernel](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.rbf_kernel.html) — scikit-learn developers (2024)
  径向基函数（RBF）核的官方文档，它是本节中展示的最大均值差异（MMD）实现的关键组成部分。
