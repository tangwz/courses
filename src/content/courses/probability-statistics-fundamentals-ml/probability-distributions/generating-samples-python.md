---
course: "probability-statistics-fundamentals-ml"
chapter: "probability-distributions"
lesson: "generating-samples-python"
sourceId: 2456
sourceUrl: "https://apxml.com/zh/courses/probability-statistics-fundamentals-ml/chapter-4-probability-distributions/generating-samples-python"
title: "使用 Python 从分布生成样本"
description: "使用 Python (NumPy/SciPy) 从各种概率分布中生成随机样本。"
order: 9
plots: ["plots/2456-0.json"]
sourceHash: "6584b8868b3ab583526fea3743d3807f8917cae8c5186741d999f9b43f34c07d"
sourceCorrections: []
---

通过生成遵循特定概率分布的数据，可以更实际地理解这些分布。Python 中的现代科学计算库，特别是 SciPy 和 NumPy，提供了强大的工具，用于从各种概率分布中生成随机数（样本）。这个过程通常称为 *抽样*。

抽样在许多任务中很有用，包括：

- 模拟随机过程。
- 直观了解分布的形态和表现。
- 检验统计方法或机器学习 (machine learning)算法。
- 生成合成数据。

我们主要使用 `scipy.stats` 模块，它提供了一个处理分布的统一接口，包括使用 `.rvs()` 方法生成随机变量（样本）。我们也会偶尔提及 `numpy.random` 中等效的函数。

我们先导入必要的库。我们需要 `scipy.stats` 来处理分布，以及 `matplotlib.pyplot`（通常导入为 `plt`）来进行基本可视化，尽管我们将使用 Plotly 格式来呈现图表以实现交互式网页显示。我们还将使用 `numpy` 进行数值运算。

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt # 我们使用这个，但输出 Plotly JSON

# 配置可视化（可选，有助于使 matplotlib 图表更好看）
# plt.style.use('seaborn-v0_8-whitegrid')
```

### 从离散分布生成样本

离散分布处理可计数的结果。我们将介绍伯努利分布和二项分布。

#### 伯努利分布

伯努利分布对具有两种可能结果的单次试验进行建模：成功（通常编码为 1），概率为 $p$；失败（通常编码为 0），概率为 $1-p$。设想一次抛硬币。

要使用 `scipy.stats` 从伯努利分布生成样本，我们使用 `stats.bernoulli.rvs()`。主要参数 (parameter)是 $p$，即成功的概率。

```python
# 参数
prob_success = 0.7  # 成功的概率（例如，正面）
num_samples = 10    # 要生成的试验（样本）数量

# 生成样本
# 每个样本为 0 或 1
bernoulli_samples = stats.bernoulli.rvs(p=prob_success, size=num_samples)

print(f"Bernoulli Samples (p={prob_success}): {bernoulli_samples}")
```

运行此代码可能会产生如下输出：`Bernoulli Samples (p=0.7): [1 1 0 1 1 1 0 1 0 1]`。每个数字代表一次试验的结果。如果生成大量样本，则 1 的比例应接近 $p$。

*(等效的 NumPy 函数：`np.random.binomial(1, p, size=num_samples)`)*

#### 二项分布

二项分布对固定数量 $n$ 次独立伯努利试验中成功的次数进行建模，每次试验成功的概率均为 $p$。例如，计算 10 次抛硬币中正面的数量。

我们使用 `stats.binom.rvs()`，指定 $n$（试验次数）和 $p$（每次试验成功的概率）。`size` 参数表示我们要运行此实验的次数（即，我们想要多少个成功*计数*的样本）。

```python
# 参数
num_trials = 10     # 一次实验中的伯努利试验次数 (n)
prob_success = 0.5  # 每次试验成功的概率 (p)
num_experiments = 1000 # 我们运行实验的次数（生成样本）

# 生成样本
# 每个样本是 'n' 次试验中成功的计数
binomial_samples = stats.binom.rvs(n=num_trials, p=prob_success, size=num_experiments)

print(f"First 10 Binomial Samples (n={num_trials}, p={prob_success}): {binomial_samples[:10]}")
# 示例输出: 前 10 个二项样本 (n=10, p=0.5): [5 6 5 4 7 5 6 5 5 3]
```

输出中的每个数字代表一组 10 次试验中获得成功的总数。为了可视化这些计数的分布，我们可以创建一个直方图。

```python
# 可视化（使用 Matplotlib 的代码）
# plt.figure(figsize=(8, 4))
# plt.hist(binomial_samples, bins=np.arange(num_trials + 2) - 0.5, density=True, alpha=0.7, color='#15aabf', edgecolor='black')
# plt.title(f'Binomial Distribution Samples (n={num_trials}, p={prob_success})')
# plt.xlabel('Number of Successes')
# plt.ylabel('Probability Density')
# plt.xticks(range(num_trials + 1))
# plt.grid(axis='y')
# plt.show()

# 实际用于直方图的 Plotly JSON 输出
hist_counts, bin_edges = np.histogram(binomial_samples, bins=np.arange(num_trials + 2) - 0.5, density=True)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
```



![模拟二项分布 (n=10, p=0.5)](plots/2456-0.json)



> 从参数为 $n=10$ 次试验和成功概率 $p=0.5$ 的二项分布中提取的 1000 个样本的直方图。其形状接近理论的二项概率质量函数（PMF），中心在 $n \times p = 5$ 附近。

### 从连续分布生成样本

连续分布描述连续范围内的结果。

#### 均匀分布

均匀分布将相等概率密度分配给指定范围 $[a, b)$ 内的所有结果。范围外的结果概率为零。

我们使用 `stats.uniform.rvs()`。它接受 `loc`（起点 $a$）和 `scale`（范围宽度 $b-a$）作为参数 (parameter)。

```python
# 参数
lower_bound = 5.0   # 区间起点 (a)
upper_bound = 10.0  # 区间终点 (b)
num_samples = 1000

# 计算 loc 和 scale
loc_param = lower_bound
scale_param = upper_bound - lower_bound

# 生成样本
uniform_samples = stats.uniform.rvs(loc=loc_param, scale=scale_param, size=num_samples)

print(f"First 10 Uniform Samples (range=[{lower_bound}, {upper_bound})): {uniform_samples[:10]}")
# 示例输出: 前 10 个均匀样本 (范围=[5.0, 10.0)): [7.82 9.21 5.34 6.78 8.89 5.01 9.98 7.11 6.05 8.43]
```

*(等效的 NumPy 函数：`np.random.uniform(low=lower_bound, high=upper_bound, size=num_samples)`)*

这些样本的直方图在区间 $[5, 10)$ 内应大致平坦。

```python
# 可视化（使用 Matplotlib 的代码）
# plt.figure(figsize=(8, 4))
# plt.hist(uniform_samples, bins=20, density=True, alpha=0.7, color='#fd7e14', edgecolor='black')
# plt.title(f'Uniform Distribution Samples (range=[{lower_bound}, {upper_bound}))')
# plt.xlabel('Value')
# plt.ylabel('Probability Density')
# plt.grid(axis='y')
# plt.show()

# 实际用于直方图的 Plotly JSON 输出
hist_counts, bin_edges = np.histogram(uniform_samples, bins=20, density=True)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
```

```plotly
{"data": [{"type": "bar", "x": bin_centers.tolist(), "y": hist_counts.tolist(), "name": "样本密度", "marker": {"color": "#fd7e14", "line": {"color": "#495057", "width": 1}}}], "layout": {"title": {"text": "模拟均匀分布 (范围=[5, 10))"}, "xaxis": {"title": {"text": "值"}}, "yaxis": {"title": {"text": "估计密度"}}, "bargap": 0.05, "width": 600, "height": 400}}
```

> 从区间 $[5, 10)$ 上的均匀分布中提取的 1000 个样本的直方图。在此范围内，密度近似保持不变。

#### 正态（高斯）分布

正态分布，常被称为钟形曲线，可能是最常见的连续分布。它的特点是均值（$\mu$，`loc`）和标准差（$\sigma$，`scale`）。该分布关于均值对称。

我们使用 `stats.norm.rvs()`，其中 `loc` 表示均值，`scale` 表示标准差。

```python
# 参数
mean_val = 0.0      # 均值 (mu)
std_dev = 1.0       # 标准差 (sigma)
num_samples = 1000

# 生成样本
normal_samples = stats.norm.rvs(loc=mean_val, scale=std_dev, size=num_samples)

print(f"First 10 Normal Samples (mean={mean_val}, std_dev={std_dev}): {normal_samples[:10]}")
# 示例输出: 前 10 个正态样本 (均值=0.0, 标准差=1.0): [-0.54  1.25  0.21 -1.87  0.88 -0.76  0.33 -0.11 -0.45  1.05]
```

*(等效的 NumPy 函数：`np.random.normal(loc=mean_val, scale=std_dev, size=num_samples)`)*

正态样本的直方图将显示其特有的钟形，中心位于均值。

```python
# 可视化（使用 Matplotlib 的代码）
# plt.figure(figsize=(8, 4))
# plt.hist(normal_samples, bins=30, density=True, alpha=0.7, color='#4263eb', edgecolor='black')
# plt.title(f'Normal Distribution Samples (mean={mean_val}, std_dev={std_dev})')
# plt.xlabel('Value')
# plt.ylabel('Probability Density')
# plt.grid(axis='y')
# plt.show()

# 实际用于直方图的 Plotly JSON 输出
hist_counts, bin_edges = np.histogram(normal_samples, bins=30, density=True)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
```

```plotly
{"data": [{"type": "bar", "x": bin_centers.tolist(), "y": hist_counts.tolist(), "name": "样本密度", "marker": {"color": "#4263eb", "line": {"color": "#495057", "width": 1}}}], "layout": {"title": {"text": "模拟正态分布 (均值=0, 标准差=1)"}, "xaxis": {"title": {"text": "值"}}, "yaxis": {"title": {"text": "估计密度"}}, "bargap": 0.05, "width": 600, "height": 400}}
```

> 从标准正态分布（$\mu=0, \sigma=1$）中提取的 1000 个样本的直方图。该分布清楚地显示了以 0 为中心的特有钟形。

"能够从这些基本分布生成样本是一项实用技能。它允许你模拟反映这些模式特征的数据，为实验、假设检验以及理解依赖概率假设的机器学习 (machine learning)模型的输入或输出提供了基础。随着遇到更复杂的分布，使用 SciPy 等库进行抽样的过程保持相似。"

## 参考资料

- [scipy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html) — SciPy Developers (2023)
  SciPy统计功能模块的官方文档，包括从各种概率分布生成随机变量（rvs()）的方法。
- [numpy.random](https://numpy.org/doc/stable/reference/random/index.html) — NumPy Developers (2023)
  Publisher: NumPy Project
  NumPy随机抽样功能的官方文档，包括从标准概率分布生成样本的方法。
- [Python for Probability, Statistics, and Machine Learning](https://doi.org/10.1007/978-3-319-30717-6) — Jose Unpingco (2016)
  Publisher: Springer International Publishing; DOI: [10.1007/978-3-319-30717-6](https://doi.org/10.1007/978-3-319-30717-6)
  探讨如何使用SciPy和NumPy等Python库实现概率和统计概念，包括从分布生成随机变量。
