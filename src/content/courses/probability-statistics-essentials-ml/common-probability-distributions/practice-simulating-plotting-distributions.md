---
course: "probability-statistics-essentials-ml"
chapter: "common-probability-distributions"
lesson: "practice-simulating-plotting-distributions"
sourceId: 1384
sourceUrl: "https://apxml.com/zh/courses/probability-statistics-essentials-ml/chapter-2-common-probability-distributions/practice-simulating-plotting-distributions"
title: "动手实践：模拟与绘制分布"
description: "从分布中生成随机样本，并使用 Python 库将其可视化。"
order: 8
plots: ["plots/1384-0.json", "plots/1384-1.json", "plots/1384-2.json"]
sourceHash: "1c5671d0a91b9d4b264c3b3ca604b7cb3cb4722e799e11a895d76d93b7faa8c1"
sourceCorrections: []
---

好的，让我们将理论付诸实践。理解概率分布的数学公式是一回事，但通过模拟和可视化观察它们的实际表现，能让你对其有更充分的直觉认知。这在你需要判断某个分布是否适合机器学习 (machine learning)中观测到的数据时，尤其有用。

我们将使用 Python，特别是 `scipy.stats` 模块，以及 `numpy` 进行数值计算，`matplotlib` 或 `seaborn` 进行绘图（常与 `plotly` 结合用于交互式网页可视化）。

首先，请确保您已安装所需的库并导入它们：

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# 可选：设置 seaborn 绘图风格
sns.set_style("whitegrid")
```

### 模拟和可视化离散分布：二项分布

我们从二项分布开始，它表示在固定次数 $n$ 的独立伯努利试验中成功次数 $k$ 的模型，每次试验的成功概率为 $p$。

假设我们抛掷一枚不均匀的硬币（$p=0.6$）20 次（$n=20$），并重复此实验多次（例如 1000 次）。我们可以使用 `scipy.stats.binom.rvs`（随机变量）模拟每次实验中获得正面的次数。

```python
# 二项分布的参数
n_binom = 20  # 试验次数
p_binom = 0.6  # 成功概率
size_binom = 1000 # 模拟次数（实验次数）

# 从二项分布中模拟随机样本
binomial_samples = stats.binom.rvs(n=n_binom, p=p_binom, size=size_binom)

# 打印前 10 个模拟结果
print(f"前 10 个模拟结果（在 {n_binom} 次试验中的成功次数）：")
print(binomial_samples[:10])
```

现在，我们使用直方图可视化模拟结果，并将其与理论概率质量函数（PMF）进行比较。PMF 表示获得恰好 $k$ 次成功的概率。

```python
# 计算理论 PMF
k_values_binom = np.arange(0, n_binom + 1)
pmf_binom = stats.binom.pmf(k=k_values_binom, n=n_binom, p=p_binom)

# 使用 Matplotlib 创建图表（或适配 Plotly）
plt.figure(figsize=(10, 6))

# 绘制模拟数据的直方图
# 使用 density=True 标准化直方图，以便与 PMF 进行比较
plt.hist(binomial_samples, bins=k_values_binom, density=True, alpha=0.6, color='#4dabf7', label='模拟数据（直方图）')

# 绘制理论 PMF
# 使用 'o-' 绘制点线
plt.plot(k_values_binom, pmf_binom, 'o-', color='#f03e3e', label='理论 PMF')

plt.xlabel(f"在 {n_binom} 次试验中的成功次数 (k)")
plt.ylabel("概率 / 密度")
plt.title(f"二项分布 (n={n_binom}, p={p_binom}) 模拟 vs. 理论")
plt.legend()
plt.grid(True)
plt.show() 
```

以下是使用 Plotly 在网页上表示的图表：



![二项分布 (n=20, p=0.6) 模拟 vs. 理论](plots/1384-0.json)



> 1000 次模拟二项分布实验（n=20, p=0.6）与理论概率质量函数的比较。直方图与 PMF 预测的形状非常接近。

请注意，我们模拟结果的直方图如何与理论 PMF 的形状非常接近。随着模拟次数 (`size_binom`) 的增加，这种近似通常会变得更好。

### 模拟和可视化连续分布：正态分布

接下来，我们来研究正态（高斯）分布，它以其均值 $\mu$ 和标准差 $\sigma$（或方差 $\sigma^2$）为特征。它可能是统计学和机器学习 (machine learning)中最常见的分布，常用于模拟自然现象或误差。

我们从标准正态分布（$\mu=0, \sigma=1$）中模拟数据，并将样本的直方图与理论概率密度函数（PDF）进行比较。

```python
# 正态分布的参数
mu_norm = 0     # 均值
sigma_norm = 1  # 标准差
size_norm = 1000 # 样本数量

# 从正态分布中模拟随机样本
normal_samples = stats.norm.rvs(loc=mu_norm, scale=sigma_norm, size=size_norm)

# 打印前 10 个模拟值
print(f"N({mu_norm}, {sigma_norm**2}) 的前 10 个模拟值：")
print(normal_samples[:10])

# 为理论 PDF 曲线生成点
x_values_norm = np.linspace(mu_norm - 4*sigma_norm, mu_norm + 4*sigma_norm, 200)
pdf_norm = stats.norm.pdf(x_values_norm, loc=mu_norm, scale=sigma_norm)

# 创建图表
plt.figure(figsize=(10, 6))

# 绘制模拟数据的直方图
plt.hist(normal_samples, bins=30, density=True, alpha=0.6, color='#74c0fc', label='模拟数据（直方图）')

# 绘制理论 PDF
plt.plot(x_values_norm, pdf_norm, color='#f76707', linewidth=2, label='理论 PDF')

plt.xlabel("数值")
plt.ylabel("密度")
plt.title(f"正态分布 (\u03bc={mu_norm}, \u03c3={sigma_norm}) 模拟 vs. 理论")
plt.legend()
plt.grid(True)
plt.show() # 同样，在网页部署时适配 Plotly
```

这是相应的 Plotly JSON 表示：



![正态分布 (μ=0, σ=1) 模拟 vs. 理论](plots/1384-1.json)



> 从标准正态分布 (N(0, 1)) 中提取的 1000 个样本的直方图，叠加了理论概率密度函数 (PDF)。模拟数据与特征性的钟形曲线非常接近。

同样，模拟数据的直方图很好地近似了理论 PDF 定义的钟形曲线。这种可视化有助于确认我们的随机样本行为符合正态分布的预期。

### 模拟和可视化另一个分布：泊松分布

我们再来看一个：泊松分布。它表示在给定平均发生率 $\lambda$ 的情况下，在固定时间或空间间隔内发生给定数量事件的概率。例子：每小时到达的电子邮件数量。

假设电子邮件以平均每小时 $\lambda=5$ 的速率到达。我们将模拟 1000 个不同小时内收到的电子邮件数量。

```python
# 泊松分布的参数
lambda_poisson = 5 # 平均发生率（例如，每小时电子邮件数）
size_poisson = 1000 # 模拟次数（观察的小时数）

# 从泊松分布中模拟随机样本
poisson_samples = stats.poisson.rvs(mu=lambda_poisson, size=size_poisson)

# 打印前 10 个模拟计数
print(f"前 10 个模拟计数（每个时间间隔的事件数，\u03bb={lambda_poisson}）：")
print(poisson_samples[:10])

# 计算理论 PMF
# 为 k 选择一个合理的上限，例如，均值 + 几个标准差
k_max_poisson = int(lambda_poisson + 4 * np.sqrt(lambda_poisson))
k_values_poisson = np.arange(0, k_max_poisson + 1)
pmf_poisson = stats.poisson.pmf(k=k_values_poisson, mu=lambda_poisson)

# 创建图表
plt.figure(figsize=(10, 6))

# 绘制模拟数据的直方图
# 为离散数据正确对齐直方图的bin
bins_poisson = np.arange(poisson_samples.min(), poisson_samples.max() + 2) - 0.5
plt.hist(poisson_samples, bins=bins_poisson, density=True, alpha=0.6, color='#69db7c', label='模拟数据（直方图）')

# 绘制理论 PMF
plt.plot(k_values_poisson, pmf_poisson, 'o-', color='#be4bdb', label='理论 PMF')

plt.xlabel(f"每个时间间隔的事件数 (k)")
plt.ylabel("概率 / 密度")
plt.title(f"泊松分布 (\u03bb={lambda_poisson}) 模拟 vs. 理论")
plt.legend()
plt.grid(True)
plt.xticks(k_values_poisson) # 确保刻度位于整数值
plt.xlim(bins_poisson.min(), bins_poisson.max())
plt.show()
```



![泊松分布 (λ=5) 模拟 vs. 理论](plots/1384-2.json)



> 1000 次模拟泊松实验（lambda=5）与理论概率质量函数的比较。直方图与 PMF 相似，在平均发生率 lambda 附近达到峰值。

这个实践练习表明，模拟如何帮助弥合抽象分布公式与具体数据之间的鸿沟。通过使用 SciPy 和 Matplotlib/Plotly 等库生成样本并对照理论函数绘制它们，你可以更好地了解常见概率分布的形状、分散情况和特点。这项技能对探索性数据分析、模型选择以及理解许多机器学习 (machine learning)算法的概率根基都很有价值。

## 参考资料

- [Statistical functions (scipy.stats)](https://docs.scipy.org/doc/scipy/reference/stats.html) — SciPy Developers (2024)
  官方文档，详细介绍了 `scipy.stats` 模块在概率分布中的使用，包括随机变量生成、PMF和PDF。
- [Python Data Science Handbook: Essential Tools for Working with Data](https://jakevdp.github.io/PythonDataScienceHandbook/) — Jake VanderPlas (2016)
  Publisher: O'Reilly Media
  实用指南，涵盖NumPy、Matplotlib和SciPy，用于数据处理和可视化，与模拟和绘制数据相关。
