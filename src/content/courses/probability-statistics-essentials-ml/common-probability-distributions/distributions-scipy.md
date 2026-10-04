---
course: "probability-statistics-essentials-ml"
chapter: "common-probability-distributions"
lesson: "distributions-scipy"
sourceId: 1381
sourceUrl: "https://apxml.com/zh/courses/probability-statistics-essentials-ml/chapter-2-common-probability-distributions/distributions-scipy"
title: "在 SciPy 中使用分布"
description: "在 Python 中使用 SciPy 库处理各种概率分布（PDF、CDF、抽样）。"
order: 7
plots: ["plots/1381-0.json", "plots/1381-1.json"]
sourceHash: "ad312b35ef761300b853d86211609dff30762f6aa4f837a2e5a6a28ba4f5a699"
sourceCorrections: []
---

SciPy 库提供了强大的工具，用于在 Python 中实际处理概率分布。具体而言，其 `stats` 模块 (`scipy.stats`) 提供了一套完整的函数，用于操作各种此类分布。这项能力对于统计建模、模拟以及各种机器学习 (machine learning)任务都是必不可少的。

`scipy.stats` 模块为许多分布（包括连续分布和离散分布）提供了一致的接口。对于每种分布，您通常可以执行以下几项操作：

- **概率密度函数 (PDF):** 对于连续分布，PDF，通常表示为 $f(x)$，表示随机变量取特定值 $x$ 的概率。使用 `.pdf()` 方法。
- **概率质量函数 (PMF):** 对于离散分布，PMF，通常表示为 $P(X=k)$，表示离散随机变量 $X$ 等于某个值 $k$ 的概率。使用 `.pmf()` 方法。
- **累积分布函数 (CDF):** CDF，$F(x) = P(X \le x)$，表示随机变量 $X$ 取值小于或等于 $x$ 的概率。使用 `.cdf()` 方法。
- **百分点函数 (PPF):** 也称为分位数函数或逆 CDF。给定一个概率 $p$，PPF 找到满足 $F(x) = p$ 的值 $x$。使用 `.ppf()` 方法。
- **随机变量抽样 (RVS):** 生成符合指定分布的随机数。使用 `.rvs()` 方法。

我们通过一些例子来看看它是如何工作的。

### 连续分布的使用：正态分布

正态（高斯）分布在统计学中普遍存在。在 `scipy.stats` 中，它由 `norm` 表示。要使用特定的正态分布，我们通常需要使用 `loc` 参数 (parameter)指定其均值 ($\mu$)，并使用 `scale` 参数指定其标准差 ($\sigma$)。请记住，正态分布通常以方差 $\sigma^2$ 参数化，但 SciPy 使用标准差作为 `scale` 参数。

```python
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt # We'll use this for basic plotting setup

# 定义一个正态分布：均值=0，标准差=2
mu = 0
sigma = 2
my_normal = norm(loc=mu, scale=sigma)

# 计算 x=1 时的 PDF
pdf_at_1 = my_normal.pdf(1)
print(f"PDF at x=1: {pdf_at_1:.4f}")

# 计算 x=1 时的 CDF (P(X <= 1))
cdf_at_1 = my_normal.cdf(1)
print(f"CDF at x=1 (P(X <= 1)): {cdf_at_1:.4f}")

# 计算概率 0.95 的 PPF（找到第 95 百分位数）
percentile_95 = my_normal.ppf(0.95)
print(f"第 95 百分位数（使 P(X <= x) = 0.95 的值 x）: {percentile_95:.4f}")

# 从该分布生成 5 个随机样本
random_samples = my_normal.rvs(size=5)
print(f"五个随机样本: {random_samples}")

# 生成用于绘制 PDF 的数据点
x_values = np.linspace(mu - 4*sigma, mu + 4*sigma, 200) # 覆盖均值附近的范围
pdf_values = my_normal.pdf(x_values)
```



![正态分布 PDF (μ=0,σ=2)](plots/1381-0.json)



> 均值为 0，标准差为 2 的正态分布的概率密度函数 (PDF)。

### 离散分布的使用：二项分布

二项分布对固定数量 $n$ 的独立伯努利试验中成功次数 $k$ 进行建模，每次试验的成功概率为 $p$。在 `scipy.stats` 中，它由 `binom` 表示。我们需要指定 $n$ 和 $p$。

```python
from scipy.stats import binom

# 定义一个二项分布：n=10 次试验，p=0.5 成功概率
n_trials = 10
prob_success = 0.5
my_binomial = binom(n=n_trials, p=prob_success)

# 计算 k=5 次成功时的 PMF (P(X=5))
pmf_at_5 = my_binomial.pmf(5)
print(f"PMF at k=5 (P(X=5)): {pmf_at_5:.4f}")

# 计算 k=5 时的 CDF (P(X <= 5))
cdf_at_5 = my_binomial.cdf(5)
print(f"CDF at k=5 (P(X <= 5)): {cdf_at_5:.4f}")

# 计算概率 0.9 的 PPF（找到使 P(X <= k) >= 0.9 的 k）
# 注意：对于离散分布，PPF 返回满足条件的最小 k 值。
quantile_90 = my_binomial.ppf(0.9)
print(f"使 P(X <= k) >= 0.9 的 k 值: {quantile_90}")

# 生成 10 个随机样本（10 次试验中的成功次数）
random_samples_binom = my_binomial.rvs(size=10)
print(f"十个随机样本（成功次数）: {random_samples_binom}")

# 生成用于绘制 PMF 的数据
k_values = np.arange(0, n_trials + 1)
pmf_values = my_binomial.pmf(k_values)
```



![二项分布 PMF ($n=10, p=0.5$)](plots/1381-1.json)



> $n=10$ 次试验、成功概率 $p=0.5$ 的二项分布的概率质量函数 (PMF)。

### 其他分布

上述模式同样适用于 `scipy.stats` 中提供的其他分布：

- **泊松分布 (Poisson):** 使用 `poisson(mu)`，`mu` 是速率参数 (parameter) $\lambda$。方法包括 `.pmf()`、`.cdf()`、`.ppf()`、`.rvs()`。
- **指数分布 (Exponential):** 使用 `expon(scale=1/lambda)`，其中 `scale` 对应于 $1/\lambda$，即速率参数 $\lambda$ 的倒数。另外，您可以使用 `loc` 来移动分布。方法包括 `.pdf()`、`.cdf()`、`.ppf()`、`.rvs()`。
- **均匀分布 (Uniform):** 使用 `uniform(loc=a, scale=b-a)` 表示区间 $[a, b)$ 上的均匀分布。`loc` 参数定义起始点 $a$，`scale` 定义宽度 $b-a$。方法包括 `.pdf()`、`.cdf()`、`.ppf()`、`.rvs()`。

例如，要计算泊松分布中观察到恰好 3 个事件的概率，其平均速率 ($\lambda$) 为每个间隔 4 个事件：

```python
from scipy.stats import poisson

lambda_rate = 4
my_poisson = poisson(mu=lambda_rate)

pmf_at_3 = my_poisson.pmf(3)
print(f"泊松 PMF 在 k=3 (lambda=4): {pmf_at_3:.4f}")
```

能够使用 SciPy 计算这些标准分布的概率并生成样本是一项基本能力。它让您能够模拟过程，检验假设（我们将在后面看到），并为依赖于概率假设的更复杂机器学习 (machine learning)模型构建组件。熟悉这些常用分布的 `scipy.stats` 接口将大有裨益。

## 参考资料

- [scipy.stats module documentation](https://docs.scipy.org/doc/scipy/reference/stats.html) — SciPy Developers (2024)
  提供SciPy中统计函数和分布的全面信息，详细说明PDF、CDF、PPF和随机变量生成等方法。
- [Python Data Science Handbook: Essential Tools for Working with Data](https://jakevdp.github.io/PythonDataScienceHandbook/) — Jake VanderPlas (2016)
  Publisher: O'Reilly Media
  一份实用的资源，用于将Python的数据科学库（包括SciPy的统计功能）应用于数据分析和机器学习任务。
