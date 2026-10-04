---
course: "probability-statistics-essentials-ml"
chapter: "common-probability-distributions"
lesson: "bernoulli-binomial-distributions"
sourceId: 1360
sourceUrl: "https://apxml.com/zh/courses/probability-statistics-essentials-ml/chapter-2-common-probability-distributions/bernoulli-binomial-distributions"
title: "伯努利分布和二项分布"
description: "了解伯努利分布和二项分布的特点及在离散结果中的用途。"
order: 1
plots: ["plots/1360-0.json"]
sourceHash: "02a7d36bf0a344c432e853d0b9c402d23fa98e992a2a61b6250d0311777c4e4a"
sourceCorrections: []
---

我们从两个处理离散结果的基础概率分布开始学习：伯努利分布和二项分布。它们通常是人们最初接触的分布，因为它们模拟简单的“是/否”或“成功/失败”情景，而这类情景在数据分析和机器学习 (machine learning)中普遍存在。

## 伯努利分布：一次试验，两种结果

想象一下最简单的随机试验，它只有两种结果：成功或失败。例如，一次硬币翻转（正面或反面），一次广告点击（已点击或未点击），或一封邮件被分类（垃圾邮件或非垃圾邮件）。伯努利分布描述了这些单次试验事件的概率。

如果随机变量 $X$ 只能取两个值，通常表示为 1（成功）和 0（失败），那么它服从伯努利分布。该分布由一个参数 (parameter) $p$ 定义，该参数表示成功的概率（$P(X=1)$）。因此，失败的概率是 $P(X=0) = 1 - p$。

**概率质量函数（PMF）：**

PMF 给出每个可能结果的概率。对于伯努利随机变量 $X$，其 PMF 为：


$$
P(X=k | p) =
\begin{cases}
p & \text{如果 } k=1 \text{ (成功)} \\
1-p & \text{如果 } k=0 \text{ (失败)}
\end{cases}
$$


这可以更简洁地写为：

$P(X=k | p) = p^k (1-p)^{1-k} \quad \text{对于 } k \in \{0, 1\}$

**特性：**

- **期望值：** 在多次试验中您期望的平均结果。对于伯努利分布，$E[X] = p$。
- **方差：** 衡量结果分散或变异程度的指标。对于伯努利分布，$Var(X) = p(1-p)$。

伯努利分布是二项分布的基本构成部分。

## 二项分布：多次独立试验

那么，如果我们*在相同条件下*重复多次伯努利试验并计算成功的次数呢？例如，抛掷一枚均匀硬币 10 次并计算正面朝上的次数，或者测试 20 个制造部件并计算有多少是缺陷产品（假设每个部件有缺陷的概率是恒定的，且测试相互独立）。这种情景由二项分布来描述。

如果随机变量 $X$ 表示 $n$ 次独立同分布的伯努利试验中成功的总次数，其中每次试验成功的概率为 $p$，则它服从二项分布。

二项分布由两个参数 (parameter)确定：

1. $n$：独立试验的总次数。
2. $p$：单次试验成功的概率。

我们将其表示为 $X \sim Binomial(n, p)$。

**概率质量函数（PMF）：**

PMF 给出在 $n$ 次试验中观察到恰好 $k$ 次成功的概率。

$P(X=k | n, p) = \binom{n}{k} p^k (1-p)^{n-k} \quad \text{对于 } k \in \{0, 1, 2, ..., n\}$

其中：

- $\binom{n}{k}$（读作“n 选 k”）是二项式系数，表示从 $n$ 次试验中选择 $k$ 次成功的不同方式的数量。它计算为 $\binom{n}{k} = \frac{n!}{k!(n-k)!}$，其中 $!$ 表示阶乘。
- $p^k$ 是获得恰好 $k$ 次成功的概率。
- $(1-p)^{n-k}$ 是获得恰好 $n-k$ 次失败的概率。

**特性：**

- **期望值：** $n$ 次试验中成功的平均次数。对于二项分布，$E[X] = np$。这符合直觉：如果您抛掷一枚硬币 10 次（$n=10$），正面朝上的概率为 $p=0.5$，那么您期望有 $10 \times 0.5 = 5$ 次正面。
- **方差：** 成功次数分散程度的指标。对于二项分布，$Var(X) = np(1-p)$。

**示例：** 假设我们抛掷一枚不均匀的硬币（正面朝上概率 $p=0.6$）5 次（$n=5$）。恰好出现 3 次正面（$k=3$）的概率是多少？

使用 PMF：
$P(X=3 | n=5, p=0.6) = \binom{5}{3} (0.6)^3 (1-0.6)^{5-3}$
$P(X=3) = \frac{5!}{3!(5-3)!} (0.6)^3 (0.4)^2$
$P(X=3) = \frac{120}{6 \times 2} (0.216) (0.16)$
$P(X=3) = 10 \times 0.216 \times 0.16 = 0.3456$

因此，在 5 次抛掷中恰好出现 3 次正面的概率约为 34.6%。

### 在 Python 中使用伯努利分布和二项分布 (SciPy)

`scipy.stats` 模块提供了用于处理这些分布的便捷函数。

```python
import numpy as np
from scipy.stats import bernoulli, binom
import matplotlib.pyplot as plt

# --- 伯努利分布示例 ---
p_success = 0.7 # 成功的概率（例如，点击率）

# 创建一个伯努利分布对象
rv_bern = bernoulli(p_success)

# 概率质量函数（PMF）
print(f"伯努利 PMF(k=1): {rv_bern.pmf(1):.4f}") # P(X=1)
print(f"伯努利 PMF(k=0): {rv_bern.pmf(0):.4f}") # P(X=0)

# 期望值和方差
print(f"伯努利 E[X]: {rv_bern.mean():.4f}")
print(f"伯努利 Var(X): {rv_bern.var():.4f}")

# 生成随机样本
print(f"伯努利样本 (10): {rv_bern.rvs(size=10)}")

print("-" * 30)

# --- 二项分布示例 ---
n_trials = 10 # 试验次数（例如，检查 10 封邮件）
p_success_bin = 0.2 # 每次试验成功的概率（例如，邮件是垃圾邮件）

# 创建一个二项分布对象
rv_binom = binom(n_trials, p_success_bin)

# k=3 次成功的概率质量函数（PMF）
k_successes = 3
print(f"二项分布 PMF(k={k_successes}): {rv_binom.pmf(k_successes):.4f}") # P(X=3)

# k<=3 次成功的累积分布函数（CDF）
print(f"二项分布 CDF(k<={k_successes}): {rv_binom.cdf(k_successes):.4f}") # P(X<=3)

# 期望值和方差
print(f"二项分布 E[X]: {rv_binom.mean():.4f}") # np
print(f"二项分布 Var(X): {rv_binom.var():.4f}") # np(1-p)

# 生成随机样本（15 次 n_trials 试验中成功的次数）
print(f"二项分布样本 (15): {rv_binom.rvs(size=15)}")

# --- 绘制二项分布 PMF ---
k_values = np.arange(0, n_trials + 1)
pmf_values = rv_binom.pmf(k_values)

# 为二项分布 PMF 生成 Plotly JSON
```



![二项分布 PMF (n=10, p=0.2)](plots/1360-0.json)



> $n=10$ 次试验和成功概率 $p=0.2$ 的二项分布概率质量函数。最有可能的结果是 2 次成功。

伯努利分布描述单个事件，而二项分布汇集了多个独立伯努利事件的结果。了解这些分布很重要，因为它们构成了分析二元结果的基础，而二元结果在分类问题（垃圾邮件/非垃圾邮件、恶性/良性）、A/B 测试结果以及许多其他与机器学习 (machine learning)相关的情景中经常出现。

## 参考资料

- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer
  讨论了统计学习方法，其中引用了基础的概率分布。
- [scipy.stats.bernoulli](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bernoulli.html) — SciPy Developers (2024)
  提供了SciPy统计模块中 `bernoulli` 函数的详细信息。
- [scipy.stats.binom](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html) — SciPy Developers (2024)
  提供了SciPy统计模块中 `binom` 函数的详细信息。
