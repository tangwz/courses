---
course: "probability-statistics-fundamentals-ml"
chapter: "probability-distributions"
lesson: "practice-distributions"
sourceId: 2458
sourceUrl: "https://apxml.com/zh/courses/probability-statistics-fundamentals-ml/chapter-4-probability-distributions/practice-distributions"
title: "练习：使用分布"
description: "练习识别分布并计算其相关概率。"
order: 10
plots: ["plots/2458-0.json"]
sourceHash: "3b03eef1e1a69d91964f25385ce5a7d77de4442f72f5d6c8e3e7eab61d79c8ab"
sourceCorrections: []
---

这些练习提供了对几种基本概率分布的实践经验。它们将帮助您巩固对何时使用特定分布、如何计算与其相关的概率以及如何使用 Python 处理它们的理解。概率分布是数据中不确定性建模的基本组成部分，这是机器学习 (machine learning)中一项常见任务。

### 练习 1：识别合适的分布

对于以下每个情景，请确定哪种概率分布（伯努利、二项、均匀或正态）最能描述所提到的随机变量。简要说明您的理由。

1. **情景：** 单个用户访问网页。随机变量是用户是否点击广告（是/否）。
   - **提示：** 思考试验次数和可能的结果。
2. **情景：** 您调查了100名学生，询问他们是喜欢在线课程还是面对面课程。随机变量是这100名学生中喜欢在线课程的*人数*。
   - **提示：** 考虑多次试验，每次试验都有两种结果，并且您在计数成功次数。
3. **情景：** 随机数生成器生成的数字在0.0到1.0之间，且任何值出现的可能性相同。随机变量是下一个生成的数字。
   - **提示：** 注意连续范围内的等可能性。
4. **情景：** 您测量一个大城市中成年男性的身高。随机变量是随机选择的一名男性的身高。
   - **提示：** 思考自然现象，以及测量值通常如何聚集在平均值附近。

**答案与理由：**

1. **伯努利分布：** 此情景涉及单次试验（一个用户访问），并有两种可能的结果（点击或不点击）。伯努利分布模拟单次试验中成功的概率（例如，点击）。
2. **二项分布：** 这涉及固定数量的独立试验（100名学生），每次试验有两种结果（喜欢在线或面对面），并且每名学生喜欢在线的概率被假定为常数。二项分布模拟固定次数伯努利试验中成功的次数。
3. **均匀分布：** 这里的重点是“等可能性”在连续区间 [0.0, 1.0] 内。均匀分布将等量的概率密度分配给指定范围内的所有结果。
4. **正态分布：** 像人类身高这样的物理测量值通常遵循钟形曲线，其中大多数值聚集在平均值附近，而偏离平均值越远的值可能性越小。正态（高斯）分布常用于模拟此类现象。

### 练习 2：计算二项概率

想象一个质量控制过程，其中制造的小部件有缺陷的概率为 $p = 0.05$。您检查了一批 $n = 10$ 个小部件。设 $X$ 为表示批次中缺陷小部件数量的随机变量。

1. 小部件*恰好*有一个有缺陷的概率是多少 ($P(X=1)$)？
2. 小部件*至多*有一个有缺陷的概率是多少 ($P(X \le 1)$)？

**解决方法：**

此情景符合二项分布，因为：

- 有固定数量的试验（$n=10$个小部件）。
- 每次试验都是独立的（假设一个小部件的缺陷不会影响其他小部件）。
- 每次试验有两种结果（有缺陷或无缺陷）。
- 成功（有缺陷）的概率是常数（$p=0.05$）。

二项分布的概率质量函数 (PMF) 给出在 $n$ 次试验中恰好观察到 $k$ 次成功的概率：


$$
P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}
$$


其中 $\binom{n}{k} = \frac{n!}{k!(n-k)!}$ 是二项式系数，表示从 $n$ 次试验中选择 $k$ 次成功的方法数。

**计算：**

1. **恰好一个缺陷小部件的概率 ($k=1$)：**

   - $n = 10$， $k = 1$， $p = 0.05$
   - $\binom{10}{1} = \frac{10!}{1!(10-1)!} = \frac{10}{1} = 10$
   - $P(X=1) = \binom{10}{1} (0.05)^1 (1-0.05)^{10-1} = 10 \times 0.05 \times (0.95)^9$
   - $P(X=1) \approx 10 \times 0.05 \times 0.6302 = 0.3151$
     因此，恰好一个缺陷小部件的概率约为31.5%。
2. **至多一个缺陷小部件的概率 ($P(X \le 1)$)：**

   - 这意味着零个小部件有缺陷 ($k=0$) 或者恰好一个有缺陷 ($k=1$)。
   - $P(X \le 1) = P(X=0) + P(X=1)$
   - 首先，计算 $P(X=0)$：
     - $\binom{10}{0} = \frac{10!}{0!(10-0)!} = 1$
     - $P(X=0) = \binom{10}{0} (0.05)^0 (1-0.05)^{10-0} = 1 \times 1 \times (0.95)^{10}$
     - $P(X=0) \approx 0.5987$
   - 现在将 $P(X=0)$ 和 $P(X=1)$ 相加：
     - $P(X \le 1) \approx 0.5987 + 0.3151 = 0.9138$
       因此，批次中一个或更少小部件有缺陷的概率约为91.4%。

**使用 Python (可选)：**

```python
import scipy.stats as stats

n = 10
p = 0.05

# 恰好 k=1 次成功的概率
prob_k1 = stats.binom.pmf(k=1, n=n, p=p)
print(f"P(X=1) = {prob_k1:.4f}") # 输出：P(X=1) = 0.3151

# 至多 k=1 次成功的概率（使用累积分布函数 - CDF）
prob_k_le_1 = stats.binom.cdf(k=1, n=n, p=p)
print(f"P(X<=1) = {prob_k_le_1:.4f}") # 输出：P(X<=1) = 0.9139
# 注意：与手动计算结果略有差异，因为手动计算有舍入
```

### 练习 3：处理正态分布

假设标准化考试的成绩服从正态分布，平均值（$\mu$）为1000，标准差（$\sigma$）为150。设 $X$ 为随机选择的学生的分数。

1. 学生得分低于850的概率是多少 ($P(X < 850)$)？
2. 学生得分在900到1100之间的概率是多少 ($P(900 < X < 1100)$)？

**解决方法：**

对于正态分布，我们通过计算概率密度函数 (PDF) 曲线下的面积来获得概率。由于我们无法手动轻松积分 PDF，因此通常将分数 ($X$) 转换为标准分数 (Z分数)，然后使用标准正态分布表或软件函数（如 SciPy 的 `cdf`）。

Z分数告诉我们一个值距离平均值有多少个标准差：


$$
Z = \frac{X - \mu}{\sigma}
$$


标准正态分布的 $\mu=0$ 且 $\sigma=1$。

**计算：**

1. **得分低于850的概率 ($P(X < 850)$)：**

   - 将 $X = 850$ 转换为 Z分数：
     $Z = (850 - 1000) / 150 = -150 / 150 = -1.0$
   - 我们需要找到 $P(Z < -1.0)$。这对应于标准正态曲线中 $Z = -1.0$ 左侧的面积。
   - 使用标准正态分布表或 `scipy.stats.norm.cdf( )`，我们找到这个概率。
   - $P(Z < -1.0) \approx 0.1587$
     因此，学生得分低于850的概率约为15.9%。
2. **得分在900到1100之间的概率 ($P(900 < X < 1100)$)：**

   - 将两个分数都转换为 Z分数：
     - $Z_1 = (900 - 1000) / 150 = -100 / 150 \approx -0.67$
     - $Z_2 = (1100 - 1000) / 150 = 100 / 150 \approx 0.67$
   - 我们需要 $P(-0.67 < Z < 0.67)$。这是标准正态曲线中 $Z = -0.67$ 和 $Z = 0.67$ 之间的面积。
   - 我们将其计算为 $P(Z < 0.67) - P(Z < -0.67)$。
   - 使用表格或软件：
     - $P(Z < 0.67) \approx 0.7486$
     - $P(Z < -0.67) \approx 0.2514$
   - $P(900 < X < 1100) = P(-0.67 < Z < 0.67) \approx 0.7486 - 0.2514 = 0.4972$
     因此，学生得分在900到1100之间的概率约为49.7%。

**使用 Python (可选)：**

```python
import scipy.stats as stats

mu = 1000
sigma = 150

# P(X < 850)
prob_below_850 = stats.norm.cdf(x=850, loc=mu, scale=sigma)
print(f"P(X < 850) = {prob_below_850:.4f}") # 输出：P(X < 850) = 0.1587

# P(900 < X < 1100) = P(X < 1100) - P(X < 900)
prob_below_1100 = stats.norm.cdf(x=1100, loc=mu, scale=sigma)
prob_below_900 = stats.norm.cdf(x=900, loc=mu, scale=sigma)
prob_between = prob_below_1100 - prob_below_900
print(f"P(900 < X < 1100) = {prob_between:.4f}") # 输出：P(900 < X < 1100) = 0.4950
# 注意：与SciPy中更精确的Z分数计算结果略有差异
```

以下是表示 $P(900 < X < 1100)$ 区域的可视化：



![正态分布 (μ=1000, σ=150)](plots/2458-0.json)



> 阴影区域表示正态分布（$\mu=1000$ 和 $\sigma=150$）的概率 $P(900 < X < 1100)$。

### 练习 4：从分布中抽样

如前所述，我们可以使用 NumPy 等 Python 库生成遵循特定概率分布的随机样本。这对于模拟以及了解数据中预期的规律很有用。

1. 从平均值 $\mu = 50$ 和标准差 $\sigma = 5$ 的正态分布中生成1000个随机样本。
2. 从 $n = 20$ 次试验、成功概率 $p = 0.7$ 的二项分布中生成1000个随机样本。
3. 如果您要为步骤1中生成的样本创建直方图，您预计它会是什么形状？步骤2中的样本又会怎样？

**Python 实现：**

```python
import numpy as np
import matplotlib.pyplot as plt # 可选：用于可视化

# 1. 正态分布样本
mu = 50
sigma = 5
normal_samples = np.random.normal(loc=mu, scale=sigma, size=1000)
# 打印“前10个正态样本:”，normal_samples[:10]

# 2. 二项分布样本
n = 20
p = 0.7
binomial_samples = np.random.binomial(n=n, p=p, size=1000)
# 打印“前10个二项样本:”，binomial_samples[:10]

# 可选：可视化直方图
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1) # 1行，2列，图1
plt.hist(normal_samples, bins=30, density=True, color='#74c0fc', alpha=0.7)
plt.title('正态样本直方图 (\u03bc=50, \u03c3=5)')
plt.xlabel('数值')
plt.ylabel('密度')

plt.subplot(1, 2, 2) # 1行，2列，图2
# 计算离散值的 bin 边界
bins = np.arange(binomial_samples.min(), binomial_samples.max() + 2) - 0.5
plt.hist(binomial_samples, bins=bins, density=True, color='#f783ac', alpha=0.7)
plt.title('二项样本直方图 (n=20, p=0.7)')
plt.xlabel('成功次数')
plt.ylabel('概率')

plt.tight_layout()
# plt.show() # 如在本地运行，取消注释以显示图表
```

**答案：**

3. - **正态样本直方图：** 您会预期正态样本的直方图大致呈**钟形**，以平均值 $\mu = 50$ 为中心。直方图的分布会反映标准差 $\sigma = 5$。拥有1000个样本，它应该很好地近似于理论正态 PDF。
   - **二项样本直方图：** 您会预期二项样本的直方图会显示成功次数的分布。由于 $p=0.7$ 大于0.5，直方图可能会**略微向左倾斜**，峰值（最常出现的结果）在 $n \times p = 20 \times 0.7 = 14$ 次成功附近。这将是一个离散直方图，条形只出现在0到20之间的整数值上。

这些练习为使用概率分布提供了一个起点。当您在机器学习 (machine learning)中遇到更复杂的情景时，您会发现理解这些基本的分布、它们的属性以及如何进行计算是很有用的。

## 参考资料

- [Probability and Statistics for Engineering and the Sciences](https://www.cengage.com/c/probability-and-statistics-for-engineering-and-the-sciences-9e-devore/9781305251809/) — Jay L. Devore (2016)
  Publisher: Cengage Learning
  一本广泛使用的教材，全面介绍概率论基础、离散和连续概率分布及其应用。
- [\`scipy.stats\` module documentation](https://docs.scipy.org/doc/scipy/reference/stats.html) — SciPy developers (2023)
  Python中统计函数的官方文档，包括各种分布的概率质量/密度函数（PMF/PDF）和累积分布函数（CDF）。
- [Random sampling (\`numpy.random\`)](https://numpy.org/doc/stable/reference/random/index.html) — NumPy developers (2023)
  使用NumPy的`random`模块从各种概率分布生成随机样本的官方文档。
- [Think Stats: Exploratory Data Analysis in Python](http://greenteapress.com/wp/think-stats-2e/) — Allen B. Downey (2014)
  Publisher: O'Reilly Media
  一本使用Python进行统计分析的实用教材，涵盖概率分布、抽样和数据可视化。
