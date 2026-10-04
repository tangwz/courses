---
course: "probability-statistics-essentials-ml"
chapter: "inferential-statistics-sampling-estimation"
lesson: "understanding-point-estimates"
sourceId: 1426
sourceUrl: "https://apxml.com/zh/courses/probability-statistics-essentials-ml/chapter-4-inferential-statistics-sampling-estimation/understanding-point-estimates"
title: "理解点估计"
description: "定义点估计及其特性（例如，偏差、方差）。"
order: 4
plots: []
sourceHash: "b44bbe29cf2a8457873c93bab463d80fdf3f84065aaf709c97742d170643283a"
sourceCorrections: []
---

您已经掌握了我们通常使用样本进行工作的原因，因为研究整个总体是不实际的。您也知道，多亏了中心极限定理等规则，从样本中计算出的统计量（如样本均值）会表现出可预测性，尤其是在样本数量增加时。现在，我们来关注如何使用特定的样本统计量对总体特征进行最佳估算。

这种从样本数据计算出的单个数值估算被称为**点估计**。这是我们最直接地确定未知总体参数 (parameter)值的方式。

这样来考虑：您想知道某个大型国家公园中所有成年长颈鹿的平均身高 ($\mu$)（总体）。测量每一只长颈鹿是不可能的。因此，您随机抽取了30只长颈鹿作为样本，并计算出它们的平均身高，比如 $\bar{x} = 5.2$ 米。这个值，5.2 米，就是您对公园中*所有*长颈鹿真实平均身高 ($\mu$) 的点估计。

### 估计量与估计值

区分用于估计的*规则*和从特定样本中获得的*结果*会有帮助。

- **估计量：** 这是从样本数据计算估计值的规则、公式或方法。它通常用包含样本中随机变量的公式来表示。例如，样本均值的公式 $\bar{X} = \frac{1}{n} \sum_{i=1}^{n} X_i$ 是总体均值 $\mu$ 的一个*估计量*。
- **估计值：** 这是当您将估计量应用于样本的实际数据时获得的具体数值。在我们的长颈鹿例子中，$\bar{x} = 5.2$ 米是 $\mu$ 的*估计值*。

如果您抽取了另一组30只长颈鹿的样本，您很可能会得到略微不同的平均身高，从而产生不同的点估计。估计量（公式）保持不变，但估计值会因样本而异。

### 常用点估计量

以下是常见总体参数 (parameter)最常用的点估计量：

1. **样本均值 ($\bar{X}$)：** 用于估计**总体均值 ($\mu$)**。
   $\bar{X} = \frac{1}{n} \sum_{i=1}^{n} X_i$
   其中 $X_i$ 是样本中的单个观测值，$n$ 是样本量。这可能是最直观的估计量。
2. **样本方差 ($S^2$)：** 用于估计**总体方差 ($\sigma^2$)**。
   $S^2 = \frac{1}{n-1} \sum_{i=1}^{n} (X_i - \bar{X})^2$
   请注意，分母是 $n-1$，而不是 $n$。这被称为贝塞尔校正。使用 $n-1$ 使 $S^2$ 成为 $\sigma^2$ 的*无偏*估计量，这是我们很快会谈到的一个理想特性。
3. **样本标准差 ($S$)：** 用于估计**总体标准差 ($\sigma$)**。
   $S = \sqrt{S^2} = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} (X_i - \bar{X})^2}$
   这只是样本方差的平方根。有意思的是，虽然 $S^2$ 是 $\sigma^2$ 的无偏估计量，但 $S$ 通常是 $\sigma$ 的*有偏*估计量（尽管偏差会随着 $n$ 的增加而减小）。尽管如此，它仍然是实践中使用的标准估计量。
4. **样本比例 ($\hat{p}$)：** 用于估计分类数据（例如，点击广告的用户比例）的**总体比例 ($p$)**。
   $\hat{p} = \frac{X}{n}$
   其中 $X$ 是样本中“成功”（具有感兴趣特征的项目）的数量，$n$ 是样本量。

### 什么是好的估计量？

我们如何判断像 $\bar{X}$ 或 $S^2$ 这样的估计量是否优良？统计学家根据一些特性来评估估计量：

1. **无偏性：** 如果估计量 $\hat{\theta}$ 的期望值等于参数 (parameter) $\theta$ 的真实值，那么 $\hat{\theta}$ 就是 $\theta$ 的无偏估计量。形式上，即 $E[\hat{\theta}] = \theta$。

   - **含义：** 如果您抽取很多很多随机样本，并从每个样本中计算估计值，那么所有这些估计值的平均值将收敛到真实的总体参数。
   - **例子：** 样本均值 $\bar{X}$ 是 $\mu$ 的无偏估计量，这意味着 $E[\bar{X}] = \mu$。样本方差 $S^2$（使用 $n-1$）是 $\sigma^2$ 的无偏估计量，因此 $E[S^2] = \sigma^2$。如果我们在样本方差公式的分母中使用 $n$，则得到的估计量将是有偏的（它会系统性地低估 $\sigma^2$）。
2. **有效性：** 在一个参数的所有无偏估计量中，方差最小的估计量被认为是最有效的。

   - **含义：** 有效的估计量与效率较低的估计量相比，更可能得到接近真实参数值的估计值。它的抽样分布更紧密地聚集在真实参数值周围。
   - **例子：** 对于正态分布总体的数据，样本均值 $\bar{X}$ 是 $\mu$ 最有效的无偏估计量。
3. **一致性：** 如果随着样本量 $n$ 的增加，估计值越来越接近总体参数的真实值，那么估计量就是一致的。更正式地说，估计量依概率收敛于该参数。

   - **含义：** 较大的样本可以得到更好的估计。这符合我们的直觉。
   - **例子：** 样本均值 $\bar{X}$ 和样本方差 $S^2$ 分别是 $\mu$ 和 $\sigma^2$ 的一致估计量。大数定律是关于样本均值一致性的一个陈述。

偏差与方差的这个认识在机器学习 (machine learning)中也很重要。有时，如果一个略有偏差的估计量具有明显更低的方差，它可能会更受青睐，从而导致更低的总体误差（通常通过均方误差 $MSE = 方差 + 偏差^2$ 来衡量）。

### 在Python中计算点估计

让我们快速看看如何使用NumPy和Pandas计算这些常见估计值。假设我们有一个存储在名为 `data` 的NumPy数组或Pandas Series中的样本。

```python
import numpy as np
import pandas as pd

# 样本数据（例如，10只抽样长颈鹿的身高，单位：米）
data = np.array([5.1, 4.9, 5.5, 5.2, 4.8, 5.3, 5.0, 5.4, 4.7, 5.1])

# 使用 NumPy
sample_mean_np = np.mean(data)
sample_var_np = np.var(data, ddof=1) # ddof=1 使用 n-1 作为分母（无偏）
sample_std_np = np.std(data, ddof=1) # ddof=1 使用 n-1 作为分母

print(f"NumPy 均值估计值: {sample_mean_np:.2f}")
print(f"NumPy 方差估计值: {sample_var_np:.2f}")
print(f"NumPy 标准差估计值: {sample_std_np:.2f}")

# 使用 Pandas（如果数据是 Pandas Series）
data_series = pd.Series(data)
sample_mean_pd = data_series.mean()
sample_var_pd = data_series.var()      # Pandas 默认使用 ddof=1
sample_std_pd = data_series.std()      # Pandas 默认使用 ddof=1

print(f"\nPandas 均值估计值: {sample_mean_pd:.2f}")
print(f"Pandas 方差估计值: {sample_var_pd:.2f}")
print(f"Pandas 标准差估计值: {sample_std_pd:.2f}")

# 比例的例子（假设10只长颈鹿中有6只是雄性）
num_success = 6
sample_size = 10
sample_proportion = num_success / sample_size
print(f"\n样本比例估计值: {sample_proportion:.2f}")
```

请注意NumPy的 `np.var` 和 `np.std` 函数中的 `ddof=1` 参数 (parameter)。`ddof` 代表“自由度调整”。将其设为1会告诉NumPy在分母中使用 $n-1$，从而得到无偏的样本方差估计值 $S^2$。默认值（`ddof=0`）使用 $n$，这对应于方差的最大似然估计但有偏。Pandas计算方差和标准差的方法默认使用 $n-1$。

### 点估计的局限性

像 $\bar{x} = 5.1$ 米这样的点估计为我们提供了总体参数 (parameter) $\mu$ 的单个最佳估算。然而，它们没有告诉我们与该估算相关的*不确定性*信息。我们有多大把握认为真实均值 $\mu$ 精确地是 5.1 米？可能把握不大。更可能的是，真实均值*大约*在 5.1 米附近。

由于任何单个样本都不太可能完美反映总体，我们的点估计几乎总是与真实总体参数略有不同。这种源于随机抽样的固有不确定性，直接引出了对*区间估计*的需求，区间估计通常称为置信区间，我们将在下一节中探讨。置信区间为参数提供了一个合理值的范围，并包含了我们估计过程中的不确定性。

## 参考资料

- [numpy.var](https://numpy.org/doc/stable/reference/generated/numpy.var.html) — NumPy Community ()
  NumPy官方文档，解释了用于计算样本方差的`ddof`参数，这对于理解NumPy中的无偏估计很重要。
- [pandas.Series.var](https://pandas.pydata.org/docs/reference/api/pandas.Series.var.html) — pandas development team ()
  Pandas官方文档，详细说明了pandas中计算样本方差时`ddof`的默认行为，与NumPy的默认行为形成对比。
