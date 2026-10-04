---
course: "probability-statistics-essentials-ml"
chapter: "descriptive-statistics-datasets"
lesson: "measures-central-tendency"
sourceId: 1388
sourceUrl: "https://apxml.com/zh/courses/probability-statistics-essentials-ml/chapter-3-descriptive-statistics-datasets/measures-central-tendency"
title: "集中趋势的度量：均值、中位数、众数"
description: "计算并理解均值、中位数和众数作为数据中集中位置的度量。"
order: 1
plots: ["plots/1388-0.json"]
sourceHash: "fa06368eaf6ee611dff7b950ef9a2f0c75c80a1b30154f7b899ba8410784f496"
sourceCorrections: []
---

了解了描述性统计的整体目标后，我们从确定数据的“中心”开始。当有人询问数据集中典型值的情况时，他们通常是在问其集中趋势。我们有三种主要方式来衡量这一点：均值、中位数和众数。每种方式都对构成“中间”或“最常见”的值提供了不同看法，了解它们的区别对于准确数据解读很重要。

### 均值：常见的平均值

均值，具体来说是*算术均值*，是最常用的集中趋势度量。它通过将数据集中所有值求和，然后除以值的数量来计算。

如果我们有一个包含 $n$ 个观测值（表示为 $x_1, x_2, ..., x_n$）的数据集，样本均值（通常用 $\bar{x}$（读作“x-bar”）表示）的计算方式如下：


$$
\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i = \frac{x_1 + x_2 + \dots + x_n}{n}
$$


例如，考虑数据集：`{2, 3, 5, 6, 9}`。
均值为 $(2 + 3 + 5 + 6 + 9) / 5 = 25 / 5 = 5$。

均值包含了数据集中的每个值，这是它的优点之一。然而，这使得它对*异常值*或极端值敏感。一个非常大或非常小的值可能会明显地将均值拉向其自身方向，从而可能会错误地代表大部分数据的中心。

**使用 Pandas 在 Python 中计算均值：**

假设您的数据存储在 Pandas Series 或 DataFrame 列中，使用 `.mean()` 方法计算均值非常简单。

```python
import pandas as pd

data = pd.Series([2, 3, 5, 6, 9, 100]) # 添加了一个异常值：100

# 计算均值
mean_value = data.mean()

print(f"数据集: {data.tolist()}")
print(f"均值: {mean_value}")
# 输出：
# 数据集: [2, 3, 5, 6, 9, 100]
# 均值: 20.833333333333332
```

请注意，当数据集为 `{2, 3, 5, 6, 9}` 时，异常值 (100) 如何导致均值相较于初始值 5 明显增加。

### 中位数：中间值

中位数是将数据集的上半部分与下半部分分开的值。要找到它，您首先需要将数据按升序排列。

- **如果数据集有奇数个观测值（$n$ 为奇数）：** 中位数是中间值。它的位置是 $(n + 1) / 2$。
- **如果数据集有偶数个观测值（$n$ 为偶数）：** 中位数是两个中间值的平均值。这些值位于 $n / 2$ 和 $(n / 2) + 1$ 的位置。

让我们回顾一下示例：

1. 数据集：`{2, 3, 5, 6, 9}` (已排序)

   - $n = 5$ (奇数)
   - 位置 = $(5 + 1) / 2 = 3$。
   - 第 3 个值是 5。中位数是 5。
2. 数据集：`{2, 3, 5, 6, 9, 100}` (已排序)

   - $n = 6$ (偶数)
   - 位置是 $6 / 2 = 3$ 和 $(6 / 2) + 1 = 4$。
   - 第 3 个值是 5，第 4 个值是 6。
   - 中位数是 $(5 + 6) / 2 = 5.5$。

中位数的主要优点是它对异常值的*稳定性*。极端值对其影响很小甚至没有影响，因为它只依赖于排序后的中间值（或中间两个值）。这使得它成为对于偏斜或包含明显异常值的数据集来说，更好的集中趋势度量。

**使用 Pandas 在 Python 中计算中位数：**

Pandas 提供了 `.median()` 方法。

```python
import pandas as pd

data_odd = pd.Series([2, 3, 5, 6, 9])
data_even = pd.Series([2, 3, 5, 6, 9, 100]) # 带异常值

# 计算中位数
median_odd = data_odd.median()
median_even = data_even.median()

print(f"数据集 1: {data_odd.tolist()}")
print(f"中位数 1: {median_odd}")
print(f"\n数据集 2 (带异常值): {data_even.tolist()}")
print(f"中位数 2: {median_even}")
# 输出：
# 数据集 1: [2, 3, 5, 6, 9]
# 中位数 1: 5.0
#
# 数据集 2 (带异常值): [2, 3, 5, 6, 9, 100]
# 中位数 2: 5.5
```

比较带异常值的数据集的均值 (20.83) 和中位数 (5.5)。中位数更能体现非异常值数据中的“典型”值。

### 众数：最常见的值

众数是数据集中出现最频繁的值。一个数据集可以有：

- **无众数：** 如果所有值出现频率相同（例如，`{1, 2, 3, 4, 5}`）。
- **一个众数（单峰）：** 如果一个值比其他任何值出现得更频繁（例如，`{1, 2, 2, 3, 4}`）。众数是 2。
- **多个众数（双峰、多峰）：** 如果两个或更多值共享最高频率（例如，`{1, 1, 2, 3, 3, 4}`）。众数是 1 和 3。

众数对于分类数据特别有用（例如，查找最常见的颜色或类别），但也可以用于数值数据，特别是离散数据。它是唯一适用于名义分类数据的集中趋势度量。与均值和中位数不同，众数不一定是唯一的。

**使用 Pandas 在 Python 中计算众数：**

Pandas 中的 `.mode()` 方法返回一个包含所有众数（因为可能不止一个）的 Series。

```python
import pandas as pd

data_unimodal = pd.Series([1, 2, 2, 3, 4, 4, 4, 5])
data_bimodal = pd.Series(['apple', 'banana', 'apple', 'orange', 'banana', 'banana'])

# 计算众数
mode_unimodal = data_unimodal.mode()
mode_bimodal = data_bimodal.mode()

print(f"数据集 1: {data_unimodal.tolist()}")
print(f"众数 1: {mode_unimodal.tolist()}")
print(f"\n数据集 2: {data_bimodal.tolist()}")
print(f"众数 2: {mode_bimodal.tolist()}")
# 输出：
# 数据集 1: [1, 2, 2, 3, 4, 4, 4, 5]
# 众数 1: [4]
#
# 数据集 2: ['apple', 'banana', 'apple', 'orange', 'banana', 'banana']
# 众数 2: ['banana']
```

### 均值、中位数或众数：如何选择？

选择在很大程度上取决于数据的性质以及您想要传达什么信息：

- **均值：** 适用于没有明显异常值的对称分布。它使用了所有数据点。
- **中位数：** 偏斜分布或含有异常值的数据首选，因为它代表了真正的中间点。
- **众数：** 最适合分类数据，且有助于找出数值数据中最常见的值。

均值、中位数和众数之间的关系也能提供关于分布的*偏度*的信息：

- **对称分布：** 均值 \u2248 中位数 \u2248 众数。
- **右偏分布（正偏）：** 均值 > 中位数 > 众数。尾部指向右侧，大值将均值向上拉。
- **左偏分布（负偏）：** 均值 < 中位数 < 众数。尾部指向左侧，小值将均值向下拉。

请看此图，显示了不同分布形状上的大致位置：



![不同分布中的均值、中位数和众数](plots/1388-0.json)



> 均值（虚线）、中位数（点线）和众数（峰值频率）的大致位置，适用于对称、右偏和左偏分布。在偏斜分布中，中位数通常介于众数和均值之间。

理解均值、中位数和众数是数据概括的第一个重要步骤。它们表示数据倾向于聚集的位置，但它们并不能说明全部情况。接下来，我们将研究如何衡量数据围绕这个中心点的离散程度。

## 参考资料

- [Practical Statistics for Data Scientists: 50+ Essential Concepts Using R and Python](https://www.oreilly.com/library/view/practical-statistics-for/9781492072935/) — Peter Bruce and Andrew Bruce (2020)
  Publisher: O'Reilly Media
  涵盖了基础描述性统计，包括均值、中位数和众数，并提供了与数据科学相关的实用示例。
- [pandas.Series API Reference](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html) — The pandas development team (2023)
  Pandas Series 方法的官方文档，包括用于计算集中趋势的 `mean()`、`median()` 和 `mode()`。
- [Think Stats: Exploratory Data Analysis in Python](https://greenteapress.com/wp/think-stats/) — Allen B. Downey (2014)
  Publisher: Green Tea Press
  提供了对描述性统计（包括集中趋势度量）的易于理解且以Python为中心的探索，并附有清晰的示例。
