# 使用 Python 计算描述性统计

来源：[原文](https://apxml.com/zh/courses/probability-statistics-fundamentals-ml/chapter-2-descriptive-statistics/calculating-stats-python)

[返回章节目录](README.md) · [返回课程目录](../README.md)

理解均值、中位数、方差及其他描述性统计量的公式是基本的，但随着数据集的增大，手动计算它们会变得不切实际。好在，Python 提供了一些强大的库，它们专为数值计算和数据分析而设，使这些计算高效且直接。

我们将主要使用 **Pandas**，它构建在 **NumPy** 之上。Pandas DataFrame 和 Series 对象（你在第 1 章加载数据时已了解）自带了用于计算大多数常见描述性统计量的方法。

首先，确保我们已导入 Pandas 和 NumPy。我们还将创建一个表示为 Pandas Series 的小型样本数据集，以便演示这些功能。设想这些数据代表学生在近期测验中的得分。

```python
import pandas as pd
import numpy as np

# 样本测验得分
scores = pd.Series([85, 92, 78, 88, 92, 65, 78, 85, 90, 75, 88])

print(scores)
```

这将输出我们的样本数据：

```
0     85
1     92
2     78
3     88
4     92
5     65
6     78
7     85
8     90
9     75
10    88
dtype: int64
```

现在，我们来计算之前学过的描述性统计量。

### 集中趋势测量

- **均值：** 平均值。使用 `.mean()` 方法。

  ```python
  mean_score = scores.mean()
  print(f"均值： {mean_score}")
  # 输出：均值： 83.27272727272727
  ```
- **中位数：** 数据排序后的中间值。使用 `.median()` 方法。

  ```python
  median_score = scores.median()
  print(f"中位数： {median_score}")
  # 输出：中位数： 85.0
  ```
- **众数：** 出现频率最高的值。使用 `.mode()` 方法。请注意，如果多个值具有相同的最高频率，众数可能会返回多个值。它返回一个 Pandas Series。

  ```python
  mode_score = scores.mode()
  print(f"众数： \n{mode_score}")
  # 输出：
  # 众数：
  # 0    78
  # 1    85
  # 2    88
  # 3    92
  # dtype: int64
  ```

  在此例中，得分 78、85、88 和 92 各出现两次，这是最高的频率，因此它们都是众数。

### 离散程度测量（分布）

- **范围：** 最大值和最小值之间的差值。我们可以使用 `.max()` 和 `.min()` 方法来计算。

  ```python
  score_range = scores.max() - scores.min()
  print(f"得分范围： {score_range}")
  # 输出：得分范围： 27
  ```
- **方差：** 偏离均值的平方差的平均值。Pandas 的 `.var()` 方法默认计算*样本方差*，它在分母中使用 $n-1$（其中 $n$ 是数据点的数量）。这通常是处理样本时所需的值。

  ```python
  variance_score = scores.var()
  print(f"样本方差： {variance_score}")
  # 输出：样本方差： 77.61818181818182
  ```
- **标准差：** 方差的平方根，衡量数据偏离均值的典型程度。Pandas 的 `.std()` 方法默认计算*样本标准差*（样本方差的平方根）。

  ```python
  std_dev_score = scores.std()
  print(f"样本标准差： {std_dev_score}")
  # 输出：样本标准差： 8.810118150017105
  ```

### 百分位数和四分位数

Pandas 提供了 `.quantile()` 方法来计算百分位数。你将所需的百分位数 `q` 作为参数 (parameter)传入（一个 0 到 1 之间的值）。

- **四分位数：**

  - Q1（第 25 百分位数）：`q=0.25`
  - Q2（第 50 百分位数，与中位数相同）：`q=0.50`
  - Q3（第 75 百分位数）：`q=0.75`

  ```python
  q1 = scores.quantile(0.25)
  median_q2 = scores.quantile(0.50) # 与 scores.median() 相同
  q3 = scores.quantile(0.75)

  print(f"Q1（第 25 百分位数）： {q1}")
  print(f"Q2（第 50 百分位数/中位数）： {median_q2}")
  print(f"Q3（第 75 百分位数）： {q3}")
  # 输出：
  # Q1（第 25 百分位数）： 78.0
  # Q2（第 50 百分位数/中位数）： 85.0
  # Q3（第 75 百分位数）： 89.0
  ```
- **四分位距 (IQR)：** Q3 和 Q1 之间的差值。

  ```python
  iqr = q3 - q1
  print(f"四分位距 (IQR)： {iqr}")
  # 输出：四分位距 (IQR)： 11.0
  ```

你还可以计算其他百分位数，例如第 90 百分位数：

```python
p90 = scores.quantile(0.90)
print(f"第 90 百分位数： {p90}")
# 输出：第 90 百分位数： 91.0
```

### 获取快速概览

通常，你会想一次性查看这些描述性统计量中的几个。Pandas 提供了一个非常方便的方法 `.describe()`，它可以一次性计算计数、均值、标准差、最小值、最大值和主要的四分位数（25%、50%、75%）。

```python
summary_stats = scores.describe()
print("描述性统计概览：")
print(summary_stats)
```

这将输出一个简洁的概览：

```
描述性统计概览：
count    11.000000
mean     83.272727
std       8.810118
min      65.000000
25%      78.000000
50%      85.000000
75%      89.000000
max      92.000000
dtype: float64
```

这个简单的命令让你能对数据的分布和集中趋势有一个很好的初步了解。

### 使用 NumPy

如果你的数据碰巧是一个 NumPy 数组，而不是 Pandas Series 或 DataFrame，NumPy 也提供了类似的功能：

```python
# 将 Pandas Series 转换为 NumPy 数组
scores_np = scores.to_numpy()

# NumPy 计算
mean_np = np.mean(scores_np)
median_np = np.median(scores_np)
# 注意：NumPy 没有像 Pandas 那样的直接众数函数。
# 为此需要 SciPy：from scipy import stats; stats.mode(scores_np)

var_np = np.var(scores_np) # 默认 ddof=0（总体方差）
std_np = np.std(scores_np) # 默认 ddof=0（总体标准差）

var_np_sample = np.var(scores_np, ddof=1) # 样本方差（与 Pandas 类似）
std_np_sample = np.std(scores_np, ddof=1) # 样本标准差（与 Pandas 类似）

q1_np = np.percentile(scores_np, 25) # 使用 np.percentile 计算四分位数

print(f"\nNumPy 均值： {mean_np}")
print(f"NumPy 中位数： {median_np}")
print(f"NumPy 总体方差： {var_np}")
print(f"NumPy 样本方差 (ddof=1)： {var_np_sample}")
print(f"NumPy Q1： {q1_np}")
```

请注意方差和标准差的一个重要区别：NumPy 的默认值 (`ddof=0`) 计算的是*总体*方差/标准差（除以 $n$），而 Pandas 的默认值 (`ddof=1`) 计算的是*样本*方差/标准差（除以 $n-1$）。在处理样本以推断总体特征时（这在机器学习 (machine learning)中很常见），通常更推荐使用样本版本（使用 `ddof=1`）。你可以在 NumPy 中使用 `ddof`（自由度调整）参数 (parameter)来控制此行为。

如你所见，Pandas 和 NumPy 等 Python 库使得计算描述性统计量极为便捷。这使你能够快速概括和理解数据集的基本特征，这是任何数据分析或机器学习任务中的一个基本步骤。

## 参考资料

- [pandas.DataFrame.describe](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html) — pandas development team (2024)
  Pandas官方文档中关于.describe()方法及相关统计函数的说明，该方法能总结关键的描述性统计量。
- [NumPy Reference: Statistical functions](https://numpy.org/doc/stable/reference/routines.statistics.html) — NumPy community (2024)
  NumPy官方文档中关于统计函数的说明，包括方差和标准差计算中'ddof'参数的细节。
- [Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython](https://www.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  Pandas创建者撰写的基础指南，展示了Pandas和NumPy在数据分析中的实际应用，包括计算描述性统计量。

---

[上一节](06-%E5%8F%AF%E8%A7%86%E5%8C%96%E6%B1%87%E6%80%BB%EF%BC%9A%E7%AE%B1%E7%BA%BF%E5%9B%BE.md) · [下一节](08-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%95%B0%E6%8D%AE%E9%9B%86%E6%B1%87%E6%80%BB.md)
