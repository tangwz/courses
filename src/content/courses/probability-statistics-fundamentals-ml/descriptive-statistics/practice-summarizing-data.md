---
course: "probability-statistics-fundamentals-ml"
chapter: "descriptive-statistics"
lesson: "practice-summarizing-data"
sourceId: 2425
sourceUrl: "https://apxml.com/zh/courses/probability-statistics-fundamentals-ml/chapter-2-descriptive-statistics/practice-summarizing-data"
title: "实践：数据集汇总"
description: "应用描述性统计技术，使用Python分析并汇总样本数据集。"
order: 8
plots: ["plots/2425-0.json", "plots/2425-1.json"]
sourceHash: "aeab0bbd4a0f6c33bfd7b5922ee8aa93ad55f4fcea0e8a8ee9e67907c0a8eea7"
sourceCorrections: []
---

使用Python的Pandas库加载一个小型数据集，计算主要描述性统计量并创建可视化图表，从而理解数据特征。

假设我们已收集了某个城市两周的每日气温数据（摄氏度）。

**我们的样本数据集：**

每日气温（°C）：`[22, 25, 19, 21, 24, 26, 23, 20, 22, 25, 28, 24, 21, 23]`

**1. 设置与加载数据**

首先，请确保已安装Pandas (`pip install pandas`)。我们将用它来高效地管理和分析数据。让我们将气温数据加载到Pandas Series中，它类似于数据的单列。

```python
import pandas as pd
import numpy as np # 通常与Pandas一同使用很有用

# 我们的气温数据
temperatures_c = [22, 25, 19, 21, 24, 26, 23, 20, 22, 25, 28, 24, 21, 23]

# 创建一个Pandas Series
temp_series = pd.Series(temperatures_c, name="每日气温 (C)")

# 显示Series
print(temp_series)
```

**2. 计算集中趋势量度**

现在，让我们使用均值、中位数和众数来找出典型气温。Pandas为此提供了直接的方法。

```python
# 计算均值
mean_temp = temp_series.mean()
print(f"平均气温：{mean_temp:.2f} °C")

# 计算中位数
median_temp = temp_series.median()
print(f"中位数气温：{median_temp:.2f} °C")

# 计算众数
# 注意：.mode() 返回一个 Series，因为可能存在多个众数。
# 如果存在，我们将取第一个。
mode_temp = temp_series.mode()
if not mode_temp.empty:
    print(f"众数气温：{list(mode_temp)} °C")
else:
    print("未找到唯一众数。")
```

- **解读：** 均值表示平均气温（$\approx 23.07^\circ C$）。中位数（$23.0^\circ C$）是数据排序后的中间值，受极端值影响较小。众数（$[21, 22, 23, 24, 25]^\circ C$）是出现频率最高的气温。在本例中，有几个气温出现了两次，这表明中心部分的分布相对平坦。均值和中位数非常接近，这表明数据分布大致对称。

**3. 计算离散程度量度**

气温的日间变化有多大？让我们计算极差、方差和标准差。

```python
# 计算极差
temp_range = temp_series.max() - temp_series.min()
print(f"气温极差：{temp_range} °C")

# 计算方差
variance_temp = temp_series.var() # 默认使用 N-1 作为分母（样本方差）
print(f"气温方差：{variance_temp:.2f} °C^2")

# 计算标准差
std_dev_temp = temp_series.std() # 默认使用 N-1 作为分母（样本标准差）
print(f"气温标准差：{std_dev_temp:.2f} °C")
```

- **解读：** 极差（$9^\circ C$）表示我们样本中最高温和最低温之间的差异。方差（$5.53^\circ C^2$）和标准差（$\approx 2.35^\circ C$）量化 (quantization)了数据点围绕均值的平均离散程度。标准差为2.35表明大多数每日气温大致落在 $23.07 \pm 2.35^\circ C$ 范围内。

**4. 计算百分位数和四分位数**

让我们找出四分位数，以更好地理解数据的分布。

```python
# 计算四分位数（第25、50、75百分位数）
quartiles = temp_series.quantile([0.25, 0.50, 0.75])
print("\n四分位数：")
print(quartiles)

# 计算四分位距（IQR）
q1 = quartiles[0.25]
q3 = quartiles[0.75]
iqr = q3 - q1
print(f"\n四分位距 (IQR)：{iqr:.2f} °C")
```

- **解读：**
  - Q1（第25百分位数）是 $21.25^\circ C$。25%的天数气温低于或等于此值。
  - Q2（第50百分位数）是 $23.0^\circ C$，与中位数相同，符合预期。
  - Q3（第75百分位数）是 $24.75^\circ C$。75%的天数气温低于或等于此值。
  - IQR（$3.5^\circ C$）表示数据中间50%的分布范围。

**5. 数据可视化**

可视化通常能提供仅凭原始数据无法获得的见解。让我们创建一个直方图和一个箱线图。我们将使用Plotly来创建交互式网页图表。

**直方图**

直方图显示了气温落在不同区间内的频率。



![每日气温分布](plots/2425-0.json)



> 直方图显示了特定气温范围内的天数计数。

- **解读：** 直方图直观地证实了我们之前的观察结果。我们看到气温在21-25°C范围附近出现峰值，与我们计算出的众数相符。分布形状大致呈钟形，尽管略有分散，但这与均值和中位数接近的情况一致。

**箱线图**

箱线图提供了分布的简洁概览，显示了中位数、四分位数和可能的异常值。



![每日气温概览](plots/2425-1.json)



> 箱线图显示了中位数（橙线）、四分位数（箱体边缘）、范围（触须）和单个数据点。

- **解读：** 箱线图清晰地显示了中位数（$23^\circ C$）、IQR（箱体本身覆盖从 $21.25^\circ C$ 到 $24.75^\circ C$ 的范围），以及通过“触须”表示的整体范围（从最低的 $19^\circ C$ 延伸到最高的 $28^\circ C$）。单个数据点也已绘制出来。在此例中，由于没有数据点远离典型范围（根据标准1.5 \* IQR规则将被归类为异常值），所以“触须”很可能延伸到最小值/最大值。

**总结**

通过应用这些描述性统计方法和可视化技术，我们已将一个简单的气温列表转化为有意义的总结。我们了解了典型气温（约 $23^\circ C$）、变异性（标准差约 $\pm 2.35^\circ C$），以及整体分布形状。数据汇总的这个过程是任何数据分析或机器学习 (machine learning)任务中的基本第一步。当您遇到更大、更复杂的数据集时，这些基本技术对于获取初步认识仍将是重要的。

## 参考资料

- [Practical Statistics for Data Scientists: 50 Essential Concepts](https://www.oreilly.com/library/view/practical-statistics-for/9781492072942/) — Peter Bruce, Andrew Bruce, and Peter Gedeck (2020)
  Publisher: O'Reilly Media
  侧重于统计概念及其在数据科学中的应用，包括集中趋势和离散度量。
- [Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython](https://www.oreilly.com/library/view/python-for-data-analysis/9781098103989/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  Python数据操作和分析的基础资源，详细介绍了Pandas Series和数据汇总。
- [Fundamentals of Data Visualization: A Primer on Making Informative and Compelling Figures](https://clauswilke.com/dataviz/) — Claus O. Wilke (2019)
  Publisher: O'Reilly Media
  讨论了有效数据可视化的原则，包括对直方图和箱线图等统计图表的详细解释。
