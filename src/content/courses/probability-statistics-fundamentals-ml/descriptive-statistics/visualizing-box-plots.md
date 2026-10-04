---
course: "probability-statistics-fundamentals-ml"
chapter: "descriptive-statistics"
lesson: "visualizing-box-plots"
sourceId: 2420
sourceUrl: "https://apxml.com/zh/courses/probability-statistics-fundamentals-ml/chapter-2-descriptive-statistics/visualizing-box-plots"
title: "可视化汇总：箱线图"
description: "使用箱线图（又称箱须图）可视化数据汇总，包括中位数、四分位数和异常值。"
order: 6
plots: ["plots/2420-0.json"]
sourceHash: "cbb069b54c75975adbb943e5e4cf2225f84a99e3e025683dab25365520f91ff7"
sourceCorrections: []
---

箱线图（也称箱须图）提供了一种紧凑的视觉汇总，能够突出数据的特定统计量。虽然直方图能有效地呈现数据的整体形态和频率，但箱线图提供了一种标准方式，基于五数概括：最小值、第一四分位数 (Q1)、中位数 (Q2)、第三四分位数 (Q3) 和最大值，来呈现数据分布。这种图表特别适用于比较不同群体的数据分布。

### 箱线图的组成部分

一个标准箱线图包含几个主要部分：

1. **箱体：** 中央箱体表示数据中间的50%。其底部边界表示第一四分位数（Q1，即25%百分位数），顶部边界表示第三四分位数（Q3，即75%百分位数）。箱体的长度因此表示四分位距（IQR），计算方式为 $IQR = Q3 - Q1$。这个范围包含你数据点的中间一半。
2. **中位线：** 箱体内的线表示数据的中位数（Q2，即50%百分位数）。这条线在箱体中的位置可以反映数据的对称性。如果中位数靠近Q1，则中位数以下的数据比中位数以上的数据更紧密，反之亦然。
3. **触须：** 从箱体向外延伸的线，通常称为触须。通常的规定是触须延伸到距离下四分位数（Q1）1.5倍IQR范围内的最低数据点，以及距离上四分位数（Q3）1.5倍IQR范围内的最高数据点。更简单地说：
   - 下触须延伸到 $max(数据最小值, Q1 - 1.5 \times IQR)$
   - 上触须延伸到 $min(数据最大值, Q3 + 1.5 \times IQR)$
     任何落在此范围之外的数据点都被视为潜在的异常值。
4. **异常值：** 落在触须定义范围之外的数据点会单独绘制，通常以点或星号表示。这些点被标记 (token)出来以备进一步检查，因为它们异常偏离数据的中心部分。

### 为何使用箱线图？

箱线图在汇总数据方面有几个优点：

- **简洁汇总：** 它们有效地展示数据的中心（中位数）、离散程度（IQR）和范围（触须）。
- **异常值识别：** 它们提供了一种识别潜在异常值的标准视觉方法。
- **比较：** 并排放置箱线图是比较不同数据集或数据集中子组分布的有效方式。你可以快速比较它们的中位数、IQR和异常值的存在情况。
- **偏斜度指示：** 中位数在箱体中的位置以及触须的相对长度可以提供数据分布偏斜度的视觉线索。中位数靠近Q1且上触须较长表示正偏斜，而中位数靠近Q3且下触须较长表示负偏斜。

### 在 Python 中创建箱线图

Matplotlib 和 Seaborn 等 Python 库使创建箱线图变得简单，特别是在处理 Pandas DataFrame 时。让我们生成一些表示两个城市（城市A和城市B）每日温度的样本数据并进行绘图。

```python
import pandas as pd
import numpy as np
import plotly.express as px

# 生成一些样本温度数据
np.random.seed(42) # 为了结果可复现
city_a_temps = np.random.normal(loc=20, scale=5, size=100) # 平均20C，标准差5C
city_b_temps = np.random.normal(loc=25, scale=8, size=100) # 平均25C，标准差8C
# 为城市A添加几个异常值
city_a_temps = np.append(city_a_temps, [3, 45])

# 创建一个 Pandas DataFrame
df = pd.DataFrame({
    'Temperature': np.concatenate([city_a_temps, city_b_temps]),
    'City': ['City A'] * len(city_a_temps) + ['City B'] * len(city_b_temps)
})

# 使用 Plotly Express 创建箱线图
fig = px.box(df, x='City', y='Temperature', 
             color='City', # 按城市为箱体着色
             points="outliers", # 显示异常值
             title="按城市划分的每日温度分布",
             labels={'Temperature': '温度 (°C)', 'City': '城市'},
             color_discrete_map={'City A': '#1f77b4', 'City B': '#ff7f0e'} # 可选自定义颜色
            )

# 若要在 Jupyter 等环境中显示图表：
# fig.show()

# 这是用于嵌入的 JSON 表示：
```



![按城市划分的每日温度分布](plots/2420-0.json)



> 并排的箱线图，比较城市A和城市B的每日温度。请注意城市A的中位线、箱体的范围（IQR）、触须以及单独标记 (token)的异常值。

### 箱线图解读

观察上面生成的图表：

- **中位数比较：** 城市B的温度中位数（橙色箱体内的线）明显高于城市A的温度中位数（蓝色箱体内的线）。
- **离散程度 (IQR) 比较：** 城市B的箱体比城市A的箱体更高，表明城市B中间50%的温度离散程度（更高的IQR）大于城市A。城市B在其中心范围内的温度变异性更大。
- **触须和整体范围：** 触须显示典型数据点（不包括异常值）的范围。城市B的触须整体覆盖范围更广，再次表明变异性更大。
- **异常值：** 城市A显示了两个单独绘制的点，分别远远高于和低于其上下触须。这些点表示我们添加的异常温度（3°C和45°C），可能需要进一步检查。在这个样本中，根据1.5 \* IQR规则，城市B没有显示异常值。
- **偏斜度：** 在城市A的图表中，中位线大致位于箱体中心，触须大致对称（忽略异常值），表示数据主体分布相对对称。城市B的中位数也显得相对居中。

箱线图提供了一种有效方式，可以快速了解数据集分布的主要特征，是探索性数据分析中必不可少的工具，特别是在比较群体时。它们补充了从均值和标准差等统计量以及直方图等可视化中获得的见解。

## 参考资料

- [Practical Statistics for Data Scientists: 50+ Essential Concepts Using R and Python](https://www.oreilly.com/library/view/practical-statistics-for/9781492072942/) — Peter Bruce, Andrew Bruce, Peter Gedeck (2020)
  Publisher: O'Reilly Media
  为统计概念（包括箱线图）提供了实践指导，并结合Python为数据科学提供了相关示例。
- [Fundamentals of Data Visualization](https://clauswilke.com/dataviz/) — Claus O. Wilke (2019)
  Publisher: O'Reilly Media
  一本关于数据可视化原理和技术的资源，提供了对箱线图及其有效使用的详细理解。
- [Plotly Express Box Plots](https://plotly.com/python/box-plots/) — Plotly (2024)
  官方文档，说明了如何使用 Plotly Express 在 Python 中创建和自定义箱线图，直接支持代码示例。
