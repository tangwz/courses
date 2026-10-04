---
course: "intro-eda-course"
chapter: "bivariate-analysis"
lesson: "numerical-vs-numerical-scatter"
sourceId: 1166
sourceUrl: "https://apxml.com/zh/courses/intro-eda-course/chapter-4-bivariate-analysis/numerical-vs-numerical-scatter"
title: "数值变量与数值变量：散点图"
description: "创建和解读散点图以可视化两个数值变量之间的关系。"
order: 1
plots: ["plots/1166-0.json", "plots/1166-1.json", "plots/1166-2.json"]
sourceHash: "b546cdab74c2e7a7755bf8976e43c728adae32685e3d7755cf8b455d3db816e9"
sourceCorrections: []
---

在通过单变量分析审视单个变量之后，我们的重点转向检查*成对*变量之间的联系。这种双变量分析对于找出数据集中涉及多个特征的联系和规律非常必要。

双变量分析中最常见且有用的情况之一是考察两个*数值*变量之间的关系。例如，发动机排量与燃油效率有何关联？或者广告支出与销售额如何相关？观察此类关系的主要可视化工具是**散点图**。

散点图在二维图上显示单个数据点。每个点的位置由两个选定的数值变量的值决定：一个变量决定水平轴（x轴）上的位置，另一个变量决定垂直轴（y轴）上的位置。这种可视化方法使我们能够直接观察两个变量之间关联的结构、方向和强度。

### 在Python中绘制散点图

Matplotlib和Seaborn等Python库提供了生成散点图的便利函数。Seaborn的`scatterplot`函数与Pandas DataFrame顺畅结合，因此特别有用。

我们从一个简单例子开始。我们将生成一些合成数据，其中两个变量大致呈线性关系，然后使用Seaborn绘制它们。

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 生成样本数据
np.random.seed(42) # 为了可重复性
x_data = np.random.rand(100) * 10
y_data = 2.5 * x_data + np.random.randn(100) * 5 # y = 2.5x + 噪声

# 创建DataFrame
df = pd.DataFrame({'Variable_X': x_data, 'Variable_Y': y_data})

# 创建散点图
plt.figure(figsize=(8, 5)) # 设置图形大小
sns.scatterplot(data=df, x='Variable_X', y='Variable_Y')

# 添加标签和标题以提高清晰度
plt.xlabel("Variable X")
plt.ylabel("Variable Y")
plt.title("Scatter Plot of Variable Y vs. Variable X")

# 显示图形
plt.show()
```

[Image 22](https://apxml.s3.amazonaws.com/content_images/section_1166_1.webp?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAZQ3DPGY5OEILO6DN%2F20261002%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20261002T070825Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=dc00605f9a8ec10e4d29d141a304510d00e1ad740b6997fa79eed0f4d2b7e1ca)
这段代码首先创建了两个NumPy数组`x_data`和`y_data`，其中`y_data`线性依赖于`x_data`并添加了一些随机噪声。然后将这些数据放入一个Pandas DataFrame中。最后，调用`sns.scatterplot`，传入DataFrame以及x轴和y轴的列名。我们还使用Matplotlib函数添加了标签和标题，以便更好地理解。

### 解读散点图

检查散点图时，寻找这些重要特征：

1. **方向：** 点的分布是否从左到右向上漂移（正向关联），向下漂移（负向关联），还是没有明显的方向趋势？
   - *正向关联：* 随着x变量值的增加，y变量的值也倾向于增加。
   - *负向关联：* 随着x变量值的增加，y变量的值倾向于减少。
2. **形式：** 这种关系最能被表示为直线（线性）还是曲线（非线性）？是否存在明显的点群？或者该图看起来像随机散布，没有可辨别的模式？
3. **强度：** 点与整体模式的贴合程度如何？如果点紧密聚集在一条清晰的线（或曲线）周围，则关系强。如果点广泛散布，则关系弱。
4. **异常值：** 是否有任何点远离主要的点群或模式？这些异常值可能代表不寻常的数据点、测量误差或值得进一步研究的特殊情况。

以下是不同模式的可视示例：



![正向线性关系](plots/1166-0.json)



> 明显的向上趋势表示正向线性关联。



![负向线性关系](plots/1166-1.json)



> 明显的向下趋势表示负向线性关联。



![无明显关系](plots/1166-2.json)



> 点随机散布，表明X和Y之间几乎没有线性关联。

### 注意事项和自定义

- **重叠绘图：** 处理大型数据集时，许多点可能重叠，遮盖了潜在的模式。可以通过以下方式缓解这种情况：
  - 使用更小的标记 (token)（`scatterplot`或`scatter`中的`s`参数 (parameter)）。
  - 使用`alpha`参数使标记半透明（例如，`alpha=0.5`）。
  - 如果数据集非常大，则绘制数据的随机样本。
- **添加背景信息：** 始终清晰地标记您的轴（`plt.xlabel`，`plt.ylabel`）并提供信息丰富的标题（`plt.title`）。这对于传达您的发现非常重要。
- **第三个变量：** 有时，您可能希望在散点图中包含第三个变量（通常是分类变量）。Seaborn的`scatterplot`允许使用`hue`（颜色）、`size`或`style`参数，根据第三个变量的值来区分点。

散点图为两个数值变量之间的潜在关系提供了非常有价值的第一印象。它们直观地总结了关联的方向、形式和强度，指导进一步的定量分析，例如计算相关系数，我们将在接下来讨论。

## 参考资料

- [seaborn.scatterplot](https://seaborn.pydata.org/generated/seaborn.scatterplot.html) — Michael Waskom and the Seaborn developers (2023)
  Python 中用于创建散点图的 Seaborn 函数官方文档。
- [Python for Data Analysis](https://www.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  一本涵盖 Pandas、NumPy、Matplotlib 和 Python 数据可视化技术的优秀书籍。
- [Practical Statistics for Data Scientists: 50+ Essential Concepts Using R and Python](https://www.oreilly.com/library/view/practical-statistics-for/9781492072935/) — Peter Bruce and Andrew Bruce (2020)
  Publisher: O'Reilly Media
  提供实用的统计方法和概念，包括双变量分析和数据可视化，与解释散点图相关。
- [Matplotlib: Python plotting](https://matplotlib.org/stable/tutorials/introductory/pyplot.html) — Matplotlib development team (2024)
  Matplotlib 库的官方文档，用于绘图自定义并支撑 Seaborn。
