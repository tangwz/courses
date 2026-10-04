---
course: "data-visualization-matplotlib-seaborn"
chapter: "seaborn-enhanced-visuals-intro"
lesson: "seaborn-scatter-plots"
sourceId: 1186
sourceUrl: "https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-4-seaborn-enhanced-visuals-intro/seaborn-scatter-plots"
title: "使用 Seaborn 创建散点图 (scatterplot)"
description: "学习使用 Seaborn 的 scatterplot 函数制作信息丰富的散点图，包括 hue 映射。"
order: 4
plots: ["plots/1186-0.json"]
sourceHash: "68f0fb61a52258db55cf4115f3ea13bcfeccad82e6705ff131bbec9540db13d5"
sourceCorrections: []
---

散点图是用于查看两个数值变量之间关系的基本工具。尽管 Matplotlib 提供了 `scatter()` 函数，但 Seaborn 提供了 `scatterplot()`，这是一个更强大、更灵活的函数，旨在与 Pandas DataFrame 配合使用，并通过颜色、大小和标记 (token)样式等视觉属性轻松加入其他变量。

基于 Seaborn 的高层方法，`scatterplot()` 简化了直接从数据结构创建美观且信息丰富的散点图的过程。

### 使用 `scatterplot` 制作基本散点图

散点图最常见的用途是将两个变量的值互相绘制。假设您已将数据加载到 Pandas DataFrame 中。Seaborn 的 `scatterplot()` 函数使这变得简单。您通常将 DataFrame 提供给 `data` 参数 (parameter)，并使用 `x` 和 `y` 参数指定 x 轴和 y 轴的列名（字符串形式）。

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 生成一些示例数据
np.random.seed(0) # 为了结果可复现
data = pd.DataFrame({
    'x_values': np.random.rand(50) * 10,
    'y_values': 2.5 * np.random.rand(50) * 10 + np.random.randn(50) * 5
})

# 创建一个基本散点图
plt.figure(figsize=(8, 5)) # 可选：调整图表大小
sns.scatterplot(x='x_values', y='y_values', data=data)

# 添加标题和标签（使用 Matplotlib 函数）
plt.title('使用 Seaborn 的基本散点图')
plt.xlabel('X 值')
plt.ylabel('Y 值')

# 显示图表
plt.show()
```

这段代码生成了一个简单的散点图，其中每个点对应 DataFrame 中的一行，并根据其“x\_values”和“y\_values”进行定位。请注意，与 Matplotlib 的基本输出相比，Seaborn 如何应用其默认样式，使图表看起来更整洁。

### 通过视觉属性增加信息：`hue`、`size` 和 `style`

`seaborn.scatterplot` 的一个重要优点是它能够将数据集中的其他变量映射到点的视觉属性。这让您可以在一个 2D 图表中呈现更多数据维度。

#### 使用 `hue` 进行类别区分

`hue` 参数 (parameter)可能是最常用的语义映射。它根据分类列中的值给点分配不同的颜色。这对于比较不同组之间 `x` 和 `y` 的关系非常有用。

让我们使用 Seaborn 内置数据集之一“tips”，其中包含有关餐厅小费的信息。我们可以查看总账单和小费金额之间的关系，并根据星期几进行区分。

```python
import seaborn as sns
import matplotlib.pyplot as plt

# 加载示例数据集
tips = sns.load_dataset("tips")

# 创建带 hue 映射的散点图
plt.figure(figsize=(9, 6))
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="day")

plt.title('总账单与小费（按星期几）')
plt.xlabel('总账单 ($)')
plt.ylabel('小费金额 ($)')

plt.show()
```



![总账单与小费（按星期几）](plots/1186-0.json)



> 散点图比较了总账单和小费金额，其中点的颜色根据星期几显示。

Seaborn 自动为每个类别（“周四”、“周五”、“周六”、“周日”）分配不同的颜色并添加图例。

#### 使用 `size` 进行数值或类别强调

`size` 参数控制标记 (token)的面积，将另一个变量映射到此视觉属性。这对于根据数值量（如用餐人数）强调点或区分组很有用。

```python
import seaborn as sns
import matplotlib.pyplot as plt

# 加载示例数据集
tips = sns.load_dataset("tips")

# 创建带 size 映射的散点图
plt.figure(figsize=(9, 6))
sns.scatterplot(data=tips, x="total_bill", y="tip", size="size") # 'size' 列指的是用餐人数

plt.title('总账单与小费（点大小按用餐人数）')
plt.xlabel('总账单 ($)')
plt.ylabel('小费金额 ($)')

# 如有需要，调整图例位置
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.tight_layout(rect=[0, 0, 0.85, 1]) # 调整布局以留出图例空间

plt.show()
```

在这里，更大的点表示更多的用餐人数。Seaborn 处理将数值“size”列映射到适当的标记大小并生成图例。

#### 使用 `style` 进行类别标记形状区分

类似于 `hue`，`style` 参数将分类变量映射到不同的标记形状（圆形、方形、交叉等）。这提供了另一种区分组的方法，特别是在与 `hue` 结合使用时或用于黑白出版物时有用。

```python
import seaborn as sns
import matplotlib.pyplot as plt

# 加载示例数据集
tips = sns.load_dataset("tips")

# 创建带 style 映射的散点图
plt.figure(figsize=(9, 6))
sns.scatterplot(data=tips, x="total_bill", y="tip", style="time") # 'time' 列：午餐或晚餐

plt.title('总账单与小费（标记样式按时间）')
plt.xlabel('总账单 ($)')
plt.ylabel('小费金额 ($)')

plt.show()
```

现在，代表“午餐”和“晚餐”的点具有不同的标记样式。

#### 组合视觉属性映射

您可以在一个图表中组合 `hue`、`size` 和 `style` 以呈现更多信息，但请注意不要使图表过于复杂且难以理解。

```python
import seaborn as sns
import matplotlib.pyplot as plt

# 加载示例数据集
tips = sns.load_dataset("tips")

# 创建组合 hue 和 style 的散点图
plt.figure(figsize=(10, 7))
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="day", style="time")

plt.title('总账单与小费（颜色按星期，样式按时间）')
plt.xlabel('总账单 ($)')
plt.ylabel('小费金额 ($)')

# 调整图例位置
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.tight_layout(rect=[0, 0, 0.85, 1]) # 调整布局

plt.show()
```

### 自定义与 Matplotlib 集成

因为 Seaborn 是基于 Matplotlib 构建的，所以您在使用 Seaborn 创建图表后，仍然可以使用 Matplotlib 函数进行自定义。`seaborn.scatterplot` 函数会返回绘制图表所用的 Matplotlib `Axes` 对象。您可以捕获此对象并调用其方法进行精细控制，或者使用 `pyplot` 函数，如 `plt.title()`、`plt.xlabel()`、`plt.ylabel()`、`plt.xlim()` 等。

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

plt.figure(figsize=(9, 6))

# 捕获 Seaborn 返回的 Axes 对象
ax = sns.scatterplot(data=tips, x="total_bill", y="tip", hue="smoker")

# 使用 Axes 对象或 plt 函数进行自定义
ax.set_title('总账单与小费（按吸烟者状态着色）')
ax.set_xlabel('总账单金额 ($)')
plt.ylabel('收到的小费 ($)') # 使用 plt 同样有效
ax.grid(True, linestyle='--', alpha=0.6) # 添加网格

plt.show()
```

总而言之，`seaborn.scatterplot` 提供了一种便捷而强大的方式，用于在 Python 中创建散点图。它与 Pandas DataFrame 的直接集成，以及其能够轻松将数据变量映射到颜色（`hue`）、大小（`size`）和标记 (token)形状（`style`）等视觉属性的能力，使其成为查看数据内部关系的绝佳工具。请记住，在需要时，您仍保留 Matplotlib 的全部自定义能力。

## 参考资料

- [seaborn.scatterplot](https://seaborn.pydata.org/generated/seaborn.scatterplot.html) — Michael Waskom and the Seaborn development team (2024)
  Seaborn `scatterplot` 函数的官方文档，提供了所有参数的完整参考和详细示例。
- [Relational plots](https://seaborn.pydata.org/tutorial/relational.html) — Michael Waskom and the Seaborn development team (2024)
  Seaborn 官方指南中的全面教程，演示了如何使用 `scatterplot` 可视化数值变量之间的关系和语义映射。
- [Python Data Science Handbook: Essential Tools for Working with Data](https://jakevdp.github.io/PythonDataScienceHandbook/04.14-visualization-with-seaborn.html) — Jake VanderPlas (2016)
  Publisher: O'Reilly Media
  第 4.14 章“使用 Seaborn 进行可视化”介绍了 Seaborn 的功能，包括 `scatterplot`、与 Matplotlib 的集成以及用于创建信息统计图的美学映射。
