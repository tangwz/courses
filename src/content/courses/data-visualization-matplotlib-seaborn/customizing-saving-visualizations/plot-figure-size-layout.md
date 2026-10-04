---
course: "data-visualization-matplotlib-seaborn"
chapter: "customizing-saving-visualizations"
lesson: "plot-figure-size-layout"
sourceId: 1210
sourceUrl: "https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-7-customizing-saving-visualizations/plot-figure-size-layout"
title: "调整图形大小和布局"
description: "学习如何设置图表的整体尺寸并管理多个子图的布局。"
order: 3
plots: []
sourceHash: "e2700b104e2837ce8531f7f77829fd2ff417cfcb6473b62806c083385ac3f6bd"
sourceCorrections: []
---

在您制作可视化图表时，您会发现默认大小通常不太合适。也许图表显得拥挤，或者对于您打算放置它的文档或网页来说，它可能过大。同样地，您可能希望在单个图像中同时显示多个相关图表以便比较。本节介绍如何控制绘图区域的整体大小以及如何在其内部排列多个图表。

### 控制整体图形大小

包含您的一个或多个图表的整个图像在 Matplotlib 中被称为 `Figure`。可以将其视为您进行绘画的画布。您可以在创建时指定此画布的大小。最常见的方法是使用 `figsize` 参数 (parameter)，该参数需要一个以英寸为单位的 `(宽度, 高度)` 元组。

您可以在最初使用 `plt.figure()` 创建图形时设置图形大小，或者，更常见的是在使用 `plt.subplots()` 创建图形及其关联的坐标轴时设置。

```python
import matplotlib.pyplot as plt
import numpy as np

# Sample data
# 示例数据
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create a figure with a specific size (e.g., 8 inches wide, 4 inches tall)
# 创建一个特定大小的图形（例如，8英寸宽，4英寸高）
fig1, ax1 = plt.subplots(figsize=(8, 4))
ax1.plot(x, y)
ax1.set_title('Figure Size: 8x4 inches') # 图形大小：8x4 英寸
ax1.set_xlabel('X-axis') # X轴
ax1.set_ylabel('Y-axis') # Y轴
plt.show()

# Create another figure with a different size (e.g., 6 inches wide, 6 inches tall)
# 创建另一个不同大小的图形（例如，6英寸宽，6英寸高）
fig2, ax2 = plt.subplots(figsize=(6, 6))
ax2.plot(x, y, color='#fa5252') # Using a red color from the palette # 使用调色板中的红色
ax2.set_title('Figure Size: 6x6 inches') # 图形大小：6x6 英寸
ax2.set_xlabel('X-axis') # X轴
ax2.set_ylabel('Y-axis') # Y轴
plt.show()
```

改变 `figsize` 会影响纵横比（宽度与高度之比）以及可用于图表元素（如标题、标签和绘图区域本身）的整体空间。尝试不同的 `figsize` 值有助于您找到最适合您的数据和演示需求的尺寸。

另一个相关参数是 `dpi`（每英寸点数），它控制图形的分辨率，在将图表保存到文件时尤为重要（在“将图表保存到文件”部分中介绍）。更高的 DPI 会生成更高分辨率的图像。您可以将其与 `figsize` 一起设置：`plt.subplots(figsize=(8, 4), dpi=150)`。

### 使用子图排列多个图表

通常，您需要并排或以网格形式显示多个图表，以便比较数据的不同视角或不同的数据集。Matplotlib 使用子图使这变得简单。`plt.subplots()` 函数是同时创建图形和子图网格（Axes 对象）的首选方法。

您将所需的行数和列数传递给 `plt.subplots()`。它会返回两样东西：`Figure` 对象和一个数组（如果您只创建一个子图，则是一个单独的对象），该数组包含每个子图的 `Axes` 对象。

```python
import matplotlib.pyplot as plt
import numpy as np

# Sample data
# 示例数据
x = np.linspace(0, 2 * np.pi, 100)
y_sin = np.sin(x)
y_cos = np.cos(x)
y_tan = np.tan(x)
y_exp = np.exp(-x)

# Create a 2x2 grid of subplots
# 创建一个2x2的子图网格
# figsize applies to the *entire* figure containing all subplots
# figsize 适用于包含所有子图的整个图形
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

# axes is now a 2D NumPy array: [[ax1, ax2], [ax3, ax4]]
# axes 现在是一个二维NumPy数组：[[ax1, ax2], [ax3, ax4]]
# Access individual axes using array indexing
# 使用数组索引访问单个坐标轴

# Top-left subplot
# 左上角子图
axes[0, 0].plot(x, y_sin, color='#4263eb') # Blue # 蓝色
axes[0, 0].set_title('Sine Wave') # 正弦波
axes[0, 0].set_ylabel('Amplitude') # 振幅
axes[0, 0].grid(True, linestyle=':', alpha=0.6)

# Top-right subplot
# 右上角子图
axes[0, 1].plot(x, y_cos, color='#12b886') # Teal # 青色
axes[0, 1].set_title('Cosine Wave') # 余弦波
axes[0, 1].grid(True, linestyle=':', alpha=0.6)

# Bottom-left subplot
# 左下角子图
axes[1, 0].plot(x, y_tan, color='#f76707') # Orange # 橙色
axes[1, 0].set_title('Tangent Wave (Limited y-range)') # 正切波（y轴范围受限）
axes[1, 0].set_ylim(-5, 5) # Limit y-axis for tangent # 限制正切波的y轴范围
axes[1, 0].set_xlabel('Radians') # 弧度
axes[1, 0].set_ylabel('Amplitude') # 振幅
axes[1, 0].grid(True, linestyle=':', alpha=0.6)

# Bottom-right subplot
# 右下角子图
axes[1, 1].plot(x, y_exp, color='#ae3ec9') # Grape # 葡萄紫
axes[1, 1].set_title('Exponential Decay') # 指数衰减
axes[1, 1].set_xlabel('Radians') # 弧度
axes[1, 1].grid(True, linestyle=':', alpha=0.6)

# Display the figure with all subplots
# 显示包含所有子图的图形
plt.show() # plt.show() is called once for the entire figure # plt.show() 只对整个图形调用一次
```

请注意我们如何使用索引（例如 `axes[0, 0]`）访问每个子图的 `Axes` 对象，然后在该特定的 `Axes` 对象上调用绘图方法（`.plot()`、`.set_title()` 等）。

#### 共享坐标轴

当子图显示相关数据时，让它们共享相同的 x 轴或 y 轴限制和刻度通常很有用。这使得比较变得容易得多。您可以使用 `plt.subplots()` 中的 `sharex` 和 `sharey` 参数 (parameter)来实现这一点。

```python
import matplotlib.pyplot as plt
import numpy as np

# Sample data
# 示例数据
years = np.arange(2010, 2021)
sales_a = np.random.rand(len(years)) * 100 + 50
sales_b = np.random.rand(len(years)) * 70 + 80

# Create a 2x1 grid, sharing the x-axis
# 创建一个2x1的网格，共享x轴
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 6), sharex=True)

# Top subplot
# 上方子图
axes[0].plot(years, sales_a, marker='o', linestyle='-', color='#228be6') # Blue # 蓝色
axes[0].set_title('Product A Sales') # 产品 A 销售额
axes[0].set_ylabel('Units Sold') # 销售数量
axes[0].grid(True, linestyle=':', alpha=0.6)

# Bottom subplot
# 下方子图
axes[1].plot(years, sales_b, marker='s', linestyle='--', color='#51cf66') # Green # 绿色
axes[1].set_title('Product B Sales') # 产品 B 销售额
axes[1].set_xlabel('Year') # 年份
axes[1].set_ylabel('Units Sold') # 销售数量
axes[1].grid(True, linestyle=':', alpha=0.6)

# Display the figure
# 显示图形
plt.show()
```

在此示例中，`sharex=True` 确保两个子图具有相同的 x 轴范围和刻度。请注意，x 轴标签和刻度仅显示在最下方的图表上，减少了冗余。如果 y 轴需要共享，您也可以类似地使用 `sharey=True`。

### 调整间距

有时，特别是在有多个子图的情况下，默认间距可能导致标题、坐标轴标签或刻度标签重叠。Matplotlib 提供了调整子图之间填充的方法。

#### 自动调整：`tight_layout()`

最简单的解决方法通常是 `plt.tight_layout()`。在创建所有图表之后，但在调用 `plt.show()` 或保存图形之前调用此函数。它会自动调整子图参数 (parameter)以提供指定的填充，通常会生成一个没有重叠的整洁布局。

```python
import matplotlib.pyplot as plt
import numpy as np

# Sample data
# 示例数据
x = np.linspace(0, 10, 50)

# Create a 1x2 grid, potentially with overlapping elements initially
# 创建一个1x2的网格，最初可能存在重叠元素
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

axes[0].plot(x, np.sin(x), color='#7048e8') # Violet # 紫罗兰色
axes[0].set_title('Sine Wave with a Long Title') # 带有长标题的正弦波
axes[0].set_xlabel('X Value') # X 值
axes[0].set_ylabel('Sine of X') # X 的正弦值

axes[1].plot(x, np.cos(x), color='#f06595') # Pink # 粉色
axes[1].set_title('Cosine Wave with another Long Title') # 带有另一个长标题的余弦波
axes[1].set_xlabel('X Value') # X 值
# axes[1].set_ylabel('Cosine of X') # Y label might overlap with left plot title without adjustment # 如果不调整，Y轴标签可能与左侧图表的标题重叠

# Apply tight_layout to adjust spacing
# 应用 tight_layout 调整间距
plt.tight_layout() # Call this before plt.show() # 在 plt.show() 之前调用此函数

plt.show()
```

如果没有 `plt.tight_layout()`，右侧图表的 y 轴标签可能与左侧图表的标题重叠。运行 `plt.tight_layout()` 通常能很好地解决这些问题。

#### 手动调整：`subplots_adjust()`

为了更精确地控制间距，您可以使用 `fig.subplots_adjust()`。此函数允许您手动设置子图周围的填充以及它们之间的空间。常见参数包括：

- `left`, `right`, `bottom`, `top`：定义子图区域的边界，作为图形宽度和高度的比例。
- `wspace`：子图列之间空间的宽度，作为平均坐标轴宽度的比例。
- `hspace`：子图行之间的高度，作为平均坐标轴高度的比例。

```python
import matplotlib.pyplot as plt
import numpy as np

# Sample data
# 示例数据
x = np.linspace(0, 10, 50)

# Create a 2x1 grid
# 创建一个2x1的网格
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 6))

axes[0].plot(x, np.sin(x), color='#15aabf') # Cyan # 青色
axes[0].set_title('Plot 1') # 图表 1

axes[1].plot(x, np.cos(x), color='#f59f00') # Yellow # 黄色
axes[1].set_title('Plot 2') # 图表 2
axes[1].set_xlabel('X Value') # X 值

# Manually adjust spacing - increase space between rows
# 手动调整间距 - 增加行之间距
fig.subplots_adjust(hspace=0.5) # Increase vertical space # 增加垂直空间

plt.show()
```

尽管 `subplots_adjust` 提供了精细控制，但 `tight_layout` 对于常见的间距问题通常足够且更易于使用，特别是对于初学者。

### 配合 Seaborn 使用布局

Seaborn 函数旨在与 Matplotlib 的 Figure 和 Axes 对象配合使用。大多数 Seaborn 绘图函数都有一个 `ax` 参数 (parameter)。您可以使用 `plt.subplots()` 创建您的图形布局，然后告诉每个 Seaborn 函数具体要在哪个 `Axes` 对象上绘制其图表。

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Sample data in a Pandas DataFrame
# Pandas DataFrame 中的示例数据
data = pd.DataFrame({
    'value': np.random.randn(200),
    'category': np.random.choice(['A', 'B'], 200)
})

# Create a 1x2 layout using Matplotlib
# 使用 Matplotlib 创建一个1x2布局
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Use Seaborn to plot on the first Axes object
# 使用 Seaborn 在第一个 Axes 对象上绘图
sns.histplot(data=data, x='value', ax=axes[0], color='#868e96') # Gray # 灰色
axes[0].set_title('Histogram (Seaborn)') # 直方图 (Seaborn)

# Use Seaborn to plot on the second Axes object
# 使用 Seaborn 在第二个 Axes 对象上绘图
sns.boxplot(data=data, x='category', y='value', ax=axes[1], palette=['#91a7ff', '#ffc078']) # Indigo, Orange # 靛蓝色，橙色
axes[1].set_title('Box Plot by Category (Seaborn)') # 按类别分组的箱线图 (Seaborn)

# Improve layout
# 改善布局
plt.tight_layout()
plt.show()
```

这种方法让您拥有 Matplotlib 布局控制的能力，结合了 Seaborn 的统计绘图功能和美学。

通过掌握图形大小和子图布局，您将能更好地控制可视化图表的呈现，使您能够创建更清晰、更具信息量且专业排版的图形。

## 参考资料

- [matplotlib.pyplot.subplots](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html) — Matplotlib Development Team (2024)
  创建图形和子图集的官方文档，详细说明了figsize、nrows、ncols、sharex和sharey等参数。
- [Matplotlib Layout Guide](https://matplotlib.org/stable/tutorials/intermediate/tight_layout_guide.html) — Matplotlib Development Team (2024)
  绘图布局定制指南，解释了如何使用`tight_layout()`进行自动间距调整，以及使用`subplots_adjust()`进行手动控制以避免重叠。
- [Figure-level vs. axes-level functions](https://seaborn.pydata.org/tutorial/function_overview.html#figure-level-vs-axes-level-functions) — Seaborn Development Team (2024)
  Seaborn文档中关于如何使用`ax`参数将Seaborn图绘制到现有Matplotlib `Axes`对象上的部分，这对于自定义布局很有用。
- [Python for Data Analysis](https://wesmckinney.com/book/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  一本广泛使用的Python数据处理和可视化书籍，其中一部分提供了对Matplotlib图形、轴和子图布局的实用理解。
