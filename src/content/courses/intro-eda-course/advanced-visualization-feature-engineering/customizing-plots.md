---
course: "intro-eda-course"
chapter: "advanced-visualization-feature-engineering"
lesson: "customizing-plots"
sourceId: 1181
sourceUrl: "https://apxml.com/zh/courses/intro-eda-course/chapter-5-advanced-visualization-feature-engineering/customizing-plots"
title: "自定义图表以提高清晰度（标题、标签、图例）"
description: "了解如何通过添加标题、坐标轴标签、图例和标注来提高图表可读性。"
order: 2
plots: []
sourceHash: "e58a04543a643c97bbd4a0495552cf5d058b3de855d29bf46ad8cddfc10f87c2"
sourceCorrections: []
---

生成直方图、散点图和配对图提供了数据的初步视角。这些可视化图表在沟通中的真正价值在于清晰的标注。没有标签或标题的图表常常含糊不清，难以正确理解。有效使用标题、坐标轴标签和图例，能将一个基本图表转变为有意义的分析成果。Matplotlib 和 Seaborn 提供了直接的方法来添加这些必要组成部分。

假设我们有一个 Pandas DataFrame `df`，包含 'feature\_A'、'feature\_B' 和 'category' 等列。我们可以生成一个散点图：

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 演示用示例数据
np.random.seed(42)
data = {
    'feature_A': np.random.rand(50) * 10,
    'feature_B': 0.5 * (np.random.rand(50) * 10) + np.random.randn(50) * 1.5,
    'category': np.random.choice(['Group 1', 'Group 2'], size=50)
}
df = pd.DataFrame(data)

# 基本散点图
plt.figure(figsize=(8, 5)) # 控制图表大小
sns.scatterplot(x='feature_A', y='feature_B', data=df)
plt.show()
```

![Image 7](https://apxml.s3.amazonaws.com/content_images/section_1181_1.webp?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAZQ3DPGY5OEILO6DN%2F20261002%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20261002T070837Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=986fc81915335355c055f359d946ca91e40a80ac171671f8b1026cbe3abc5031)
这个初始图表展示了关系，但缺少背景信息。我们来改进它。

### 添加标题

标题概括了图表的主要信息或内容。在 Matplotlib 中（因此在使用 Seaborn 时也经常如此，因为 Seaborn 在 Matplotlib 坐标轴上绘图），您可以使用 `plt.title()` 或 Axes 对象方法 `ax.set_title()`。

```python
# 创建图表并获取 Axes 对象
plt.figure(figsize=(8, 5))
ax = sns.scatterplot(x='feature_A', y='feature_B', data=df)

# 添加描述性标题
ax.set_title('特征 A 与特征 B 之间的关系')

plt.show()
```

![Image 8](https://apxml.s3.amazonaws.com/content_images/section_1181_2.webp?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAZQ3DPGY5OEILO6DN%2F20261002%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20261002T070837Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=f1c1806498771720cff5c434c5653bd279df7b3d7bc168a4a23a0d7d958c1051)
添加 `ax.set_title()` 可立即提供图表所表示内容的背景信息。选择标题时要简洁且信息量大。

### 设置坐标轴标签

坐标轴标签对于理解 x 轴和 y 轴所表示的内容非常重要。没有它们，刻度和变量将是未知的。使用 `plt.xlabel()`、`plt.ylabel()` 或 Axes 对象方法 `ax.set_xlabel()`、`ax.set_ylabel()`。如果适用，最好包含单位（例如，'温度 (u00b0C)'，'收入 (USD)'）。

```python
# 创建图表并获取 Axes 对象
plt.figure(figsize=(8, 5))
ax = sns.scatterplot(x='feature_A', y='feature_B', data=df)

# 添加标题和坐标轴标签
ax.set_title('特征 A 与特征 B 之间的关系')
ax.set_xlabel('特征 A (单位)') # 如果知道单位，请具体说明
ax.set_ylabel('特征 B (响应)')

plt.show()
```

![Image 9](https://apxml.s3.amazonaws.com/content_images/section_1181_3.webp?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAZQ3DPGY5OEILO6DN%2F20261002%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20261002T070837Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=5df9a3985d75ee32ee192c1eb0fc74dc11d70ca9fe945493afa77d9437fbf511)
现在，任何查看图表的人都能准确了解哪些变量绘制在哪个轴上。

### 管理图例

当您的图表包含通过颜色、标记 (token)样式或线条类型区分的多个组或类别时，图例是必要的。当您使用 `hue`、`style` 或 `size` 等参数 (parameter)时，Seaborn 通常会自动添加图例。

我们来修改散点图，按 'category' 为点着色：

```python
# 为类别创建带色调的图表
plt.figure(figsize=(9, 6)) # 略微增大图表大小以容纳图例
ax = sns.scatterplot(x='feature_A', y='feature_B', hue='category', data=df)

# 添加标题和坐标轴标签
ax.set_title('特征 B 对特征 A，按类别划分')
ax.set_xlabel('特征 A (单位)')
ax.set_ylabel('特征 B (响应)')

# 自定义图例位置（可选）
# ax.legend(loc='upper left', title='类别类型') # 示例自定义

plt.show()
```

![Image 10](https://apxml.s3.amazonaws.com/content_images/section_1181_4.webp?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAZQ3DPGY5OEILO6DN%2F20261002%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20261002T070837Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=cf34894521de98abc80c4037a30067b8eab3ad0dfee790238e0ee3896328282c)
因为我们使用了 `hue` 参数，Seaborn 自动添加了图例。图例将颜色映射到类别名称（'Group 1'，'Group 2'）。

有时您可能需要对图例进行更多控制。Matplotlib 的 `ax.legend()` 提供了位置 (`loc`)、为图例添加标题 (`title`)、移除边框 (`frameon=False`) 等选项。常见的 `loc` 值包括 'best'、'upper right'、'upper left'、'lower left'、'lower right'、'center left'、'center right'、'lower center'、'upper center'、'center'。Seaborn 有时可能会将图例放置在不合适的位置，因此调整 `loc` 可以提高可读性。

### 组合自定义

有效的可视化涉及将这些元素一起应用。以下是完全自定义的示例：

```python
# 创建带色调的图表并自定义
plt.figure(figsize=(9, 6))
ax = sns.scatterplot(x='feature_A', y='feature_B', hue='category', data=df, s=60) # 增大了点的大小

# 添加标题和坐标轴标签
ax.set_title('特征 B 对特征 A，按类别分组', fontsize=14, fontweight='bold')
ax.set_xlabel('特征 A (单位)', fontsize=12)
ax.set_ylabel('特征 B (响应)', fontsize=12)

# 提高网格可见性
ax.grid(True, linestyle='--', alpha=0.6)

# 自定义图例
ax.legend(title='客户组', loc='upper left', frameon=True)

# 改进刻度标签外观（可选）
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# 调整布局以防止标签重叠
plt.tight_layout()

plt.show()
```

![Image 11](https://apxml.s3.amazonaws.com/content_images/section_1181_5.webp?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAZQ3DPGY5OEILO6DN%2F20261002%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20261002T070837Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=881912678fc4fa4a27b34555a0d0ec3514de75cbfc8cf2a1e815f7cd9c2387ea)
这个最终图表明显更加清晰：

- 标题清楚地说明了图表的用途和背景。
- 坐标轴标签定义了变量及其可能的单位。
- 图例解释了不同类别的颜色映射。
- 网格线和字体大小等额外调整提高了可读性。

花时间用标题、标签和图例自定义图表是 EDA 中的重要步骤。它确保您的视觉分析不仅有助于您自己的理解，而且可以有效地传达给他人，为讨论和进一步分析奠定基础，包括我们接下来将讨论的特征工程步骤。

## 参考资料

- [Matplotlib Legend Guide](https://matplotlib.org/stable/tutorials/intermediate/legend_guide.html) — Matplotlib Developers (2024)
  Publisher: Matplotlib Developers
  一个教程，解释如何在Matplotlib图中有效创建、自定义和放置图例，这对于区分数据系列至关重要。
- [Matplotlib Text in Matplotlib](https://matplotlib.org/stable/tutorials/text/text_intro.html) — Matplotlib Developers (2024)
  涵盖添加和自定义各种文本元素，包括图表标题、轴标签和注释，以提高可视化图的可读性。
- [Customizing plots (Seaborn documentation)](https://seaborn.pydata.org/tutorial/aesthetics.html#controlling-plot-aesthetics) — Michael Waskom (2024)
  Publisher: Michael Waskom
  展示如何调整Seaborn图的各种视觉属性，包括如何使用Matplotlib函数设置标题、标签和图例。
- [Python Data Science Handbook: Essential Tools for Working with Data](https://jakevdp.github.io/PythonDataScienceHandbook/) — Jake VanderPlas (2016)
  Publisher: O'Reilly Media
  一份关于Python数据科学的指南，其中包含专门的Matplotlib和Seaborn数据可视化章节，详细介绍了自定义选项。
