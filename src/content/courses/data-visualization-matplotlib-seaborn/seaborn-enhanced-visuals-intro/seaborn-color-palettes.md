---
course: "data-visualization-matplotlib-seaborn"
chapter: "seaborn-enhanced-visuals-intro"
lesson: "seaborn-color-palettes"
sourceId: 1185
sourceUrl: "https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-4-seaborn-enhanced-visuals-intro/seaborn-color-palettes"
title: "Seaborn 调色板"
description: "了解并应用 Seaborn 丰富且吸引人的调色板到您的可视化中。"
order: 3
plots: ["plots/1185-0.json"]
sourceHash: "58bdc6a6217ed42577e7b2bd50befb3dd85383f6ce34b3ff476701e93de3e265"
sourceCorrections: []
---

颜色是数据可视化中的一个重要工具。它有助于区分不同数据组，表示数据值，并简单地使图表更具视觉吸引力且更易理解。Matplotlib 提供了基本的颜色选项，而 Seaborn 则通过其调色板提供了一个更精致、更灵活的颜色使用系统。这些调色板是预设的颜色集合，旨在美观且感知效果良好。

选择合适的颜色对有效传达信息很重要。好的颜色选择可以突显模式并使图表直观，而糟糕的选择则可能模糊信息甚至误导观看者。Seaborn 通过提供生成和应用适合不同可视化任务的各类调色板的函数来简化此过程。

### 调色板类型

Seaborn 的调色板通常分为三大类：

1. **类别（或定性）调色板：** 当您想区分没有固有顺序的离散类别时，最适合使用这些调色板。调色板中的每种颜色都应清晰可辨。Seaborn 提供了几种内置类别调色板，包括 `deep`、`muted`、`pastel`、`bright`、`dark` 和 `colorblind`。`colorblind` 调色板专门设计用于帮助有常见色盲问题的人区分颜色。

   您可以使用 `seaborn.palplot()` 函数可视化调色板：

   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt

   # 可视化“deep”类别调色板
   print("Seaborn 'deep' 调色板：")
   sns.palplot(sns.color_palette("deep"))
   plt.show()

   # 可视化“colorblind”调色板
   print("Seaborn 'colorblind' 调色板：")
   sns.palplot(sns.color_palette("colorblind"))
   plt.show()
   ```
2. **顺序调色板：** 这些调色板适用于表示从低到高（或反之）变化的数值数据。顺序调色板中的颜色通常在明度或饱和度上有所不同，从浅色表示低值，到深色表示高值（或反向）。示例包括感知均匀调色板，如 `viridis`、`plasma`、`magma` 和 `cividis`，以及基于单一色调的调色板，如 `Blues`、`Greens` 或 `Reds`。

   ```python
   # 可视化“Blues”顺序调色板
   print("Seaborn 'Blues' 顺序调色板：")
   sns.palplot(sns.color_palette("Blues"))
   plt.show()

   # 可视化“viridis”顺序调色板
   print("Seaborn 'viridis' 顺序调色板：")
   sns.palplot(sns.color_palette("viridis", n_colors=8)) # 指定颜色数量
   plt.show()
   ```
3. **发散调色板：** 这些调色板用于数值数据，其中数值从一个有意义的中心点（通常是零）向两边发散。它们通常在两端使用两种不同的色调，并在中心点附近逐渐过渡到中性色。这有助于强调数值从中心点向两个方向的偏离。示例包括 `coolwarm`、`RdBu`（红-蓝）、`PiYG`（粉-黄绿）和 `vlag`。

   ```python
   # 可视化“coolwarm”发散调色板
   print("Seaborn 'coolwarm' 发散调色板：")
   sns.palplot(sns.color_palette("coolwarm", n_colors=7))
   plt.show()

   # 可视化“RdBu”发散调色板
   print("Seaborn 'RdBu' 发散调色板：")
   sns.palplot(sns.color_palette("RdBu", n_colors=9))
   plt.show()
   ```

### 在图表中使用调色板

Seaborn 使其易于将这些调色板应用到您的可视化中。大多数 Seaborn 绘图函数都接受 `palette` 参数 (parameter)，您可以在其中指定内置调色板的名称或提供自定义颜色列表。

以下是如何在 `scatterplot` 中使用类别调色板，其中点的颜色是基于类别着色的：

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 创建示例数据
np.random.seed(42)
data = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100),
    'category': np.random.choice(['A', 'B', 'C'], 100)
})

# 设置图表大小以更好地在网页显示
plt.figure(figsize=(6, 4))
# 使用“bright”调色板创建散点图
sns.scatterplot(data=data, x='x', y='y', hue='category', palette='bright')
plt.title('带有类别调色板的散点图')
plt.xlabel('X 值')
plt.ylabel('Y 值')
# 调整布局
plt.tight_layout()
plt.show()
```



![带有类别调色板的散点图](plots/1185-0.json)



> 散点图，显示使用Seaborn 'bright' 调色板按类别 'A'、'B' 和 'C' 着色的随机数据点。

### 设置默认调色板

如果您希望在脚本或笔记本中的多个图表中使用相同的调色板，而无需每次都指定，则可以使用 `seaborn.set_palette()`。

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 创建示例数据
np.random.seed(10)
data = pd.DataFrame({
    'category': ['X', 'Y', 'Z', 'X', 'Y', 'Z', 'X', 'Y'],
    'value': np.random.rand(8) * 10
})

# 将默认调色板设置为“pastel”
sns.set_palette("pastel")

# 创建一个条形图——它将自动使用“pastel”调色板
plt.figure(figsize=(5, 4))
sns.barplot(data=data, x='category', y='value')
plt.title('带有默认“pastel”调色板的条形图')
plt.xlabel('类别')
plt.ylabel('值')
plt.tight_layout()
plt.show()

# 创建另一个图表（例如，箱线图）——它也使用“pastel”
plt.figure(figsize=(5, 4))
sns.boxplot(data=data, x='category', y='value')
plt.title('带有默认“pastel”调色板的箱线图')
plt.xlabel('类别')
plt.ylabel('值')
plt.tight_layout()
plt.show()

# 如果后续代码需要，重置为默认调色板
# sns.set_palette("deep")
```

> 条形图和箱线图，展示了使用 `sns.set_palette("pastel")` 自动将“pastel”配色方案应用到后续 Seaborn 图表的过程。

### 选择合适的调色板

调色板的选择很大程度上取决于您正在可视化的数据类型以及您想表达的信息：

- **类别数据：** 使用类别调色板（例如 `deep`、`muted`、`colorblind`）来区分不同的组，不暗示顺序。确保您的类别有足够的区别色。
- **数值数据（有序）：** 使用顺序调色板（例如 `viridis`、`Blues`、`YlGnBu`）来表示大小。浅色通常表示较低的值，深色表示较高的值。
- **数值数据（发散）：** 当数据具有有意义的中心点（如零）并且您想强调高于和低于此点的偏离时，使用发散调色板（例如 `coolwarm`、`RdBu`）。

如果您的可视化需要被广泛受众理解，请通过测试调色板来考虑可访问性，尤其是 `colorblind` 选项。

尝试不同的调色板通常是可视化过程的一部分。Seaborn 的内置选项为创建清晰、有信息量且美观的图表提供了很好的起点。

## 参考资料

- [Choosing color palettes](https://seaborn.pydata.org/tutorial/color_palettes.html) — Michael Waskom (2023)
  Seaborn 官方指南，介绍调色板系统、类型及其使用方法。
- [Fundamentals of Data Visualization](https://clauswilke.com/dataviz/) — Claus O. Wilke (2019)
  Publisher: O'Reilly Media
  为有效的数据可视化提供了全面的基础，包含专门章节讲解如何为不同数据类型（分类、顺序、发散）选择和使用调色板，并考虑感知均匀性和可访问性。
