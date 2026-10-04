---
course: "data-visualization-matplotlib-seaborn"
chapter: "customizing-saving-visualizations"
lesson: "seaborn-themes-customization"
sourceId: 1212
sourceUrl: "https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-7-customizing-saving-visualizations/seaborn-themes-customization"
title: "应用 Seaborn 主题"
description: "了解不同的内置 Seaborn 主题，并学习如何进一步自定义它们。"
order: 5
plots: []
sourceHash: "feff91f13d8565c656ce2b733790a466e8937cfb170389ba4ad4ecec705b3847"
sourceCorrections: []
---

虽然创建具有良好数据表现的图表必不可少，但整体外观和感觉会很大程度上影响您的可视化图表被看待和理解的方式。Seaborn 提供了一种直接的方法来管理图表的整体外观，它使用预定义的主题和样式。这些主题处理诸如背景颜色、网格线、字体样式和坐标轴刻度等方面，让您无需手动调整大量的 Matplotlib 参数 (parameter)，即可在不同的视觉风格之间快速切换。

可以将 Seaborn 主题视为图表的样式模板。应用某个主题会立即更改您在会话或脚本中创建的后续图表的默认设置。

### 使用内置的 Seaborn 主题

Seaborn 内置了几种美观的主题。最常用的有：

- `darkgrid`: 灰色背景，白色网格线（通常是默认设置）。适用于需要从坐标轴读取特定值的图表。
- `whitegrid`: 白色背景，灰色网格线。与 `darkgrid` 相似，但外观更清晰、明亮。
- `dark`: 深灰色背景，无网格线。
- `white`: 纯白色背景，无网格线。常用于出版物。
- `ticks`: 与 `white` 相似，但会在坐标轴上添加小刻度线。

您可以使用 `sns.set_theme()` 函数为您的整个脚本或 Jupyter Notebook 全局设置主题。通常在脚本开始时，导入库之后立即调用此函数。

```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 示例数据
x = np.linspace(0, 10, 50)
y1 = np.sin(x)
y2 = np.cos(x)

# 设置 'whitegrid' 主题
sns.set_theme(style="whitegrid")

# 创建图表 - 它将使用 'whitegrid' 样式
plt.figure(figsize=(6, 4)) # 控制图表大小
sns.lineplot(x=x, y=y1, label='sin(x)')
sns.lineplot(x=x, y=y2, label='cos(x)')
plt.title("使用 'whitegrid' 主题的图表")
plt.xlabel("X 值")
plt.ylabel("Y 值")
plt.legend()
plt.show()

# 现在，我们尝试 'ticks' 主题
sns.set_theme(style="ticks")

# 重新创建图表 - 它现在将使用 'ticks' 样式
plt.figure(figsize=(6, 4))
sns.lineplot(x=x, y=y1, label='sin(x)')
sns.lineplot(x=x, y=y2, label='cos(x)')
plt.title("使用 'ticks' 主题的图表")
plt.xlabel("X 值")
plt.ylabel("Y 值")
plt.legend()
plt.show()
```

请注意，仅仅更改一行代码 (`sns.set_theme(style=...)`) 就明显改变了图表的外观，包括背景、网格和坐标轴边框。

### 临时设置样式

有时，您可能只想对一个或少数几个图表应用特定样式，而不是更改全局默认设置。Seaborn 允许使用 Python 的 `with` 语句和 `sns.axes_style()` 函数来实现这一点。此上下文 (context)管理器仅在 `with` 代码块内临时更改样式设置。

```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 示例数据
x = np.linspace(0, 10, 50)
y1 = np.sin(x)

# 假设默认或之前设置的主题处于活动状态
sns.set_theme(style="darkgrid") # 设置一个基础主题

# 图表 1 使用默认的 'darkgrid' 样式
plt.figure(figsize=(6, 4))
sns.lineplot(x=x, y=y1)
plt.title("使用默认主题 ('darkgrid') 的图表")
plt.show()

# 图表 2 使用临时 'white' 样式
with sns.axes_style("white"):
    plt.figure(figsize=(6, 4))
    sns.lineplot(x=x, y=y1)
    plt.title("临时设置为 'white' 样式的图表")
    plt.show()

# 图表 3 返回默认的 'darkgrid' 样式
plt.figure(figsize=(6, 4))
sns.lineplot(x=x, y=y1)
plt.title("返回默认主题 ('darkgrid') 的图表")
plt.show()
```

这种方法有助于在保持一致的默认样式的同时，偶尔为特定可视化图表进行样式偏离。

### 进一步自定义主题元素

虽然内置主题提供了很好的起点，但您可能希望进行更精细的控制。

- **基本自定义仍然适用：** 请记住 Seaborn 是建立在 Matplotlib 之上的。您仍然可以使用 Matplotlib 函数，例如 `plt.title()`、`plt.xlabel()`、`plt.ylabel()`、`plt.xlim()`、`plt.ylim()`，以及 Seaborn 函数内部的参数 (parameter)（如 `color`、`linestyle`）来在应用主题后自定义元素。
- **`sns.set_theme()` 参数：** 除了 `style` 之外，`set_theme()` 函数还接受其他参数。例如，`palette` 控制默认的颜色序列（之前已介绍），`font_scale` 则会整体上将所有文本元素（坐标轴标签、标题、图例）的大小进行缩放，这对于全局调整文本大小非常方便。

  ```python
  # 示例：使用 'whitegrid' 样式和更大的字体
  sns.set_theme(style="whitegrid", font_scale=1.5)

  plt.figure(figsize=(6, 4))
  sns.lineplot(x=x, y=y1, label='sin(x)')
  sns.lineplot(x=x, y=y2, label='cos(x)')
  plt.title("带有缩放字体的图表")
  plt.show()

  # 如有需要，为后续图表重置为默认比例
  sns.set_theme(style="whitegrid", font_scale=1.0)
  ```
- **`sns.set_style()` 用于精细控制：** 如果您想修改样式的特定元素（例如背景或网格线的精确颜色），可以将字典传递给 `sns.set_style()`。这会使用 Matplotlib 的运行时配置 (rc) 参数。有时需要查阅 Matplotlib 文档才能找到正确的参数名称，因此这是一种更高级的技术，通常在标准主题不完全适用时使用。

  ```python
  # 示例：从 'darkgrid' 开始，但更改背景颜色
  custom_style = sns.axes_style("darkgrid")
  custom_style['axes.facecolor'] = '#e9ecef' # 我们的调色板中的浅灰色

  sns.set_style(custom_style)

  plt.figure(figsize=(6, 4))
  ```

sns.lineplot(x=x, y=y1, label='sin(x)')
plt.title("具有自定义背景的图表")
plt.show()

````
# 如有需要，之后重置为标准主题
sns.set_theme(style="whitegrid")
```
对于初学者来说，坚持使用主要的 `style` 名称（`'whitegrid'`、`'dark'` 等）并使用 `font_scale` 通常就足够了。
````

### 选择合适的主题

最佳主题取决于您的数据以及图表的用途：

- `darkgrid` / `whitegrid`: 在数据查看阶段或当网格线有助于精确读取数值时很有用。
- `white` / `ticks`: 常用于报告、演示文稿或出版物，因为它们提供了整洁、不那么杂乱的外观。没有网格线会使注意力更多地放在数据本身的模式上。
- `dark`: 视觉上可能很吸引人，尤其是在深色主题的演示文稿或仪表板中，但要确保有足够的对比度以保证可读性。

尝试不同的主题，看看哪一个最符合您的可视化目的。应用主题是美化图表的快速第一步，之后您可以进行更具体的自定义或保存图表以供分享。

## 参考资料

- [seaborn.set_theme - seaborn 0.13.2 documentation](https://seaborn.pydata.org/generated/seaborn.set_theme.html) — Michael Waskom and the Seaborn development team (2024)
  Seaborn中设置全局绘图主题和样式的官方API参考，包括字体缩放。
- [Plot customization - seaborn 0.13.2 documentation](https://seaborn.pydata.org/tutorial/aesthetics.html) — Michael Waskom and the Seaborn development team (2024)
  官方教程，概述了Seaborn的美学功能、内置主题和临时样式设置。
- [Customizing Matplotlib with style sheets and rcParams](https://matplotlib.org/stable/tutorials/introductory/customizing.html) — Matplotlib development team (2024)
  Publisher: The Matplotlib development team
  Matplotlib的官方指南，解释了Seaborn主题调整的运行时配置参数（rcParams）。
- [Python for Data Analysis](https://wesmckinney.com/book/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  一本基础书籍，涵盖了Python中的数据处理和可视化，包括Seaborn和Matplotlib的样式设置。
