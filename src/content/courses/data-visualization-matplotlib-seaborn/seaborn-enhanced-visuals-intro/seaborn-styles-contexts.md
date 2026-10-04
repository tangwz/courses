---
course: "data-visualization-matplotlib-seaborn"
chapter: "seaborn-enhanced-visuals-intro"
lesson: "seaborn-styles-contexts"
sourceId: 1182
sourceUrl: "https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-4-seaborn-enhanced-visuals-intro/seaborn-styles-contexts"
title: "设定 Seaborn 样式与上下文"
description: "了解如何轻松改变 Seaborn 图表的整体外观（样式）和缩放（上下文）。"
order: 2
plots: []
sourceHash: "6b34d2e073925093393da8f1b52d57ede4db05fb463328eea6359c8ac3b8b59b"
sourceCorrections: []
---

Seaborn 的一大优势在于它能轻松生成视觉上美观的图表。这主要归功于其内置的主题，这些主题控制着背景色、网格线和字体选择等整体视觉属性。Seaborn 将这些控制分为两大类：**样式**和**上下文 (context)**。

### 理解 Seaborn 样式

样式主要影响图表背景、网格和坐标轴元素（如轴脊和刻度）的视觉呈现。Seaborn 提供了多种预设样式，您可以轻松切换。在您的会话或脚本中，为所有后续图表全局设置样式最常见的方法是使用 `set_theme()` 函数，并指定 `style` 参数 (parameter)。

五种可用的预设样式是：

1. `darkgrid`: 灰色背景，白色网格线（通常是默认设置）。适用于数据点较多的图表。
2. `whitegrid`: 白色背景，灰色网格线。类似于 `darkgrid`，但在某些应用中外观更简洁。
3. `dark`: 深灰色背景，无网格线。
4. `white`: 纯白色背景，无网格线。最接近标准的 Matplotlib 外观，但通常有细微优化。
5. `ticks`: 白色背景，坐标轴上添加刻度线，这有助于判断数值，尤其是在没有网格线的情况下。

让我们看看如何应用样式。首先，我们需要标准导入和一些简单数据：

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 生成一些简单数据
x = np.linspace(0, 10, 50)
y = np.sin(x)

# 默认 Matplotlib 图表（用于比较）
plt.figure() # 开始一个新的图表
plt.plot(x, y)
plt.title("默认 Matplotlib 样式")
plt.show()

# 现在，设置一个 Seaborn 样式并再次绘图
sns.set_theme(style="whitegrid") # 应用 'whitegrid' 样式

plt.figure() # 开始另一个新的图表
plt.plot(x, y)
plt.title("Seaborn 'whitegrid' 样式")
plt.show()

# 尝试另一种样式
sns.set_theme(style="ticks") # 应用 'ticks' 样式

plt.figure()
plt.plot(x, y)
plt.title("Seaborn 'ticks' 样式")
plt.show()

# 重置为默认主题参数（对 Notebook 有用）
# sns.reset_defaults()
```

请注意 `sns.set_theme(style=...)` 是如何改变在其调用*之后*创建的图表外观的。`'whitegrid'` 样式添加了白色背景和有用的网格线，而 `'ticks'` 则在坐标轴上添加了小标记 (token)。选择样式取决于您的数据以及图表的显示位置。

如果您只想为特定图表临时应用样式，可以在 Python 的 `with` 语句中使用 `axes_style()` 函数。这可以避免改变全局设置。

```python
# 生成数据
x = np.linspace(0, 10, 50)
y = np.cos(x)

# 首先设置一个全局样式（例如 darkgrid）
sns.set_theme(style="darkgrid")

# 使用全局样式绘图
plt.figure()
plt.plot(x, y)
plt.title("全局样式 ('darkgrid')")
plt.show()

# 为一个图表临时使用 'white' 样式
with sns.axes_style("white"):
    plt.figure()
    plt.plot(x, y)
    plt.title("临时样式 ('white')")
    plt.show()

# 再次绘图 - 它使用全局 'darkgrid' 样式
plt.figure()
plt.plot(x, y)
plt.title("回到全局样式 ('darkgrid')")
plt.show()
```

### 使用上下文 (context)缩放图表

样式控制着图表的美学*类型*（网格、背景），而**上下文**则控制图表元素的*缩放*。这包括标签大小、线条宽度和标记 (token)大小。上下文旨在帮助您根据不同情况调整图表，例如图表是用于演示文稿幻灯片（`talk`、`poster`）还是嵌入 (embedding)到文档中（`paper`、`notebook`）。

四种预设上下文，按从小到大缩放顺序排列，分别是：

1. `paper`: 最小元素，适合包含在书面文档中。
2. `notebook`: 默认上下文，为交互式 Notebook 使用而平衡。
3. `talk`: 较大元素，适合观众可能距离较远的演示文稿。
4. `poster`: 最大元素，适用于远距离观看的显示屏。

您可以在 `set_theme()` 中使用 `context` 参数 (parameter)来设置上下文，也可以使用单独的 `set_context()` 函数。与样式类似，`set_context()` 全局适用，而 `plotting_context()` 可在 `with` 语句中用于临时更改。

让我们看看不同上下文的效果：

```python
# 生成数据
x = np.linspace(0, 10, 50)
y = np.sin(x) * np.exp(x/10)

# 使用默认的 'notebook' 上下文（通常是隐式设置）
sns.set_theme(style="whitegrid", context="notebook") # 显式设置 notebook
plt.figure()
plt.plot(x, y)
plt.title("上下文: 'notebook' (默认)")
plt.xlabel("X轴标签")
plt.ylabel("Y轴标签")
plt.show()

# 切换到 'paper' 上下文
sns.set_theme(style="whitegrid", context="paper")
plt.figure()
plt.plot(x, y)
plt.title("上下文: 'paper'")
plt.xlabel("X轴标签")
plt.ylabel("Y轴标签")
plt.show()

# 切换到 'talk' 上下文
sns.set_theme(style="whitegrid", context="talk")
plt.figure()
plt.plot(x, y)
plt.title("上下文: 'talk'")
plt.xlabel("X轴标签")
plt.ylabel("Y轴标签")
plt.show()

# 如果需要，重置主题为默认
# sns.reset_defaults()
```

观察 `'paper'` 上下文如何使标签、标题和线条粗细变小，而 `'talk'` 上下文则使它们与默认的 `'notebook'` 上下文相比明显大很多。底层图表数据保持不变，只有呈现比例发生变化。

### 组合样式和上下文 (context)

您可以，也通常会使用 `set_theme()` 同时设置样式和上下文：

```python
sns.set_theme(style="darkgrid", context="talk")

plt.figure()
plt.plot(x, y**2) # 绘制一些略有不同的内容
plt.title("样式: 'darkgrid', 上下文: 'talk'")
plt.xlabel("X轴")
plt.ylabel("Y轴平方")
plt.show()
```

这种组合使您能够精准控制 Seaborn 可视化的基本美学（样式）和缩放（上下文），从而使您能够快速为目标受众和媒介调整图表。尝试这些设置通常是自定义 Seaborn 图表的第一步。为了对特定图表元素（如字体大小或特定颜色）进行更详细的控制，Seaborn 允许通过 `rc` 字典覆盖参数 (parameter)，但 `set_theme` 提供了一种方便的方式来管理整体外观和感受。

## 参考资料

- [Python for Data Analysis](https://wesmckinney.com/book/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  一本综合性指南，其中包含Matplotlib和Seaborn部分，为Python数据可视化提供基础知识。
- [Customizing Matplotlib with style sheets and rcParams](https://matplotlib.org/stable/tutorials/introductory/customizing.html) — Matplotlib Developers (2023)
  此文档解释了Matplotlib美学控制的底层机制，Seaborn的样式构建于此之上。
