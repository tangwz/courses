---
course: "data-visualization-matplotlib-seaborn"
chapter: "customizing-saving-visualizations"
lesson: "matplotlib-oo-api-control"
sourceId: 1211
sourceUrl: "https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-7-customizing-saving-visualizations/matplotlib-oo-api-control"
title: "使用 Matplotlib 面向对象 API 实现更精细的控制"
description: "了解并使用 Matplotlib 的面向对象接口以实现更详细的图表自定义。"
order: 4
plots: []
sourceHash: "a04f2620d44e5cdbd13e45b53ad4a2f4278fe76643ee7289975c085fff5d2898"
sourceCorrections: []
---

Matplotlib 的 `pyplot` 接口是生成图表的常用方式。像 `plt.plot()`、`plt.title()` 和 `plt.xlabel()` 这样的函数提供了一种便捷的方式来快速生成图表。这个接口非常适合简单的图示和交互式操作。然而，Matplotlib 也提供了一种更结构化、更强大、更灵活的绘图方式：面向对象 (OO) 应用程序编程接口 (API)。

掌握 OO API 在您需要对图表的各个方面进行更详细的控制时特别有用，尤其是在一个图表中处理多个子图时，或者在创建需要精确元素放置和修改的复杂图示时。它使您的绘图代码通常更明确，并且可以说更易于维护，适用于复杂的图表。

### Figure 和 Axes 对象

还记得第 2 章中的“Matplotlib 图表结构”吗？我们讨论了 `Figure`（即绘制所有内容的整个窗口或画布）和 `Axes`（带有 x 轴、y 轴、数据等的独立绘图区域）。OO API 的核心是显式创建和操作这些对象。

您通常会显式创建它们，而不是依赖 `pyplot` 隐式管理当前的 Figure 和 Axes：

```python
import matplotlib.pyplot as plt
import numpy as np

# 创建一些示例数据
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# 创建一个 Figure 对象和一个 Axes 对象
# plt.subplots() 是一个实现此功能的便捷函数
fig, ax = plt.subplots()

print(type(fig))
# 输出: <class 'matplotlib.figure.Figure'>

print(type(ax))
# 输出: <class 'matplotlib.axes._axes.Axes'>
```

在这里，`fig` 是您的 `Figure` 实例，`ax` 是您的 `Axes` 实例。`plt.subplots()`（注意末尾的 's'）是同时生成这两者的标准函数。如果您想要多个子图，可以指定 `nrows` 和 `ncols`，`plt.subplots()` 将返回一个 Figure 和一个 Axes 对象*数组*。

### 通过 Axes 方法绘图和自定义

一旦您有了 `Axes` 对象（如上面示例中的 `ax`），您就可以直接在该对象上调用方法来绘制数据并自定义其外观，而不是调用 `pyplot` 中的函数。

比较 `pyplot` 方法与 OO 方法：

**Pyplot 方法：**

```python
# 创建一些示例数据（如果尚未创建）
# x = np.linspace(0, 10, 100)
# y1 = np.sin(x)

plt.figure() # 隐式创建一个 Figure
plt.plot(x, y1) # 在“当前”Axes 上绘图（隐式创建）
plt.title("简单正弦波 (pyplot)")
plt.xlabel("X 值")
plt.ylabel("Sine(X)")
plt.grid(True)
plt.show()
```

**面向对象方法：**

```python
# 创建一些示例数据（如果尚未创建）
# x = np.linspace(0, 10, 100)
# y1 = np.sin(x)

# 显式创建 Figure 和 Axes
fig, ax = plt.subplots()

# 直接在 Axes 对象 'ax' 上调用方法
ax.plot(x, y1)
ax.set_title("简单正弦波 (OO API)") # 注意: set_title()
ax.set_xlabel("X 值")                  # 注意: set_xlabel()
ax.set_ylabel("Sine(X)")                  # 注意: set_ylabel()
ax.grid(True)                           # grid() 方法也存在于 Axes 上

plt.show() # plt.show() 仍然用于显示图表
```

请注意主要区别：在 OO 方法中，我们直接在 `ax` 对象上调用 `ax.plot()`、`ax.set_title()`、`ax.set_xlabel()` 和 `ax.set_ylabel()` 等方法。许多 `pyplot` 自定义函数都有对应的 `Axes` 方法，通常以 `set_` 为前缀。

这种显式方法让您更清楚地知道正在修改图表的*哪个*部分，这在处理多个子图时变得极其有用。

### 自定义的优点

为什么要采用这种略显冗长的方法呢？

1. **显式控制：** 您总是清楚地知道您正在修改哪个 `Axes` 对象。这消除了歧义，尤其是在存在多个图表时。
2. **访问元素：** `Axes` 对象（以及 `Figure` 对象）持有其中所有元素（线条、文本、补丁等）的引用。OO API 提供了方法，可以在绘图后直接访问和修改这些元素，提供了 `pyplot` 不易实现的细粒度控制。例如，您可以获取绘制在 `Axes` 上的线条列表，并单独更改它们的属性。
3. **更适合函数/脚本：** 在编写生成图表的通用函数时，将 `Axes` 对象作为参数 (parameter)传递，允许函数在调用者提供的特定子图上绘图，使您的代码更具模块化。
4. **多个子图：** 管理多个子图要简单得多。

### 使用多个子图

让我们演示 OO API 在单个 Figure 中处理多个子图（Axes）的能力。

```python
# 创建一些示例数据（如果尚未创建）
# x = np.linspace(0, 10, 100)
# y1 = np.sin(x)
# y2 = np.cos(x)

# 创建一个 Figure 和 1x2 的 Axes 对象网格
# 'axes' 将是一个包含两个 Axes 对象的 NumPy 数组
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4)) # 请求 1 行，2 列

# axes[0] 是第一个子图（左侧）
axes[0].plot(x, y1, color='#1c7ed6') # 蓝色
axes[0].set_title('正弦波')
axes[0].set_xlabel('X')
axes[0].set_ylabel('sin(X)')
axes[0].grid(True)

# axes[1] 是第二个子图（右侧）
axes[1].plot(x, y2, color='#f03e3e') # 红色
axes[1].set_title('余弦波')
axes[1].set_xlabel('X')
axes[1].set_ylabel('cos(X)')
axes[1].grid(True)

# 优化布局以防止标题/标签重叠
fig.tight_layout()

plt.show()
```

在这个例子中，`plt.subplots(1, 2)` 返回 `Figure` (`fig`) 和一个包含两个 `Axes` 对象的 NumPy 数组 `axes`。我们通过索引（`axes[0]`、`axes[1]`）访问每个 `Axes`，并调用其特定的绘图和自定义方法。这允许对每个子图进行独立控制。`fig.tight_layout()` 是一个有用的 `Figure` 方法，用于自动调整间距。

### 与 Seaborn 集成

Seaborn 建立在 Matplotlib 之上，通常在内部使用 Matplotlib 对象。许多 Seaborn 绘图函数实际上会返回绘制图表所用的 Matplotlib `Axes` 对象。这使得您可以使用 Seaborn 的高级统计绘图功能，然后使用 Matplotlib 的 OO API 对图表进行微调 (fine-tuning)。

```python
import seaborn as sns
import pandas as pd

# DataFrame 中的示例数据
data = pd.DataFrame({'x_val': x, 'y_val': y1 + np.random.randn(100) * 0.1})

# 显式创建 Figure 和 Axes
fig, ax = plt.subplots(figsize=(8, 5))

# 使用 Seaborn 创建图表，并传入 Axes 对象
# Seaborn 在我们提供的 'ax' 上绘图
sns.scatterplot(data=data, x='x_val', y='y_val', ax=ax, color='#7048e8') # 紫色

# 现在在 'ax' 上使用 Matplotlib OO API 方法进行进一步自定义
ax.set_title("带有 Matplotlib 自定义的 Seaborn 图表")
ax.set_xlabel("X 值")
ax.set_ylabel("带噪声的正弦值")
ax.grid(True)

# 使用 Axes 方法添加批注
ax.text(2, 0.75, "通过 Matplotlib 添加的批注", fontsize=10, color='#495057') # 灰色

plt.show()
```

在这里，我们首先使用 `plt.subplots()` 创建了 `fig` 和 `ax`。然后，我们通过 `ax=ax` 参数 (parameter)告诉 Seaborn 的 `scatterplot` 函数在我们的特定 `ax` 上绘图。在 Seaborn 创建了基本图表后，我们使用了 `ax.set_title()`、`ax.set_xlabel()`、`ax.set_ylabel()`、`ax.grid()` 甚至 `ax.text()` 这些 `Axes` 对象的方法，添加了更多自定义设置。

### 总结

尽管 `pyplot` 接口对于简单的图表很方便，但 Matplotlib 面向对象 API 提供了一种更明确、更强大的方式来控制图表的每一个细节。通过直接使用 `Figure` 和 `Axes` 对象并调用它们的方法（如 `ax.plot()`、`ax.set_title()`、`ax.set_xlabel()` 等），您可以获得更精细的控制，这在以下情况下特别有用：

- 复杂自定义。
- 在一个图表中管理多个子图。
- 创建可重复使用的绘图函数。
- 与 Seaborn 等库顺畅集成以进行进一步优化。

随着您在数据图示方面的进步，熟悉 OO API 将使您能够为分析和沟通需求创建更精细、更精确的定制图表。对于快速的、临时性的图表，`pyplot` 仍然是一个很好的工具，但对于精良、复杂的图表，OO API 通常是更受推荐的方法。

## 参考资料

- [Anatomy of a figure](https://matplotlib.org/stable/gallery/showcase/anatomy.html) — Matplotlib development team (2024)
  Publisher: Matplotlib development team
  一份基础的官方指南，概述了 Matplotlib 图形的核心组件，例如 Figure 和 Axes，这些组件是使用面向对象 API 的核心。
- [Python Data Science Handbook: Essential Tools for Working with Data](https://jakevdp.github.io/PythonDataScienceHandbook/04.00-introduction-to-matplotlib.html) — Jake VanderPlas (2016)
  Publisher: O'Reilly Media; Pages: 221-270 (Chapter 4, Matplotlib)
  一本备受推崇的教科书，提供了 Matplotlib 的全面介绍，对比了其 pyplot 和面向对象接口，并展示了高级自定义技术。
