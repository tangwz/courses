# Matplotlib 绘图基本原理

来源：[原文](https://apxml.com/zh/courses/intermediate-python-programming-ml/chapter-4-data-visualization-matplotlib-seaborn/matplotlib-fundamentals)

[返回章节目录](README.md) · [返回课程目录](../README.md)

Matplotlib 是一个基础库，用于在 Python 中创建静态、交互式和动态可视化。尽管 Seaborn（我们很快会讲到）等其他库为特定统计图表提供了更高级的接口，但理解 Matplotlib 的结构很重要，因为许多其他绘图库都是在其基础上构建的。可以将 Matplotlib 视为支撑 Python 大部分 2D 绘图功能的核心。

为了有效使用 Matplotlib，理解其主要组成部分会有所帮助。可视化是逐层构建的，从整体容器开始，逐步添加特定的绘图元素。

### Matplotlib 绘图的结构

在最高层级，我们有 `Figure`。可以把它想象成绘制所有内容的整个窗口或页面。它是所有绘图元素的顶层容器，包括标题、图例，以及最重要的是，一个或多个 `Axes`。

一个 `Axes` 对象代表一个独立的图或图表（不要混淆复数 `Axes` 和单数 `Axis`）。它是用 X 轴和 Y 轴（或 3D 中的其他坐标）绘制数据的区域。一个 `Figure` 可以包含多个 `Axes` 对象，通常排列成网格。这对于并排比较数据的不同视图很有用。每个 `Axes` 都有自己的标题、X 轴标签和 Y 轴标签。

在一个 `Axes` 内部，有 `Axis` 对象（对于 2D 图通常有两个：X 轴和 Y 轴）。这些对象处理数据范围（限制）的缩放、刻度线（沿轴的位置）以及刻度标签（表示刻度处值的文本）。

以下是其分解：

> 该图表展示了层次结构：一个 `Figure` 包含一个或多个 `Axes`，每个 `Axes` 包含 `Axis` 对象和实际绘制的数据（线条、点等），以及标题和标签等描述性元素。

### 两种接口：`pyplot` 和面向对象

Matplotlib 提供了两种主要的绘图方式：

1. **`pyplot` 模块：** 这提供了一组函数（`plt.plot()`、`plt.title()` 等），它们隐式地作用于“当前”的图形和坐标系。它常用于快速、交互式绘图，因为它启动所需代码量少。它维护一个内部状态，在需要时自动创建图形和坐标系。
2. **面向对象 (OO) 接口：** 这涉及显式地创建 `Figure` 和 `Axes` 对象，然后直接在这些对象上调用方法（例如 `ax.plot()`、`ax.set_title()`）。这种方法通常推荐用于更复杂的图表、可重用函数，或者需要对图形元素进行更精细控制的情况。它使代码更清楚地指示正在修改哪个图。

尽管 `pyplot` 对于简单任务很方便，但本课程将主要侧重于面向对象的方法，因为它更符合构建更结构化、可能更复杂的视觉效果的需求，这在机器学习 (machine learning)工作流中常会用到。

### 创建你的第一个图（面向对象风格）

让我们看看如何使用 OO 接口创建一个基本图。标准的第一个步骤是导入必要的库：

```python
import matplotlib.pyplot as plt
import numpy as np
```

我们通常将 `matplotlib.pyplot` 导入为 `plt`。即使在使用 OO 接口时，`pyplot` 仍用于创建图形和坐标系（`plt.subplots()`）以及显示图表（`plt.show()`）等函数。

接下来，我们创建一个 `Figure` 和一组 `Axes`。`plt.subplots()` 函数是完成此操作的便捷方法。它返回一个包含 `Figure` 对象和一个或多个 `Axes` 对象的元组。

```python
# 示例数据
x = np.linspace(0, 10, 100) # 从 0 到 10 的 100 个点
y = np.sin(x)

# 创建一个 Figure 和一个 Axes 对象
fig, ax = plt.subplots()

# 在 Axes 上绘图
ax.plot(x, y, color='#1c7ed6', linewidth=2) # 蓝色线条

# 使用 Axes 方法自定义图表
ax.set_title("简单的正弦波")
ax.set_xlabel("X 值")
ax.set_ylabel("Y 值 (sin(x))")
ax.grid(True, linestyle=':', color='#adb5bd') # 添加一个浅灰色网格

# 显示图表
plt.show()
```

此代码片段执行以下操作：

1. 导入 `matplotlib.pyplot` 和 `numpy`。
2. 使用 NumPy 生成一些示例数据。
3. `plt.subplots()` 创建一个 `Figure`（赋值给 `fig`）和该图形内的一个 `Axes`（赋值给 `ax`）。
4. `ax.plot(x, y, ...)` 直接在 `Axes` 对象 `ax` 上绘制折线图。我们指定数据并可选地自定义其外观（颜色、线宽）。
5. `ax.set_title()`、`ax.set_xlabel()` 和 `ax.set_ylabel()` 为图表添加描述性文本，同样是通过在 `ax` 对象上调用方法。
6. `ax.grid(True, ...)` 添加背景网格以便于阅读。
7. `plt.show()` 显示包含我们图表的图形。在某些环境（如 Jupyter Notebooks）中，图表可能会自动渲染，但明确调用 `plt.show()` 是一个好习惯，尤其是在脚本中。

这种使用 `fig, ax = plt.subplots()` 然后在 `ax` 上调用方法（如 `ax.plot`、`ax.set_title`）的基本结构，构成了使用 Matplotlib 创建各种可视化的基础，我们将在接下来的章节中在此基础上进行构建。理解 `Figure` 和 `Axes` 对象的作用是掌握 Python 数据可视化的第一步。

## 参考资料

- [Anatomy of a Matplotlib Plot](https://matplotlib.org/stable/tutorials/introductory/usage.html#anatomy-of-a-figure) — The Matplotlib Development Team (2024)
  详细介绍了Matplotlib图表的基本组成部分，例如Figure、Axes和Axis对象，它们构成了可视化的基础。
- [Usage Guide](https://matplotlib.org/stable/tutorials/introductory/usage.html) — The Matplotlib Development Team (2024)
  介绍了Matplotlib中绘图的两种主要接口（pyplot和面向对象），阐明了它们的使用场景和典型工作流程。
- [Python for Data Analysis](https://www.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  一本标准资源，其中包含专门的Matplotlib章节，在数据分析的更广阔背景下解释其数据可视化用途。

---

[上一节](../03-Pandas%20%E6%95%B0%E6%8D%AE%E6%93%8D%E4%BD%9C/09-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%BD%BF%E7%94%A8%20Pandas%20%E6%95%B4%E7%90%86%E6%95%B0%E6%8D%AE.md) · [下一节](02-%E5%88%9B%E5%BB%BA%E5%B8%B8%E7%94%A8%E5%9B%BE%E8%A1%A8%E7%B1%BB%E5%9E%8B.md)
