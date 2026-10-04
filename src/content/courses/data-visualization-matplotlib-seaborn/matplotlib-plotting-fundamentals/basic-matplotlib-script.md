---
course: "data-visualization-matplotlib-seaborn"
chapter: "matplotlib-plotting-fundamentals"
lesson: "basic-matplotlib-script"
sourceId: 1150
sourceUrl: "https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-2-matplotlib-plotting-fundamentals/basic-matplotlib-script"
title: "创建基础绘图脚本"
description: "了解使用 Matplotlib 创建图表的 Python 脚本的典型结构。"
order: 2
plots: ["plots/1150-0.json"]
sourceHash: "b0994b5524270992a3af0437c16f04d31fc7395a0c257877702e1ab0db701061"
sourceCorrections: []
---

Figure 和 Axes 是在实际 Python 脚本中用于数据可视化的组件。虽然可以交互式地创建图表（特别是在 Jupyter Notebook 等环境中），但了解典型绘图脚本的结构，对于制作可复用且更复杂的图表非常重要。

大多数 Matplotlib 绘图脚本遵循以下标准步骤顺序：

1. **导入所需库：** 主要是 Matplotlib 的 `pyplot` 模块。
2. **准备数据：** 这可能涉及创建 Python 列表、使用 NumPy 数组，或通过 Pandas 加载数据（我们将在后面介绍）。
3. **创建 Figure 和 Axes：** 这会设置画布以及实际绘制图表的区域。最常见的方法是使用 `plt.subplots()`。
4. **绘制数据：** 在 `Axes` 对象上调用绘图方法（例如 `plot()`、`scatter()`、`bar()` 等）。
5. **自定义图表：** 添加标题、标签、图例、更改颜色、设置范围等，同样使用与 `Axes` 对象相关的方法。
6. **显示图表：** 使用 `plt.show()` 显示 Figure。

我们来看一个最简脚本，它将这些步骤结合起来，创建一个简单的折线图。

```python
# 1. 导入所需库
import matplotlib.pyplot as plt

# 2. 准备数据
x_values = [0, 1, 2, 3, 4, 5]
y_values = [0, 1, 4, 9, 16, 25] # 简单的 y = x^2 数据

# 3. 创建 Figure 和 Axes
fig, ax = plt.subplots() # 创建一个 Figure 和一个 Axes 对象

# 4. 在 Axes 上绘制数据
ax.plot(x_values, y_values)

# 5. 自定义图表（可选但推荐）
ax.set_title("Simple Plot: y = x^2")
ax.set_xlabel("X Values")
ax.set_ylabel("Y Values (Squared)")

# 6. 显示图表
plt.show()
```

我们来分解一下主要部分：

- `import matplotlib.pyplot as plt`：这会导入 `pyplot` 模块，它为 Matplotlib 库提供了一个便捷的接口。我们使用标准的别名 `plt`。
  "\* `x_values = ...`、`y_values = ...`：在这里，我们只是定义包含数据点的简单 Python 列表。在实际场景中，这些数据可能来自文件、数据库或计算。"
- `fig, ax = plt.subplots()`：这是一种非常常见且重要的模式。`plt.subplots()` 函数会为我们创建两个对象：
  - `fig`：代表整个 Figure 或窗口。您以后可能会用它来执行保存图表或调整 Figure 整体属性等任务。
  - `ax`：代表 Axes，即实际进行绘图的子图。您将使用的大多数绘图命令（`plot()`、`scatter()`、`set_title()`、`set_xlabel()` 等）都是此 `Axes` 对象的方法。使用 `ax` 可以使您的代码明确您正在 *哪个位置* 绘图。
- `ax.plot(x_values, y_values)`：这是实际绘制折线图的命令。我们在创建的 `ax` 对象上调用 `plot()` 方法，并传入 x 和 y 数据。
- `ax.set_title(...)`、`ax.set_xlabel(...)`、`ax.set_ylabel(...)`：这些方法同样在 `ax` 对象上调用，它们为我们的图表添加描述性元素，使其更易于理解。
- `plt.show()`：此函数告知 Matplotlib 渲染并显示您创建的 Figure。在某些交互式环境（如 Jupyter）中，图表可能会自动显示，但显式调用 `plt.show()` 可以确保您的脚本在运行时（例如从标准 Python 文件运行时）显示图表。这是一个好习惯，建议包含它。

执行此脚本将打开一个显示折线图的窗口：



![简单图表：y = x^2](plots/1150-0.json)



> 一个由基础脚本生成的简单折线图，显示 $x$ 从 0 到 5 时的 $y=x^2$。

这种基本结构为使用 Matplotlib 创建几乎所有图表提供了基础框架。虽然还有其他方式来生成图表（有时您可能会看到直接使用 `plt.plot()` 的代码），但通常建议使用 `fig, ax = plt.subplots()` 的方法并在 `ax` 对象上调用方法（即面向对象风格）。它提供了更好的控制，特别是当您开始在同一 Figure 上处理多个子图时，并且使您的代码更清晰地表示您正在修改可视化的哪个部分。

在接下来的部分中，我们将学习 `ax.plot()` 和 `ax.scatter()` 等特定的绘图函数，并检查通过 `Axes` 对象方法提供的各种自定义选项。

## 参考资料

- [Matplotlib Usage Guide](https://matplotlib.org/stable/contents.html) — Matplotlib Development Team (2024)
  提供了Matplotlib核心概念（包括创建图和轴的面向对象方法）的基础知识，这对于编写脚本至关重要。
- [Python for Data Analysis](https://wesmckinney.com/book/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  一本广受认可的书籍，提供了Pandas数据处理和使用Matplotlib进行数据可视化的全面介绍，详细说明了面向对象的绘图API。
- [Figure and Axes Objects](https://matplotlib.org/3.7.5/api/index.html) — Matplotlib Development Team (2023)
  Publisher: Matplotlib
  具体解释了图（Figure）和轴（Axes）对象的关联性和功能，它们是构建Matplotlib图的主要组成部分。
