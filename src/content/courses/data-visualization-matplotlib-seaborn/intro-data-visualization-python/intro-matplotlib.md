---
course: "data-visualization-matplotlib-seaborn"
chapter: "intro-data-visualization-python"
lesson: "intro-matplotlib"
sourceId: 1138
sourceUrl: "https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-1-intro-data-visualization-python/intro-matplotlib"
title: "Matplotlib 介绍"
description: "了解 Matplotlib，它是 Python 中数据可视化的基础库。"
order: 3
plots: []
sourceHash: "e186e3796da5d64fdf6d03a180fd724666f9ff33d0a945aa9cb7130bf0aabf80"
sourceCorrections: []
---

在明确了数据可视化的重要性之后，我们来关注一个在 Python 中将要使用的主要工具：Matplotlib。可以将 Matplotlib 看作是 Python 绘图库的**鼻祖**。它是一个功能完备的库，用于在 Python 中创建静态、动画和交互式可视化图表。

Matplotlib 由 John D. Hunter 于 2003 年最初构思，旨在在 Python 中复现 MATLAB 的绘图功能。多年来，它已成为 Python 科学计算社区中绘图的事实上的*标准*，并作为许多其他可视化库（包括我们接下来要讨论的 Seaborn）的基础。

### Matplotlib 的功能

其核心是，Matplotlib 让你能够精确控制图表的几乎每个方面。你可以将图表（figure）看作是包含你所有绘图的整个画布或窗口。在这个图表中，你可以有一个或多个子图，在 Matplotlib 术语中通常称为 *Axes*。Matplotlib 允许你：

1. **创建多种多样的图表：** 从简单的折线图和散点图到复杂的条形图、直方图、饼图、误差图，甚至基本的 3D 图。
2. **自定义每个元素：** 你可以更改线条样式、颜色、标记 (token)类型、添加文本注释、配置坐标轴标签和刻度、添加图例、调整图表大小等等。
3. **输出多种格式：** 将可视化图表保存为高质量图像（如 PNG、JPG），用于网页或报告，或者保存为矢量图形（如 PDF、SVG），这些图形在缩放时不会损失分辨率，非常适合出版物。

### Matplotlib 的常见用法

Matplotlib 提供两种主要绘图方式：

1. **`pyplot` 接口：** 这是一组命令式函数，使得 Matplotlib 的工作方式类似于 MATLAB。每个 `pyplot` 函数都会对图表进行一些更改：例如，创建一个图表、在图表中创建一个绘图区域、在绘图区域中绘制一些线条、用标签装饰图表等。此接口常用于快速、交互式绘图和简单脚本。我们将从这种方法开始。
2. **面向对象接口：** 这需要显式创建和操作 Figure 和 Axes 对象。虽然对于简单绘图来说它可能稍微冗长，但它提供了更大的控制力和灵活性，特别是在同一图表上处理多个绘图或创建复杂可视化图表时。随着你经验的增长，我们将在课程后期介绍这一点。

### 为什么从 Matplotlib 开始？

尽管像 Seaborn 这样的新库提供了更简单的方法来创建某些类型的统计侧重图表，但理解 Matplotlib 具有重要意义，因为：

- **它是核心：** 许多其他绘图库在底层使用 Matplotlib。了解 Matplotlib 有助于你理解这些其他库的工作方式，并在需要时进一步自定义其输出。
- **它提供完全控制：** 当你需要微调 (fine-tuning)图表的每个细节时，Matplotlib 让你能够做到这一点。
- **它被广泛使用：** 你将在科学 Python 文档、示例和现有项目中经常遇到 Matplotlib 代码。

Matplotlib 通常与 NumPy 结合使用，NumPy 是 Python 中用于数值计算的基本包。你通常会使用 NumPy 数组（或 Pandas DataFrame，我们稍后介绍）来准备数据，然后将其传递给 Matplotlib 函数进行绘图。

本质上，Matplotlib 是一个功能强大且多功能的库，它为 Python 中的数据可视化提供构建模块。尽管其语法有时会显得冗长，但掌握其基本知识将使你能够创建几乎所有可以想象的静态图表。我们将从使用其更简单的 `pyplot` 接口开始，以帮助你快速上手。

## 参考资料

- [Matplotlib Documentation](https://matplotlib.org/) — John Hunter (1968–2012) and the project's many contributors (2024)
  Matplotlib库的官方用户指南和API参考。
- [Python for Data Analysis](https://wesmckinney.com/book/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  一本广泛使用的书籍，涵盖使用Pandas进行数据处理和使用Matplotlib进行数据可视化，包括核心概念。
