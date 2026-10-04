# Matplotlib 图的结构：Figure 与 Axes

来源：[原文](https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-2-matplotlib-plotting-fundamentals/matplotlib-plot-anatomy)

[返回章节目录](README.md) · [返回课程目录](../README.md)

要使用 Matplotlib 创建有效的图表，了解绘图的基本构成非常重要。可以把它想象成建造事物：你需要一个根基，然后在上面搭建主要结构。这种结构在 Matplotlib 中主要包含两个核心部分：`Figure`（图形对象）和 `Axes`（坐标系对象）。

### `Figure` 对象

想象你是一位艺术家。在你开始绘画之前，你需要一块画布。在 Matplotlib 中，`Figure` 对象就是这块画布。它是与你的图表相关的所有内容的顶层容器。这包括所有的绘图区域、标题、可能应用于整个图形的图例，以及在画布上绘制的任何其他元素。

你可以把 `Figure` 看作是你的图表将出现的整体窗口或页面。它不包含实际绘制的数据本身，但它保留了绘图发生的空间。通常，你会首先创建一个 `Figure`。虽然 Matplotlib 经常可以隐式地为你创建一个，但了解它的存在对于后续获得更多控制权很有帮助，尤其是在同时制作多个图表时。

`Figure` 的重要特点：

- 它是所有图表元素的最外层容器。
- 它追踪一个或多个 `Axes` 对象。
- 整体图形大小、背景颜色和保存选项等属性通常与 `Figure` 相关联。

### `Axes` 对象

现在，思考你想要在画布上绘制的实际图像。这就是 `Axes` 对象发挥作用的地方。尽管名称听起来是复数，但一个单独的 `Axes` 对象表示 `Figure` *内*的一个特定绘图区域。这是你的数据被绘制的区域——你会看到线条、点、柱状图等。

`Axes` 对象包含你通常与图表相关联的大多数元素：

- X轴和Y轴（以及3D图的潜在Z轴）。
- 坐标轴上的刻度和刻度标签。
- 坐标轴标签（例如，'时间', '温度'）。
- 此绘图区域的特定图表标题。
- 你的数据的视觉呈现（线条、散点、直方图）。
- 解释此图表中数据的任何图例。

**重要提示：** 不要混淆 `Axes`（以 'es' 结尾）和 `Axis`（以 'is' 结尾）。一个 `Axes` 对象包含两个（或3D图的三个）`Axis` 对象（X轴、Y轴、Z轴）。大多数时候，你会与 `Axes` 对象交互，使用 `plot()`、`scatter()`、`hist()` 等函数来创建你的图表。

### `Figure` 与 `Axes` 的关系

这种关系是直接的：一个 `Figure` 包含一个或多个 `Axes` 对象。

- **简单情况：** 最基本的图表通常涉及一个 `Figure` 包含一个单独的 `Axes`。你最初最常使用的就是这种情况。
- **多个图表（子图）：** 一个单独的 `Figure` 可以容纳多个 `Axes` 对象，它们以网格形式排列。这就是你创建子图的方式，允许你在同一个画布上同时显示多个相关的图表。

了解这种层级结构——`Figure` 作为容器，`Axes` 作为实际绘图区域——非常重要。它为有效组织和自定义你的图表提供了框架。当你调用绘图函数时，你通常是调用属于 `Axes` 对象的方法，告诉 Matplotlib *在哪里*绘制数据，在整体 `Figure` 画布内。

> 该图表展示了层级关系。`Figure` 作为整体容器，容纳一个或多个 `Axes` 对象，实际数据图表在此发生。

理解整体 `Figure` 画布与特定 `Axes` 绘图区域的区别是掌握 Matplotlib 的第一步。随着学习的推进，你会看到直接操作这些对象可以让你对图表的各个方面进行精细控制。

## 参考资料

- [Anatomy of a Matplotlib plot](https://matplotlib.org/3.10.6/gallery/showcase/anatomy.html) — The Matplotlib development team (2025)
  Publisher: The Matplotlib development team
  官方教程，阐释了 Matplotlib 图形的基本组成部分，包括 Figure 和 Axes。
- [Python Data Science Handbook: Essential Tools for Working with Data - Chapter 4.00 - Introduction to Matplotlib](https://jakevdp.github.io/PythonDataScienceHandbook/04.00-introduction-to-matplotlib.html) — Jake VanderPlas (2016)
  Publisher: O'Reilly Media
  一本广受好评的书籍，清晰全面地介绍了 Matplotlib 的面向对象接口和绘图结构。
- [Usage Guide](https://matplotlib.org/stable/tutorials/introductory/usage.html) — Matplotlib Development Team (2024)
  Publisher: Matplotlib Development Team
  官方指南，详细介绍了如何与 Matplotlib 的面向对象接口交互，演示了 Figure 和 Axes 对象的程序化创建和使用。

---

[上一节](../01-%E6%95%B0%E6%8D%AE%E5%8F%AF%E8%A7%86%E5%8C%96%E5%85%A5%E9%97%A8/08-%E6%82%A8%E7%9A%84%E7%AC%AC%E4%B8%80%E4%B8%AA%E7%AE%80%E5%8D%95%E5%9B%BE%E8%A1%A8.md) · [下一节](02-%E5%88%9B%E5%BB%BA%E5%9F%BA%E7%A1%80%E7%BB%98%E5%9B%BE%E8%84%9A%E6%9C%AC.md)
