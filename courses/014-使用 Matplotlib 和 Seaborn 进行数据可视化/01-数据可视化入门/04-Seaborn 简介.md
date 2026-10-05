# Seaborn 简介

来源：[原文](https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-1-intro-data-visualization-python/intro-seaborn)

[返回章节目录](README.md) · [返回课程目录](../README.md)

尽管 Matplotlib 提供了在 Python 中绘制图形的基本构成要素，但你可能经常希望创建更具统计倾向或视觉效果更佳的图形，而无需编写大量代码。这时 Seaborn 就派上用场了。

Seaborn 是一个基于 Matplotlib 构建的 Python 数据可视化库。你可以将其视为一个补充工具，它提供了一个更高级别的接口，专门用于创建信息丰富且美观的统计图形。如果说 Matplotlib 让你能够对图形的每个元素进行精细控制，那么 Seaborn 则为常见的可视化任务，特别是那些涉及统计分析和数据查看的任务，提供了合理的默认设置和便捷函数。

### Seaborn 的特点:

1. **统计侧重：** Seaborn 擅长将统计关系可视化。其函数通常被设计为自动执行必要的统计估计和聚合，以生成概括数据或显示不确定性的信息图形。例子包括可视化分布、绘制线性回归模型，或使用均值或中位数等估计值比较类别。
2. **复杂图形的简化语法：** 与仅使用 Matplotlib 实现相比，使用 Seaborn 创建多面板分类比较、详细分布视图（如小提琴图或核密度估计图）或热力图等图形通常需要明显更少的代码。
3. **美观的默认样式和调色板：** Seaborn 自带多种内置主题和调色板，旨在美观且能有效传达信息。这使得立即创建精美的可视化效果变得更容易。
4. **与 Pandas 的良好集成：** Seaborn 与 Pandas DataFrame 配合得非常好。大多数 Seaborn 函数直接接受 DataFrame 作为输入，允许你通过引用列名来指定图形，这简化了绘图的数据处理。

通常，你不会在 Matplotlib *或* Seaborn 之间进行选择。相反，你会将它们结合使用。Seaborn 简化了许多常见统计图类型的创建，而 Matplotlib 则提供了底层引擎和在需要时进行更精细定制的工具。许多数据分析师和工程师使用 Seaborn 进行快速查看和绘制标准统计图，然后使用 Matplotlib 函数微调 (fine-tuning)最终外观（例如调整标签、标题或添加特定注释）。

简而言之，Seaborn 有助于弥补对数据进行统计分析与视觉化呈现这些分析结果之间的差距，在这些特定任务中，它通常比单独使用 Matplotlib 更方便，默认美观度也更好。后续章节中，我们将学习如何使用 Seaborn 的功能，通常会与 Pandas DataFrame 结合使用。

## 参考资料

- [Seaborn: Statistical Data Visualization](https://seaborn.pydata.org/) — Michael Waskom and the Seaborn development team (2024)
  Seaborn库的主要官方文档，提供全面的指南和API参考。
- [Seaborn: a Python library for visualization statistical models](https://joss.theoj.org/papers/10.21105/joss.03021) — Michael L. Waskom (2021)
  Journal: Journal of Open Source Software; Publisher: The Open Journal; Volume: 6; Pages: 3021; DOI: [10.21105/joss.03021](https://doi.org/10.21105/joss.03021)
  介绍Seaborn库，详细阐述其设计原则和统计数据可视化功能。
- [Python Data Science Handbook: Essential Tools for Working with Data](https://jakevdp.github.io/PythonDataScienceHandbook/) — Jake VanderPlas (2016)
  Publisher: O'Reilly Media
  提供Python数据科学工具的广泛概述，包含关于Matplotlib和Seaborn的数据可视化专用章节。

---

[上一节](03-Matplotlib%20%E4%BB%8B%E7%BB%8D.md) · [下一节](05-Matplotlib%20%E4%B8%8E%20Seaborn%20%E7%9A%84%E5%85%B3%E7%B3%BB.md)
