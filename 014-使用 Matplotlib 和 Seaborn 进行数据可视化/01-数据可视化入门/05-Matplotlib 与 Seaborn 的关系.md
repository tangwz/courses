# Matplotlib 与 Seaborn 的关系

来源：[原文](https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-1-intro-data-visualization-python/matplotlib-seaborn-relationship)

[返回章节目录](README.md) · [返回课程目录](../README.md)

您现在已经了解了 Matplotlib 和 Seaborn。了解这两个库如何相互关联很重要，因为您经常会一起使用它们。

可以把 Matplotlib 看作是 Python 中绘图的基础引擎。它提供生成各种静态、动态和交互式可视化图表所需的核心对象和函数。它让您能够细致地控制图表的几乎每个方面，从线条粗细和标记 (token)样式到文本位置和坐标轴属性。

另一方面，Seaborn 是直接构建在 Matplotlib *之上*的。它旨在让创建常见的、具有信息量且美观的统计图表类型变得容易得多。它基于 Matplotlib 的方式如下：

1. **更高级的接口：** Seaborn 为特定图表类型（如分布图、分类图和回归模型图）提供函数，这些图表在 Matplotlib 中常需多步或复杂代码。使用 Seaborn，您通常只需一次函数调用即可生成这些图表。
2. **合理的默认设置：** Seaborn 带有多个内置主题和调色板，这些主题和调色板旨在开箱即用，美观且统计信息丰富。虽然 Matplotlib 的默认设置功能实用，但 Seaborn 的默认设置通常只需更少的工作就能显得更精致。
3. **统计估计：** 许多 Seaborn 绘图函数会自动进行必要的统计估计（如计算均值、置信区间或核密度估计），并直接可视化结果。
4. **Pandas DataFrame 集成：** Seaborn 专门设计用于高效处理 Pandas DataFrame。您通常可以将整个 DataFrame 传递给 Seaborn 函数，并使用字符串指定列名，这使得处理结构化数据时的代码更清晰、更易读。

然而，由于 Seaborn 基于 Matplotlib 构建，它不能替代后者。相反，它们相互补充。

- Seaborn 生成的每个图表本质上都是一个 Matplotlib 图表。
- 您可以使用 Matplotlib 函数来自定义通过 Seaborn 生成的图表。例如，在使用 Seaborn 函数创建图表后，您可以使用 Matplotlib 的函数来调整坐标轴标签、添加标题、更改图表大小，或将图表保存到文件。

> 此图显示了 Seaborn 函数如何作为便捷的接口，使用底层的 Matplotlib 引擎生成图表，通常直接从 Pandas DataFrame 获取数据。您的代码可以与这两个库进行交互。

在实践中，许多数据科学家和工程师在生成标准统计可视化图表时，会使用 Seaborn，因为它易于使用且有美观的默认设置。然后，他们会转而使用 Matplotlib 来调整细节、添加复杂注释，或创建高度定制或非标准图表类型。作为初学者，了解这种关系有助于您明白为什么 Seaborn 让某些任务更容易，同时也会知道 Matplotlib 的能力始终可用于更细致的控制。在本课程中，您将看到独立使用和联合使用这两个库的示例。

## 参考资料

- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html) — The Matplotlib Development Team (2024)
  Matplotlib的官方指南，提供核心功能和详细自定义选项的信息，作为Python绘图的基础。
- [Seaborn: statistical data visualization](https://seaborn.pydata.org/index.html) — Michael Waskom (2024)
  Seaborn的官方指南，解释其高层接口、统计绘图功能以及它如何与Matplotlib集成并在此基础上构建。
- [Python for Data Analysis](https://wesmckinney.com/book/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  关于使用Python进行数据分析的基础文本，包含Matplotlib和Seaborn的实际应用，特别是与Pandas DataFrames的结合使用。
- [seaborn: statistical data visualization](https://joss.theoj.org/papers/10.21105/joss.03021) — Michael L. Waskom (2021)
  Journal: Journal of Open Source Software; Publisher: The Open Journal; Volume: 6; Pages: 3021; DOI: [10.21105/joss.03021](https://doi.org/10.21105/joss.03021)
  介绍Seaborn的学术论文，详细说明其设计原则以及它通过在Matplotlib生态系统上构建对统计可视化的贡献。

---

[上一节](04-Seaborn%20%E7%AE%80%E4%BB%8B.md) · [下一节](06-%E8%AE%BE%E7%BD%AE%E6%82%A8%E7%9A%84Python%E7%8E%AF%E5%A2%83.md)
