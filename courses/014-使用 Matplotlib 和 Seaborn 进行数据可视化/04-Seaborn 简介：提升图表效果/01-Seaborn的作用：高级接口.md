# Seaborn的作用：高级接口

来源：[原文](https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-4-seaborn-enhanced-visuals-intro/seaborn-role-high-level)

[返回章节目录](README.md) · [返回课程目录](../README.md)

在了解Seaborn时，理解它与Matplotlib（一个基础的Python绘图库）的关系至关重要。Seaborn并非Matplotlib的替代品；相反，它是建立在Matplotlib*之上*的。Matplotlib提供绘图的基本构建块，例如图形、坐标轴、线条和点。Seaborn使用这些构建块，但提供了一种更结构化的方法，尤其适用于创建常见的统计图表。

那么，Seaborn作为“高级接口”意味着什么？实质上，与直接使用Matplotlib完成类似任务相比，Seaborn函数的设计目标是使用更少的代码处理更复杂的任务。

以下是Seaborn作用的主要方面：

1. **复杂图表的简化函数调用：** 许多标准统计图表在Matplotlib中需要多个步骤才能正确创建（例如，计算组统计量、处理分类数据、选择合适的颜色）。Seaborn常将这些步骤打包到一个函数调用中。例如，创建一个条形图，显示变量在不同类别间的平均值，并带有表示置信区间的误差棒，通常只需一个Seaborn函数即可完成，而Matplotlib则需要手动数据聚合和多次函数调用。
2. **内置统计估算：** Seaborn函数经常包含统计计算。当您请求一个`barplot`时，Seaborn会自动计算均值（或其他估计量）并引导置信区间以供显示。类似地，像`kdeplot`这样的函数会自动计算核密度估计。这种对统计表示的侧重是一个核心特性。
3. **与Pandas DataFrames的良好衔接：** 尽管Matplotlib可以绘制Pandas DataFrames中的数据，但Seaborn被明确设计为与它们良好协作。大多数Seaborn函数通过`data`参数 (parameter)直接接受DataFrame，并允许您通过传递列名字符串来简单指定变量（例如，`x="column_name"`，`y="other_column"`）。这通常使绘图代码更清晰、更易读，尤其是在处理整洁数据格式时。
4. **美观的默认样式和调色板：** Matplotlib提供极高的灵活性，但通常需要大量自定义才能制作出出版质量的图表。Seaborn自带多种内置主题和调色板，提供开箱即用的美观默认设置。这使得快速创建视觉吸引力强的图表变得更容易，同时您仍然可以根据需要使用Matplotlib函数进行广泛自定义。

实质上，Seaborn作为Matplotlib之上的一个智能层，简化了信息丰富且美观的统计图形的创建。它自动化常见任务，与数据分析工作流（特别是使用Pandas的工作流）顺畅地集成，并提供合理的默认设置。这让您能够更侧重于数据解读，而不是绘图机制，尤其是在处理标准统计图表模式时。

在学习本章时，您将看到这些优势的实际效果，我们将了解Seaborn用于设置样式、使用调色板以及创建散点图和折线图等基本图表的函数。请记住，由于Seaborn底层使用Matplotlib，您通常可以使用Matplotlib函数进一步优化由Seaborn创建的图表，从而获得两全其美的效果：高级别的便利性以及在需要时的低级别控制。

## 参考资料

- [Seaborn: statistical data visualization](https://seaborn.pydata.org/index.html) — Michael Waskom (2023)
  官方文档，解释了Seaborn的设计原则、其作为高级接口的功能，以及它基于Matplotlib构建的特性。
- [Seaborn: statistical data visualization](https://joss.theoj.org/papers/10.21105/joss.03021) — Michael L. Waskom (2021)
  Journal: Journal of Open Source Software; Publisher: The Open Journal; Volume: 6; Pages: 3021; DOI: [10.21105/joss.03021](https://doi.org/10.21105/joss.03021)
  正式介绍Seaborn的原始出版物，概述了其设计目标和统计图形功能。
- [Python Data Science Handbook: Essential Tools for Working with Data](https://jakevdp.github.io/PythonDataScienceHandbook/) — Jake VanderPlas (2016)
  Publisher: O'Reilly Media
  一本备受推崇的书，涵盖了Matplotlib和Seaborn，详细介绍了Seaborn如何在Matplotlib的基础上提供更高级的API进行统计绘图。

---

[上一节](../03-%E5%9F%BA%E6%9C%AC%20Matplotlib%20%E5%9B%BE%E8%A1%A8%E7%B1%BB%E5%9E%8B/07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E7%BB%98%E5%88%B6%E4%B8%8D%E5%90%8C%E6%95%B0%E6%8D%AE%E7%B1%BB%E5%9E%8B%E5%9B%BE%E8%A1%A8.md) · [下一节](02-%E8%AE%BE%E5%AE%9A%20Seaborn%20%E6%A0%B7%E5%BC%8F%E4%B8%8E%E4%B8%8A%E4%B8%8B%E6%96%87.md)
