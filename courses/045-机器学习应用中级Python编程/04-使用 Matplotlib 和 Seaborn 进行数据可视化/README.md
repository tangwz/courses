# 第 4 章：使用 Matplotlib 和 Seaborn 进行数据可视化

来源：[原章节](https://apxml.com/zh/courses/intermediate-python-programming-ml/chapter-4-data-visualization-matplotlib-seaborn)

[返回课程目录](../README.md)

在用 NumPy 和 Pandas 处理完数据结构后，弄清数据内容是接下来自然要做的事。数据可视化提供了以图形方式呈现数据的手段，能更方便地看出原始表格中可能遗漏的模式、趋势、异常值和数据间的关联。这种做法对描述性数据分析和有效传达结果都非常重要。

本章主要介绍两个用于制作静态图表的 Python 库：Matplotlib 和 Seaborn。你将从 Matplotlib 的基本原理学起，学习如何生成折线图、条形图、直方图和散点图等常用图表。我们会讲解如何通过标签、标题、颜色和样式来调整这些图表，以及如何使用子图将多个图表排布在一个图形中。

接着，你将接触到 Seaborn，一个在 Matplotlib 基础上构建的库，它提供了一个更高级的接口，专门用于制作内容丰富、美观的统计图形。你将学会用简洁的代码制作更复杂的图表，例如热力图、成对散点图和分布图。最后，我们会讲解保存你制作的图表的实际操作，你可以将它们保存为多种格式，方便纳入报告和演示文稿中。本章结束后，你将能够针对不同的数据类型和分析目标选择合适的图表，并使用 Matplotlib 和 Seaborn 来实现它们。

## 小节

- 1. [Matplotlib 绘图基本原理](01-Matplotlib%20%E7%BB%98%E5%9B%BE%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86.md)
- 2. [创建常用图表类型](02-%E5%88%9B%E5%BB%BA%E5%B8%B8%E7%94%A8%E5%9B%BE%E8%A1%A8%E7%B1%BB%E5%9E%8B.md)
- 3. [图表自定义](03-%E5%9B%BE%E8%A1%A8%E8%87%AA%E5%AE%9A%E4%B9%89.md)
- 4. [使用子图](04-%E4%BD%BF%E7%94%A8%E5%AD%90%E5%9B%BE.md)
- 5. [用于统计数据可视化的 Seaborn 简介](05-%E7%94%A8%E4%BA%8E%E7%BB%9F%E8%AE%A1%E6%95%B0%E6%8D%AE%E5%8F%AF%E8%A7%86%E5%8C%96%E7%9A%84%20Seaborn%20%E7%AE%80%E4%BB%8B.md)
- 6. [使用 Seaborn 创建高级图表](06-%E4%BD%BF%E7%94%A8%20Seaborn%20%E5%88%9B%E5%BB%BA%E9%AB%98%E7%BA%A7%E5%9B%BE%E8%A1%A8.md)
- 7. [分布与关系的可视化](07-%E5%88%86%E5%B8%83%E4%B8%8E%E5%85%B3%E7%B3%BB%E7%9A%84%E5%8F%AF%E8%A7%86%E5%8C%96.md)
- 8. [保存绘图用于报告和演示文稿](08-%E4%BF%9D%E5%AD%98%E7%BB%98%E5%9B%BE%E7%94%A8%E4%BA%8E%E6%8A%A5%E5%91%8A%E5%92%8C%E6%BC%94%E7%A4%BA%E6%96%87%E7%A8%BF.md)
- 9. [动手实践：数据可视化查看](09-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%95%B0%E6%8D%AE%E5%8F%AF%E8%A7%86%E5%8C%96%E6%9F%A5%E7%9C%8B.md)

章节测验：[在线测验](https://apxml.com/zh/courses/intermediate-python-programming-ml/chapter-4-data-visualization-matplotlib-seaborn/quiz)
