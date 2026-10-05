# 第 2 章：常用概率分布

来源：[原章节](https://apxml.com/zh/courses/probability-statistics-essentials-ml/chapter-2-common-probability-distributions)

[返回课程目录](../README.md)

在上一章概率基础之上，我们现在将重点放在描述随机变量不同结果发生可能性的特定数学函数上：概率分布。这些分布是数据中固有不确定性建模的重要工具，这也是机器学习中常见的一项工作。

本章将介绍统计分析中常用到、并作为机器学习模型组成部分的几种常见分布。您将学习如何：

*   识别并说明主要的离散分布：伯努利分布、二项分布（$n, p$）和泊松分布（$\lambda$）。
*   识别并说明主要的连续分布：均匀分布、正态分布（$\mu, \sigma^2$）和指数分布（$\lambda$）。
*   理解使每种分布适合对特定类型数据或现象进行建模的属性。
*   应用Python库（特别是SciPy）来计算概率（例如概率密度/质量函数和累积分布函数）、生成随机样本以及可视化这些分布。

在本章结束时，您将能够识别并使用这些标准分布，从而更好地分析数据并理解机器学习中使用的统计方法。

## 小节

- 1. [伯努利分布和二项分布](01-%E4%BC%AF%E5%8A%AA%E5%88%A9%E5%88%86%E5%B8%83%E5%92%8C%E4%BA%8C%E9%A1%B9%E5%88%86%E5%B8%83.md)
- 2. [泊松分布](02-%E6%B3%8A%E6%9D%BE%E5%88%86%E5%B8%83.md)
- 3. [均匀分布](03-%E5%9D%87%E5%8C%80%E5%88%86%E5%B8%83.md)
- 4. [正态（高斯）分布](04-%E6%AD%A3%E6%80%81%EF%BC%88%E9%AB%98%E6%96%AF%EF%BC%89%E5%88%86%E5%B8%83.md)
- 5. [指数分布](05-%E6%8C%87%E6%95%B0%E5%88%86%E5%B8%83.md)
- 6. [数据建模中的性质与应用](06-%E6%95%B0%E6%8D%AE%E5%BB%BA%E6%A8%A1%E4%B8%AD%E7%9A%84%E6%80%A7%E8%B4%A8%E4%B8%8E%E5%BA%94%E7%94%A8.md)
- 7. [在 SciPy 中使用分布](07-%E5%9C%A8%20SciPy%20%E4%B8%AD%E4%BD%BF%E7%94%A8%E5%88%86%E5%B8%83.md)
- 8. [动手实践：模拟与绘制分布](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%A8%A1%E6%8B%9F%E4%B8%8E%E7%BB%98%E5%88%B6%E5%88%86%E5%B8%83.md)

章节测验：[在线测验](https://apxml.com/zh/courses/probability-statistics-essentials-ml/chapter-2-common-probability-distributions/quiz)
