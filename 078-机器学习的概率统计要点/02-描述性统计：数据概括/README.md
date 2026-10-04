# 第 2 章：描述性统计：数据概括

来源：[原章节](https://apxml.com/zh/courses/probability-statistics-fundamentals-ml/chapter-2-descriptive-statistics)

[返回课程目录](../README.md)

原始数据，无论是实验数据还是日志数据，通常表现为大量数字或分类的集合。为了理解这些数据，我们需要方法来概括其主要特点。本章主要讲述*描述性统计*，即用于描述和概括数据集特点的方法。

您将学习如何计算和解读：
*   集中趋势量度：*均值*、*中位数*和*众数*，用于确定数据的中心位置。
*   离散程度量度（或离散程度）：*全距*、*方差*和*标准差*，描述数据点之间的差异程度。
*   位置量度：*百分位数*和*四分位数*，有助于理解数据分布情况。

我们还将介绍基本的数据可视化技术，例如*直方图*和*箱线图*，作为理解这些概括结果的视觉辅助。在本章中，您将看到如何使用Python的NumPy和Pandas库高效地计算这些统计量。

## 小节

- 1. [衡量中心：均值、中位数和众数](01-%E8%A1%A1%E9%87%8F%E4%B8%AD%E5%BF%83%EF%BC%9A%E5%9D%87%E5%80%BC%E3%80%81%E4%B8%AD%E4%BD%8D%E6%95%B0%E5%92%8C%E4%BC%97%E6%95%B0.md)
- 2. [衡量变异性：极差](02-%E8%A1%A1%E9%87%8F%E5%8F%98%E5%BC%82%E6%80%A7%EF%BC%9A%E6%9E%81%E5%B7%AE.md)
- 3. [衡量离散程度：方差和标准差](03-%E8%A1%A1%E9%87%8F%E7%A6%BB%E6%95%A3%E7%A8%8B%E5%BA%A6%EF%BC%9A%E6%96%B9%E5%B7%AE%E5%92%8C%E6%A0%87%E5%87%86%E5%B7%AE.md)
- 4. [理解百分位数与四分位数](04-%E7%90%86%E8%A7%A3%E7%99%BE%E5%88%86%E4%BD%8D%E6%95%B0%E4%B8%8E%E5%9B%9B%E5%88%86%E4%BD%8D%E6%95%B0.md)
- 5. [可视化分布：直方图](05-%E5%8F%AF%E8%A7%86%E5%8C%96%E5%88%86%E5%B8%83%EF%BC%9A%E7%9B%B4%E6%96%B9%E5%9B%BE.md)
- 6. [可视化汇总：箱线图](06-%E5%8F%AF%E8%A7%86%E5%8C%96%E6%B1%87%E6%80%BB%EF%BC%9A%E7%AE%B1%E7%BA%BF%E5%9B%BE.md)
- 7. [使用 Python 计算描述性统计](07-%E4%BD%BF%E7%94%A8%20Python%20%E8%AE%A1%E7%AE%97%E6%8F%8F%E8%BF%B0%E6%80%A7%E7%BB%9F%E8%AE%A1.md)
- 8. [实践：数据集汇总](08-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%95%B0%E6%8D%AE%E9%9B%86%E6%B1%87%E6%80%BB.md)

章节测验：[在线测验](https://apxml.com/zh/courses/probability-statistics-fundamentals-ml/chapter-2-descriptive-statistics/quiz)
