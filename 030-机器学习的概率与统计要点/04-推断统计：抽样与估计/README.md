# 第 4 章：推断统计：抽样与估计

来源：[原章节](https://apxml.com/zh/courses/probability-statistics-essentials-ml/chapter-4-inferential-statistics-sampling-estimation)

[返回课程目录](../README.md)

到目前为止，我们一直专注于使用描述性统计来总结手头的数据。现在，我们将注意力转向仅根据一小部分数据，即*样本*，来对更大的总体做出有依据的推断。这很必要，因为获取或分析整个总体（例如，所有潜在客户，所有传感器读数）通常是不可能或成本过高的。

本章介绍*推断统计*的基本原理。你将学习如何：

*   区分总体和样本，并理解抽样的必要性。
*   识别不同的抽样方法。
*   理解*中心极限定理* (CLT) 及其在统计推断中的重要作用，特别是关于样本均值的分布。
*   计算总体参数的*点估计*，例如使用样本均值 $\bar{x}$ 来估计总体均值 $\mu$。
*   构建和解释*置信区间*，它根据样本数据为总体参数提供一个合理的值范围。

我们将把这些内容应用于实践，使用 Python 模拟抽样分布并计算估计值，从而连接理论与应用。

## 小节

- 1. [总体和样本](01-%E6%80%BB%E4%BD%93%E5%92%8C%E6%A0%B7%E6%9C%AC.md)
- 2. [抽样方法概述](02-%E6%8A%BD%E6%A0%B7%E6%96%B9%E6%B3%95%E6%A6%82%E8%BF%B0.md)
- 3. [中心极限定理](03-%E4%B8%AD%E5%BF%83%E6%9E%81%E9%99%90%E5%AE%9A%E7%90%86.md)
- 4. [理解点估计](04-%E7%90%86%E8%A7%A3%E7%82%B9%E4%BC%B0%E8%AE%A1.md)
- 5. [置信区间说明](05-%E7%BD%AE%E4%BF%A1%E5%8C%BA%E9%97%B4%E8%AF%B4%E6%98%8E.md)
- 6. [计算均值的置信区间](06-%E8%AE%A1%E7%AE%97%E5%9D%87%E5%80%BC%E7%9A%84%E7%BD%AE%E4%BF%A1%E5%8C%BA%E9%97%B4.md)
- 7. [动手实践：抽样模拟与区间估计](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%8A%BD%E6%A0%B7%E6%A8%A1%E6%8B%9F%E4%B8%8E%E5%8C%BA%E9%97%B4%E4%BC%B0%E8%AE%A1.md)

章节测验：[在线测验](https://apxml.com/zh/courses/probability-statistics-essentials-ml/chapter-4-inferential-statistics-sampling-estimation/quiz)
