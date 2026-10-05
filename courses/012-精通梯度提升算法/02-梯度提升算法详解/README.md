# 第 2 章：梯度提升算法详解

来源：[原章节](https://apxml.com/zh/courses/mastering-gradient-boosting-algorithms/chapter-2-gradient-boosting-algorithm-depth)

[返回课程目录](../README.md)

在了解了集成方法和加法模型的基本原理后，本章将着重讲解标准梯度提升机 (GBM) 算法的运作机制。我们将从一般思路出发，循序渐进地理解这些模型是如何构建的。

您将学会从优化角度看待提升过程，特别是在函数空间中进行的梯度下降。我们将正式推导通用GBM算法，阐明残差或伪残差如何指导每个顺序基学习器（通常是决策树）的训练。

涵盖的主要方面包括：

*   GBM算法的数学推导。
*   对常见损失函数（例如回归中的平方误差或分类中的对数损失）的分析，以及它们的梯度（$ \nabla L $）如何推动模型拟合过程。
*   收缩（即学习率参数）在控制每棵树贡献和辅助正则化方面的作用。
*   诸如行和列子采样（随机梯度提升）等技术，以提高模型泛化能力和计算效率。
*   使用Scikit-learn的`GradientBoostingRegressor`和`GradientBoostingClassifier`进行实际实现。

学完本章，您将对经典的GBM算法有一个详细的操作理解，为您后续讨论的更高级实现做好准备。一个实践环节将指导您使用Python构建您的第一个GBM模型。

## 小节

- 1. [函数梯度下降](01-%E5%87%BD%E6%95%B0%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D.md)
- 2. [推导通用GBM算法](02-%E6%8E%A8%E5%AF%BC%E9%80%9A%E7%94%A8GBM%E7%AE%97%E6%B3%95.md)
- 3. [回归中常见的损失函数](03-%E5%9B%9E%E5%BD%92%E4%B8%AD%E5%B8%B8%E8%A7%81%E7%9A%84%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.md)
- 4. [分类任务的常用损失函数](04-%E5%88%86%E7%B1%BB%E4%BB%BB%E5%8A%A1%E7%9A%84%E5%B8%B8%E7%94%A8%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.md)
- 5. [收缩（学习率）的作用](05-%E6%94%B6%E7%BC%A9%EF%BC%88%E5%AD%A6%E4%B9%A0%E7%8E%87%EF%BC%89%E7%9A%84%E4%BD%9C%E7%94%A8.md)
- 6. [抽样方法 (随机梯度提升)](06-%E6%8A%BD%E6%A0%B7%E6%96%B9%E6%B3%95%20%28%E9%9A%8F%E6%9C%BA%E6%A2%AF%E5%BA%A6%E6%8F%90%E5%8D%87%29.md)
- 7. [使用 Scikit-learn 实现 GBM](07-%E4%BD%BF%E7%94%A8%20Scikit-learn%20%E5%AE%9E%E7%8E%B0%20GBM.md)
- 8. [实践：构建基础GBM模型](08-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9E%84%E5%BB%BA%E5%9F%BA%E7%A1%80GBM%E6%A8%A1%E5%9E%8B.md)
