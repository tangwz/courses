# 第 1 章：泛化能力的挑战

来源：[原章节](https://apxml.com/zh/courses/deep-learning-regularization-optimization/chapter-1-generalization-challenge)

[返回课程目录](../README.md)

构建深度学习模型的主要目标不仅仅是在训练数据上的表现；更是要确保模型在新的、未见过的数据上也能表现良好。这种能力被称为泛化。泛化效果不佳的模型常常会遇到过拟合（对训练数据学习得过于具体，甚至包含了噪声）或欠拟合（模型过于简单，无法捕捉数据的规律）的问题。

在本章中，我们将为理解和改进泛化能力奠定开端。我们将定义泛化、过拟合和欠拟合。您将了解到深度学习背景下的偏差-方差权衡，以及如何使用学习曲线作为诊断工具。我们还将介绍正则化和优化在处理泛化问题中的作用，并引导您设置后续实践工作所需的软件环境。最后，您将练习通过视觉方式识别过拟合。

## 小节

- 1. [模型泛化介绍](01-%E6%A8%A1%E5%9E%8B%E6%B3%9B%E5%8C%96%E4%BB%8B%E7%BB%8D.md)
- 2. [理解欠拟合与过拟合](02-%E7%90%86%E8%A7%A3%E6%AC%A0%E6%8B%9F%E5%90%88%E4%B8%8E%E8%BF%87%E6%8B%9F%E5%90%88.md)
- 3. [深度学习中的偏差-方差权衡](03-%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B8%AD%E7%9A%84%E5%81%8F%E5%B7%AE-%E6%96%B9%E5%B7%AE%E6%9D%83%E8%A1%A1.md)
- 4. [诊断模型表现：学习曲线](04-%E8%AF%8A%E6%96%AD%E6%A8%A1%E5%9E%8B%E8%A1%A8%E7%8E%B0%EF%BC%9A%E5%AD%A6%E4%B9%A0%E6%9B%B2%E7%BA%BF.md)
- 5. [验证与交叉验证策略](05-%E9%AA%8C%E8%AF%81%E4%B8%8E%E4%BA%A4%E5%8F%89%E9%AA%8C%E8%AF%81%E7%AD%96%E7%95%A5.md)
- 6. [正则化与优化的作用](06-%E6%AD%A3%E5%88%99%E5%8C%96%E4%B8%8E%E4%BC%98%E5%8C%96%E7%9A%84%E4%BD%9C%E7%94%A8.md)
- 7. [配置开发环境](07-%E9%85%8D%E7%BD%AE%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83.md)
- 8. [动手：过拟合的可视化](08-%E5%8A%A8%E6%89%8B%EF%BC%9A%E8%BF%87%E6%8B%9F%E5%90%88%E7%9A%84%E5%8F%AF%E8%A7%86%E5%8C%96.md)

章节测验：[在线测验](https://apxml.com/zh/courses/deep-learning-regularization-optimization/chapter-1-generalization-challenge/quiz)
