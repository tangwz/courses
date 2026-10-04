# 第 5 章：微积分的运用：简单优化

来源：[原章节](https://apxml.com/zh/courses/calculus-fundamentals-machine-learning/chapter-5-calculus-action-optimization)

[返回课程目录](../README.md)

学完导数、优化原理和梯度后，我们现在将这些知识点与一个实际的机器学习场景关联起来。本章将说明微积分如何通过优化来辅助模型训练过程。

在此，您将学习如何：
*   运用代价函数的原理到一个简单的线性模型。
*   计算代价函数相对于模型参数的梯度。
*   观察这些梯度如何在梯度下降的步骤中被用于更新参数。
*   理解学习率在此过程中的作用。

我们将通过一个使用简单线性回归（$y = mx + b$）的例子来学习。您将看到，计算代价函数的偏导数如何使我们能够系统地调整 $m$ 和 $b$ 以最小化误差，这说明了训练许多机器学习模型背后的主要运作方式。

## 小节

- 1. [回顾：优化目标与梯度下降](01-%E5%9B%9E%E9%A1%BE%EF%BC%9A%E4%BC%98%E5%8C%96%E7%9B%AE%E6%A0%87%E4%B8%8E%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D.md)
- 2. [示例：简单线性回归模型](02-%E7%A4%BA%E4%BE%8B%EF%BC%9A%E7%AE%80%E5%8D%95%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92%E6%A8%A1%E5%9E%8B.md)
- 3. [为线性回归定义成本函数](03-%E4%B8%BA%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92%E5%AE%9A%E4%B9%89%E6%88%90%E6%9C%AC%E5%87%BD%E6%95%B0.md)
- 4. [计算成本函数的梯度](04-%E8%AE%A1%E7%AE%97%E6%88%90%E6%9C%AC%E5%87%BD%E6%95%B0%E7%9A%84%E6%A2%AF%E5%BA%A6.md)
- 5. [执行梯度下降的单次更新](05-%E6%89%A7%E8%A1%8C%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E7%9A%84%E5%8D%95%E6%AC%A1%E6%9B%B4%E6%96%B0.md)
- 6. [学习率参数](06-%E5%AD%A6%E4%B9%A0%E7%8E%87%E5%8F%82%E6%95%B0.md)
- 7. [整合：优化过程](07-%E6%95%B4%E5%90%88%EF%BC%9A%E4%BC%98%E5%8C%96%E8%BF%87%E7%A8%8B.md)
- 8. [实践操作：手动梯度计算](08-%E5%AE%9E%E8%B7%B5%E6%93%8D%E4%BD%9C%EF%BC%9A%E6%89%8B%E5%8A%A8%E6%A2%AF%E5%BA%A6%E8%AE%A1%E7%AE%97.md)

章节测验：[在线测验](https://apxml.com/zh/courses/calculus-fundamentals-machine-learning/chapter-5-calculus-action-optimization/quiz)
