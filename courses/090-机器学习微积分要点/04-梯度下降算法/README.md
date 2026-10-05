# 第 4 章：梯度下降算法

来源：[原章节](https://apxml.com/zh/courses/calculus-essentials-machine-learning/chapter-4-gradient-descent-algorithms)

[返回课程目录](../README.md)

基于上一章对梯度的理解，我们现在关注如何使用它们进行模型优化。本章介绍梯度下降，这是一种迭代算法，用于寻找函数的最小值，在机器学习中通常是成本函数$J(\theta)$。

您将掌握其原理以及通过沿着梯度$\nabla J(\theta)$相反的方向移动来更新模型参数的具体步骤。我们将考察学习率$\alpha$的作用，比较批量梯度下降、随机梯度下降和Mini-batch梯度下降等不同方法，并讨论常见的优化问题，例如局部最小值。本章包含一个关于实现基本算法的实践部分。

## 小节

- 1. [梯度下降的直观原理](01-%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E7%9A%84%E7%9B%B4%E8%A7%82%E5%8E%9F%E7%90%86.md)
- 2. [梯度下降算法的步骤](02-%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E7%AE%97%E6%B3%95%E7%9A%84%E6%AD%A5%E9%AA%A4.md)
- 3. [学习率参数](03-%E5%AD%A6%E4%B9%A0%E7%8E%87%E5%8F%82%E6%95%B0.md)
- 4. [批梯度下降](04-%E6%89%B9%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D.md)
- 5. [随机梯度下降 (SGD)](05-%E9%9A%8F%E6%9C%BA%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%20%28SGD%29.md)
- 6. [小批量梯度下降](06-%E5%B0%8F%E6%89%B9%E9%87%8F%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D.md)
- 7. [挑战：局部最小值与鞍点](07-%E6%8C%91%E6%88%98%EF%BC%9A%E5%B1%80%E9%83%A8%E6%9C%80%E5%B0%8F%E5%80%BC%E4%B8%8E%E9%9E%8D%E7%82%B9.md)
- 8. [动手实践：实现简单的梯度下降](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%9E%E7%8E%B0%E7%AE%80%E5%8D%95%E7%9A%84%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D.md)

章节测验：[在线测验](https://apxml.com/zh/courses/calculus-essentials-machine-learning/chapter-4-gradient-descent-algorithms/quiz)
