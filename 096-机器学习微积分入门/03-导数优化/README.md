# 第 3 章：导数优化

来源：[原章节](https://apxml.com/zh/courses/calculus-fundamentals-machine-learning/chapter-3-optimization-with-derivatives)

[返回课程目录](../README.md)

在了解了导数作为瞬时变化率之后，我们现在转向一个重要应用：使用它们来找出函数中的最优点。这个被称为优化的过程，是许多机器学习模型从数据中学习的一个重要途径。

在本章中，你将学习导数如何用于定位函数的最小值或最大值，这通常是通过找出导数为零 ($f'(x) = 0$) 的点来实现的。我们将考察机器学习中的成本函数（或损失函数），它们衡量模型表现如何，并理解为什么最小化这些函数是目标。你将了解到梯度下降，这是一种用于迭代最小化成本函数的基本算法。我们将解释导数如何提供梯度下降所需的“方向”，并通过可视化这个过程来帮助你建立直观理解。

## 小节

- 1. [寻找最大值和最小值点](01-%E5%AF%BB%E6%89%BE%E6%9C%80%E5%A4%A7%E5%80%BC%E5%92%8C%E6%9C%80%E5%B0%8F%E5%80%BC%E7%82%B9.md)
- 2. [优化：为什么最小化或最大化？](02-%E4%BC%98%E5%8C%96%EF%BC%9A%E4%B8%BA%E4%BB%80%E4%B9%88%E6%9C%80%E5%B0%8F%E5%8C%96%E6%88%96%E6%9C%80%E5%A4%A7%E5%8C%96%EF%BC%9F.md)
- 3. [机器学习中的代价函数](03-%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E4%B8%AD%E7%9A%84%E4%BB%A3%E4%BB%B7%E5%87%BD%E6%95%B0.md)
- 4. [目标：最小化成本函数](04-%E7%9B%AE%E6%A0%87%EF%BC%9A%E6%9C%80%E5%B0%8F%E5%8C%96%E6%88%90%E6%9C%AC%E5%87%BD%E6%95%B0.md)
- 5. [梯度下降简介](05-%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E7%AE%80%E4%BB%8B.md)
- 6. [导数如何指导梯度下降](06-%E5%AF%BC%E6%95%B0%E5%A6%82%E4%BD%95%E6%8C%87%E5%AF%BC%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D.md)
- 7. [梯度下降的可视化](07-%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E7%9A%84%E5%8F%AF%E8%A7%86%E5%8C%96.md)

章节测验：[在线测验](https://apxml.com/zh/courses/calculus-fundamentals-machine-learning/chapter-3-optimization-with-derivatives/quiz)
