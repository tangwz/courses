# 第 4 章：反向传播和高级优化

来源：[原章节](https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-4-backpropagation-advanced-optimization)

[返回课程目录](../README.md)

在上一章中，我们明确了训练目标：通过梯度下降调整网络权重，以最小化损失函数 $L$。然而，在多层网络中有效计算关于*所有*权重的梯度 $ \nabla L $ 需要一种特定的方法。

本章将介绍反向传播算法，它是计算这些梯度的标准方法。我们将审视反向传播在微积分链式法则中的原理，并通过计算图来描绘其过程。随后，我们将超越基本的梯度下降，学习更精密的优化算法。这些算法包括 Momentum、RMSprop 和 Adam，它们有助于加速收敛，并能更有效地应对复杂的损失曲面。完成本章后，你将理解梯度是如何计算并反向传播通过网络的，以及高级优化器如何改进训练过程。

## 小节

- 1. [计算梯度：链式法则](01-%E8%AE%A1%E7%AE%97%E6%A2%AF%E5%BA%A6%EF%BC%9A%E9%93%BE%E5%BC%8F%E6%B3%95%E5%88%99.md)
- 2. [计算图](02-%E8%AE%A1%E7%AE%97%E5%9B%BE.md)
- 3. [反向传播算法详解](03-%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD%E7%AE%97%E6%B3%95%E8%AF%A6%E8%A7%A3.md)
- 4. [前向传播与反向传播](04-%E5%89%8D%E5%90%91%E4%BC%A0%E6%92%AD%E4%B8%8E%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD.md)
- 5. [带有动量的梯度下降](05-%E5%B8%A6%E6%9C%89%E5%8A%A8%E9%87%8F%E7%9A%84%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D.md)
- 6. [RMSprop 优化器](06-RMSprop%20%E4%BC%98%E5%8C%96%E5%99%A8.md)
- 7. [Adam 优化器](07-Adam%20%E4%BC%98%E5%8C%96%E5%99%A8.md)
- 8. [选择优化算法](08-%E9%80%89%E6%8B%A9%E4%BC%98%E5%8C%96%E7%AE%97%E6%B3%95.md)
- 9. [动手实践：反向传播逐步解析](09-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD%E9%80%90%E6%AD%A5%E8%A7%A3%E6%9E%90.md)

章节测验：[在线测验](https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-4-backpropagation-advanced-optimization/quiz)
