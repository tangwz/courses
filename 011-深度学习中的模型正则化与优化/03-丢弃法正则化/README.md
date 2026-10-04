# 第 3 章：丢弃法正则化

来源：[原章节](https://apxml.com/zh/courses/deep-learning-regularization-optimization/chapter-3-dropout-regularization)

[返回课程目录](../README.md)

在我们了解了 L1 和 L2 等惩罚大权重值的权重正则化方法之后，接下来我们介绍一种截然不同的防止过拟合的方法：丢弃法。这种方法在每次训练更新中，通过随机将一部分神经元输出设为零来发挥作用。这有助于避免神经元之间出现复杂的相互依赖，即神经元过度依赖特定其他神经元的情况。

本章将讲解丢弃法的工作原理，包括它在训练时如何运作，以及在推理时（测试阶段）激活值如何调整。我们将介绍常见的“倒置丢弃法”实现，讨论作为可调超参数的丢弃率 $p$，并提及在卷积网络和循环网络中应用丢弃法的考量。最后，你将通过实际例子了解如何使用标准深度学习框架将丢弃层整合到模型中。

## 小节

- 1. [Dropout介绍：避免协同适应](01-Dropout%E4%BB%8B%E7%BB%8D%EF%BC%9A%E9%81%BF%E5%85%8D%E5%8D%8F%E5%90%8C%E9%80%82%E5%BA%94.md)
- 2. [Dropout 在训练时的工作原理](02-Dropout%20%E5%9C%A8%E8%AE%AD%E7%BB%83%E6%97%B6%E7%9A%84%E5%B7%A5%E4%BD%9C%E5%8E%9F%E7%90%86.md)
- 3. [在测试时调整激活值](03-%E5%9C%A8%E6%B5%8B%E8%AF%95%E6%97%B6%E8%B0%83%E6%95%B4%E6%BF%80%E6%B4%BB%E5%80%BC.md)
- 4. [反转Dropout实现](04-%E5%8F%8D%E8%BD%ACDropout%E5%AE%9E%E7%8E%B0.md)
- 5. [Dropout 比率作为超参数](05-Dropout%20%E6%AF%94%E7%8E%87%E4%BD%9C%E4%B8%BA%E8%B6%85%E5%8F%82%E6%95%B0.md)
- 6. [卷积层和循环层的使用考量](06-%E5%8D%B7%E7%A7%AF%E5%B1%82%E5%92%8C%E5%BE%AA%E7%8E%AF%E5%B1%82%E7%9A%84%E4%BD%BF%E7%94%A8%E8%80%83%E9%87%8F.md)
- 7. [在实践中应用Dropout](07-%E5%9C%A8%E5%AE%9E%E8%B7%B5%E4%B8%AD%E5%BA%94%E7%94%A8Dropout.md)
- 8. [动手实践：添加 Dropout 层](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%B7%BB%E5%8A%A0%20Dropout%20%E5%B1%82.md)

章节测验：[在线测验](https://apxml.com/zh/courses/deep-learning-regularization-optimization/chapter-3-dropout-regularization/quiz)
