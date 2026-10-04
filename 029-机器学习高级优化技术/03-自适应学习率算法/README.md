# 第 3 章：自适应学习率算法

来源：[原章节](https://apxml.com/zh/courses/optimization-techniques-ml/chapter-3-adaptive-learning-rate-algorithms)

[返回课程目录](../README.md)

标准梯度下降法常使用固定的学习率，$\eta$。找到一个好的$\eta$需要仔细调整；过小的学习率会减慢收敛，而过大的学习率则可能阻碍收敛。本章介绍的算法通过在训练期间自动调整学习率来解决这个问题。

我们将考察几种常用的自适应学习率方法。您将学习AdaGrad的原理，它根据历史梯度调整学习率，以及RMSprop，它改进了AdaGrad的方法以避免过度衰减。接着，我们将学习Adam（自适应矩估计），一个结合了自适应学习率和动量估计的优化器。我们还将分析Adamax、Nadam和AMSGrad等变体，了解它们的具体改进之处。本章还会介绍如何将自适应方法与学习率调度结合使用。

学完本章，您将理解这些自适应技术的理论和实际应用，从而能够选择并运用它们来更高效地训练模型。

## 小节

- 1. [固定学习率的局限性](01-%E5%9B%BA%E5%AE%9A%E5%AD%A6%E4%B9%A0%E7%8E%87%E7%9A%84%E5%B1%80%E9%99%90%E6%80%A7.md)
- 2. [AdaGrad：根据过往梯度调整学习率](02-AdaGrad%EF%BC%9A%E6%A0%B9%E6%8D%AE%E8%BF%87%E5%BE%80%E6%A2%AF%E5%BA%A6%E8%B0%83%E6%95%B4%E5%AD%A6%E4%B9%A0%E7%8E%87.md)
- 3. [RMSprop：处理AdaGrad学习率递减的问题](03-RMSprop%EF%BC%9A%E5%A4%84%E7%90%86AdaGrad%E5%AD%A6%E4%B9%A0%E7%8E%87%E9%80%92%E5%87%8F%E7%9A%84%E9%97%AE%E9%A2%98.md)
- 4. [Adam：结合动量与RMSprop](04-Adam%EF%BC%9A%E7%BB%93%E5%90%88%E5%8A%A8%E9%87%8F%E4%B8%8ERMSprop.md)
- 5. [Adamax 和 Nadam 变体](05-Adamax%20%E5%92%8C%20Nadam%20%E5%8F%98%E4%BD%93.md)
- 6. [AMSGrad：提升 Adam 的收敛性](06-AMSGrad%EF%BC%9A%E6%8F%90%E5%8D%87%20Adam%20%E7%9A%84%E6%94%B6%E6%95%9B%E6%80%A7.md)
- 7. [了解学习率调整策略](07-%E4%BA%86%E8%A7%A3%E5%AD%A6%E4%B9%A0%E7%8E%87%E8%B0%83%E6%95%B4%E7%AD%96%E7%95%A5.md)
- 8. [实践操作：比较自适应优化器](08-%E5%AE%9E%E8%B7%B5%E6%93%8D%E4%BD%9C%EF%BC%9A%E6%AF%94%E8%BE%83%E8%87%AA%E9%80%82%E5%BA%94%E4%BC%98%E5%8C%96%E5%99%A8.md)
