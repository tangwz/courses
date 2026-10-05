# 第 6 章：自适应优化算法

来源：[原章节](https://apxml.com/zh/courses/deep-learning-regularization-optimization/chapter-6-adaptive-optimizers)

[返回课程目录](../README.md)

随机梯度下降（SGD）及其动量变体等优化器相比基本梯度下降有了显著改进，但它们通常依赖于一个学习率，这个学习率对所有参数一视同仁，或按照预设方案衰减。然而，深度网络中不同的参数可能需要不同的学习率调整，根据其梯度的历史信息。

本章介绍自适应优化算法，旨在自动为每个参数独立调整学习率。我们将考察几种常用方法：

*   **AdaGrad：** 根据每个参数的梯度平方和的历史数据，来调整学习率。
*   **RMSprop：** 解决了AdaGrad学习率下降过快的问题，通过使用梯度平方的移动平均。
*   **Adam（自适应矩估计）：** 结合了动量和RMSprop的思路，存储了梯度及其平方的移动平均值。

您将学习自适应方法背后的动机，AdaGrad、RMSprop和Adam的具体更新机制，包括它们的数学基础和偏差修正技术。我们将讨论它们的优点、缺点，以及在标准深度学习框架中的实现细节。最后，我们将提供实用建议，用于为您的模型选择合适的优化器。

## 小节

- 1. [自适应学习率的必要性](01-%E8%87%AA%E9%80%82%E5%BA%94%E5%AD%A6%E4%B9%A0%E7%8E%87%E7%9A%84%E5%BF%85%E8%A6%81%E6%80%A7.md)
- 2. [AdaGrad：调整每个参数的学习率](02-AdaGrad%EF%BC%9A%E8%B0%83%E6%95%B4%E6%AF%8F%E4%B8%AA%E5%8F%82%E6%95%B0%E7%9A%84%E5%AD%A6%E4%B9%A0%E7%8E%87.md)
- 3. [AdaGrad 的局限性：学习率衰减](03-AdaGrad%20%E7%9A%84%E5%B1%80%E9%99%90%E6%80%A7%EF%BC%9A%E5%AD%A6%E4%B9%A0%E7%8E%87%E8%A1%B0%E5%87%8F.md)
- 4. [RMSprop：处理AdaGrad的局限性](04-RMSprop%EF%BC%9A%E5%A4%84%E7%90%86AdaGrad%E7%9A%84%E5%B1%80%E9%99%90%E6%80%A7.md)
- 5. [Adam：自适应矩估计](05-Adam%EF%BC%9A%E8%87%AA%E9%80%82%E5%BA%94%E7%9F%A9%E4%BC%B0%E8%AE%A1.md)
- 6. [Adam算法细致分析](06-Adam%E7%AE%97%E6%B3%95%E7%BB%86%E8%87%B4%E5%88%86%E6%9E%90.md)
- 7. [Adamax 和 Nadam 变体（简要概述）](07-Adamax%20%E5%92%8C%20Nadam%20%E5%8F%98%E4%BD%93%EF%BC%88%E7%AE%80%E8%A6%81%E6%A6%82%E8%BF%B0%EF%BC%89.md)
- 8. [优化器选择指南](08-%E4%BC%98%E5%8C%96%E5%99%A8%E9%80%89%E6%8B%A9%E6%8C%87%E5%8D%97.md)
- 9. [实现 Adam 和 RMSprop](09-%E5%AE%9E%E7%8E%B0%20Adam%20%E5%92%8C%20RMSprop.md)
- 10. [动手实践：优化器比较实验](10-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%BC%98%E5%8C%96%E5%99%A8%E6%AF%94%E8%BE%83%E5%AE%9E%E9%AA%8C.md)

章节测验：[在线测验](https://apxml.com/zh/courses/deep-learning-regularization-optimization/chapter-6-adaptive-optimizers/quiz)
