# 第 5 章：构建一个基础自动编码器

来源：[原章节](https://apxml.com/zh/courses/introduction-autoencoders-feature-learning/chapter-5-building-basic-autoencoder)

[返回课程目录](../README.md)

之前的章节详细介绍了自动编码器的核心原理，涵盖了其架构和学习机制。本章将从这些原理转向实际应用。在此，我们将提供一份分步指南，指导您构建您的第一个基础自动编码器。

您将学会：
*   搭建适合深度学习任务的Python环境。
*   使用PyTorch和Keras来定义和构建神经网络模型。
*   加载、理解和预处理一个常见数据集（例如MNIST），用于训练自动编码器。
*   构建简单自动编码器的编码器和解码器层。
*   配置训练过程，包括选择优化器和损失函数，例如均方误差（MSE）。MSE衡量原始输入$x$和重建输出$\hat{x}$之间的平均平方差，通常表示为 $$L(x, \hat{x}) = \frac{1}{N} \sum_{i=1}^{N} (x_i - \hat{x}_i)^2$$
*   训练自动编码器，然后评估其重建数据的能力。
*   可视化原始输入及其重建版本以评估性能。

通过学习本章，您将获得实现自动编码器和观察其数据重建能力的直接经验。这项实际练习将巩固您对这些网络如何运作的理解。

## 小节

- 1. [深度学习的 Python 环境配置](01-%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%9A%84%20Python%20%E7%8E%AF%E5%A2%83%E9%85%8D%E7%BD%AE.md)
- 2. [PyTorch和Keras入门](02-PyTorch%E5%92%8CKeras%E5%85%A5%E9%97%A8.md)
- 3. [加载并理解一个基础数据集](03-%E5%8A%A0%E8%BD%BD%E5%B9%B6%E7%90%86%E8%A7%A3%E4%B8%80%E4%B8%AA%E5%9F%BA%E7%A1%80%E6%95%B0%E6%8D%AE%E9%9B%86.md)
- 4. [自动编码器的数据预处理](04-%E8%87%AA%E5%8A%A8%E7%BC%96%E7%A0%81%E5%99%A8%E7%9A%84%E6%95%B0%E6%8D%AE%E9%A2%84%E5%A4%84%E7%90%86.md)
- 5. [构建一个简单的自编码器模型](05-%E6%9E%84%E5%BB%BA%E4%B8%80%E4%B8%AA%E7%AE%80%E5%8D%95%E7%9A%84%E8%87%AA%E7%BC%96%E7%A0%81%E5%99%A8%E6%A8%A1%E5%9E%8B.md)
- 6. [配置模型以进行训练](06-%E9%85%8D%E7%BD%AE%E6%A8%A1%E5%9E%8B%E4%BB%A5%E8%BF%9B%E8%A1%8C%E8%AE%AD%E7%BB%83.md)
- 7. [执行训练过程](07-%E6%89%A7%E8%A1%8C%E8%AE%AD%E7%BB%83%E8%BF%87%E7%A8%8B.md)
- 8. [评估重建质量](08-%E8%AF%84%E4%BC%B0%E9%87%8D%E5%BB%BA%E8%B4%A8%E9%87%8F.md)
- 9. [可视化重建输出：动手实践](09-%E5%8F%AF%E8%A7%86%E5%8C%96%E9%87%8D%E5%BB%BA%E8%BE%93%E5%87%BA%EF%BC%9A%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5.md)
- 10. [检查编码数据：实践](10-%E6%A3%80%E6%9F%A5%E7%BC%96%E7%A0%81%E6%95%B0%E6%8D%AE%EF%BC%9A%E5%AE%9E%E8%B7%B5.md)

章节测验：[在线测验](https://apxml.com/zh/courses/introduction-autoencoders-feature-learning/chapter-5-building-basic-autoencoder/quiz)
