# 第 3 章：自动编码器如何学习

来源：[原章节](https://apxml.com/zh/courses/introduction-autoencoders-feature-learning/chapter-3-how-autoencoders-learn)

[返回课程目录](../README.md)

前面章节介绍了自动编码器及其主要组成部分。现在，我们审视这些网络如何从数据中学习。主要目标是最小化重建误差，即原始输入与自动编码器输出之间的差异。

本章将说明：
*   **训练目的**: 自动编码器如何致力于精确重建输入。
*   **损失函数**: 用于衡量重建误差的指标，例如均方误差 ($MSE$) 和二元交叉熵 ($BCE$)。
*   **优化**: 像梯度下降这样的基本过程，用于调整自动编码器内部设置以提高性能。
*   **数据流**: 理解前向传播（从输入到输出）和反向传播（误差调整）的概览。
*   **训练周期**: 诸如周期（epochs）和批次（batches）等组织学习过程的要点。

我们还将简要介绍过拟合和欠拟合。到本章结束时，您将了解准备自动编码器进行训练所涉及的步骤。

## 小节

- 1. [训练目标：减少重建误差](01-%E8%AE%AD%E7%BB%83%E7%9B%AE%E6%A0%87%EF%BC%9A%E5%87%8F%E5%B0%91%E9%87%8D%E5%BB%BA%E8%AF%AF%E5%B7%AE.md)
- 2. [自编码器的损失函数 (MSE, BCE)](02-%E8%87%AA%E7%BC%96%E7%A0%81%E5%99%A8%E7%9A%84%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%20%28MSE%2C%20BCE%29.md)
- 3. [学习过程：优化基础](03-%E5%AD%A6%E4%B9%A0%E8%BF%87%E7%A8%8B%EF%BC%9A%E4%BC%98%E5%8C%96%E5%9F%BA%E7%A1%80.md)
- 4. [数据流：前向传播说明](04-%E6%95%B0%E6%8D%AE%E6%B5%81%EF%BC%9A%E5%89%8D%E5%90%91%E4%BC%A0%E6%92%AD%E8%AF%B4%E6%98%8E.md)
- 5. [从错误中学习：反向传播（宏观视角）](05-%E4%BB%8E%E9%94%99%E8%AF%AF%E4%B8%AD%E5%AD%A6%E4%B9%A0%EF%BC%9A%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD%EF%BC%88%E5%AE%8F%E8%A7%82%E8%A7%86%E8%A7%92%EF%BC%89.md)
- 6. [训练周期：迭代次数与批次](06-%E8%AE%AD%E7%BB%83%E5%91%A8%E6%9C%9F%EF%BC%9A%E8%BF%AD%E4%BB%A3%E6%AC%A1%E6%95%B0%E4%B8%8E%E6%89%B9%E6%AC%A1.md)
- 7. [过拟合与欠拟合初识](07-%E8%BF%87%E6%8B%9F%E5%90%88%E4%B8%8E%E6%AC%A0%E6%8B%9F%E5%90%88%E5%88%9D%E8%AF%86.md)
- 8. [构建自编码器的准备工作](08-%E6%9E%84%E5%BB%BA%E8%87%AA%E7%BC%96%E7%A0%81%E5%99%A8%E7%9A%84%E5%87%86%E5%A4%87%E5%B7%A5%E4%BD%9C.md)

章节测验：[在线测验](https://apxml.com/zh/courses/introduction-autoencoders-feature-learning/chapter-3-how-autoencoders-learn/quiz)
