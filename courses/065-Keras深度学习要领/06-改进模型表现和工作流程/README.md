# 第 6 章：改进模型表现和工作流程

来源：[原章节](https://apxml.com/zh/courses/deep-learning-fundamentals-keras/chapter-6-improving-model-performance-workflow)

[返回课程目录](../README.md)

构建和训练神经网络通常只是一个开始。确保模型在新数据上表现良好，并改进开发工作流程，是后续的必要步骤。本章侧重于诊断并处理常见的训练问题、提高模型泛化能力，以及更有效地管理训练过程的方法。

您将学会识别过拟合和欠拟合，这是模型开发中常遇到的两个问题。我们将介绍解决这些问题的方法，包括$L_1$、$L_2$和Dropout等正则化技术，以及数据增强策略。我们还会讨论使用Keras回调函数改进工作流程，例如`ModelCheckpoint`（用于在训练期间保存最佳模型）和`EarlyStopping`（以避免不必要的计算）。此外，您将学习如何正确保存和加载训练好的模型以供后续使用，并熟悉TensorBoard，用于查看训练进度和模型结构。最后，我们将提及超参数调整背后的原理。

## 小节

- 1. [理解过拟合和欠拟合](01-%E7%90%86%E8%A7%A3%E8%BF%87%E6%8B%9F%E5%90%88%E5%92%8C%E6%AC%A0%E6%8B%9F%E5%90%88.md)
- 2. [正则化方法：L1、L2、Dropout](02-%E6%AD%A3%E5%88%99%E5%8C%96%E6%96%B9%E6%B3%95%EF%BC%9AL1%E3%80%81L2%E3%80%81Dropout.md)
- 3. [数据增强](03-%E6%95%B0%E6%8D%AE%E5%A2%9E%E5%BC%BA.md)
- 4. [使用回调](04-%E4%BD%BF%E7%94%A8%E5%9B%9E%E8%B0%83.md)
- 5. [保存和加载模型](05-%E4%BF%9D%E5%AD%98%E5%92%8C%E5%8A%A0%E8%BD%BD%E6%A8%A1%E5%9E%8B.md)
- 6. [TensorBoard 简介](06-TensorBoard%20%E7%AE%80%E4%BB%8B.md)
- 7. [超参数调优概念](07-%E8%B6%85%E5%8F%82%E6%95%B0%E8%B0%83%E4%BC%98%E6%A6%82%E5%BF%B5.md)
- 8. [动手实践：应用改进技术](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%BA%94%E7%94%A8%E6%94%B9%E8%BF%9B%E6%8A%80%E6%9C%AF.md)

章节测验：[在线测验](https://apxml.com/zh/courses/deep-learning-fundamentals-keras/chapter-6-improving-model-performance-workflow/quiz)
