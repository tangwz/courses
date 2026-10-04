# 第 3 章：构建神经网络架构

来源：[原章节](https://apxml.com/zh/courses/julia-deep-learning/chapter-3-constructing-nn-architectures)

[返回课程目录](../README.md)

在掌握了前一章Flux.jl的核心组成部分后，我们将着重介绍这些元素在构建多种神经网络架构中的具体运用。本章将指导您从零开始搭建几种常用的神经网络类型。

您将学到：

*   使用Julia工具准备和预处理专门用于深度学习模型的数据，包括利用`MLUtils.jl`高效处理数据，以进行批处理和迭代。
*   构建多层感知机（MLPs），它们是全连接网络的基本形式。
*   实现卷积神经网络（CNNs），它们对于图像处理等涉及网格状数据的任务必不可少。
*   开发循环神经网络（RNNs），包括长短期记忆（LSTM）单元，它们适用于文本或时间序列等序列数据。
*   运用嵌入层在网络中高效表示类别或文本数据。
*   通过创建自定义神经网络层来扩展Flux.jl的功能，以支持特定操作。
*   使用序列化技术（例如使用`BSON.jl`）保存您训练好的模型，并加载它们以供后续推断或继续训练。

本章最后是一个实践练习，您将在此应用这些技术来构建和配置一个CNN以解决图像分类问题，从而巩固您对这些架构组成部分如何组合在一起的理解。

## 小节

- 1. [Julia 中的数据准备与预处理](01-Julia%20%E4%B8%AD%E7%9A%84%E6%95%B0%E6%8D%AE%E5%87%86%E5%A4%87%E4%B8%8E%E9%A2%84%E5%A4%84%E7%90%86.md)
- 2. [处理数据集：使用 MLUtils.jl 的迭代器和加载器](02-%E5%A4%84%E7%90%86%E6%95%B0%E6%8D%AE%E9%9B%86%EF%BC%9A%E4%BD%BF%E7%94%A8%20MLUtils.jl%20%E7%9A%84%E8%BF%AD%E4%BB%A3%E5%99%A8%E5%92%8C%E5%8A%A0%E8%BD%BD%E5%99%A8.md)
- 3. [构建多层感知器（MLP）](03-%E6%9E%84%E5%BB%BA%E5%A4%9A%E5%B1%82%E6%84%9F%E7%9F%A5%E5%99%A8%EF%BC%88MLP%EF%BC%89.md)
- 4. [卷积神经网络（CNN）与 Flux](04-%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%EF%BC%88CNN%EF%BC%89%E4%B8%8E%20Flux.md)
- 5. [Flux 中的循环神经网络（RNN）与长短期记忆网络（LSTM）](05-Flux%20%E4%B8%AD%E7%9A%84%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%EF%BC%88RNN%EF%BC%89%E4%B8%8E%E9%95%BF%E7%9F%AD%E6%9C%9F%E8%AE%B0%E5%BF%86%E7%BD%91%E7%BB%9C%EF%BC%88LSTM%EF%BC%89.md)
- 6. [处理序列数据的嵌入](06-%E5%A4%84%E7%90%86%E5%BA%8F%E5%88%97%E6%95%B0%E6%8D%AE%E7%9A%84%E5%B5%8C%E5%85%A5.md)
- 7. [在 Flux 中创建自定义层](07-%E5%9C%A8%20Flux%20%E4%B8%AD%E5%88%9B%E5%BB%BA%E8%87%AA%E5%AE%9A%E4%B9%89%E5%B1%82.md)
- 8. [模型序列化：保存和加载Flux模型](08-%E6%A8%A1%E5%9E%8B%E5%BA%8F%E5%88%97%E5%8C%96%EF%BC%9A%E4%BF%9D%E5%AD%98%E5%92%8C%E5%8A%A0%E8%BD%BDFlux%E6%A8%A1%E5%9E%8B.md)
- 9. [实践：为图像分类实现一个卷积神经网络](09-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%B8%BA%E5%9B%BE%E5%83%8F%E5%88%86%E7%B1%BB%E5%AE%9E%E7%8E%B0%E4%B8%80%E4%B8%AA%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C.md)
