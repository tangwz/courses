# 第 2 章：搭建神经网络：从 Keras 到 torch.nn

来源：[原章节](https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-2-pytorch-nn-module-for-keras-users)

[返回课程目录](../README.md)

如果你习惯使用 TensorFlow 的 Keras API 构建神经网络，本章将帮助你把这些技能应用到 PyTorch 的 `torch.nn` 模块。本章侧重于 PyTorch 中神经网络结构的搭建、配置和管理，并会持续与 Keras 的方法进行比较。

你将学习到：

*   将 Keras `Layer` 对象与其在 `torch.nn.Module`（PyTorch 中所有神经网络模块的基本类别）内对应的部分联系起来。
*   区分 Keras 用于模型定义的 Sequential 和 Functional API，以及 PyTorch 灵活的 `nn.Module` 子类化方法。
*   实现并理解常见的层类型，例如密集（线性）、卷积和循环层，同时注意它们在 PyTorch 和 TensorFlow 之间的区别与相似之处。
*   识别并使用标准激活函数，这些函数可通过 `torch.nn.functional` 或特定的 `nn.Module` 类别获取。
*   了解各种与 PyTorch 模型相关的权重初始化技术，并将其与常见的 TensorFlow 做法进行比较。
*   检查、访问和修改 PyTorch 模型中的参数和子模块，这对于开发和调试非常重要。

通过这些比较性说明和例子，你将获得在 PyTorch 框架中有效应用 Keras 模型构建知识的实际能力，从而理解通用之处以及 PyTorch 在网络设计方面的特定特性。

## 小节

- 1. [定义网络组件：Keras 层与 torch.nn.Module](01-%E5%AE%9A%E4%B9%89%E7%BD%91%E7%BB%9C%E7%BB%84%E4%BB%B6%EF%BC%9AKeras%20%E5%B1%82%E4%B8%8E%20torch.nn.Module.md)
- 2. [模型架构：Keras API 与 PyTorch 的 nn.Module](02-%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84%EF%BC%9AKeras%20API%20%E4%B8%8E%20PyTorch%20%E7%9A%84%20nn.Module.md)
- 3. [常见层类型：对比实现](03-%E5%B8%B8%E8%A7%81%E5%B1%82%E7%B1%BB%E5%9E%8B%EF%BC%9A%E5%AF%B9%E6%AF%94%E5%AE%9E%E7%8E%B0.md)
- 4. [激活函数：比较与分析](04-%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0%EF%BC%9A%E6%AF%94%E8%BE%83%E4%B8%8E%E5%88%86%E6%9E%90.md)
- 5. [PyTorch中的权重初始化方法](05-PyTorch%E4%B8%AD%E7%9A%84%E6%9D%83%E9%87%8D%E5%88%9D%E5%A7%8B%E5%8C%96%E6%96%B9%E6%B3%95.md)
- 6. [访问和修改模型参数及层](06-%E8%AE%BF%E9%97%AE%E5%92%8C%E4%BF%AE%E6%94%B9%E6%A8%A1%E5%9E%8B%E5%8F%82%E6%95%B0%E5%8F%8A%E5%B1%82.md)
- 7. [动手实践：构建等效模型](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9E%84%E5%BB%BA%E7%AD%89%E6%95%88%E6%A8%A1%E5%9E%8B.md)

章节测验：[在线测验](https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-2-pytorch-nn-module-for-keras-users/quiz)
