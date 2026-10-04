# 第 3 章：数据加载与预处理：从 tf.data 到 torch.utils.data

来源：[原章节](https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-3-pytorch-data-loading-for-tf-users)

[返回课程目录](../README.md)

有效的数据管理是任何机器学习项目的重要组成部分。作为一名 TensorFlow 开发者，您可能对使用 `tf.data` 构建输入管道非常熟悉。本章侧重介绍 PyTorch 处理数据加载和预处理的方式，帮助您调整现有技能。

我们将学习 PyTorch 的 `torch.utils.data` 模块，它为此目的提供了主要工具。您将学习如何使用 `Dataset` 类定义自定义数据源，并使用 `DataLoader` 有效地批量处理您的数据。我们还将介绍 `torchvision.transforms`，用于应用常见的数据预处理和增强技术。在本章结束时，您将能够为您的 PyTorch 模型构建灵活且高性能的数据管道，并结合您在 TensorFlow 的经验进行比较。

## 小节

- 1. [数据结构：tf.data.Dataset 与 torch.utils.data.Dataset](01-%E6%95%B0%E6%8D%AE%E7%BB%93%E6%9E%84%EF%BC%9Atf.data.Dataset%20%E4%B8%8E%20torch.utils.data.Dataset.md)
- 2. [批处理与迭代：TensorFlow DataLoaders 与 PyTorch DataLoaders](02-%E6%89%B9%E5%A4%84%E7%90%86%E4%B8%8E%E8%BF%AD%E4%BB%A3%EF%BC%9ATensorFlow%20DataLoaders%20%E4%B8%8E%20PyTorch%20DataLoaders.md)
- 3. [数据增强：TensorFlow 方法与 torchvision.transforms](03-%E6%95%B0%E6%8D%AE%E5%A2%9E%E5%BC%BA%EF%BC%9ATensorFlow%20%E6%96%B9%E6%B3%95%E4%B8%8E%20torchvision.transforms.md)
- 4. [在 PyTorch 中实现自定义数据集](04-%E5%9C%A8%20PyTorch%20%E4%B8%AD%E5%AE%9E%E7%8E%B0%E8%87%AA%E5%AE%9A%E4%B9%89%E6%95%B0%E6%8D%AE%E9%9B%86.md)
- 5. [使用 PyTorch 变换进行数据预处理](05-%E4%BD%BF%E7%94%A8%20PyTorch%20%E5%8F%98%E6%8D%A2%E8%BF%9B%E8%A1%8C%E6%95%B0%E6%8D%AE%E9%A2%84%E5%A4%84%E7%90%86.md)
- 6. [在 PyTorch 中构建高效数据管道](06-%E5%9C%A8%20PyTorch%20%E4%B8%AD%E6%9E%84%E5%BB%BA%E9%AB%98%E6%95%88%E6%95%B0%E6%8D%AE%E7%AE%A1%E9%81%93.md)
- 7. [动手实践：创建自定义数据集和数据加载器](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%88%9B%E5%BB%BA%E8%87%AA%E5%AE%9A%E4%B9%89%E6%95%B0%E6%8D%AE%E9%9B%86%E5%92%8C%E6%95%B0%E6%8D%AE%E5%8A%A0%E8%BD%BD%E5%99%A8.md)

章节测验：[在线测验](https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-3-pytorch-data-loading-for-tf-users/quiz)
