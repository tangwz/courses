# 第 2 章：为神经网络准备数据

来源：[原章节](https://apxml.com/zh/courses/introduction-to-neural-networks/chapter-2-data-preparation-neural-networks)

[返回课程目录](../README.md)

神经网络处理数字数据，它们的有效学习能力受输入数据的格式和特性影响很大。原始数据集很少能直接使用。本章介绍准备和组织数据以进行最佳神经网络训练的必要步骤。

您将了解如何表示不同类型的输入特征，使用归一化（$x' = \frac{x - min(x)}{max(x) - min(x)}$）和标准化（$x' = \frac{x - \mu}{\sigma}$）等方法对数值数据进行缩放的重要性，以及将分类特征转换为适合网络使用的数值格式（例如独热编码）的方法。此外，我们还将介绍如何将数据分成批次以在训练期间高效处理，以及将数据集划分为训练集、验证集和测试集以进行模型开发和可靠评估的常用做法。本章结束时，您将明白如何将原始数据转换为结构化格式，以促进神经网络有效学习。

## 小节

- 1. [理解输入数据表示](01-%E7%90%86%E8%A7%A3%E8%BE%93%E5%85%A5%E6%95%B0%E6%8D%AE%E8%A1%A8%E7%A4%BA.md)
- 2. [特征缩放：归一化与标准化](02-%E7%89%B9%E5%BE%81%E7%BC%A9%E6%94%BE%EF%BC%9A%E5%BD%92%E4%B8%80%E5%8C%96%E4%B8%8E%E6%A0%87%E5%87%86%E5%8C%96.md)
- 3. [处理分类数据：编码技术](03-%E5%A4%84%E7%90%86%E5%88%86%E7%B1%BB%E6%95%B0%E6%8D%AE%EF%BC%9A%E7%BC%96%E7%A0%81%E6%8A%80%E6%9C%AF.md)
- 4. [构建训练数据批次](04-%E6%9E%84%E5%BB%BA%E8%AE%AD%E7%BB%83%E6%95%B0%E6%8D%AE%E6%89%B9%E6%AC%A1.md)
- 5. [数据划分：训练集、验证集和测试集](05-%E6%95%B0%E6%8D%AE%E5%88%92%E5%88%86%EF%BC%9A%E8%AE%AD%E7%BB%83%E9%9B%86%E3%80%81%E9%AA%8C%E8%AF%81%E9%9B%86%E5%92%8C%E6%B5%8B%E8%AF%95%E9%9B%86.md)
- 6. [实践操作：预处理样本数据](06-%E5%AE%9E%E8%B7%B5%E6%93%8D%E4%BD%9C%EF%BC%9A%E9%A2%84%E5%A4%84%E7%90%86%E6%A0%B7%E6%9C%AC%E6%95%B0%E6%8D%AE.md)

章节测验：[在线测验](https://apxml.com/zh/courses/introduction-to-neural-networks/chapter-2-data-preparation-neural-networks/quiz)
