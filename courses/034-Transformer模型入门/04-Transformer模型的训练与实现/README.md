# 第 4 章：Transformer模型的训练与实现

来源：[原章节](https://apxml.com/zh/courses/introduction-to-transformer-models/chapter-4-training-implementing-transformers)

[返回课程目录](../README.md)

在前面章节对Transformer模型的架构，包括注意力机制和编解码器结构进行了详细说明后，本章我们将把这些原理付诸实践。本章将讨论训练和实现Transformer模型的必要步骤。

您将了解如何为Transformer准备数据，包括常见的分词技术，如字节对编码（BPE），以及如何创建格式正确的输入批次，包括填充和注意力掩码。之后，我们将考察训练过程本身，讨论合适的损失函数（如交叉熵）、Transformer模型常用的优化算法（如Adam）、学习率调度技术以及像Dropout这样的正则化方法。最后，我们将概述如何将前面讨论的组件组装成一个基本可运行的模型，并简要介绍提供预训练Transformer实现方式的库的运用。

## 小节

- 1. [数据准备：分词](01-%E6%95%B0%E6%8D%AE%E5%87%86%E5%A4%87%EF%BC%9A%E5%88%86%E8%AF%8D.md)
- 2. [构建输入批次](02-%E6%9E%84%E5%BB%BA%E8%BE%93%E5%85%A5%E6%89%B9%E6%AC%A1.md)
- 3. [序列任务的损失函数](03-%E5%BA%8F%E5%88%97%E4%BB%BB%E5%8A%A1%E7%9A%84%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.md)
- 4. [优化策略](04-%E4%BC%98%E5%8C%96%E7%AD%96%E7%95%A5.md)
- 5. [正则化方法](05-%E6%AD%A3%E5%88%99%E5%8C%96%E6%96%B9%E6%B3%95.md)
- 6. [基本实现概述](06-%E5%9F%BA%E6%9C%AC%E5%AE%9E%E7%8E%B0%E6%A6%82%E8%BF%B0.md)
- 7. [使用预训练模型库（简述）](07-%E4%BD%BF%E7%94%A8%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B%E5%BA%93%EF%BC%88%E7%AE%80%E8%BF%B0%EF%BC%89.md)
- 8. [实践：组装一个基本Transformer](08-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E7%BB%84%E8%A3%85%E4%B8%80%E4%B8%AA%E5%9F%BA%E6%9C%ACTransformer.md)

章节测验：[在线测验](https://apxml.com/zh/courses/introduction-to-transformer-models/chapter-4-training-implementing-transformers/quiz)
