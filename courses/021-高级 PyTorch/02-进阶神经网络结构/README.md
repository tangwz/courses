# 第 2 章：进阶神经网络结构

来源：[原章节](https://apxml.com/zh/courses/advanced-pytorch/chapter-2-advanced-network-architectures)

[返回课程目录](../README.md)

虽然像卷积神经网络（CNN）和循环神经网络（RNN）这样的基本网络设计能有效处理许多任务，但某些问题场景需要更专业的结构。本章将着重介绍如何使用PyTorch实现多种进阶神经网络模型。

您将学习重要的现代结构及其实现细节。我们将逐个组件地介绍Transformer模型的构建，包括注意力机制。我们还将使用图神经网络（GNN）处理图结构数据，并运用PyTorch Geometric等库。此外，本章会介绍用于生成任务的归一化流（Normalizing Flows）、用于连续深度建模的神经常微分方程（Neural ODEs），以及针对少样本场景的元学习方法。侧重于理解这些构成要素，并在代码中构建这些复杂的模型。

## 小节

- 1. [从组件构建Transformer模型](01-%E4%BB%8E%E7%BB%84%E4%BB%B6%E6%9E%84%E5%BB%BATransformer%E6%A8%A1%E5%9E%8B.md)
- 2. [高级注意力机制](02-%E9%AB%98%E7%BA%A7%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6.md)
- 3. [使用 PyTorch Geometric 的图神经网络](03-%E4%BD%BF%E7%94%A8%20PyTorch%20Geometric%20%E7%9A%84%E5%9B%BE%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C.md)
- 4. [用于生成建模的归一化流](04-%E7%94%A8%E4%BA%8E%E7%94%9F%E6%88%90%E5%BB%BA%E6%A8%A1%E7%9A%84%E5%BD%92%E4%B8%80%E5%8C%96%E6%B5%81.md)
- 5. [神经常微分方程](05-%E7%A5%9E%E7%BB%8F%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B.md)
- 6. [元学习算法](06-%E5%85%83%E5%AD%A6%E4%B9%A0%E7%AE%97%E6%B3%95.md)
- 7. [实践：实现自定义GNN层](07-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%9E%E7%8E%B0%E8%87%AA%E5%AE%9A%E4%B9%89GNN%E5%B1%82.md)
