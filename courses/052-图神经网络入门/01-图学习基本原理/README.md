# 第 1 章：图学习基本原理

来源：[原章节](https://apxml.com/zh/courses/introduction-to-graph-neural-networks/chapter-1-foundations-graph-based-learning)

[返回课程目录](../README.md)

在构建图神经网络之前，我们需要先了解其处理的数据对象：图。本章讲解在机器学习环境下处理图结构数据的基本要点。

我们先从定义图的构成开始，并分析为什么 CNN 和 RNN 等传统神经网络不适合处理这类数据。接着，我们将介绍图上的主要机器学习任务，例如节点分类、链路预测和图分类。

本章的大部分篇幅用于说明如何为了计算而表示图。你将学习如何使用邻接矩阵 ($A$) 和节点特征矩阵 ($X$) 等标准格式来编码图的结构和属性。最后，我们将通过 NetworkX 库的简要介绍将这些想法付诸实践，使用该库加载并查看图数据集。

## 小节

- 1. [什么是图数据？](01-%E4%BB%80%E4%B9%88%E6%98%AF%E5%9B%BE%E6%95%B0%E6%8D%AE%EF%BC%9F.md)
- 2. [标准神经网络在图数据上的局限性](02-%E6%A0%87%E5%87%86%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9C%A8%E5%9B%BE%E6%95%B0%E6%8D%AE%E4%B8%8A%E7%9A%84%E5%B1%80%E9%99%90%E6%80%A7.md)
- 3. [常见图机器学习任务](03-%E5%B8%B8%E8%A7%81%E5%9B%BE%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E4%BB%BB%E5%8A%A1.md)
- 4. [图的表示：邻接矩阵与特征矩阵](04-%E5%9B%BE%E7%9A%84%E8%A1%A8%E7%A4%BA%EF%BC%9A%E9%82%BB%E6%8E%A5%E7%9F%A9%E9%98%B5%E4%B8%8E%E7%89%B9%E5%BE%81%E7%9F%A9%E9%98%B5.md)
- 5. [图的属性与度量](05-%E5%9B%BE%E7%9A%84%E5%B1%9E%E6%80%A7%E4%B8%8E%E5%BA%A6%E9%87%8F.md)
- 6. [NetworkX 库简介](06-NetworkX%20%E5%BA%93%E7%AE%80%E4%BB%8B.md)
- 7. [动手实践：加载并检查图数据集](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%8A%A0%E8%BD%BD%E5%B9%B6%E6%A3%80%E6%9F%A5%E5%9B%BE%E6%95%B0%E6%8D%AE%E9%9B%86.md)

章节测验：[在线测验](https://apxml.com/zh/courses/introduction-to-graph-neural-networks/chapter-1-foundations-graph-based-learning/quiz)
