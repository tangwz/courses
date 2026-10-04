# 第 4 章：GNN 模型训练

来源：[原章节](https://apxml.com/zh/courses/introduction-to-graph-neural-networks/chapter-4-training-gnn-models)

[返回课程目录](../README.md)

在了解 GNN 架构后，接下来的重点是学习如何训练这些模型。本章将介绍从确定 GNN 架构到完成特定图数据任务的全过程。

你将学习如何为节点分类等应用配置模型，包括将最终生成的节点嵌入传入分类层。接着，我们会说明如何选择合适的目标函数（如交叉熵损失）来衡量模型误差。本章的主要内容是构建标准的训练循环：包括执行前向传播、计算损失以及通过反向传播更新模型权重。

我们还会讲解图学习特有的操作。这包括数据划分时“转导式”（transductive）和“归纳式”（inductive）的区别，以及它们对模型评估的影响。你将学习使用准确率等标准指标来评估模型效果。最后，我们会介绍 dropout 等正则化方法，以提升模型的泛化能力。本章最后附带动手练习，让你应用上述步骤来训练之前构建的 GCN 模型。

## 小节

- 1. [为节点分类任务搭建 GNN](01-%E4%B8%BA%E8%8A%82%E7%82%B9%E5%88%86%E7%B1%BB%E4%BB%BB%E5%8A%A1%E6%90%AD%E5%BB%BA%20GNN.md)
- 2. [图任务的损失函数](02-%E5%9B%BE%E4%BB%BB%E5%8A%A1%E7%9A%84%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.md)
- 3. [GNN 的训练循环](03-GNN%20%E7%9A%84%E8%AE%AD%E7%BB%83%E5%BE%AA%E7%8E%AF.md)
- 4. [图数据的切分：直推式与归纳式](04-%E5%9B%BE%E6%95%B0%E6%8D%AE%E7%9A%84%E5%88%87%E5%88%86%EF%BC%9A%E7%9B%B4%E6%8E%A8%E5%BC%8F%E4%B8%8E%E5%BD%92%E7%BA%B3%E5%BC%8F.md)
- 5. [节点分类的评估指标](05-%E8%8A%82%E7%82%B9%E5%88%86%E7%B1%BB%E7%9A%84%E8%AF%84%E4%BC%B0%E6%8C%87%E6%A0%87.md)
- 6. [GNN 中的过拟合与正则化](06-GNN%20%E4%B8%AD%E7%9A%84%E8%BF%87%E6%8B%9F%E5%90%88%E4%B8%8E%E6%AD%A3%E5%88%99%E5%8C%96.md)
- 7. [实践：训练与评估你的 GCN](07-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E8%AE%AD%E7%BB%83%E4%B8%8E%E8%AF%84%E4%BC%B0%E4%BD%A0%E7%9A%84%20GCN.md)

章节测验：[在线测验](https://apxml.com/zh/courses/introduction-to-graph-neural-networks/chapter-4-training-gnn-models/quiz)
