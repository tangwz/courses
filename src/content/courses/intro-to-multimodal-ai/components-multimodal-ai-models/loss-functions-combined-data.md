---
course: "intro-to-multimodal-ai"
chapter: "components-multimodal-ai-models"
lesson: "loss-functions-combined-data"
sourceId: 6528
sourceUrl: "https://apxml.com/zh/courses/intro-to-multimodal-ai/chapter-4-components-multimodal-ai-models/loss-functions-combined-data"
title: "衡量性能：组合数据的损失函数"
description: "简要介绍专为涉及多种数据模态的任务定制的损失函数。"
order: 5
plots: []
sourceHash: "2f37e89beaec661e8d9f3a392616be105accf563a9765a25671ab58dae22c8bb"
sourceCorrections: []
---

为了评估多模态 (multimodal)AI模型是否有效学习并表现良好，一个重要问题随之而来：我们如何衡量其进展？损失函数 (loss function)在此发挥着重要作用。它们是训练任何AI模型的主要部分，在多模态系统中，它们帮助我们在处理组合数据类型时衡量性能。

### 什么是损失函数 (loss function)？AI模型的成绩单

想象你在教学生一项新技能。你给他们一个测试，他们的分数会告诉你他们对所学内容的掌握程度。**损失函数**（有时也称成本函数）对AI模型来说，扮演着类似的角色。

训练期间，模型进行预测（例如为图片生成文字描述）。损失函数随后将模型的预测与正确答案（“真实值”）进行比较。它会计算一个数值分数，表示模型的预测有多“不准确”或“偏离目标”。高的损失值表示模型犯了重大错误，而低的损失值则表明其预测接近目标。

训练AI模型的主要目标是**最小化这个损失值**。通过反复调整其内部设置（参数 (parameter)）以减少损失，模型会逐渐学会做出更好的预测。

### 单一类型数据的损失函数 (loss function)：快速回顾

在我们看这如何应用于组合数据之前，让我们简要回顾一下模型只处理一种数据类型（单模态系统）时如何使用损失函数：

- **对于文本任务：** 如果模型对文本进行分类（例如，积极或消极情绪），常用损失函数是*交叉熵损失*。它衡量模型的预测概率与实际类别之间的差异。
- **对于图像任务：** 如果模型标记 (token)图片（例如，“猫”或“狗”），也会使用交叉熵损失。如果它预测数值，例如图片中物体的坐标，则可能会使用*均方误差 (MSE)*，它计算预测值与实际值之间的平均平方差。
- **对于音频任务：** 根据任务的不同，例如语音转录（音频转文本）或声音分类，可以应用各种损失函数，包括与文本和图像任务相似的那些。

具体的损失函数总是取决于任务的性质以及模型应该产生的输出类型。

### 多模态 (multimodal)难题：组合数据的性能评估

那么，当我们的AI系统同时处理多种类型数据时会怎样呢？例如，一个图片描述模型接收图像（输入1）并生成文本描述（输出1）。一个视觉问答（VQA）系统接收图像（输入1）和文本问题（输入2）并生成文本答案（输出1）。

当模型的成功取决于理解和处理来自这些不同来源的信息，并可能产生涉及多种模态的输出时，我们如何计算一个单一的“错误分数”呢？

这就是我们需要定义组合数据损失函数 (loss function)的策略的地方。

### 组合损失：加权求和方法

一种常用且直接的方法是为多模态 (multimodal)任务的不同方面计算单独的损失值，然后将它们组合起来，通常是作为加权求和。

假设我们的模型有两个与两种模态相关的主要任务。例如，在一个模型需要同时理解图像和文本特征才能进行预测的任务中：

- $L_{image}$ 可以是与图像特征处理或表示效果相关的损失。
- $L_{text}$ 可以是与文本特征处理或表示效果相关的损失，或者与文本输出的准确性相关的损失。

然后我们可以将这些组合成总损失 $L_{total}$：


$$
L_{total} = w_{image} \cdot L_{image} + w_{text} \cdot L_{text}
$$


在这个公式中：

- $L_{image}$ 和 $L_{text}$ 是各个损失值。
- $w_{image}$ 和 $w_{text}$ 是**权重 (weight)**。这些是我们选择的数值，它们决定了每个单独损失部分在总损失中的重要性。

你可以把它想象成一个食谱。如果对于特定应用来说，文本部分的完善比图像部分更重要，你可能会给 $L_{text}$ 更高的权重（$w_{text} > w_{image}$）。如果两者同样重要，它们的权重可能相等（例如，$w_{image} = 1$ 且 $w_{text} = 1$）。

找到这些权重的恰当平衡有时需要通过实验。如果某个权重过高，模型可能会过多地侧重于最小化该部分的损失而忽视其他部分。

下图说明了如何组合单独的损失：

> 此图显示了多模态模型如何处理输入（图像和文本），产生与不同方面相关的输出，计算这些方面的单独损失，然后使用权重将它们组合成总损失。这个总损失指导优化器提升模型。

### 定义多模态 (multimodal)损失的其他方法

虽然加权求和是常用方法，特别是对初学者而言，但也存在其他方法：

- **任务特定的端到端损失：** 对于某些任务，可以设计一个单一的、更复杂的损失函数 (loss function)来直接衡量整体性能。例如，在旨在学习联合表示（其中图像和文本特征映射到共享空间）的任务中，*对比损失*很受欢迎。这些函数试图将匹配的图像-文本对的表示拉近，并将不匹配的对在该共享空间中推开。这本身就一次性考虑了两种模态。
- **多任务学习损失：** 如果你的多模态系统设计为同时执行多个不同的任务（例如，自动驾驶系统可能需要使用摄像头检测物体、理解语音指令并预测行人路径），每个任务将有其自己的损失函数。总损失通常是这些单独任务损失的总和（通常是加权的）。

### 损失如何指导提升

无论总损失如何计算，它的值都是训练过程（通常使用称为*梯度下降 (gradient descent)*或其变体的算法）试图最小化的对象。优化器利用损失函数 (loss function)的信息（特别是其梯度）来找出如何调整模型的内部数值（神经网络 (neural network)层中的权重 (weight)和偏差），以便下次其预测会更好一点，损失也会更低一点。

### 关于多模态 (multimodal)损失需要记住什么

思考处理组合数据类型模型的损失函数 (loss function)时，请记住以下几点：

1. **目的：** 主要目标是创建一个单一的数值分数，告诉模型在整体多模态任务上的表现如何。
2. **通常需要平衡：** 如果你正在组合损失（如加权求和方法中），每种模态或任务的贡献需要平衡。你不希望模型完善对图像的理解却完全忽视文本，反之亦然，除非那是具体的意图。
3. **与任务的相关性：** 损失函数必须准确反映“良好表现”对你的特定应用意味着什么。如果你的损失函数没有衡量你真正关心的事情，模型可能会完全学会做别的事情。
4. **调试的起点：** 如果你的多模态模型学习效果不佳，检查组合损失的各个组成部分（如果适用）有时可以提供关于模型在任务的哪一部分或哪种模态上表现最差的线索。

理解如何通过这些损失函数来衡量性能，是理解多模态AI模型如何学习解释和连接来自不同来源信息的重要一步。正如我们将在下一节中看到，这个损失值是驱动训练过程的引擎。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 5, Chapter 8
  全面解释了损失函数、优化算法及其在深度学习模型中的应用。
- [Learning Transferable Visual Models From Natural Language Supervision](https://proceedings.mlr.press/v139/radford21a.html) — Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever (2021)
  Journal: Proceedings of the 38th International Conference on Machine Learning; Publisher: PMLR; Volume: 139; Pages: 8748-8763
  介绍了一种广泛认可的多模态模型（CLIP），它利用对比学习作为任务特定的端到端损失函数，以对齐图像和文本表示。
- [An Overview of Multi-Task Learning in Deep Neural Networks](https://arxiv.org/abs/1706.05098) — Sebastian Ruder (2017)
  Journal: arXiv preprint arXiv:1706.05098; DOI: [10.48550/arXiv.1706.05098](https://doi.org/10.48550/arXiv.1706.05098)
  综述了深度神经网络中的多任务学习，详细介绍了如何组合不同任务的损失以提升模型的泛化能力。
- [CS231n Convolutional Neural Networks for Visual Recognition: Neural Networks, Backpropagation and Training](http://cs231n.github.io/neural-networks-3/) — Stanford University (2024)
  解释了各种损失函数（如交叉熵、均方误差）以及神经网络中反向传播和优化的原理。
