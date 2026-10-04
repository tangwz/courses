---
course: "applied-autoencoders-feature-extraction"
chapter: "advanced-autoencoder-architectures"
lesson: "layer-wise-training-stacked-autoencoders"
sourceId: 6470
sourceUrl: "https://apxml.com/zh/courses/applied-autoencoders-feature-extraction/chapter-4-advanced-autoencoder-architectures/layer-wise-training-stacked-autoencoders"
title: "堆叠自编码器的分层训练"
description: "讨论了贪婪分层预训练的理念，以有效初始化堆叠自编码器。"
order: 7
plots: []
sourceHash: "c1e04010b4f5166cc0b1e7bbea1a5ca226eb95f986b24dccde20e8b82c3a2e35"
sourceCorrections: []
---

堆叠自编码器，如我们所介绍的，旨在通过构建具有多个隐藏层的深度网络来学习分层特征表示。其设想是每个后续层基于前一层的输出，学习更复杂和抽象的特征。然而，直接从随机初始化开始训练这类深度神经网络 (neural network)可能存在难度。梯度消失或梯度爆炸等问题，或者优化过程陷入不佳的局部最优解，曾是重要的障碍，特别是在深度学习 (deep learning)早期。

为解决这些挑战并有效初始化深度堆叠自编码器的权重 (weight)，发展出一种称为**贪婪分层预训练 (pre-training)**的技术。这种方法将训练深度网络的复杂问题分解为一系列更简单、更易于处理的步骤，一次训练一层。

### 贪婪分层预训练 (pre-training)过程

核心设想是单独训练堆叠自编码器的每一层，将其作为一个浅层自编码器。一旦一层训练完毕，其学习到的权重 (weight)即被固定，其输出（编码表示）随后用作训练下一层的输入。我们来回顾一下这些步骤：

1. **训练第一个自编码器 (AE1)**:

   - 使用原始输入数据，我们称之为 $X$。
   - 训练一个简单的自编码器，包含一个隐藏层（编码器1，解码器1）以重建 $X$。目标是最小化重建误差，例如，$L(X, \text{解码器1}(\text{编码器1}(X)))$。
   - AE1 训练完毕后，保存其编码器（编码器1）和解码器（解码器1）的权重。编码器1 将 $X$ 转换为低维表示 $H_1 = \text{编码器1}(X)$。
2. **训练第二个自编码器 (AE2)**:

   - 使用输出 $H_1$（来自 AE1 隐藏层的激活），并将其作为 *新的* 简单自编码器（AE2，包含编码器2和解码器2）的输入数据。
   - 训练 AE2 以重建 $H_1$。即，最小化 $L(H_1, \text{解码器2}(\text{编码器2}(H_1)))$。
   - AE2 训练完毕后，保存编码器2和解码器2的权重。编码器2 将 $H_1$ 转换为新的、可能更紧凑的表示 $H_2 = \text{编码器2}(H_1)$。
3. **重复以增加层数**:

   - 如果想在堆叠自编码器中增加更多层，可继续此过程。对于第 $k$ 个自编码器 (AE\_k)，其输入将是 $H_{k-1}$（来自编码器\_{k-1}的激活），并会训练它以重建 $H_{k-1}$。
4. **组装堆叠自编码器**:

   - 所有单独的层预训练完毕后，组装完整的堆叠自编码器。
   - 堆叠自编码器的**编码器部分**通过按顺序堆叠所有预训练的编码器来形成：$\text{堆叠编码器} = \text{编码器}_N \circ \dots \circ \text{编码器}_2 \circ \text{编码器}_1$。
   - 堆叠自编码器的**解码器部分**通过按相反顺序堆叠所有预训练的解码器来形成：$\text{堆叠解码器} = \text{解码器}_1 \circ \text{解码器}_2 \circ \dots \circ \text{解码器}_N$。
   - 因此，输入 $X$ 依次经过编码器1、编码器2等，生成最终的潜在表示。这个潜在表示随后依次经过解码器N、解码器(N-1)等，回到解码器1，以重建原始输入 $X$。
5. **微调 (fine-tuning)整个堆叠**:

   - 通过贪婪分层预训练获得的权重为深度网络提供了良好的初始化。
   - 最后一步是将整个堆叠自编码器视为一个单一模型并进行端到端训练。这意味着您将原始输入 $X$ 馈送给堆叠编码器，将结果通过堆叠解码器得到 $\hat{X}$，然后使用反向传播 (backpropagation)同时更新*所有*层（包括编码器和解码器）中的*所有*权重，以最小化全局重建误差 $L(X, \hat{X})$。
   - 这一微调步骤使得所有层中的权重能够进行微调，以便为整体重建任务更紧密配合，通常会提升表现。

使用“贪婪”一词是因为每一层都旨在优化其局部任务（重建其自身输入），最初并未考虑整个深度堆叠的总体全局目标。

下方是一张图表，说明了针对具有两个隐藏层的堆叠自编码器的贪婪分层预训练过程，随后是组装和微调。

> 贪婪分层预训练的步骤。每个自编码器层按顺序训练。然后，编码器和解码器被组装成一个深度堆叠，随后进行端到端微调。

### 为何分层训练有效？

此策略特别有益，因为：

- **良好的初始化**：与随机初始化相比，它为深度网络的权重 (weight)提供了一个更好的起始点。每一层都已准备好从其输入中提取有用的特征。
- **分层特征学习**：通过逐层训练，网络自然地学习到特征的分层结构。第一层从原始数据中学习基础特征，第二层从第一层的特征中学习更复杂的特征，以此类推。
- **减轻优化问题**：一次训练一个较浅的网络通常更容易，并且不太容易出现梯度消失等问题，这些问题会困扰从头开始的超深度网络的端到端训练。每一层的训练都引导参数 (parameter)进入搜索空间的一个合理区域。

### 在现代深度学习 (deep learning)中的意义

尽管更好的激活函数 (activation function)（如 ReLU）、先进的优化算法（例如 Adam、RMSprop）、归一化 (normalization)技术（如批量归一化）以及新的架构设计（如残差连接）的出现使得深度网络的直接端到端训练更具可行性，但贪婪分层预训练 (pre-training)并未过时。

- 在处理超深度自编码器时，或当进行端到端训练所需的广泛超参数 (parameter) (hyperparameter)搜索的计算资源有限时，它仍然是一种有价值的技术。
- 它提供了一种直观的方法来构建和理解分层特征提取器。
- 在某些情形下，特别是在数据有限时，预训练可以帮助正则化 (regularization)模型并带来更好的泛化能力。

理解分层预训练使您能够了解深度表示如何逐步构建，并为您的工具箱增添了另一项工具，用于构建和训练有效的堆叠自编码器以进行特征提取。

## 参考资料

- [A Fast Learning Algorithm for Deep Belief Nets](https://doi.org/10.1162/neco.2006.18.7.1527) — Geoffrey E. Hinton, Simon Osindero, and Yee-Whye Teh (2006)
  Journal: Neural Computation; Publisher: MIT Press; Volume: 18; Pages: 1527-1554; DOI: [10.1162/neco.2006.18.7.1527](https://doi.org/10.1162/neco.2006.18.7.1527)
  这篇论文提出了一种开创性的深度信念网络贪婪逐层训练方法，为有效训练像堆叠自编码器这样的深度神经网络架构奠定了基础。
- [Greedy Layer-Wise Training of Deep Networks](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHGJdLnz1Z5XMUZ8wnSeM9CqAnPXZQjqb71w0uKlcTS9ZpwX8FsfGdRmodAjErbo0mTltU0wiEGLfQoHnLTVR2W8mBIAyOpqq2HLa28SREbbPtbWa2MQV-PZa-HtPofjKK23MHLOIAIabV7xGKUCA1Np-YfpOloWB8B_Rv8j_8rk684Jd1PxV2o) — Yoshua Bengio, Pascal Lamblin, Dan Popovici, Hugo Larochelle (2007)
  Journal: Advances in Neural Information Processing Systems 19; Volume: 19; Pages: 153-160
  这项工作进一步阐述并推广了贪婪逐层训练方法，展示了其在预训练包括堆叠自编码器在内的各种深度网络架构方面的有效性。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  这本书提供了深度学习的全面论述，详细解释了堆叠自编码器、贪婪逐层预训练以及它们的历史背景和重要性。
