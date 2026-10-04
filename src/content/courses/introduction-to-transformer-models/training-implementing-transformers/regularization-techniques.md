---
course: "introduction-to-transformer-models"
chapter: "training-implementing-transformers"
lesson: "regularization-techniques"
sourceId: 3030
sourceUrl: "https://apxml.com/zh/courses/introduction-to-transformer-models/chapter-4-training-implementing-transformers/regularization-techniques"
title: "正则化方法"
description: "解释Dropout在Transformer架构中的应用。"
order: 5
plots: []
sourceHash: "5e9b0fc5307db1c6f18f08ce6eda453ff3f0bb31a20ae9e65b72582298923764"
sourceCorrections: []
---

训练大型神经网络 (neural network)（如Transformer）需要应对过拟合 (overfitting)的风险。当模型过度学习训练数据，连同其噪声和特定特征时，就会发生过拟合，导致在新数据上表现不佳。虽然优化技术有助于模型收敛，但正则化 (regularization)方法对于提升其泛化能力是必需的。

标准Transformer架构中采用的主要正则化技术是**Dropout**。

### 理解Dropout

Dropout是一种简单而有效的正则化 (regularization)方法。在训练期间，对于每次前向传播，Dropout会随机地将层中一部分神经元（或隐藏单元）的输出设为零。这种“舍弃”是临时且具有随机性的；在每个训练步骤中，都会有不同的神经元集合被舍弃。

假设你有一个团队正在进行项目。如果在某一天，一些团队成员随机缺席，那么剩下的成员必须学会顶替他们，并且不能过度依赖任何一个成员。同样地，Dropout阻止神经元变得过度专业化或相互依赖于特定其他神经元。它促使网络学习更多冗余表示，因为它不能依赖于任何特定的神经元子集始终处于活跃状态。

### Dropout在Transformer架构中的应用

在原始的“Attention is All You Need”论文中，Dropout在Transformer模型的几个特定位置被应用：

1. **在嵌入 (embedding)和位置编码 (positional encoding)相加之后：** Dropout应用于组合后的输入嵌入和位置编码，然后将它们输入到第一个编码器或解码器层。

   ```
   输入 -> 输入嵌入 -> 位置编码 -> 相加 -> **Dropout** -> 编码器/解码器层
   ```
2. **在每个子层之后（在相加与归一化 (normalization)之前）：** 在编码器和解码器层内，Dropout应用于每个子层（多头注意力 (multi-head attention)机制 (attention mechanism)和逐位置前馈网络）的输出，紧接在残差连接（相加）和层归一化（归一化）步骤之前。

   ```
   子层输入 -> 多头注意力 -> **Dropout** -> 相加与归一化 -> 输出
   子层输入 -> 前馈网络 -> **Dropout** -> 相加与归一化 -> 输出
   ```
3. **在注意力权重 (weight)上（有时）：** 一些实现也会将Dropout直接应用于多头注意力机制中的注意力权重（$softmax(QK^T/\sqrt{d_k})$）。这有助于模型避免过度关注少数特定位置，从而避免过度自信。

将Dropout放置在每个主要处理模块（嵌入、注意力、前馈）之后，可以确保噪声贯穿整个网络深度被引入，有助于在不同表示学习层面上提升模型的适应能力。

### Dropout在训练和推理 (inference)期间的工作方式

- **训练：** 在训练期间，对于给定的一次前向传播，每个神经元都有概率 $p$（即dropout率，通常在0.1到0.5之间）被设为零。其余神经元的输出通常会按 $1/(1-p)$ 的比例放大。这种被称为*倒置Dropout*的技术，确保了训练期间输出的期望和与推理期间保持一致，简化了模式间的转换。
- **推理/评估：** 在推理或评估期间（在新数据上进行预测时），Dropout是关闭的。所有神经元都处于活跃状态，它们的输出不进行缩放（假设训练期间使用了倒置Dropout）。这使得模型能够使用其完全学到的能力进行预测。

当你使用PyTorch和TensorFlow等现代深度学习 (deep learning)框架的内置Dropout层时，它们会自动处理训练和推理期间的不同行为。你通常只需将dropout率 $p$ 指定为一个超参数 (parameter) (hyperparameter)即可。

### Dropout为何对Transformer有帮助

Transformer，特别是大型Transformer，拥有数百万甚至数十亿个参数 (parameter)。这种高容量使其容易记住训练数据。Dropout作用类似于一种模型平均。由于在每个步骤中实际训练的是一个不同的“瘦身”网络，最终的网络表现得像许多小型网络的集成，这通常会带来更好的泛化能力。它防止了神经元高度依赖于特定其他神经元的复杂相互适应，促使学习到单独来看更具信息量的特征。

虽然Dropout是标准Transformer中最主要的显式正则化 (regularization)技术，但其他因素也有助于防止过拟合 (overfitting)：

- **层归一化 (normalization)：** 有助于稳定学习过程，并可能产生轻微的正则化作用。
- **权重 (weight)衰减（L2正则化）：** 通常通过AdamW等优化器隐式应用，惩罚大权重。
- **学习率调度：** 例如学习率预热后接衰减等技术，可以帮助模型收敛到更好、更平坦的最小值，这通常对应于泛化能力更好的解决方案。
- **数据增强：** 尽管对于文本不如图像常见，但应用于输入数据的技术也可作为正则化手段。

选择适当的dropout率 $p$ 有着很大的作用。过低的dropout率可能无法提供足够的正则化，而过高的dropout率则可能通过移除过多信息而阻碍学习（导致欠拟合 (underfitting)）。这个比率通常根据在验证数据集上的表现进行调整。

通过在特定位置应用Dropout，Transformer架构有效地平衡了其学习复杂模式的高容量与对新数据良好泛化的需要，这是其在各种自然语言处理任务中取得成功的一个主要原因。

## 参考资料

- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://jmlr.org/papers/v15/srivastava14a.html) — Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov (2014)
  Journal: Journal of Machine Learning Research; Publisher: JMLR; Volume: 15; Pages: 1929-1958
  介绍了神经网络正则化中的Dropout技术。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems; Pages: 15; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  提出了Transformer架构并描述了其中Dropout的应用。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  全面解释了正则化技术，包括Dropout、层归一化和权重衰减。
