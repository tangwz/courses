---
course: "foundations-transformers-architecture"
chapter: "implementation-details-optimization"
lesson: "transformer-weight-initialization"
sourceId: 2058
sourceUrl: "https://apxml.com/zh/courses/foundations-transformers-architecture/chapter-7-implementation-details-optimization/transformer-weight-initialization"
title: "权重初始化策略"
description: "Transformer层中权重初始化的推荐做法，旨在实现稳定训练。"
order: 2
plots: ["plots/2058-0.json"]
sourceHash: "2ecf585d820f19d025d37cd14a075e765f1dacb43b273193baf0dc4decc5f123"
sourceCorrections: []
---

正确初始化神经网络 (neural network)的权重 (weight)是实现稳定高效训练的**主要步骤**。对于Transformer这类层数较多的架构，梯度通过多层传播，不当的初始化很容易导致梯度消失或梯度爆炸，在学习过程开始前就使其停滞。虽然现代的优化器和归一化 (normalization)层可以缓解这些问题中的一部分，但周全的权重初始化仍然是实际操作中一个**主要组成部分**。

大多数标准初始化方案的目标是，在激活值和梯度在网络中向前和向后传播时，保持其方差不变。如果方差随每层呈指数增长，梯度就会爆炸；如果呈指数减小，梯度就会消失。

### 线性层初始化

Transformer模型在多头注意力 (multi-head attention)模块（用于Q、K、V投影和最终输出投影）和逐位置前馈网络（FFNs）中大量使用线性（或全连接）层。这些层最常用且有效的初始化策略是Glorot (Xavier) 和 He 初始化。

- **Glorot (Xavier) 初始化：** 由Glorot和Bengio（2010年）提出，此方法旨在保持激活值和梯度在各层间的方差恒定。它对于使用对称激活函数 (activation function)（如$tanh$）的层尤其有效。对于将输入大小为$fan_{in}$转换为输出大小为$fan_{out}$的线性层，权重 (weight)$W$通常从均匀分布中抽取：

  
  $$
  W \sim U\left[-\sqrt{\frac{6}{fan_{in} + fan_{out}}}, \sqrt{\frac{6}{fan_{in} + fan_{out}}}\right]
  $$
  

  或者，也可以使用正态分布：

  
  $$
  W \sim N\left(0, \frac{2}{fan_{in} + fan_{out}}\right)
  $$
  
- **He 初始化：** 由He等人（2015年）提出，此方法专为后接修正线性单元（ReLU）或其变体的层设计，这在Transformer FFN中很常见。它考虑了ReLU将一半激活值归零从而减小方差的事实。权重通常从正态分布中抽取：

  
  $$
  W \sim N\left(0, \frac{2}{fan_{in}}\right)
  $$
  

  也存在均匀分布版本。

在实践中，许多深度学习 (deep learning)框架在预期使用ReLU激活时，默认对线性层采用He初始化，否则采用Glorot初始化。对于Transformer模型，为FFN层使用He初始化，而为注意力投影层（这些层在进一步组合之前通常不会对其直接输出立即应用ReLU）使用Glorot/Xavier初始化，是一个合理的起点。像最初的“Attention Is All You Need”论文中的实现就使用了Glorot均匀初始化。

### 嵌入 (embedding)层初始化

输入和输出嵌入层将离散的令牌ID映射到稠密向量 (vector)。这些层通常与标准线性层的初始化方式不同。一个常见的做法是从均值为0、标准差相对较小的正态分布（例如$N(0, 0.02^2)$）初始化嵌入权重 (weight)。这可以避免初始嵌入值过大，从而可能导致后续计算不稳定，尤其是位置编码 (positional encoding)的加入。

一些实现可能会绑定输入和输出嵌入权重，为两者共享相同的矩阵（可能经过转置），这也会影响初始化方法的选择。

### 层归一化 (normalization)初始化

层归一化层也有可训练参数 (parameter)：增益（$\gamma$）和偏差（$\beta$）。这些参数通常初始化为$\gamma = 1$和$\beta = 0$。这确保了层归一化在初始时仅将激活值归一化为零均值和单位方差，而不应用任何缩放或平移，从而让网络在训练过程中学习合适的变换。

### 实际考量

大多数现代深度学习 (deep learning)库（如PyTorch、TensorFlow、JAX）为其标准层提供了合理的默认初始化，通常与Glorot或He方法一致。例如，Hugging Face Transformers库通常使用标准差由`initializer_range`配置参数 (parameter)（通常默认为0.02）指定的正态分布初始化权重 (weight)，并将偏差初始化为零，而LayerNorm的权重设置为1，偏差设置为0。



![权重初始化分布](plots/2058-0.json)



> 比较了层具有512个输入/输出单元时，使用小标准差正态分布初始化和Xavier均匀初始化所产生的权重分布。请注意，与Xavier方法的更宽泛分布相比，N(0, 0.02^2)方法在零点附近聚集更紧密。

虽然这些默认设置通常表现良好，但理解其基本原理有助于在训练变得不稳定时进行更明智的调试。偶尔，可以尝试对初始化标准差进行微小调整，特别是对于嵌入 (embedding)层或最终输出层，但对于核心线性层而言，大幅偏离Glorot或He等既定方法通常是不必要的，并可能适得其反。

## 参考资料

- [Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) — Xavier Glorot, Yoshua Bengio (2010)
  Journal: Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics (AISTATS); Volume: 9; Pages: 249-256
  提出Xavier（Glorot）初始化方法，旨在维持各层激活和梯度方差，对tanh等对称激活函数尤其有用。
- [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/pdf/1502.01852.pdf) — Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (2015)
  Journal: Proceedings of the IEEE International Conference on Computer Vision (ICCV); Pages: 1026-1034; DOI: [10.1109/ICCV.2015.122](https://doi.org/10.1109/ICCV.2015.122)
  提出He初始化，专门为使用ReLU或其变体的神经网络设计，以解决这些激活函数导致的方差减小问题。
- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Volume: 30; Pages: 5998-6008
  介绍Transformer架构的论文，文中提到其权重矩阵使用Glorot均匀初始化，与Transformer实现细节有关。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  一本全面的教材，涵盖深度学习的基本概念，包括各种权重初始化策略的详细理论和实践说明。
- [Transformers Library Documentation: PreTrainedModel](https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel) — Hugging Face (2024)
  提供关于Hugging Face Transformers库中默认权重初始化参数（如`initializer_range`）的实用指导，这对于实际实现至关重要。
