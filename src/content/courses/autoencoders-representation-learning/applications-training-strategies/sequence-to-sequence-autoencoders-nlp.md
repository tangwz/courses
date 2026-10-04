---
course: "autoencoders-representation-learning"
chapter: "applications-training-strategies"
lesson: "sequence-to-sequence-autoencoders-nlp"
sourceId: 2926
sourceUrl: "https://apxml.com/zh/courses/autoencoders-representation-learning/chapter-7-applications-training-strategies/sequence-to-sequence-autoencoders-nlp"
title: "序列到序列自编码器概述"
description: "简要概述将自编码器思想应用于自然语言处理中的序列数据。"
order: 5
plots: []
sourceHash: "1ab1f74eda061abbea6e253152e832560eafd3d928c2bbabc7ac5812c227067e"
sourceCorrections: []
---

“标准自编码器在学习固定大小向量 (vector)数据的表示方面表现出色，但许多数据集涉及长度可变的序列，例如文本文档、时间序列数据或音频信号。序列到序列（Seq2Seq）自编码器扩展了自编码器的核心思想，以有效地处理此类序列信息。这种架构调整在自然语言处理（NLP）方面尤为重要。”

其基本思路与经典自编码器相似：一个编码器将输入映射到潜在表示，一个解码器从该表示重建输入。然而，在序列到序列自编码器中，编码器和解码器通常都是循环神经网络 (neural network)（RNN），例如LSTM或GRU，它们能够处理序列数据。

### 架构概述

1. **编码器：** 编码器RNN逐步处理输入序列（例如，句子中的词语）。在每一步，它会更新其内部隐藏状态。处理完整个输入序列后，最终的隐藏状态（或其转换形式）将作为整个序列的压缩、固定大小表示。这个向量 (vector)通常被称为“上下文 (context)向量”或“思维向量”，类似于标准自编码器中瓶颈层的输出。
2. **上下文向量：** 这个固定大小的向量封装了输入序列中被认为重要的信息，以便重建该序列。它作为解码器的初始隐藏状态。
3. **解码器：** 解码器RNN接收上下文向量并逐步生成输出序列。其目标是重建原始输入序列。在训练期间，解码器常使用“教师强制”等方法，将目标序列的实际前一个元素作为每一步的输入，这有助于稳定训练。在推理 (inference)期间，上一步生成的输出通常作为下一步的输入。

下图说明了这种流程：

> 序列到序列自编码器架构。编码器将输入序列 ($x_1, ..., x_T$) 处理为一个固定大小的上下文向量 $z$。解码器使用 $z$ 重建原始序列 ($y'_1, ..., y'_T$)。

### 训练与表示

该模型通过最小化适用于序列的重建损失函数 (loss function)进行端到端训练，例如用于离散符号（如词语）的交叉熵损失或用于连续值的均方误差。目标是使输出序列 $y'$ 尽可能接近输入序列 $x$。

对表示学习的意义在于上下文 (context)向量 (vector) $z$。通过强制网络将整个输入序列压缩到这个单一向量中，并成功重建它，$z$ 隐式地学习捕捉输入序列的显著语义和句法特征。这种学到的表示可以随后用于下游任务。

### 在自然语言处理中的应用

序列到序列自编码器服务于多种目的：

- **无监督预训练 (pre-training)：** 类似于标准自编码器可以为图像分类器预训练层一样，序列到序列自编码器可以为机器翻译或文本摘要等序列任务预训练编码器和解码器网络。预训练的权重 (weight)提供了良好的初始化，通常在监督任务上带来更好的性能和更快的收敛，尤其是在标注数据有限的情况下。
- **学习序列嵌入 (embedding)：** 上下文 (context)向量 (vector) $z$ 本身可用作整个输入序列的固定大小嵌入（例如，句子嵌入）。这些嵌入捕捉序列的含义，并可用于语义相似度比较、聚类，或作为后续分类器的特征。
- **数据生成（有限）：** 尽管在高保真生成方面不如VAE或GAN常见，但解码器在给定上下文向量（可能经过采样或操作）的情况下可以生成序列。这为更先进的生成式序列模型提供了起点。
- **序列中的异常检测：** 类似于标准自编码器，序列到序列自编码器可用于序列数据（例如，系统日志、传感器读数）中的异常检测。产生高重建误差的序列可能会被标记 (token)为异常。

### 增强：注意力机制 (attention mechanism)

基本序列到序列架构的一个局限是需要将整个输入序列压缩成一个单一的固定大小上下文 (context)向量 (vector)，这对于长序列来说可能成为瓶颈。引入注意力机制是为了解决这个问题。在自编码器的情境下，注意力机制允许解码器在每个解码步骤中选择性地关注*编码器的隐藏状态*的不同部分，而不是仅仅依赖最终的上下文向量。这提供了一种更灵活有效的方法来处理重建长序列时的依赖关系。

### 实现注意事项

实现序列到序列自编码器通常涉及使用TensorFlow或PyTorch等深度学习 (deep learning)框架中提供的RNN层（如`LSTM`或`GRU`）作为编码器和解码器组件。在处理不同长度序列批次时，通常需要仔细处理序列填充和掩码。

总而言之，序列到序列自编码器调整了自编码器的核心原理以适应序列数据，从而实现有意义的序列表示的无监督学习 (supervised learning) (unsupervised learning)。这些主要通过上下文 (context)向量 (vector)捕获的表示，对于初始化下游模型或直接用于需要序列理解的任务都很有价值。

## 参考资料

- [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) — Ilya Sutskever, Oriol Vinyals, Quoc V. Le (2014)
  Journal: arXiv; DOI: [10.48550/arXiv.1409.3215](https://doi.org/10.48550/arXiv.1409.3215)
  本文介绍了基于LSTM的序列到序列学习框架，它是序列到序列自编码器架构的基础。
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio (2014)
  Journal: International Conference on Learning Representations (ICLR) 2015; DOI: [10.48550/arXiv.1409.0473](https://doi.org/10.48550/arXiv.1409.0473)
  本文引入了注意力机制，解决了序列到序列模型中固定大小上下文向量的瓶颈问题。
- [Deep Learning](http://www.deeplearningbook.org) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  本书提供了自编码器、循环神经网络和序列模型的全面内容，为序列到序列自编码器提供了理论背景。
- [Stanford CS224n: Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/) — Diyi Yang, Tatsunori Hashimoto (2025)
  Publisher: Stanford University
  一门高级课程，提供了关于序列模型、注意力机制及其在自然语言处理中应用的详细讲座和材料。
