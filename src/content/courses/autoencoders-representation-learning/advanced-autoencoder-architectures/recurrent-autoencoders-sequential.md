---
course: "autoencoders-representation-learning"
chapter: "advanced-autoencoder-architectures"
lesson: "recurrent-autoencoders-sequential"
sourceId: 2873
sourceUrl: "https://apxml.com/zh/courses/autoencoders-representation-learning/chapter-5-advanced-autoencoder-architectures/recurrent-autoencoders-sequential"
title: "循环自编码器处理序列数据"
description: "在自编码器中使用RNN或LSTM来建模序列信息。"
order: 2
plots: []
sourceHash: "dda9f785f73f331bce42c0edaa5ff90f2e9a441c0e4a38640071b4f991adf90c"
sourceCorrections: []
---

卷积自编码器擅长捕获图像等网格状数据的空间层级，但许多重要数据集具有固有的序列结构。时间序列数据（股票价格、传感器读数）、自然语言文本、音频信号和视频都是元素顺序具有重要意义的例子。标准的馈送式自编码器，甚至在固定窗口上运行的卷积自编码器，都难以有效建模这些时间依赖性。

循环神经网络 (neural network)（RNN）可以结合到自编码器架构中，从而形成循环自编码器（RAE）。其基本思路是将编码器和解码器中的馈送层替换为循环层，例如简单RNN、LSTM（长短期记忆）或GRU（门控循环单元 (GRU)）。

### 循环自编码器架构

RAE通常遵循熟悉的编码器-瓶颈-解码器结构，但已适用于序列：

1. **编码器：** 循环网络（例如LSTM）逐步处理输入序列$x = (x_1, x_2, ..., x_T)$。在每个时间步$t$，RNN根据当前输入$x_t$和上一个隐藏状态$h_{t-1}$更新其隐藏状态$h_t$：
   $h_t = f_{enc}(h_{t-1}, x_t)$
   最终的隐藏状态$h_T$（或有时是所有隐藏状态的函数）用作捕获整个输入序列信息的压缩表示或上下文 (context)向量 (vector)$z$。这个向量$z$代表瓶颈层。
2. **瓶颈：** 固定大小的上下文向量$z = h_T$。此向量旨在概括输入序列的基本信息。
3. **解码器：** 另一个循环网络（通常在架构上与编码器相似）接收上下文向量$z$并生成输出序列$\hat{x} = (\hat{x}_1, \hat{x}_2, ..., \hat{x}_T')$。解码器通常用上下文向量$z$初始化（例如，$h'_0 = z$），并逐个元素生成序列，通常将先前生成的元素作为下一个步骤的输入反馈（或在训练期间使用教师强制）：
   $h'_t = f_{dec}(h'_{t-1}, \hat{x}_{t-1})$
   $\hat{x}_t = g_{dec}(h'_t)$
   目标通常是重建原始输入序列，因此通常$T' = T$，目标是$\hat{x} \approx x$。

该结构与机器翻译或文本摘要中常用的序列到序列（Seq2Seq）模型非常相似，但在此处以无监督方式应用于表示学习或重建。

> 循环自编码器图示。编码器处理输入序列$(x_1, ..., x_T)$，生成一个上下文向量$z$。解码器利用$z$生成重建序列$(\hat{x}_1, ..., \hat{x}_T)$。虚线表示依赖关系或初始化。

### LSTM和GRU的优点

尽管简单的RNN可以捕获时间模式，但它们通常受梯度消失或梯度爆炸问题影响，使得学习长距离依赖性变得困难。对于大多数应用，RAE中更青睐LSTM或GRU。它们的门控机制允许它们在更长的时间尺度上选择性地记忆或遗忘信息，从而产生更好的序列数据表示。

### 实现考量

- **单元选择：** LSTM或GRU单元通常比简单RNN更推荐。
- **双向性：** 对于编码时（例如特征提取）过去和未来元素的上下文 (context)均有益的任务，编码器中可以使用双向RNN。这会双向处理序列，并连接两个方向的隐藏状态。
  "\* **变长序列：** 序列数据通常具有不同的长度。将序列填充到最大长度以及使用掩码层（以在计算和损失计算期间忽略填充步骤）是必要的。"
- **训练：** 教师强制，即解码器使用真实的前一个元素$x_{t-1}$作为输入而不是生成的$\hat{x}_{t-1}$，通常在训练期间用于稳定学习。课程学习方法可能会逐渐降低使用教师强制的概率。
- **损失函数 (loss function)：** 重建损失通常在序列中逐元素计算，然后聚合（例如，求和或平均）。对于数值序列，每个时间步的均方误差（MSE）很常见。对于类别序列（如文本），每个时间步的交叉熵损失是适用的。

### 循环自编码器应用

RAE特别适用于：

- **无监督特征提取：** 上下文 (context)向量 (vector)$z$可以用作变长序列的固定大小特征表示，适合作为下游分类器或聚类算法的输入。
- **时间序列异常检测：** 通过在正常序列上训练RAE，可以通过观察未见序列与所学模式的偏离所导致的高重建误差来检测异常。
- **序列压缩/重建：** 重建输入序列的核心任务。
- **数据去噪：** 类似于去噪自编码器，RAE可以训练从受损输入中重建干净序列。

### 局限性

基本RAE架构的主要局限是固定大小的瓶颈$z$。对于非常长的序列，将所有必要信息压缩到这个单一向量 (vector)中可能具有挑战性，可能导致信息丢失。高级Seq2Seq技术，例如注意力机制 (attention mechanism)，允许解码器回顾特定的编码器隐藏状态，可以缓解此问题，但这超出了标准RAE的定义范围。

总之，循环自编码器扩展了自编码器理念，通过利用循环神经网络 (neural network) (RNN)的能力来有效建模序列数据。它们为无监督表示学习、异常检测以及涉及时间序列、文本和其他有序数据的重建任务提供了一种有价值的工具。

## 参考资料

- [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735) — Sepp Hochreiter, Jürgen Schmidhuber (1997)
  Journal: Neural Computation; Publisher: MIT Press; Volume: 9; Pages: 1735-1780; DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
  提出了长短期记忆（LSTM）网络架构，这是循环自编码器中有效建模长期依赖的关键组成部分。
- [Sequence to Sequence Learning with Neural Networks](https://neurips.cc/paper_files/NeurIPS2014/hash/a14ac55a4f27472c1110e755508c70ff-Abstract.html) — Ilya Sutskever, Oriol Vinyals, Quoc V. Le (2014)
  Journal: Advances in Neural Information Processing Systems 27; Volume: 27; Pages: 3104-3112
  提出了使用LSTM的开创性编码器-解码器架构，通过学习将输入序列映射到目标序列，为循环自编码器奠定了基础，直接反映了RAE的无监督重建结构。
- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://aclanthology.org/D14-1179.pdf) — Kyunghyun Cho, Bart van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio (2014)
  Journal: Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP); Publisher: Association for Computational Linguistics; Pages: 1724-1734; DOI: [10.3115/v1/D14-1179](https://doi.org/10.3115/v1/D14-1179)
  介绍了门控循环单元（GRU）和RNN编码器-解码器结构，用于学习变长序列的固定维度表示，与循环自编码器的目标高度相关。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  一本基础教材，全面涵盖了循环神经网络、LSTM、GRU、自编码器和序列建模，为循环自编码器提供了理论背景。
