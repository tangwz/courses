---
course: "introduction-to-transformer-models"
chapter: "sequence-modeling-attention-fundamentals"
lesson: "rnn-limitations"
sourceId: 2957
sourceUrl: "https://apxml.com/zh/courses/introduction-to-transformer-models/chapter-1-sequence-modeling-attention-fundamentals/rnn-limitations"
title: "传统循环神经网络方法的局限性"
description: "讨论RNN中的梯度消失问题、长序列处理困难和顺序计算瓶颈。"
order: 3
plots: []
sourceHash: "f67c70bea8496801856f9c1d7236c75761a549458f450da69d18686b70ac7970"
sourceCorrections: []
---

循环神经网络 (neural network)（RNN）及其变体，如LSTM和GRU，在序列建模方面取得了显著进展。然而，它们在处理任务中遇到的复杂且通常较长的序列时，仍面临一些实际问题。这些固有的局限性正是Transformer等新型架构得以发展的原因。

### 长距离依赖的难题

简单RNN最常讨论的局限之一是难以捕捉序列中相距较远元素之间的依赖关系。这主要源于**梯度消失问题**。

在训练过程中，RNN使用时间反向传播 (backpropagation)（BPTT）。梯度（误差信号）需要从输出一直向后流到早期时间步以更新权重 (weight)。在深度网络或长序列中，这些梯度在反向传播过程中，由于激活函数 (activation function)（如sigmoid或tanh）和权重矩阵的影响，会被反复乘以小于1的值，从而变得极其微小。

试想一下，如果想根据序列中较晚发生的错误来更新网络。如果梯度信号在到达相关的早期步骤时变得消失般微小，网络就实际未能学习到早期输入与后期输出之间的关联。过去信息的影响消退得太快了。


$$
\frac{\partial E}{\partial W} = \sum_{t=1}^{T} \frac{\partial E_t}{\partial W} = \sum_{t=1}^{T} \sum_{k=1}^{t} \frac{\partial E_t}{\partial y_t} \frac{\partial y_t}{\partial h_t} \frac{\partial h_t}{\partial h_k} \frac{\partial h_k}{\partial W}
$$


项 $\frac{\partial h_t}{\partial h_k}$ 涉及循环权重矩阵和激活函数导数的重复乘法。如果这些因子持续很小，那么来自遥远的过去步骤（$k \ll t$）的总体梯度贡献就会减小。

反之，梯度也可能**爆炸**（变得过大），导致训练不稳定。虽然像梯度裁剪这样的技术可以处理梯度爆炸，但梯度消失对学习长期模式构成了更根本的障碍。

尽管LSTM和GRU专门设计了门控机制，以缓解梯度消失问题并更好地控制信息流动，但它们仍然按顺序处理信息，并且与基于注意力的模型相比，在处理极长依赖关系时可能会遇到困难。

### 顺序计算阻碍并行化

RNN按顺序逐元素处理序列。时间步$t$处的隐状态$h_t$的计算根本上依赖于前一时间步的隐状态$h_{t-1}$：


$$
h_t = f(W_{hh} h_{t-1} + W_{xh} x_t + b_h)
$$


这种固有的顺序依赖性意味着，在$h_{t-1}$可用之前，无法计算$h_t$。尽管可以在批次中对*不同序列之间*的计算进行并行化，但无法对*单个序列内部*的计算进行跨时间步并行化。

这对于长序列来说是一个重要的瓶颈。GPU和TPU等现代硬件非常适合大规模并行计算。RNN由于其逐步的性质，无法充分利用这种能力，导致与允许更多并行处理序列元素的架构相比，训练时间更长。

### 固定大小编码的局限（在基本Seq2Seq中）

在使用RNN构建的经典序列到序列（seq2seq）模型中，编码器处理整个输入序列，并将其所有信息压缩成一个单一的、固定大小的上下文 (context)向量 (vector)（通常是编码器RNN的最终隐状态）。然后，这个向量被传递给解码器RNN，解码器RNN使用它来生成输出序列。

试图将潜在的非常长和复杂的输入句子（例如，用于翻译的一段文字）的含义压缩到一个固定大小的向量中是困难的。这充当了一个信息瓶颈。网络很难保留所有相关细节，特别是长输入序列开头的具体细节或信息。解码器只能访问这个压缩后的摘要，可能丢失原始输入中的重要细节。

这些局限，即因梯度问题导致的长距离依赖困难、无法在序列内部并行化计算，以及基本seq2seq模型中固定大小上下文向量的信息瓶颈，促使人们寻找替代方法。这一寻找直接促成了注意力机制 (attention mechanism)的出现，该机制允许模型在每一步回顾整个输入序列，并选择性地关注最相关部分，从而解决了RNN的许多这些不足。

## 参考资料

- [On the difficulty of training recurrent neural networks](http://proceedings.mlr.press/v28/pascanu13.pdf) — Razvan Pascanu, Caglar Gulcehre, Kyunghyun Cho, Yoshua Bengio (2013)
  Journal: Proceedings of the 30th International Conference on Machine Learning (ICML); Pages: 1310-1318; DOI: [10.1137/1.9781611973001.27](https://doi.org/10.1137/1.9781611973001.27)
  讨论了循环神经网络训练中的挑战，特别是梯度消失和梯度爆炸。
- [Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — Sepp Hochreiter, Jürgen Schmidhuber (1997)
  Journal: Neural Computation; Volume: 9; Pages: 1735-1780; DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
  介绍了长短期记忆（LSTM）架构，用于处理循环神经网络中的长程依赖。
- [Sequence to Sequence Learning with Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2014/file/a14ac55a4f27472c112fd97523550b0c-Paper.pdf) — Ilya Sutskever, Oriol Vinyals, Quoc V. Le (2014)
  Journal: Advances in Neural Information Processing Systems (NIPS) 27; Pages: 3104-3112
  提出了一种通用的端到端序列学习方法，通过固定大小的上下文向量展示了编码器-解码器结构。
- [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems (NIPS) 30; Publisher: Curran Associates, Inc.; Volume: 30; Pages: 5998-6008; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  介绍了Transformer模型，旨在克服循环模型在序列计算和长程依赖捕捉方面的局限。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  为循环神经网络、通过时间反向传播及相关训练难题提供了基础。
