---
course: "rnns-and-sequence-modeling"
chapter: "rnn-training-challenges"
lesson: "impact-long-range-dependencies"
sourceId: 2556
sourceUrl: "https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-4-rnn-training-challenges/impact-long-range-dependencies"
title: "长期依赖学习的影响"
description: "讨论梯度消失/爆炸如何阻止循环神经网络有效捕获长期模式"
order: 3
plots: []
sourceHash: "cc2faccdeafdd0e3bfdcfac9df50c7aed2af986b5e7ba66577eb13091a65d1b8"
sourceCorrections: []
---

循环神经网络 (neural network) (RNN) (Recurrent Neural Networks) 通过时间反向传播 (backpropagation) (BPTT) 进行训练，这实际上是将网络展开并应用标准的反向传播算法。其主要原理是计算网络误差相对于其权重 (weight)的变化量，从而能够使用梯度下降 (gradient descent)来调整这些权重以提升性能。梯度代表着指导学习过程的信号。

然而，这种学习过程的有效性取决于梯度信号的质量。在处理序列，特别是长序列时，梯度信号必须反向传播经过许多时间步。这正是梯度消失和梯度爆炸问题变得特别有害的地方，直接影响了循环神经网络学习序列中相距较远事件之间联系的能力，即所谓的长期依赖。

### 梯度消失的难题

试想学习一个长段落的第一个词与其表达的最终情感之间的联系。BPTT需要将误差信号从序列的末端一直传播回起始处。

梯度消失问题产生的原因是，在BPTT过程中，梯度在反向传播 (backpropagation)的每一步都会重复乘以循环权重 (weight)矩阵 $W_{hh}$ (并经过激活函数 (activation function)导数的处理)。如果这些矩阵中（或其导数中）的相关值持续小于1，梯度信号在时间反向传播时会呈指数级缩小。

设损失 $L$ 在时间步 $T$ 的梯度，相对于远早的时间步 $t$ 处的隐藏状态 $h_t$ (其中 $t \ll T$)。这涉及到雅可比矩阵的乘积：


$$
\frac{\partial L_T}{\partial h_t} = \frac{\partial L_T}{\partial h_T} \frac{\partial h_T}{\partial h_{T-1}} \frac{\partial h_{T-1}}{\partial h_{T-2}} \dots \frac{\partial h_{t+1}}{\partial h_t}
$$


如果雅可比矩阵的范数 $\frac{\partial h_{k}}{\partial h_{k-1}}$ 持续较小 (例如，$< 1$)，整体梯度 $\frac{\partial L_T}{\partial h_t}$ 将随着时间距离 $T-t$ 的增加而非常迅速地趋近于零。

> 梯度信号在反向传播过程中迅速衰减，随着它反向传播经过时间步。

这在实践中意味着什么？与早期时间步相关的梯度变得非常小。当优化器试图根据这些微小的梯度更新网络权重时，变化可以忽略不计。


$$
W_{new} = W_{old} - \eta \nabla_W L
$$


如果对于与遥远过去输入相关的连接，梯度 $\nabla_W L$ 几乎为零，那些权重 ($W_{old}$) 基本上从未得到更新。网络表现得好像患有健忘症，无法学习跨越几个时间步的关联或依赖。这极大地限制了它在需要理解长上下文 (context)的任务中的实用性，例如文档分类、机器翻译或长期预测。

### 梯度爆炸的危害

反之，梯度爆炸问题发生于反向传播 (backpropagation)链中雅可比矩阵的范数持续大于1时。在这种情况下，梯度信号在反向传播时呈指数级增长。


$$
\frac{\partial L_T}{\partial h_t} = \frac{\partial L_T}{\partial h_T} \underbrace{\frac{\partial h_T}{\partial h_{T-1}} \frac{\partial h_{T-1}}{\partial h_{T-2}} \dots \frac{\partial h_{t+1}}{\partial h_t}}_{\text{乘积可能变得非常大}}
$$


大梯度会导致训练过程中权重 (weight)的大幅更新。这可能导致优化过程变得不稳定：

1. **震荡：** 权重可能在参数 (parameter)空间中剧烈波动，无法收敛到损失函数 (loss function)的良好最小值。
2. **数值溢出：** 梯度值可能变得非常大，超出标准浮点数的表示能力，导致出现 `NaN` (非数字) 值。这会有效地导致训练过程崩溃。

> 梯度信号在反向传播过程中失控增长，导致不稳定。

即使训练没有因为NaN而完全停止，不稳定的更新也使得网络难以学习细粒度的依赖，包括长期依赖。学习过程过于混乱，无法可靠地收敛到能够捕获复杂时间模式的解。

### 结果：序列理解受损

梯度消失和梯度爆炸，虽然在数值表现上相反，却有一个共同的有害结果：它们阻止了简单循环神经网络 (neural network) (RNN)有效学习长期依赖。梯度消失使网络对遥远的过去视而不见，而梯度爆炸则使学习过程过于不稳定，无法捕获一致的长期模式。

"这种局限性很重要，因为许多序列建模任务都严重依赖于对长期上下文 (context)的理解。标准循环神经网络架构在其循环特性理论上应提供最大优势的地方遇到了困难。了解这些难题是体会为解决这些问题而开发的方案的第一步，例如梯度裁剪（用于梯度爆炸）以及更复杂的循环单元，如长短期记忆网络 (LSTM) (LSTMs) 和门控循环单元 (GRU) (GRUs)（主要用于梯度消失），这些我们接下来将进行探讨。"

## 参考资料

- [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735) — Sepp Hochreiter, Jürgen Schmidhuber (1997)
  Journal: Neural Computation; Publisher: MIT Press; Volume: 9; Pages: 1735-1780; DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
  提出了长短期记忆（LSTM）架构，这是旨在克服梯度消失问题并有效学习循环神经网络中长期依赖的重大进展。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  第10章详细介绍了循环神经网络的理论和实践，包括对梯度消失和梯度爆炸以及各种解决方案的全面解释。
- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://aclanthology.org/D14-1179/) — Kyunghyun Cho, Bart van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio (2014)
  Journal: Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP); Publisher: Association for Computational Linguistics; Pages: 1724–1734; DOI: [10.3115/v1/D14-1179](https://doi.org/10.3115/v1/D14-1179)
  这篇论文提出了门控循环单元（GRU）作为LSTM在序列建模中的高效替代方案，尤其解决了梯度消失等问题。
