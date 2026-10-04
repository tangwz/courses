---
course: "rnns-and-sequence-modeling"
chapter: "gated-recurrent-units-gru"
lesson: "gru-computational-efficiency"
sourceId: 2594
sourceUrl: "https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-6-gated-recurrent-units-gru/gru-computational-efficiency"
title: "计算效率考量"
description: "分析GRU与LSTM相比因门控数量较少而带来的计算成本。"
order: 8
plots: ["plots/2594-0.json"]
sourceHash: "9ed32e57db90d92c1eacfc8df6bc771b1188feb5b2ad795dd3386d6d592f3fbd"
sourceCorrections: []
---

开发门控循环单元（GRU）的一个原因是为了构建一种门控循环单元，它能够像LSTM一样处理长距离依赖，但结构更简单，因此计算效率更高。这种效率主要源于GRU相对于LSTM单元的复杂度降低。造成这种差异的因素得到分析。

### 参数 (parameter)数量更少

影响GRU效率的最直接原因在于它与相同隐藏状态大小的LSTM单元相比，参数数量更少。我们回顾一下它们的结构：

- **LSTM：** 有三个独立的门（遗忘门 $f_t$、输入门 $i_t$、输出门 $o_t$）和一个候选细胞状态计算（$\tilde{C}_t$）。它们中的每一个通常都包含针对输入 $x_t$ 和前一个隐藏状态 $h_{t-1}$ 的独立权重 (weight)矩阵和偏置 (bias)向量 (vector)。这导致这些变换有四组参数。
- **GRU：** 只有两个门（重置门 $r_t$、更新门 $z_t$）。候选隐藏状态 $\tilde{h}_t$ 的计算使用重置门，最终隐藏状态 $h_t$ 的计算使用更新门。这种结构仅需三组参数进行这些变换。

设 $d$ 为输入特征维度，$h$ 为隐藏状态维度。

对于一个LSTM单元，近似的参数（权重和偏置）数量是：
$\text{参数数量}_{\text{LSTM}} \approx 4 \times (h \times (d + h) + h)$
“4”来自于四种变换（输入门、遗忘门、输出门、候选细胞状态）。每种变换都涉及将连接后的输入 $x_t$（维度 $d$）和前一个隐藏状态 $h_{t-1}$（维度 $h$）映射到隐藏维度 $h$ 的权重，再加上一个维度为 $h$ 的偏置向量。

对于一个GRU单元，近似的参数数量是：
$\text{参数数量}_{\text{GRU}} \approx 3 \times (h \times (d + h) + h)$
“3”来自于三种变换（重置门、更新门、候选隐藏状态）。结构相似，但与LSTM相比少了一种与门控相关的变换。

这意味着一个GRU单元通常比相同隐藏状态大小的LSTM单元少大约 **25% 的参数**。



![每个单元的相对参数数量](plots/2594-0.json)



> 比较LSTM和GRU单元中核心变换所涉及的近似参数组数量。

### 计算负荷降低

参数 (parameter)数量更少直接意味着每个时间步所需的计算量更少。LSTM和GRU的主要运算都涉及矩阵乘法（输入/隐藏状态与权重 (weight)矩阵之间）和元素级操作（用于门控激活和状态更新）。

由于GRU每个时间步执行三次主要矩阵乘法运算，而LSTM是四次，因此它需要的浮点运算（FLOPs）更少。这种减少适用于前向传播（计算隐藏状态）和反向传播 (backpropagation)（通过时间反向传播在训练期间计算梯度）。

尽管确切的速度提升取决于特定的硬件、软件实现（如cuDNN优化）和模型维度，GRU通常在每个时间步执行得更快。

### 对训练和推理 (inference)的影响

- **训练速度：** 每个前向和反向传播 (backpropagation)的计算量更少，训练GRU网络可能比训练同等LSTM网络更快，特别是对于大型数据集或循环层计算成本较高的复杂模型。这意味着你可能在更少的实际时间内达到收敛，或者在给定时间内运行更多训练周期。
- **推理速度：** 降低的计算负荷在推理时（使用训练好的模型进行预测时）也很有益。更快的预测通常是人们所希望的，特别是在实时应用中，如即时翻译或响应式用户界面。
- **内存占用：** 参数 (parameter)更少的模型占用的内存也更少。这在训练时（可能允许在内存受限的GPU上使用更大的批次大小或模型）和部署时都很有益，特别是在内存资源通常受限的边缘设备或手机上。

### 效率与性能

重要的是要记住，计算效率只是模型选择的一个方面。尽管GRU通常比LSTM更快、更轻量，但LSTM凭借其独立的细胞状态和输出门，有时在需要建模特别复杂或长距离依赖的任务上能提供稍好的性能。GRU和LSTM之间的选择通常需要进行实际评估，权衡计算资源需求与期望的预测性能，我们将在下一节中进一步讨论。

## 参考资料

- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) — Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio (2014)
  Journal: Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP); Pages: 1724-1734; DOI: [10.48550/arXiv.1406.1078](https://doi.org/10.48550/arXiv.1406.1078)
  介绍门控循环单元 (GRU) 作为 LSTM 的简化替代方案，重点介绍其架构和在序列建模中的早期应用。
- [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735) — Sepp Hochreiter and Jürgen Schmidhuber (1997)
  Journal: Neural Computation; Publisher: MIT Press; Volume: 9; Pages: 1735-1780; DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
  首次提出长短期记忆 (LSTM) 架构的论文，GRU 是对其的简化。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本标准教材，详细阐述了循环神经网络的理论和应用，包括 LSTM 和 GRU 的比较及其计算特性。
