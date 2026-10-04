# 门控循环单元 (GRU) 架构

来源：[原文](https://apxml.com/zh/courses/foundations-transformers-architecture/chapter-1-revisiting-sequence-modeling-limitations/gru-architecture)

[返回章节目录](README.md) · [返回课程目录](../README.md)

门控循环单元 (GRU)，由 Cho 等人于 2014 年提出，提供了一种门控循环架构中的替代方案。这些单元与长短期记忆 (LSTM) 网络类似，旨在缓解简单循环神经网络 (neural network) (RNN) 中固有的梯度问题。GRU 的目标是实现类似的效果，即控制信息随时间流动，但与 LSTM 相比，它通过一种略微简化的结构来实现。这种简化通常会减少参数 (parameter)，并可能加快计算速度，同时其性能常与 LSTM 相当。

### GRU 门和状态更新

GRU 单元不使用独立的细胞状态，而是直接通过两个主要门控机制来修改其隐藏状态 $h_t$：重置门和更新门。我们来分析它们的作用和计算方式。

#### 重置门 ($r_t$)

重置门决定了在提出新的候选隐藏状态时，前一个隐藏状态 $h_{t-1}$ 有多少应该被有效地“遗忘”或忽略。如果重置门输出接近 0 的值，它允许单元丢弃过去被认为与当前计算无关的信息。反之，接近 1 的值则保留了前一状态的大部分信息。

计算涉及当前输入 $x_t$ 和前一个隐藏状态 $h_{t-1}$。一个 Sigmoid 函数 $\sigma$ 将输出压缩到 [0, 1] 范围：


$$
r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r)
$$


这里，$W_r$、$U_r$ 和 $b_r$ 是针对重置门学习到的权重 (weight)矩阵和偏置 (bias)向量 (vector)。

#### 更新门 ($z_t$)

更新门的作用类似于 LSTM 中遗忘门和输入门的组合。它决定了前一个隐藏状态 $h_{t-1}$ 有多少信息应该传递到新的隐藏状态 $h_t$。同时，它也控制着新计算出的*候选*隐藏状态 $\tilde{h}_t$ 有多少应该被纳入。

它的计算结构与重置门相似：


$$
z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z)
$$


$W_z$、$U_z$ 和 $b_z$ 是用于更新门学习到的参数 (parameter)。

#### 候选隐藏状态 ($\tilde{h}_t$)

候选隐藏状态表示在时间步 $t$ 新隐藏状态的一个提议。它的计算受重置门 $r_t$ 的影响。具体来说，前一个隐藏状态 $h_{t-1}$ 的贡献通过重置门的输出进行调节（按元素相乘，表示为 $\odot$），然后与处理后的输入 $x_t$ 结合。通常使用双曲正切函数 ($\tanh$) 作为激活函数 (activation function)：


$$
\tilde{h}_t = \tanh(W_h x_t + U_h (r_t \odot h_{t-1}) + b_h)
$$


$W_h$、$U_h$ 和 $b_h$ 是用于计算候选状态的学习参数。按元素相乘 $r_t \odot h_{t-1}$ 是允许 GRU 根据 $r_t$ 选择性地丢弃前一状态部分内容的机制。

#### 最终隐藏状态 ($h_t$)

当前时间步的最终隐藏状态 $h_t$ 通过对前一个隐藏状态 $h_{t-1}$ 和候选隐藏状态 $\tilde{h}_t$ 进行线性插值计算得出。更新门 $z_t$ 决定了这种插值的平衡。


$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$


当 $z_t$ 接近 1 时，候选状态 $\tilde{h}_t$ 贡献更多，有效地用新信息更新隐藏状态。当 $z_t$ 接近 0 时，前一状态 $h_{t-1}$ 大部分被保留，允许信息远距离传递。

> 门控循环单元 (GRU) 单元在时间步 $t$ 内的数据流。$x_t$ 是输入，$h_{t-1}$ 是前一个隐藏状态。重置门 ($r_t$) 和更新门 ($z_t$) 控制着候选状态 ($ilde{h}_t$) 和最终隐藏状态 ($h_t$) 的计算。虚线表示某个值在计算中的使用。

### GRU 与 LSTM 对比

GRU 架构可被视为 LSTM 的一种简化：

1. **门：** GRU 使用两个门（重置、更新），而 LSTM 使用三个（输入、遗忘、输出）。
2. **细胞状态：** GRU 没有 LSTM 中独立的细胞状态 ($c_t$)。隐藏状态 $h_t$ 直接包含了控制长期依赖的机制。
3. **参数 (parameter)数量：** 由于门的数量减少且没有细胞状态路径，GRU 通常比具有相同隐藏状态维度的 LSTM 具有更少的训练参数。这使得它们在计算上可能略便宜，并且在较小数据集上可能更不容易过拟合 (overfitting)。

从经验来看，两种架构在所有任务上都没有哪一个持续表现更优。LSTM 和 GRU 之间的选择通常取决于特定问题的实验结果，尽管当计算资源或参数效率是主要考量时，GRU 可能更受青睐。

### 依然存在的局限性

尽管有精密的门控，GRU 仍保留了循环模型的基本特点：顺序计算。信息必须按序列长度一步步传播。这种固有的顺序性限制了训练时的并行化，使得它们在处理非常长的序列时，与 Transformer 等架构相比训练速度明显更慢。此外，尽管它们在捕捉更长距离的依赖关系方面远优于简单的 RNN，但依赖于将过去信息总结到一个固定大小的隐藏状态中，对于复杂依赖关系跨越很长的距离的极长序列，这仍然可能成为瓶颈。这些尚存的挑战促使了基于注意力的机制出现，我们接下来将讨论这些机制。

## 参考资料

- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://aclanthology.org/D14-1179.pdf) — Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio (2014)
  Journal: Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP); Pages: 1724-1734; DOI: [10.3115/v1/D14-1179](https://doi.org/10.3115/v1/D14-1179)
  这篇基础论文介绍了门控循环单元（GRU）架构，作为编码器-解码器框架内序列建模的有效替代方案。
- [Deep Learning (Chapter 10: Sequence Modeling: Recurrent and Recursive Networks)](https://www.deeplearningbook.org/contents/rnn.html) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: 367-418
  这本基础教材的第10章详细解释了循环神经网络，包括LSTM和GRU的详细工作原理。
- [CS224n: Natural Language Processing with Deep Learning - Lecture Notes (Recurrent Neural Networks, LSTMs and GRUs)](http://web.stanford.edu/class/cs224n/) — Christopher Manning, John Hewitt (2023)
  Publisher: Stanford University
  这些来自知名大学课程的讲义提供了循环神经网络的学术讨论，涵盖了自然语言处理中LSTM和GRU的结构和功能。

---

[上一节](03-%E9%95%BF%E7%9F%AD%E6%9C%9F%E8%AE%B0%E5%BF%86%EF%BC%88LSTM%EF%BC%89%E9%97%A8%E6%8E%A7%E6%9C%BA%E5%88%B6.md) · [下一节](05-%E8%BF%9C%E8%B7%9D%E7%A6%BB%E4%BE%9D%E8%B5%96%E7%9A%84%E6%8C%91%E6%88%98.md)
