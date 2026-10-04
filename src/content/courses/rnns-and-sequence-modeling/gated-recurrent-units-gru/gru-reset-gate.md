---
course: "rnns-and-sequence-modeling"
chapter: "gated-recurrent-units-gru"
lesson: "gru-reset-gate"
sourceId: 2586
sourceUrl: "https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-6-gated-recurrent-units-gru/gru-reset-gate"
title: "重置门"
description: "说明重置门在决定遗忘多少过去信息方面的作用。"
order: 4
plots: []
sourceHash: "01c061a62123b3f8d0c55abe1bbb9348ba6e3de0ef91ca76548ec42f567db55c"
sourceCorrections: []
---

**重置门**在门控循环单元 (GRU) 中扮演着一个具体而重要的角色，用于管理信息流动。它决定在计算*新的*候选隐藏状态 ($ilde{h}_t$) 时，应忽略或“重置”多少由前一个隐藏状态 ($h_{t-1}$) 携带的*过去*信息。这个门可以看作是一个过滤器，用于决定过去背景信息对提议更新的记忆状态的关联度。

### 计算重置门激活值

像更新门一样，重置门的激活值，记为 $r_t$，是根据当前输入 $x_t$ 和前一个隐藏状态 $h_{t-1}$ 计算的。它使用 sigmoid 激活函数 (activation function)，确保其输出值在 0 到 1 之间。

该计算涉及学习单独的权重 (weight)矩阵 ($W_{xr}$ 和 $W_{hr}$) 和一个偏置 (bias)项 ($b_r$)：


$$
r_t = \sigma(W_{xr} x_t + W_{hr} h_{t-1} + b_r)
$$


这里：

- $x_t$ 是当前时间步 $t$ 的输入向量 (vector)。
- $h_{t-1}$ 是前一时间步 $t-1$ 的隐藏状态向量。
- $W_{xr}$ 是连接输入到重置门的权重矩阵。
- $W_{hr}$ 是连接前一个隐藏状态到重置门的权重矩阵。
- $b_r$ 是重置门的偏置项。
- $\sigma$ 代表 sigmoid 激活函数，$\sigma(z) = 1 / (1 + e^{-z})$。

输出 $r_t$ 是一个与隐藏状态维度相同的向量。 $r_t$ 中的每个元素对应隐藏状态的一个维度，充当该特定维度的门控值。

### 重置门如何调整信息

重置门向量 (vector) $r_t$ 中的值直接控制着在计算候选隐藏状态 $\tilde{h}_t$ 时前一个隐藏状态 $h_{t-1}$ 的影响。 $r_t$ 中某个特定维度接近 0 的值会有效地“重置”或抵消 $h_{t-1}$ 中相应维度的贡献。相反，接近 1 的值则允许前一个隐藏状态的该部分基本不变地通过。

这种机制通过重置门 $r_t$ 与前一个隐藏状态 $h_{t-1}$ 之间的逐元素乘法 ($\odot$) 实现。这种被调整过的先前状态随后用于计算候选隐藏状态：


$$
\tilde{h}_t = \tanh(W_{xh} x_t + W_{hh} (r_t \odot h_{t-1}) + b_h)
$$


请注意 $r_t \odot h_{t-1}$ 如何准确地决定前一个状态 $h_{t-1}$ 的哪些部分与当前输入 $x_t$ 结合以形成候选状态 $\tilde{h}_t$。如果 $r_t$ 中的一个元素为 0，则 $h_{t-1}$ 中对应的元素在 $\tanh$ 函数内部的加权和之前被有效地清零。

> 此图展示了重置门 $r_t$ 的计算过程，及其与前一个隐藏状态 $h_{t-1}$ 的逐元素乘法 ($\odot$) 如何影响候选隐藏状态 $\tilde{h}_t$。

### 重置门的意义

重置门赋予 GRU 单元动态调整提议的新状态 ($\tilde{h}_t$) 对紧邻的过去状态 ($h_{t-1}$) 依赖程度的能力。如果当前输入 $x_t$ 表明与 $h_{t-1}$ 中编码的内容相比，背景或主题发生了显著变化，重置门可以学习激活接近 0。这有效地让单元在计算候选状态时“重新开始”，更多地侧重于当前输入 $x_t$，而不是将其与可能不相关的过去信息混合。

例如，在语言建模中，如果网络遇到句子的结尾（可能由 $x_t$ 中的标点符号表示），重置门可能会强烈激活（值接近 0），以减少前一个句子的隐藏状态在计算下一个句子开头候选状态时的影响。

总之，重置门充当一个控制器，在计算候选隐藏状态之前选择性地减弱前一个隐藏状态的某些部分。这使得 GRU 能够有效地忘记对紧邻下一步被认为不相关的信息，有助于其处理时间依赖的能力。

## 参考资料

- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://aclanthology.org/D14-1179/) — Kyunghyun Cho, Bart van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio (2014)
  Journal: Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP); Publisher: Association for Computational Linguistics; Pages: 1724-1734; DOI: [10.3115/v1/D14-1179](https://doi.org/10.3115/v1/D14-1179)
  本文介绍了门控循环单元（GRU），并详细说明了其门控机制，包括重置门的功能。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  提供了循环神经网络（包括GRU及其门控机制）的详细说明。
- [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](https://arxiv.org/abs/1412.3555) — Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, and Yoshua Bengio (2014)
  Journal: arXiv preprint arXiv:1412.3555; DOI: [10.48550/arXiv.1412.3555](https://doi.org/10.48550/arXiv.1412.3555)
  本文对GRU和LSTM进行了实证评估，并进一步描述了GRU架构，增强了对其门控机制的理解。
- [Sequence Models (Course 5 of Deep Learning Specialization)](https://www.coursera.org/learn/nlp-sequence-models) — Andrew Ng, Kian Katanforoosh, Younes Bensouda Mourri (2017)
  Publisher: DeepLearning.AI
  对GRU提供了直观实用的解释，阐明了重置门在序列建模中的作用。
