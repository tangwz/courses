---
course: "rnns-and-sequence-modeling"
chapter: "gated-recurrent-units-gru"
lesson: "gru-final-hidden-state"
sourceId: 2590
sourceUrl: "https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-6-gated-recurrent-units-gru/gru-final-hidden-state"
title: "计算最终隐藏状态"
description: "显示如何使用更新门计算最终隐藏状态。"
order: 6
plots: []
sourceHash: "386fbe51a8e08553d9cdf8af1f14830830d20b77971f8f306db149b25ea30640"
sourceCorrections: []
---

给定时间步 $t$ 的 GRU 单元内，计算实际的隐藏状态 $h_t$ 是最后一步。这个计算整合了更新门 ($z_t$)、重置门 ($r_t$) 和候选隐藏状态 ($ilde{h}_t$)。这个状态表示 GRU 将传递到下一个时间步的信息。

最终隐藏状态 $h_t$ 的计算中，更新门 $z_t$ 起着核心作用。回想一下，$z_t$ 决定了之前隐藏状态 $h_{t-1}$ 应保留多少，以及新的候选状态 $\tilde{h}_t$ 应包含多少。GRU 通过直接插值机制来实现这一点。

最终隐藏状态 $h_t$ 的公式是：


$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$


我们来分析这个公式：

1. **$(1 - z_t) \odot h_{t-1}$**: 此项计算应保留*之前*隐藏状态 $h_{t-1}$ 的多少。更新门 $z_t$ 的值介于 0 到 1 之间（因为使用了 Sigmoid 激活函数 (activation function)）。如果 $z_t$ 的某个值接近 0，则 $(1 - z_t)$ 中的对应值将接近 1，这意味着之前状态 $h_{t-1}$ 中的对应元素在很大程度上得以保留。反之，如果 $z_t$ 的某个值接近 1，则 $(1 - z_t)$ 中的对应值接近 0，从而有效地遗忘之前状态的该部分。符号 $\odot$ 表示元素级乘法（哈达玛积）。
2. **$z_t \odot \tilde{h}_t$**: 此项计算应包含*候选*隐藏状态 $\tilde{h}_t$ 的多少。如果 $z_t$ 的某个值接近 1，则候选状态 $\tilde{h}_t$ 中的对应元素（包含为此时间步提议的新信息，受重置门影响）会被强烈包含。如果 $z_t$ 的某个值接近 0，则候选状态中的对应元素大部分会被忽略。
3. **加法 (+)**: 两个所得向量 (vector)进行元素级相加。这个加法完成了插值操作。最终隐藏状态 $h_t$ 的每个元素都是之前状态 $h_{t-1}$ 和候选状态 $\tilde{h}_t$ 中对应元素的加权和，权重 (weight)由更新门 $z_t$ 控制。

这种机制使得 GRU 单元能够动态调整从过去流向的信息量以及在每个时间步引入的新信息量。如果更新门认为之前状态仍然相关（$z_t$ 值较低），它可以使其在很大程度上保持不变地通过。如果它认定新信息更重要（$z_t$ 值较高），它会更多地包含候选状态。这种门控机制比 LSTM 的独立单元状态和输出门更简单，但提供了一种有效的方式来管理信息流并解决梯度消失问题。

以下图表说明了之前状态、候选状态和更新门如何组合形成最终隐藏状态：

> 最终隐藏状态 $h_t$ 是通过对之前隐藏状态 $h_{t-1}$ 和候选隐藏状态 $\tilde{h}_t$ 进行插值计算而得出的。更新门 $z_t$ 控制着这种插值的平衡。

所得向量 $h_t$ 有两个作用：它是当前时间步 $t$ 的 GRU 单元的输出（通常传递给后续层或用于预测），并且它成为*下一个*时间步 $t+1$ 的“之前隐藏状态” $h_{t-1}$。这种循环连接使得 GRU 能够逐步处理序列，根据其遇到的序列元素来维护和更新其内部状态。

## 参考资料

- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) — Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio (2014)
  Journal: EMNLP 2014; Pages: 1724-1734; DOI: [10.48550/arXiv.1406.1078](https://doi.org/10.48550/arXiv.1406.1078)
  介绍门控循环单元（GRU）架构的开创性论文，详细阐述了其组件，包括更新门和最终隐藏状态计算。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: 394-396
  一本全面的教科书，对循环神经网络进行了学术论述，其中专门一节解释了 GRU 的结构和门控机制，用于管理信息流。
- [Dive into Deep Learning](https://d2l.ai/chapter_recurrent-neural-networks/gru.html) — Aston Zhang, Zachary C. Lipton, Mu Li, and Alex Smola (2024)
  Publisher: Cambridge University Press
  一本交互式在线教材，提供了清晰易懂的 GRU 解释，包括隐藏状态计算的数学公式，通常辅以代码示例。
