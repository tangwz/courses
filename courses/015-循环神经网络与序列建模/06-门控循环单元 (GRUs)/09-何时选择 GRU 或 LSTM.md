# 何时选择 GRU 或 LSTM

来源：[原文](https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-6-gated-recurrent-units-gru/when-to-choose-gru-lstm)

[返回章节目录](README.md) · [返回课程目录](../README.md)

LSTM 和 GRU 都旨在解决梯度消失问题并捕获长期依赖关系，这是对简单 RNN 的很大进步。它们通过使用调节信息流动的门控机制来实现这一点。但是，你如何决定为你的特定序列建模任务使用哪一种呢？没有一种答案适用于所有情况，但理解它们的不同之处可以指导你的决策。

### 架构区别再谈

我们快速回顾一下主要的结构区别：

- **LSTM（长短期记忆）：** 使用三个不同的门（遗忘门、输入门、输出门），并在隐藏状态 ($h_t$) 之外保持一个独立的单元状态 ($c_t$)。这种分离使得单元状态可以作为一个更受保护的记忆通道，可能更有效地在更长时间内保存信息。
- **GRU（门控循环单元 (GRU)）：** 使用两个门（重置门、更新门），并将单元状态和隐藏状态组合成一个单一的状态向量 (vector) ($h_t$)。更新门 ($z_t$) 承担与 LSTM 中遗忘门和输入门相似的职责，决定从前一状态中遗忘什么以及添加什么新信息。重置门 ($r_t$) 确定在提出新的候选状态时忽略多少前一状态。

GRU 的这种结构简化对复杂性和计算有直接影响。

> LSTM 和 GRU 单元的内部复杂度和状态管理的比较。

### 计算效率和模型大小

源于架构的最直接区别是计算开销。

- **参数 (parameter)更少：** GRU 具有更少的权重 (weight)矩阵和偏置 (bias)，因为它们有更少的门并组合了单元/隐藏状态。对于给定的隐藏维度大小，一个 GRU 层将具有大约是等效 LSTM 层可训练参数数量的三分之二。这使得 GRU 模型本身更小。
- **计算更快：** 由于每个时间步的计算量更少（主要是因为只有两个门而不是三个），GRU 通常比 LSTM 训练更快，推理 (inference)也更迅速，前提是隐藏层大小和数据集相同。这种区别会很明显，尤其是在深度网络或非常长的序列中。

如果你需要快速迭代模型设计，在计算能力受限的设备（如手机）上部署模型，或仔细管理内存使用，GRU 的效率使其成为一个有吸引力的选择。

### 性能和泛化能力

更简单的 GRU 架构是否会导致更差的性能？通常情况下，答案是否定的。

- **经验性能：** 许多研究和实际应用已经在语言建模、机器翻译、时间序列分析和情感分类等各种任务中比较了 LSTM 和 GRU。结果经常显示出可比的性能。两种架构都没有在所有可能的问题上持续超越对方。有时 LSTM 取得稍微好一些的分数，有时 GRU 也是，而且这种差异通常很小，并很大程度上取决于特定的数据集、任务和超参数 (parameter) (hyperparameter)调整。
- **数据量考量：** 一个常见的经验法则表明，GRU 参数较少，可能更不容易过拟合 (overfitting)，尤其是在较小的数据集上。理论上，在这种情况下它们的泛化能力可能稍微好一点，因为需要拟合训练数据中噪声的参数更少。相反，LSTM 拥有更多参数和可能更复杂的动态（独立的单元状态），在数据充足时，可能有更高的能力来建模复杂模式。然而，请将其视为指导，而非严格规定。适当的正则化 (regularization)技术（如 dropout 和循环 dropout）对管理两种架构的过拟合都很重要，无论数据集大小如何。

### 如何选择：实际考量

考虑到这些权衡，你如何为你的项目选择 LSTM 和 GRU 呢？这里有一个实用的决策过程：

1. **从默认选项开始（或许是 LSTM）：** LSTM 历史更长，并且得到了广泛的文档记录和研究。它们的单元状态明确分离，可能为某些非常复杂的序列动态提供稍微更多的灵活性或表达能力。如果你没有关于计算或数据大小的严格限制，从 LSTM 开始是一个完全合理的默认选择。
2. **优先考虑速度或效率？（选择 GRU）：** 如果训练时间、推理 (inference)速度、计算开销或模型内存占用对你的项目来说是重要限制，GRU 通常是更好的第一个选择。它更简单的结构会带来显著的效率提升。
3. **数据量有限？（考虑 GRU）：** 如果你的数据集相对较小，GRU 减少的参数 (parameter)数量可能在泛化方面提供优势。与相同隐藏大小的 LSTM 相比，它可能更不容易过拟合 (overfitting)，并可能带来更稳定的模型。
4. **需要最大表达能力来处理非常复杂的任务？（考虑 LSTM）：** 如果任务涉及极长的依赖关系或高度复杂的时序模式，且计算资源不是主要瓶颈，LSTM 额外的门控复杂性和独立的单元状态记忆*可能*在建模能力上提供轻微优势。
5. **拿不准时，就去尝试：** 最可靠的选择方法通常是通过经验评估。训练一个基于 LSTM 的模型和一个基于 GRU 的模型。最初保持其他因素一致（例如，隐藏单元数量、层数、优化器、学习率）。比较它们在你的验证集上的性能指标（准确率、损失、困惑度等）。选择在你的特定任务和数据上表现更好的架构。请记住，仔细的超参数 (hyperparameter)调整对于获得两种架构的最佳性能非常重要。



[交互图表：权衡：LSTM 与 GRU](https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-6-gated-recurrent-units-gru/when-to-choose-gru-lstm#plot-1q2rvyw)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": {
      "text": "权衡：LSTM 与 GRU",
      "font": {
        "size": 12
      }
    },
    "xaxis": {
      "title": "模型复杂性 / 参数",
      "tickvals": [
        0.66,
        1
      ],
      "ticktext": [
        "较低 (GRU)",
        "较高 (LSTM)"
      ]
    },
    "yaxis": {
      "title": "潜在表达能力 / 数据需求",
      "tickvals": [
        1,
        1.5
      ],
      "ticktext": [
        "较低 (GRU)",
        "较高 (LSTM)"
      ]
    },
    "legend": {
      "orientation": "h",
      "yanchor": "bottom",
      "y": -0.35,
      "xanchor": "center",
      "x": 0.5
    },
    "font": {
      "family": "sans-serif",
      "size": 10
    },
    "margin": {
      "l": 50,
      "r": 20,
      "t": 40,
      "b": 80
    },
    "paper_bgcolor": "#f8f9fa",
    "plot_bgcolor": "#f8f9fa"
  },
  "data": [
    {
      "x": [
        1
      ],
      "y": [
        1.5
      ],
      "mode": "markers+text",
      "marker": {
        "size": 25,
        "color": "#748ffc"
      },
      "text": [
        "LSTM"
      ],
      "textposition": "middle center",
      "name": "LSTM"
    },
    {
      "x": [
        0.66
      ],
      "y": [
        1
      ],
      "mode": "markers+text",
      "marker": {
        "size": 25,
        "color": "#69db7c"
      },
      "text": [
        "GRU"
      ],
      "textposition": "middle center",
      "name": "GRU"
    }
  ]
}
```

</details>



> GRU 通常比 LSTM 复杂性较低（参数更少）且计算开销更低。LSTM 由于其独立的单元状态，可能提供更高的表达能力，这在更大的数据集或更复杂的模式下可能更有益。

最终，LSTM 和 GRU 都代表着对简单 RNN 的很大进步，并且是你的序列建模工具集中的强大工具。理解它们的架构权衡有助于做出明智的初步选择，但经验验证通常仍然是确定你的特定序列建模问题和限制的最佳架构的最佳方法。

## 参考资料

- [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735) — Sepp Hochreiter, Jürgen Schmidhuber (1997)
  Journal: Neural Computation; Publisher: Massachusetts Institute of Technology; Volume: 9; Pages: 1735-1780; DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
  介绍长短期记忆 (LSTM) 网络架构的原始论文，详细说明了其门控和单元状态如何解决梯度消失问题。
- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://aclanthology.org/D14-1179.pdf) — Kyunghyun Cho, Bart van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio (2014)
  Journal: Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP); Publisher: Association for Computational Linguistics; Pages: 1724-1734; DOI: [10.3115/v1/D14-1179](https://doi.org/10.3115/v1/D14-1179)
  本文介绍了门控循环单元（GRU），作为LSTM的简化替代方案，概述了其带有重置门和更新门的架构。
- [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](https://arxiv.org/abs/1412.3555) — Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, Yoshua Bengio (2014)
  Journal: NIPS 2014 Deep Learning and Representation Learning Workshop; DOI: [10.48550/arXiv.1412.3555](https://doi.org/10.48550/arXiv.1412.3555)
  对GRU、LSTM和传统RNN在各种序列建模任务上的性能和效率进行了实证比较，提供了相关见解。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  一本综合性教科书，其中有一个详细的章节（第10章）介绍序列建模，包括对循环神经网络、LSTM和GRU的解释。

---

[上一节](08-%E8%AE%A1%E7%AE%97%E6%95%88%E7%8E%87%E8%80%83%E9%87%8F.md) · [下一节](../07-%E5%AE%9E%E7%8E%B0%20LSTM%20%E5%92%8C%20GRU/01-%E5%9C%A8%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A1%86%E6%9E%B6%E4%B8%AD%E4%BD%BF%E7%94%A8LSTM%E5%B1%82.md)
