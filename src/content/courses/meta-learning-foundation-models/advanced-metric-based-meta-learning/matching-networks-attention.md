---
course: "meta-learning-foundation-models"
chapter: "advanced-metric-based-meta-learning"
lesson: "matching-networks-attention"
sourceId: 4328
sourceUrl: "https://apxml.com/zh/courses/meta-learning-foundation-models/chapter-3-advanced-metric-based-meta-learning/matching-networks-attention"
title: "带有注意力机制的匹配网络"
description: "将注意力机制整合到匹配网络中，以改进上下文建模。"
order: 3
plots: []
sourceHash: "b75b77b74e361afc8a0bbdfac23a955a868350a0d8bfe6d203bb2632b27b5391"
sourceCorrections: []
---

匹配网络在基于度量的元学习中提供了一种独特的方法，将少样本预测视为在学习到的嵌入 (embedding)空间中的一种加权最近邻形式。虽然一些方法计算固定类别原型，但匹配网络直接将查询（测试）样本 $x'$ 与支持集 $S = \{(x_j, y_j)\}_{j=1}^{k \times N}$ 中的每个可用支持样本 $(x_i, y_i)$ 进行比较，用于N类k样本任务。对 $x'$ 的预测是支持集标签 $y_i$ 的加权和，权重 (weight)则反映了查询样本和每个支持样本之间的相似性。

最初的公式通常依赖于由嵌入函数 $f_\phi$ 生成的查询和支持样本嵌入之间的简单余弦相似性。这种方法虽然有效，但它假设一个固定的相似性足以应对所有比较。然而，对于复杂的任务以及从基础模型获得的高维嵌入，支持样本 $x_i$ 对查询 $x'$ 的相关性可能很大程度上取决于由**其他**支持样本提供的上下文 (context)或查询样本自身的特定特征。

这就是注意力机制 (attention mechanism)带来明显提升的地方。不使用静态相似性函数，我们可以学习一个注意力机制 $a(\cdot, \cdot)$，它动态计算每个支持样本 $x_i$ 相对于查询样本 $x'$ 的重要性（权重）$\alpha_i$。对查询样本 $x'$ 的预测 $\hat{y}'$ 变为：


$$
\hat{y}' = \sum_{i=1}^{k \times N} \alpha_i y_i
$$


注意力权重 $\alpha_i$ 通常通过查询和支持嵌入之间的相似度分数的 softmax 计算得到：


$$
\alpha_i = \frac{\exp(a(f_\phi(x'), g_\phi(x_i)))}{\sum_{j=1}^{k \times N} \exp(a(f_\phi(x'), g_\phi(x_j)))}
$$


在这里，$f_\phi$ 嵌入查询样本，$g_\phi$ 嵌入支持样本。这些嵌入函数可以是相同的网络，也可以是不同的网络。注意力函数 $a(\cdot, \cdot)$ 本身可以从简单的余弦相似性（恢复原始匹配网络）到更复杂的学习函数，例如缩放点积或接受一对嵌入作为输入的小型神经网络 (neural network)。

### 完全上下文 (context)嵌入 (embedding) (FCE)

随匹配网络引入的一个强大思想是使用**完全上下文嵌入 (FCE)**。其设想是让每个样本的嵌入依赖于**整个支持集上下文**。这使得模型能够捕获支持集提供的任务定义中更丰富的关系和依赖性。

通常，FCE 通过双向 LSTM 或类似的循环架构实现。

1. **支持集嵌入 $g_\phi(x_i, S)$：** 一个 LSTM 处理支持样本嵌入序列 $\{f_\phi(x_j)\}_{j=1}^{k \times N}$。对应于 $x_i$ 的隐藏状态在处理完完整序列（通常是前向和后向）后，成为其上下文感知嵌入 $g_\phi(x_i, S)$。这使得 $x_i$ 的表示能够受到 $S$ 中所有其他 $x_j$ 的影响。
2. **查询嵌入 $f_\phi(x', S)$：** 同样，查询嵌入 $f_\phi(x')$ 也可以通过一个 LSTM 处理，该 LSTM 以支持集 $S$（或其处理后的状态）作为上下文。这产生了感知由 $S$ 定义的特定任务的查询嵌入 $f_\phi(x', S)$。

然后使用这些上下文感知嵌入计算注意力权重 (weight)：


$$
\alpha_i = \frac{\exp(\text{余弦}(f_\phi(x', S), g_\phi(x_i, S)))}{\sum_{j=1}^{k \times N} \exp(\text{余弦}(f_\phi(x', S), g_\phi(x_j, S)))}
$$


尽管 FCE 通过将完整的支持集上下文融入每个嵌入中，明显提高了表示能力，但它也由于涉及的循环处理而引入了大量的计算开销，特别是当支持集大小 ($k \times N$) 增加时。

> 带有注意力机制 (attention mechanism)的匹配网络中的信息流。支持和查询样本被嵌入（可能使用 FCE），注意力权重根据查询-支持相似性计算，最终预测是支持标签的加权和。

### 优点与考量

**优点：**

- **动态加权：** 注意力机制 (attention mechanism)允许网络根据特定的查询-支持对动态分配重要性分数，可能捕获比固定原型或简单余弦相似性更细致的关系。
- **上下文 (context)关联（带FCE）：** FCE 使模型能够将支持集作为一个整体进行推理 (inference)，可能产生更充分的嵌入 (embedding)和比较。
- **端到端训练：** 嵌入函数 ($f_\phi, g_\phi$) 和注意力机制 ($a$) 通常在元学习循环中进行端到端训练。

**考量：**

- **计算复杂性：** 为每个查询计算支持集中成对的注意力分数比原型网络等方法需要更多的计算量。FCE 的循环组件增加了额外的开销。
- **对大型支持集的可扩展性：** 注意力计算的每查询 $O(k \times N)$ 复杂度可能成为具有许多样本或许多类别的任务的瓶颈。
- **架构选择：** 性能可能很大程度上取决于嵌入架构（例如，CNN、Transformer）、注意力函数 $a(\cdot, \cdot)$ 的选择以及是否使用 FCE。

### 与基础模型的集成

带有注意力的匹配网络可以有效使用来自大型基础模型的嵌入 (embedding)。基础模型可以作为主要的特征提取器 ($f_\phi$, $g_\phi$)，在元训练期间可能被冻结或进行少量调整。注意力机制 (attention mechanism)随后对这些丰富的、高维的嵌入进行操作。在注意力计算之前应用投影层等技术可能是必要的，以管理维度或使嵌入适应注意力函数的特定需求。使用预训练 (pre-training)嵌入减少了在元训练期间从零开始学习复杂特征提取器的负担，使重心放在学习度量和注意力空间以实现快速适应。在静态嵌入上使用简单注意力与在基础模型特征之上实现 FCE 之间选择，涉及性能提升和计算成本之间的权衡。

## 参考资料

- [Matching Networks for One Shot Learning](https://proceedings.neurips.cc/paper_files/paper/2016/file/90e1357833654983612fb05e3ec9148c-Paper.pdf) — Oriol Vinyals, Charles Blundell, Timothy Lillicrap, Koray Kavukcuoglu, Daan Wierstra (2016)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 29; Pages: 3637-3645; DOI: [10.5555/3157382.3157463](https://doi.org/10.5555/3157382.3157463)
  介绍匹配网络、全上下文嵌入（FCE）及其在单样本学习中的应用。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Pages: 15; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  介绍Transformer架构和缩放点积注意力机制，与高级注意力函数相关。
