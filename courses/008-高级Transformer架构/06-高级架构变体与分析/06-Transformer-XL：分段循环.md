# Transformer-XL：分段循环

来源：[原文](https://apxml.com/zh/courses/foundations-transformers-architecture/chapter-6-advanced-architectural-variants-analysis/transformer-xl-recurrence)

[返回章节目录](README.md) · [返回课程目录](../README.md)

标准的Transformer架构以固定长度、独立的块或窗口处理序列，但这种方式会遇到一个明显局限性，即上下文 (context)碎片化。在处理长度超出固定窗口限制的长文档或序列数据流时，模型必须将输入分成独立的分段进行处理。在处理新分段时，来自前序分段的信息通常会丢失，这阻碍了模型捕获跨越分段边界的长距离依赖关系的能力。

Transformer-XL（意为具有超长上下文的Transformer）通过引入分段级别的循环机制直接解决了这一限制。Transformer-XL不再孤立地处理每个分段，而是重复使用从先前分段计算得到的隐藏状态。

### 分段级别状态复用

主要思想简单但有效。当模型处理一个分段（例如分段 $\tau$）时，它会在每个层计算一系列隐藏状态，这与标准Transformer类似。这些隐藏状态随后会被缓存或存储在内存中。当模型移动到下一个分段 $\tau+1$ 时，各层不仅可以关注当前分段 $\tau+1$ 内的隐藏状态，还可以关注来自前一个分段 $\tau$ 的缓存隐藏状态。

设 $h_{\tau}^n \in \mathbb{R}^{L \times d}$ 表示第 $\tau$ 个分段中第 $n$ 个Transformer层产生的隐藏状态序列，其中 $L$ 是分段长度，$d$ 是隐藏维度。在计算下一个分段的隐藏状态 $h_{\tau+1}^n$ 时，第 $(n)$ 层接收源自 $h_{\tau+1}^{n-1}$（当前分段下方层的输出）和 $h_{\tau}^{n-1}$（前一个分段下方层的输出）的输入。

具体而言，分段 $\tau+1$ 中层 $n$ 的扩展上下文 (context)是通过将来自前一个分段的缓存状态与来自当前分段的状态沿着序列长度维度连接而形成的：


$$
\tilde{h}_{\tau+1}^{n-1} = [SG(h_{\tau}^{n-1}) \circ h_{\tau+1}^{n-1}]
$$


这里，$SG(\cdot)$ 表示一个停止梯度操作，这意味着梯度不会通过缓存状态 $h_{\tau}^{n-1}$ 反向传播 (backpropagation)。这很要紧：它防止了计算图变得过长，并避免了相关的优化困难。$\circ$ 运算符表示沿着序列长度维度的连接。

层 $n$ 内的注意力机制 (attention mechanism)随后仅基于当前分段的表示 $h_{\tau+1}^{n-1}$ 计算其查询 ($Q$)，而键 ($K$) 和值 ($V$) 则源自扩展上下文 $\tilde{h}_{\tau+1}^{n-1}$：


$$
Q_{\tau+1}^n = h_{\tau+1}^{n-1} W_q^n \\
K_{\tau+1}^n = \tilde{h}_{\tau+1}^{n-1} W_k^n \\
V_{\tau+1}^n = \tilde{h}_{\tau+1}^{n-1} W_v^n
$$


$$
AttentionOutput_{\tau+1}^n = Attention(Q_{\tau+1}^n, K_{\tau+1}^n, V_{\tau+1}^n)
$$


这使得当前分段 $\tau+1$ 中的每个位置都能够关注其自身内部以及前一个分段 $\tau$ 中的所有位置，有效地将每一步可用的上下文长度加倍，同时不跨分段边界传播梯度。

> Transformer-XL 中的信息流。来自分段 $\tau$ 的隐藏状态被缓存，并用作处理分段 $\tau+1$ 的扩展上下文，同时不通过缓存反向传播梯度。

### 相对位置编码 (positional encoding)

状态复用机制给标准位置编码（如第4章中描述的正弦或学习到的绝对嵌入 (embedding)）带来了挑战。如果我们只是简单地将相同的绝对位置编码添加到每个分段，那么一个位置索引（例如，第10个词元 (token)）将具有相同的编码，无论它是第一个分段的第10个词元还是第二个分段的第10个词元。这种位置歧义性使得模型难以区分跨分段的时间顺序。

Transformer-XL 通过采用一种**相对位置编码**方案来解决这个问题。它不编码词元的绝对位置 $i$，而是编码位于位置 $i$ 的查询词元与位于位置 $j$ 的词元之间的相对距离（或偏移）$i-j$。这种相对信息被直接注入到注意力分数计算中。

在查询 $q_i$ 和键 $k_j$ 的标准自注意力 (self-attention)分数计算中，我们计算 $q_i^T k_j$。在带有相对位置编码的Transformer-XL中，这个计算被修改为包含仅依赖于相对距离 $i-j$ 的项。具体公式涉及将键向量 (vector)中的绝对位置信息替换为相对位置嵌入。这确保了注意力机制 (attention mechanism)能够知晓词元之间的距离，而不受它们在可能非常长的、分段处理的序列中绝对位置的影响。

### Transformer-XL的优势

引入分段级别循环和相对位置编码 (positional encoding)带来了多项益处：

1. **建模更长距离的依赖关系：** 通过复用状态，Transformer-XL 可以学习跨越比单个分段长度更远距离的依赖关系，缓解了上下文 (context)碎片化问题。其有效上下文长度可以明显大于标准Transformer在训练时因内存限制而能达到的长度。
2. **更快的评估：** 在推理 (inference)或评估期间，前一个分段的隐藏状态可以计算一次并重复用于后续分段。与标准Transformer中为实现类似效果而重新处理重叠分段的朴素方法相比，这避免了冗余计算，从而带来明显的加速，特别是对于自回归 (autoregressive)生成任务。
3. **提高连贯性：** 能够访问过去上下文使得生成更加连贯，并能更好地理解长文本 (long context)或时间序列。

Transformer-XL 代表了使Transformer能够有效地处理更长序列的重要一步，为涉及长篇文档、文章或连续数据流的应用创造了途径，在这些应用中，保持长距离连贯性很要紧。尽管它引入了缓存状态的开销，但在建模能力和评估速度方面的益处对于特定任务通常会超过此成本。

## 参考资料

- [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860) — Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc V. Le, Ruslan Salakhutdinov (2019)
  Journal: ACL 2019; DOI: [10.48550/arXiv.1901.02860](https://doi.org/10.48550/arXiv.1901.02860)
  引入分段级循环和相对位置编码以解决Transformer中上下文碎片问题的开创性论文。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS 2017); DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  引入Transformer架构的开创性论文，为包括Transformer-XL在内的所有后续Transformer变体奠定了基础。
- [Natural Language Processing with Transformers](https://oreil.ly/nlp-with-transformers) — Lewis Tunstall, Leandro von Werra, and Thomas Wolf (2022)
  Publisher: O'Reilly Media; Pages: 406
  一本关于Transformer模型的综合指南，包括在更广泛的高级架构背景下对Transformer-XL及其机制的解释。

---

[上一节](05-%E4%BD%8E%E7%A7%A9%E6%8A%95%E5%BD%B1%E6%96%B9%E6%B3%95%EF%BC%88Linformer%EF%BC%89.md) · [下一节](07-%E7%9B%B8%E5%AF%B9%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81.md)
