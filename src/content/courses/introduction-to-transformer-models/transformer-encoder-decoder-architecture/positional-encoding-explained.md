---
course: "introduction-to-transformer-models"
chapter: "transformer-encoder-decoder-architecture"
lesson: "positional-encoding-explained"
sourceId: 2997
sourceUrl: "https://apxml.com/zh/courses/introduction-to-transformer-models/chapter-3-transformer-encoder-decoder-architecture/positional-encoding-explained"
title: "位置编码说明"
description: "说明用于位置编码的正弦和余弦函数，以及它们如何与嵌入向量相加。"
order: 4
plots: ["plots/2997-0.json"]
sourceHash: "11b987b5e3981b6df1d6879983b134b0527b0d5a02e2afe8284e049b5b8d2693"
sourceCorrections: []
---

Transformer中自注意力 (self-attention)机制 (attention mechanism)的核心本身无法自然地理解序列中词元 (token)的顺序。如果打乱输入词语的顺序，特定词语的自注意力输出（在加入位置信息之前）会改变，但模型没有内置方式知道*顺序*本身已不同。这与循环神经网络 (neural network)（RNN）不同，RNN逐步处理序列，自然地包含顺序信息。

为解决此问题，Transformer模型会将关于每个词元位置的信息直接注入到输入嵌入 (embedding)向量 (vector)中。这是通过使用**位置编码 (positional encoding)**实现的。其思路是为序列中每个位置创建一个独特的向量，并将此向量加到相应词元的嵌入向量上。因此，这种组合后的嵌入向量既携带了词元的语义含义，也携带了它在序列中的位置。

尽管位置编码有多种方法，最初的Transformer论文提出了一种广泛采用的技术，该技术基于不同频率的正弦和余弦函数。对于序列中位置为 $pos$ 的词元，以及嵌入向量中维度为 $i$ 的分量（嵌入向量大小为 $d_{\text{model}}$），位置编码 $PE$ 的计算方式如下：


$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$


$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$


我们来分析一下：

- $pos$：这是词元在序列中的位置索引（例如，第一个词元为0，第二个为1，依此类推）。
- $i$：这是嵌入向量中的维度索引。它的范围从 $0$ 到 $d_{\text{model}}/2 - 1$。请注意，每对维度 $(2i, 2i+1)$ 使用相同的频率，但函数不同（正弦和余弦）。
- $d_{\text{model}}$：这是输入嵌入向量的维度（也是位置编码向量的维度）。一个常见的值可能是512或768。
- $10000$：这个值是作者选择的一个超参数 (parameter) (hyperparameter)；它决定了所用频率的范围。

主要思路是正弦/余弦波的波长在不同维度上有所不同。对于较小的 $i$ 值（初始维度），波长较短（高频）。对于较大的 $i$ 值（后续维度），波长变得很长（低频，对于短序列接近一个常数）。因此，每个位置 $pos$ 都会在 $d_{\text{model}}$ 维上获得一组独特的正弦值组合。

所生成的位置编码向量 $PE_{(pos)}$ 与输入嵌入向量具有相同的维度（$d_{\text{model}}$）。这个位置编码向量随后会逐元素地*添加*到词元的输入嵌入向量中：


$$
\text{最终嵌入}_{(pos)} = \text{输入嵌入}_{(pos)} + PE_{(pos)}
$$


这种添加方式使得模型能够学习表示，这些表示结合了词元的含义和其位置。

这种特定的正弦方法有几个吸引人的特性：

1. **独特性**：它为序列中的每个位置生成一个独特的编码向量。
2. **确定性**：这些编码是固定的，不在训练期间学习，这使得它们可预测且一致。
3. **相对位置信息**：一个重要优点是，未来位置 $pos+k$ 的位置编码可以表示为位置 $pos$ 编码的线性变换。这一特性使得模型的注意力机制更容易学习基于相对位置的关系。
4. **有界值**：正弦和余弦函数自然地将编码值保持在 [-1, 1] 范围内，这有助于稳定嵌入向量的数值特性。
5. **泛化能力**：模型可能能够泛化到比训练时遇到的序列更长的长度，尽管如果差异很大，性能可能会下降。

以下是这些位置编码值在不同位置和维度上的可视化表示：



![位置编码值（前8个维度，10个位置）](plots/2997-0.json)



> 位置编码值的可视化。每行代表序列中的一个位置，每列代表编码向量中的一个维度。请注意，正弦模式在不同维度上的频率不同，以及每个位置独特的向量。

这种位置信息的注入只发生一次，就在最开始，在输入嵌入向量被送入编码器（或解码器）堆栈的第一层之前。Transformer模型中的后续层随后会处理这些带有位置信息的嵌入向量。没有这一步，Transformer模型本质上会处理一个“词袋”，从而失去语言或其他有序数据的序列性质。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems; Volume: 30; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  介绍了Transformer架构和原始的正弦位置编码方法。
- [Speech and Language Processing (3rd edition draft)](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky, James H. Martin (2025)
  Publisher: Stanford University
  Transformer模型的综合教科书式解释，包括第10章对位置编码的详细讨论。
- [Lecture 9: Transformers and Large Language Models (Winter 2023)](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGv91LgjnY6wuKYf6udoALiDNbesHz0Qn7QBQOl3GuAxsNMSTpmEOJraZVzZ3CAX1D3Pc71Om8fmPTX6_shO4hCbAffSW0imJglkQ2uzRAkTz_q2IvH-hJXSRmCHuUwz03KYzjpUjV8kXMjHIOJvZIV-Mjm-8BRpd66aaAxKtyC2qooy0emL3_whejZWaUO54vuK323K-X45qXuNsdurueCnV2C) — John Hewitt, Anna Goldie (2023)
  Publisher: Stanford University (CS224N)
  来自顶尖大学课程的讲义，对Transformer和位置编码提供了清晰的解释和可视化。
