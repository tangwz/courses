---
course: "foundations-transformers-architecture"
chapter: "positional-encoding-embedding-layer"
lesson: "sinusoidal-encoding-formulation"
sourceId: 1989
sourceUrl: "https://apxml.com/zh/courses/foundations-transformers-architecture/chapter-4-positional-encoding-embedding-layer/sinusoidal-encoding-formulation"
title: "正弦型位置编码：公式表述"
description: "基于正弦和余弦的位置编码的详细数学表述。"
order: 3
plots: ["plots/1989-0.json"]
sourceHash: "3a30fb2c806dc2b314b8a0aee98d2cb189863757a041f0a73915b5ea0322409c"
sourceCorrections: []
---

自注意力 (self-attention)机制 (attention mechanism)在处理输入元素时，无法自然地理解它们在序列中的顺序。这种置换不变性在捕获无关距离的关系方面表现出色，但未能对语言及其他有序数据中必需的序列特性进行建模。为此，Transformer架构引入了位置编码 (positional encoding)，这些向量 (vector)被添加到输入嵌入 (embedding)中，用以向模型提供每个token的位置信息。

原始Transformer论文（“Attention Is All You Need”）提出了一种固定的、无需学习的方法，该方法利用不同频率的正弦和余弦函数。这种被称作正弦型位置编码的方法，能为序列中直至预设最大长度的每个位置生成独特的编码向量。

### 数学表述

对于序列中的给定位置 $pos$（其中 $pos = 0, 1, 2, ...$），以及 $d_{model}$ 维嵌入 (embedding)向量 (vector)中的给定维度 $i$（其中 $i = 0, 1, ..., d_{model}-1$），位置编码 (positional encoding) $PE$ 的定义是：对偶数维度索引 ($2i$) 应用正弦函数，对奇数维度索引 ($2i+1$) 应用余弦函数：


$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$


$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$


其中：

- $pos$ 表示token在输入序列中的位置索引。
- $i$ 表示编码向量中的维度索引。请注意，这里的 $i$ 实际范围是从 $0$ 到 $d_{model}/2 - 1$，因为每个 $i$ 定义了一对维度（$2i$ 和 $2i+1$）。
- $d_{model}$ 是整个Transformer模型中使用的嵌入向量的维度（例如，512，768）。

项 $1 / 10000^{2i/d_{model}}$ 决定了正弦和余弦波的角频率。我们来分析其特性：

- 当 $i$ 较小（代表编码向量的起始维度）时，指数 $2i/d_{model}$ 接近于 0。底数 $10000^{2i/d_{model}}$ 接近于 $10000^0 = 1$。这导致参数 (parameter) $pos/1$ 相对较大，意味着这些维度上的正弦/余弦函数在位置上以高频率（短波长）振荡。
- 随着 $i$ 增加并接近 $d_{model}/2$，指数 $2i/d_{model}$ 接近于 1。底数 $10000^{2i/d_{model}}$ 接近于 $10000^1 = 10000$。这导致参数 $pos/10000$ 较小，意味着这些后续维度上的函数在位置上以极低频率（长波长）振荡。

这种设计创建了位置编码向量，其中每个维度都对应特定频率的正弦波。这些正弦波在所有维度上的组合为每个位置 $pos$ 生成了独特的标识。



![正弦型位置编码可视化](plots/1989-0.json)



> 热图显示了前50个位置（每隔2个位置采样）以及前128个维度（每隔4个维度采样）的正弦型位置编码值。每行代表一个维度索引，每列代表序列中的一个位置。颜色深浅表示编码值（范围从-1到1）。请注意，沿维度轴（y轴）的振荡频率有所不同。

从这些三角恒等式得出的一个重要性质是，位置 $pos + k$ 的编码可以表示为位置 $pos$ 编码的线性变换。这种线性性质可能有助于模型轻松地根据相对位置进行注意力计算，因为相距固定距离 $k$ 的token编码之间的关系，无论其绝对位置如何，都保持一致。

总之，正弦型位置编码提供了一种确定性且计算高效的方法，用以向Transformer传入序列顺序信息。它能生成独特的位置标识，并具备有利于建模相对位置的性质，同时无需额外的可学习参数。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems (NIPS 2017); DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  介绍了 Transformer 架构和原始正弦位置编码的基础论文。
- [Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  Publisher: Stanford University; Pages: Chapter 10
  在其专有章节中对 Transformer 架构和位置编码提供了易懂且全面的解释。
