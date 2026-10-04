---
course: "speech-recognition-synthesis-asr-tts"
chapter: "advanced-acoustic-modeling-asr"
lesson: "transformer-asr"
sourceId: 3131
sourceUrl: "https://apxml.com/zh/courses/speech-recognition-synthesis-asr-tts/chapter-2-advanced-acoustic-modeling-asr/transformer-asr"
title: "用于自动语音识别的Transformer架构"
description: "将自注意力机制和Transformer模块应用于声学建模难题。"
order: 5
plots: []
sourceHash: "92968569952315dfad4975f443e132a22fce8973f2a8b7b9eec418bd3cc17c29"
sourceCorrections: []
---

Transformer架构（最初用于机器翻译）在自动语音识别中变得非常有影响力。这种影响力源于它们对注意力机制 (attention mechanism)的有效利用，该机制此前已在序列到序列任务中展现出显著成功。与顺序处理输入的循环神经网络 (neural network)（RNN）不同，Transformer完全依靠自注意力 (self-attention)机制来计算其输入和输出的表示，从而大大增加了并行计算并更好地捕获长距离依赖。

### 自注意力 (self-attention)机制 (attention mechanism)

自注意力机制是Transformer的核心。回顾编码器-解码器模型中使用的注意力机制，其中解码器关注编码器的输出。自注意力机制将此思想应用于单个序列内部（无论是输入音频特征还是输出文本序列）。它允许序列中的每个位置关注所有其他位置，根据相关性计算其表示的加权和。这使得模型能够直接关联语音信号的不同部分，而无需考虑它们之间的距离。

最常见的形式是缩放点积注意力。对于给定的序列元素，我们计算其查询 ($Q$)、键 ($K$) 和值 ($V$) 向量 (vector)（通常通过线性投影）。注意力输出的计算方式如下：


$$
\text{注意力}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$


这里，$d_k$ 是键向量的维度。缩放因子 $\sqrt{d_k}$ 防止点积过大，这可能导致softmax函数饱和并阻碍学习。

### 多头自注意力 (self-attention)

Transformer不执行单一注意力函数，而是采用*多头*注意力。$Q$、$K$和$V$向量 (vector)通过不同的、学习到的线性投影进行多次投影（每个“头”一次）。每个头的注意力独立地并行计算，然后结果被连接起来并再次投影。

> 多头自注意力机制 (attention mechanism)的流程。输入表示被投影为多个头的查询、键和值。注意力在每个头内部独立计算，然后结果被拼接并投影以形成最终输出。

这使得模型能够共同关注不同位置上来自不同表示子空间的信息。这样一来，不同的头可以学习不同类型的关系（例如，局部声学模式、长距离依赖）。

### 用于ASR的Transformer架构

一个典型的基于Transformer的ASR模型包含一个编码器和（可选的）一个解码器。

#### 编码器

编码器将声学特征（例如，对数梅尔滤波器组）的输入序列 $X = (x_1, ..., x_T)$ 映射到上下文 (context)表示序列 $Z = (z_1, ..., z_T)$。它通常是一堆相同的层。每层有两个主要子层：

1. **多头自注意力 (self-attention)：** 将自注意力应用于层的输入，允许所有位置进行交互。
2. **逐位置前馈网络 (FFN)：** 一个全连接前馈网络，独立应用于每个位置。它通常由两个线性变换构成，中间带有一个ReLU或GeLU激活函数 (activation function)：
   
   $$
   \text{前馈网络}(x) = \max(0, xW_1 + b_1)W_2 + b_2
   $$
   

在两个子层周围都使用了残差连接，随后进行层归一化 (normalization)。每个子层的输出是 $\text{层归一化}(x + \text{子层}(x))$，其中 $\text{子层}(x)$ 是子层本身实现的功能。

由于自注意力机制 (attention mechanism)本身不处理序列顺序，因此必须注入位置信息。这通过*位置编码 (positional encoding)*完成，位置编码被添加到编码器堆栈底部的输入嵌入 (embedding)中。常见方式包括固定的正弦编码或学习到的位置嵌入。

#### 解码器 (用于序列到序列ASR)

当用于端到端序列到序列设置（如基于注意力的编码器-解码器模型）时，也使用Transformer解码器。它一次生成一个标记 (token)的输出序列（字符、音素、词语）（自回归 (autoregressive)地）。除了编码器中自注意力层和FFN子层外，解码器还插入了第三个子层：
3. **多头交叉注意力：** 该层关注编码器 ($Z$) 的输出。查询来自前一个解码器层，而键和值来自编码器输出。这使得解码器在预测下一个输出标记时能够结合来自输入语音信号的信息。

解码器中的自注意力子层经过*掩码*处理，以防止位置关注后续位置。这保证了位置 $i$ 的预测只能依赖于小于 $i$ 的已知输出，从而保持了自回归特性。

#### 带CTC的Transformer编码器

另外，Transformer编码器可以作为一个高效的特征提取器，其输出被馈送到一个最终的线性层，然后是softmax，并使用前面讨论的CTC损失函数 (loss function)进行训练。这种方式使用了Transformer在获取上下文信息方面的能力，而无需自回归解码器，通常会带来更简单、更快的推断。

### 优点与挑战

**优点：**

- **并行性：** 每个Transformer层内的计算在时间维度上高度并行化，与顺序RNN相比，训练速度大幅加快。
- **长距离上下文 (context)：** 自注意力 (self-attention)机制 (attention mechanism)在序列中的任意两个位置之间提供了直接连接，使得对长距离依赖进行建模比RNN中的梯度消失问题更容易。
- **表现：** 基于Transformer的模型在许多ASR基准测试上取得了当前最优或有竞争力的表现。

**挑战：**

- **计算复杂度：** 自注意力的计算成本随输入序列长度 $T$ 呈二次方增长，即 $O(T^2 \cdot d)$，其中 $d$ 是模型维度。这对于ASR中非常常见的长音频序列来说，计算成本高且内存占用大。存在多种近似方法（例如，稀疏注意力、线性注意力）来减轻此问题。
- **数据需求：** Transformer通常需要大量训练数据才能良好泛化。
- **位置信息：** 它们完全依赖于位置编码 (positional encoding)，因为自注意力机制本身是置换不变的。
- **局部关联：** 标准Transformer在处理细粒度局部模式（如特定的音素转换）方面可能不如CNN擅长。

### Conformer：结合Transformer和CNN

为更好地解决局部关联建模问题，**Conformer**架构被提出。它将卷积神经网络 (neural network)（CNN）模块直接集成到Transformer块结构中。一个典型的Conformer块用一个序列替换标准FFN子层：一个FFN模块、一个多头自注意力 (self-attention)模块、一个卷积模块和另一个FFN模块，所有这些都带有适当的归一化 (normalization)和残差连接。

> Conformer块的简化视图，说明了前馈网络、多头自注意力以及卷积模块与残差连接的组合。注意：实际的残差连接通常涉及将主要组件夹在中间的半步前馈模块。

Conformer旨在兼顾Transformer的全局上下文 (context)建模优势和CNN的局部特征提取能力，它已成为当前最优ASR系统中声学建模的一种广泛采用且非常有效的架构。

总而言之，Transformer架构，特别是Conformer，代表着ASR声学建模的一大进步。它们通过自注意力机制 (attention mechanism)有效建模长距离依赖的能力，结合计算并行性（以及Conformer中的卷积增强），使其成为现代语音识别系统的根本。这些模型的实际实现细节和相关训练策略将在动手实践部分和后续章节中进一步讨论。

## 参考资料

- [Attention Is All You Need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems 30; Publisher: Curran Associates, Inc.; Pages: 5998-6008; DOI: [10.5591/978-1-57766-068-1.5998](https://doi.org/10.5591/978-1-57766-068-1.5998)
  这篇论文介绍了Transformer架构、自注意力机制和多头注意力机制，为后续序列处理模型奠定了基础。
- [Conformer: Convolution-augmented Transformer for Speech Recognition](https://doi.org/10.21437/Interspeech.2020-1168) — Anmol Gulati, James Qin, C. C. Chiu, Haikal Hassoun, Ruoming Pang, Karthik Ramasubramanian, Justin Salpekar, Kevin Webster, Yonghui Wu (2020)
  Journal: INTERSPEECH 2020; Publisher: International Speech Communication Association (ISCA); Pages: 5036-5040; DOI: [10.21437/Interspeech.2020-1168](https://doi.org/10.21437/Interspeech.2020-1168)
  本文提出了Conformer架构，它将卷积模块融入Transformer块中，以提高语音识别性能。
- [Speech-Transformer: A No-Recurrence Sequence-to-Sequence Model for Speech Recognition](https://doi.org/10.21437/Interspeech.2018-1971) — Li-Wen Dong, Shu-wen Yang, Bo-Cheng Lai, Hung-yi Lee (2018)
  Journal: INTERSPEECH 2018; Publisher: ISCA; Pages: 3123-3127; DOI: [10.21437/Interspeech.2018-1971](https://doi.org/10.21437/Interspeech.2018-1971)
  Transformer架构在端到端自动语音识别任务中的早期应用，展示了其在该任务中的可行性。
