---
course: "introduction-to-transformer-models"
chapter: "transformer-encoder-decoder-architecture"
lesson: "decoder-stack"
sourceId: 3007
sourceUrl: "https://apxml.com/zh/courses/introduction-to-transformer-models/chapter-3-transformer-encoder-decoder-architecture/decoder-stack"
title: "解码器堆栈"
description: "详细说明单个解码器层的组成部分：遮蔽多头自注意力、编码器-解码器注意力以及前馈网络。"
order: 8
plots: []
sourceHash: "f143e3bc1ad2936cfbab6cc7980b4ba8a841a4f9a3c28395bcbaba55e62b6273"
sourceCorrections: []
---

编码器的作用是处理整个输入序列并构建有用的表示，而解码器的职责是逐个元素地生成输出序列。这种生成过程通常是*自回归 (autoregressive)的*，这意味着在预测下一个元素（例如翻译中的一个词）时，解码器会考虑编码器的输出以及它已经生成的元素。

解码器由$N$个相同的层堆叠而成。其每一层的结构都较为复杂，因为它需要处理两种类型的注意力机制 (attention mechanism)。

### 单个解码器层的结构

让我们分析一个解码器层内部的组成部分。第一个解码器层的输入是目标序列嵌入 (embedding)（右移，我们稍后会讨论）加上位置编码 (positional encoding)。后续层的输入是来自前一个解码器层的输出。重要的是，每个解码器层还会接收来自*最终*编码器层的键($K$)和值($V$)矩阵。

单个解码器层由三个主要子层组成，每个子层后面都跟着一个“Add & Norm”（加和归一化 (normalization)）步骤（残差连接加上层归一化）：

1. **遮蔽多头自注意力 (self-attention):** 该子层运行方式与编码器中的自注意力机制 (attention mechanism)类似，但有一个重要的修改：遮蔽。在训练期间，解码器接收完整的输出序列作为输入（具体来说，是向右移动一个位置的版本，因此对位置 $i$ 的预测仅依赖于到 $i-1$ 的输出）。然而，为了保持自回归 (autoregressive)特性并防止模型通过“偷看”它应该预测的未来令牌来“作弊”，我们应用了注意力遮罩。这个遮罩在softmax计算之前，将后续位置的注意力分数有效设置为负无穷大，确保未来令牌的注意力概率变为零。这使得解码器中的每个位置只能关注输出序列中的先前位置（包括当前位置）。该子层的查询($Q$)、键($K$)和值($V$)向量 (vector)源自前一个解码器层的输出（或第一个层的目标序列嵌入）。
2. **多头编码器-解码器注意力（交叉注意力）:** 这是解码器与编码器堆栈的输出进行交互的地方。该子层使得解码器能够聚焦于输入序列的相关部分，以帮助预测下一个输出令牌。工作方式如下：

   - 查询($Q$)是从*前一个*解码器子层（遮蔽自注意力之后接加和归一化）的输出生成的。
   - 键($K$)和值($V$)直接来自*编码器*堆栈的输出。这些是所有解码器层使用的相同$K$和$V$矩阵。
     通过比较自身的当前状态（通过$Q$）与输入序列的编码表示（通过$K$和$V$），解码器可以确定输入中哪些部分与生成当前输出令牌最相关。例如，在翻译句子时，此机制有助于将目标语言中的词语与源语言中的对应词语对齐 (alignment)。
3. **位置前馈网络:** 该子层结构上与编码器层中的相同。它由两个线性变换组成，中间带有ReLU激活函数 (activation function)：

   
   $$
   \text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
   $$
   

   该网络独立应用于每个位置，处理来自编码器-解码器注意力子层的输出。它提供额外的非线性变换，以进一步细化每个位置的表示。

这三个子层中的每一个都带有残差连接，后面跟着层归一化，就像编码器中一样。这有助于梯度流动并稳定深度模型的训练。

> 单个Transformer解码器层内部的数据流。它包含遮蔽自注意力、编码器-解码器注意力（使用编码器输出）和一个前馈网络，每个都紧接着加和归一化。

堆栈中最终解码器层的输出代表了处理过的目标序列信息，该信息受到输入序列（通过编码器）和先前生成的目标令牌的影响。这个输出张量随后通常会传递给一个最终的线性层和一个softmax函数，以生成词汇表 (vocabulary)上的概率分布，用于预测下一个令牌，这部分我们将在后面介绍。

## 参考资料

- [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Publisher: Curran Associates, Inc.; Volume: 30; Pages: 5998-6008; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  这篇基础论文介绍了Transformer架构，详细说明了解码器的组成及其功能，包括带掩码的自注意力机制和编码器-解码器注意力机制。
- [Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky, James H. Martin (2025)
  一本关于自然语言处理的全面学术著作，包含对Transformer架构及其解码器堆栈的详细讨论。
- [Natural Language Processing with Transformers](https://www.oreilly.com/library/view/natural-language-processing/9781098136789/) — Lewis Tunstall, Leandro von Werra, and Thomas Wolf (2022)
  Publisher: O'Reilly Media
  提供了使用Transformer模型的实践指导，清晰地解释了其架构，包括解码器的作用和子层。
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Jay Alammar (2018)
  采用图表和视觉效果来解释Transformer架构，使解码器组件及其相互作用易于理解和掌握。
