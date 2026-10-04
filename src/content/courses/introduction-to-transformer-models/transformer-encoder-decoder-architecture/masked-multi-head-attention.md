---
course: "introduction-to-transformer-models"
chapter: "transformer-encoder-decoder-architecture"
lesson: "masked-multi-head-attention"
sourceId: 3010
sourceUrl: "https://apxml.com/zh/courses/introduction-to-transformer-models/chapter-3-transformer-encoder-decoder-architecture/masked-multi-head-attention"
title: "带掩码的多头自注意力"
description: "说明掩码如何保证解码器预测仅基于先前位置的输出。"
order: 9
plots: []
sourceHash: "349f8670001762a2b76fd12831b5f26978478deee02e2192189e9728ed95eceb"
sourceCorrections: []
---

在我们考察解码器堆栈时，请回想它的主要作用：一次生成一个元素地输出序列。在机器翻译等任务中，这意味着逐词生成翻译后的句子。这种按序生成过程对解码器内部使用的自注意力 (self-attention)机制 (attention mechanism)提出了特定要求。与编码器可以同时处理整个输入序列不同，解码器必须仅根据*已生成*的令牌来预测下一个令牌。它不能预见当前正在构建的输出序列中的未来令牌。

这就是**带掩码的多头自注意力**发挥作用的地方。它是我们在第2章中讨论的多头自注意力机制的一种修改，专门设计用于在训练和推理 (inference)期间强制执行这种“不可预见”的约束。

### 为什么需要掩码

考虑序列到序列模型的训练过程。我们通常使用“教师强制”机制，即解码器被输入*正确的*目标序列（向右移位，通常带有序列开始令牌）作为输入，以预测每个位置的下一个令牌。例如，为了预测目标翻译的第三个词，解码器会接收到前两个正确的词。

如果在此处使用标准自注意力 (self-attention)，位置`i`的注意力机制 (attention mechanism)将能够整合目标序列中*所有*位置的信息，包括位置`i+1`、`i+2`等。这就像作弊；模型可以简单地从输入中复制下一个词，而不是学习根据前面的词和编码器的输出来预测它。模型需要学习条件概率分布$P(\text{输出}_t | \text{输出}_{<t}, \text{编码器\_输出})$。允许对未来令牌的注意力会破坏这种条件依赖性。

在推理 (inference)期间（生成新序列时），未来令牌本来就是未知的。因此，注意力机制在训练和推理之间必须保持一致，只关注先前生成的令牌。

### 掩码机制

带掩码的多头自注意力 (self-attention)通过修改每个注意力头内部的缩放点积注意力计算来实现这一约束。在对注意力分数应用softmax函数之前，会添加一个*掩码*。

回顾缩放点积注意力公式：


$$
\text{分数} = \frac{QK^T}{\sqrt{d_k}}
$$


在带掩码的自注意力中，我们修改如下：


$$
\text{带掩码的分数} = \frac{QK^T}{\sqrt{d_k}} + M
$$


$$
\text{注意力权重} = \text{softmax}(\text{带掩码的分数})
$$


$$
\text{注意力}(Q, K, V) = \text{注意力权重} V
$$


掩码$M$通常是一个矩阵，其中模型*不允许*关注的位置的元素被设置为一个非常大的负数（实际上是负无穷大），而允许关注的位置的元素被设置为零。

对于长度为$L$的目标序列，掩码$M$将是一个$L \times L$矩阵。对于第$i$个令牌（$Q$矩阵的第$i$行），掩码确保它只关注令牌$j$，其中$j \le i$（$K$矩阵的第$0$到$i$列）。其中$j > i$的元素$M_{ij}$被设置为$-\infty$，而$j \le i$的元素被设置为$0$。

当一个大的负数被添加到未来位置的注意力分数时，随后的softmax函数会给这些位置分配一个非常接近零的概率。这有效阻止了任何信息从未来令牌流入当前令牌的表示中。

让我们可视化长度为4的序列的掩码：

```
   关注目标 -->   位置 1  位置 2  位置 3  位置 4
查询自位置 1: [  0      -inf   -inf   -inf ]
查询自位置 2: [  0       0     -inf   -inf ]
查询自位置 3: [  0       0      0     -inf ]
查询自位置 4: [  0       0      0      0   ]
```

这里，`0`代表一个允许的注意力连接（添加此掩码后，原始分数不变），`-inf`代表一个被掩码的连接（添加此掩码后，分数实际上变为负无穷大）。

下图说明了输出序列“The model predicts well”中位置3（“predicts”）的令牌允许的注意力连接，假设它是逐步生成的。

> 该图显示，来自位置3（“predicts”）的查询可以关注位置1、2和3的键/值，但对位置4（“well”）的注意力被掩盖掉了。

### 与多头注意力 (multi-head attention)的结合

这种掩码过程在多头结构的*每个*注意力头内部独立应用。整体过程与标准多头注意力相同：

1. 将输入嵌入 (embedding)投影到多组Q、K、V向量 (vector)中（每头一组）。
2. 对于每个头，计算缩放点积注意力分数。
3. **应用注意力掩码**，将掩码矩阵$M$添加到分数中。
4. 应用softmax函数以获得注意力权重 (weight)。
5. 计算V向量的加权和。
6. 连接所有头的输出。
7. 应用最终的线性投影。

与编码器的自注意力 (self-attention)相比，唯一的区别是步骤3，即掩码的应用。

### 总结

带掩码的多头自注意力 (self-attention)是标准自注意力机制 (attention mechanism)的一种修改，对于Transformer的解码器部分不可或缺。通过阻止位置关注输出序列中的后续位置，它确保了模型的预测是自回归 (autoregressive)的，这意味着当前步骤的预测仅依赖于先前生成的步骤和输入序列。这是通过在softmax计算之前，向注意力分数中添加一个掩码矩阵（包含零和负无穷大）来实现的，从而有效地将未来令牌的权重 (weight)归零。这种机制应用于解码器自注意力层中多头注意力 (multi-head attention)结构的每个头部。这与编码器的自注意力形成对比，编码器自注意力允许每个位置关注输入序列中的所有位置。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems; Volume: 30; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  介绍Transformer架构及其掩码多头自注意力机制，对解码器的自回归生成至关重要。
- [Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  一本全面的教材，详细解释了Transformer模型，包括掩码自注意力的功能和实现。
- [Stanford CS224N: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/archive/winter2023/) — Christopher Manning, Abigail See, John Hewitt, Tatsunori Hashimoto (2023)
  Publisher: Stanford University
  课程资料提供了关于Transformer架构和掩码注意力在解码器中作用的学术视角。
- [The Hugging Face Course: Transformers](https://huggingface.co/course/chapter1/1) — Hugging Face (2023)
  Publisher: Hugging Face
  一个易于理解且实用的在线课程，以清晰互动的方式解释了Transformer架构和掩码注意力的具体作用。
