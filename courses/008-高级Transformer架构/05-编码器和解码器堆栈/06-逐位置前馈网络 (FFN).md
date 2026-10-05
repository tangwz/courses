# 逐位置前馈网络 (FFN)

来源：[原文](https://apxml.com/zh/courses/foundations-transformers-architecture/chapter-5-encoder-decoder-stacks/position-wise-ffn)

[返回章节目录](README.md) · [返回课程目录](../README.md)

逐位置前馈网络（FFN）是每个编码器和解码器层中的一个重要组成部分。虽然注意力机制 (attention mechanism)旨在处理序列间的关联并整合来自不同位置的信息，但FFN提供额外的计算深度和非线性，对序列中的每个位置独立进行操作。

可以将注意力子层视为通信中心，允许令牌交换信息。然后，FFN则作为每个令牌的独立处理单元，转换通过注意力接收到的信息。

### 结构和运作

FFN通常是一个简单的全连接前馈网络，包含两个线性变换，中间夹有一个非线性激活函数 (activation function)。原始Transformer论文中，激活函数的标准选择是修正线性单元（ReLU）。

FFN在特定位置 $t$ 的输入是来自前一个子层的输出向量 (vector) $z_t$（无论是自注意力 (self-attention)还是交叉注意力，且在Add & Norm步骤之后，具体取决于Post-LN或Pre-LN等层配置）。计算过程如下：

1. **第一次线性变换：** 输入向量 $z_t$（维度为 $d_{model}$）被投影到更高维空间 (high-dimensional space)，通常是 $d_{ff}$。这种扩展通过权重 (weight)矩阵 $W_1$（形状为 $d_{model} \times d_{ff}$）和偏置 (bias)向量 $b_1$（形状为 $d_{ff}$）实现。
   
   $$
   \text{线性}_1(z_t) = z_t W_1 + b_1
   $$
   
2. **非线性激活：** 第一次变换的结果通过ReLU激活函数（或在较新模型中有时使用GeLU等其他激活函数）处理。ReLU引入非线性，使得网络能够学习更复杂的函数。
   
   $$
   \text{ReLU}(x) = \max(0, x)
   $$
   
3. **第二次线性变换：** 激活后的输出使用第二个权重矩阵 $W_2$（形状为 $d_{ff} \times d_{model}$）和偏置向量 $b_2$（形状为 $d_{model}$）重新投影回原始模型维度 $d_{model}$。
   
   $$
   \text{线性}_2(x) = x W_2 + b_2
   $$
   

综合这些步骤，对于单个位置 $t$ 的完整FFN操作可以表示为：


$$
FFN(z_t) = \text{ReLU}(z_t W_1 + b_1) W_2 + b_2
$$


或者，更一般地使用 $f$ 代表激活函数：


$$
FFN(z_t) = f(z_t W_1 + b_1) W_2 + b_2
$$


### 逐位置应用

该网络的一个重要方面是其“逐位置”应用。虽然在序列的所有位置都使用相同的FFN（指完全相同的权重 (weight)矩阵 $W_1, W_2$ 和偏置 (bias) $b_1, b_2$），但它被独立应用于每个位置的向量 (vector)表示。

如果注意力子层后的输入序列表示是 $Z = [z_1, z_2, ..., z_n]$，其中 $n$ 是序列长度，则FFN计算结果为：


$$
FFN_{output} = [FFN(z_1), FFN(z_2), ..., FFN(z_n)]
$$


这与注意力机制 (attention mechanism)形成鲜明对比，后者明确地建模不同位置*之间*的关联。FFN独立处理每个位置的表示，使得模型能够学习单个令牌表示的复杂变换，这些变换由通过注意力收集的上下文 (context)信息来辅助。

> 该图显示了相同的两个线性层（具有共享权重 $W_1, b_1$ 和 $W_2, b_2$）和ReLU激活函数 (activation function)如何独立应用于每个序列位置的输入向量 ($z_t$)。

### 维度和实现

在原始Transformer论文《Attention Is All You Need》中，模型维度 $d_{model}$ 为512，FFN的内层维度 $d_{ff}$ 设置为2048。这种四倍扩展（$d_{ff} = 4 \times d_{model}$）是一种常用的启发式方法，尽管也存在变体。这种扩展使得FFN能够将表示投影到更高维空间 (high-dimensional space)，在那里可以更容易地学习复杂模式，然后再将其投影回标准模型维度。

从实现角度看，在每个位置独立应用相同的线性变换是计算高效的。它可以使用1x1卷积实现。如果将向量 (vector)序列 $Z$（形状：序列长度 $\times$ $d_{model}$）视为一个高度为1、宽度等于序列长度的“图像”，那么FFN的线性变换对应于核大小为1x1、输入通道为 $d_{model}$、输出通道为 $d_{ff}$（对于第一层）或 $d_{model}$（对于第二层）的卷积。这种形式使得深度学习 (deep learning)框架能够利用高度优化的卷积实现，以进行跨序列长度的并行处理。

FFN子层与注意力子层一起，构成每个编码器和解码器层中的核心计算块。理解其结构和逐位置操作对于掌握Transformer如何在序列交互层面（注意力）和单个令牌层面（FFN）处理信息是必要的。此子层之后是残差连接和层归一化 (normalization)，我们将在接下来讨论。

## 参考资料

- [Attention Is All You Need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS) 30; Publisher: Curran Associates, Inc.; Pages: 5998-6008
  介绍Transformer架构的奠基性论文，详细阐述了位置前馈网络的结构、作用和原始参数。
- [Transformers for Natural Language Processing: From GPT-2 to BERT and Beyond](https://www.packtpub.com/product/transformers-for-natural-language-processing-second-edition/9781804616239) — Karthikeyan Vijayakumar, Sudharsan Ravichandiran, and Vignesh Jaganathan (2023)
  Publisher: Packt Publishing
  一本全面书籍，提供Transformer组件的易懂解释，包括在整体架构中对FFN的专门讨论。
- [Gaussian Error Linear Units (GELUs)](https://arxiv.org/pdf/1606.08415.pdf) — Dan Hendrycks and Kevin Gimpel (2017)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1606.08415](https://doi.org/10.48550/arXiv.1606.08415)
  介绍了GeLU激活函数，本节内容中指出它是现代Transformer模型中ReLU的替代方案。

---

[上一节](05-%E7%BC%96%E7%A0%81%E5%99%A8-%E8%A7%A3%E7%A0%81%E5%99%A8%E4%BA%A4%E5%8F%89%E6%B3%A8%E6%84%8F%E5%8A%9B.md) · [下一节](07-%E6%AE%8B%E5%B7%AE%E8%BF%9E%E6%8E%A5%20%28%E7%9B%B8%E5%8A%A0%29.md)
