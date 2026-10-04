---
course: "foundations-transformers-architecture"
chapter: "encoder-decoder-stacks"
lesson: "final-linear-softmax"
sourceId: 2028
sourceUrl: "https://apxml.com/zh/courses/foundations-transformers-architecture/chapter-5-encoder-decoder-stacks/final-linear-softmax"
title: "最终线性层和Softmax输出"
description: "将解码器输出向量转换为词汇表上的概率分布。"
order: 10
plots: []
sourceHash: "7cb74d63f4a7e95994d5b7f683f1c392b199a211348c8b9d6468ff3d0293abef"
sourceCorrections: []
---

Transformer架构中的最终解码器层对序列进行处理，它会整合来自输入序列的信息（通过交叉注意力）和之前生成的输出标记 (token)（通过遮蔽自注意力 (self-attention)）。这个过程会产生一系列高维表示向量 (vector)。对于输出序列中的每个位置，解码器堆栈会生成一个维度为 $d_{model}$ 的向量。尽管这些向量编码了丰富的上下文 (context)信息，但它们不能直接解读为目标词汇表 (vocabulary)上的概率分布。这种分布对于机器翻译或文本生成等任务是必不可少的。

Transformer解码器架构的最后一步涉及将这些输出向量转换为可用的概率。这通常通过两个连续操作来实现：一个最终线性变换，然后是一个softmax激活函数 (activation function)。

### 最终线性变换

顶部解码器层的输出是一个形状为 (batch\_size, target\_sequence\_length, $d_{model}$) 的张量。最终线性层的作用是将每个位置上的这个 $d_{model}$ 维表示向量 (vector)投影到一个维度与目标词汇表 (vocabulary)大小相等的向量上，我们将这个大小记作 $V$。

这是一个标准的、不带激活函数 (activation function)（或有时视为具有恒等激活函数）的全连接线性层。如果我们将最终解码器层的输出表示为 $H_{dec} \in \mathbb{R}^{\text{batch\_size} \times \text{target\_sequence\_length} \times d_{model}}$，将线性层的权重 (weight)矩阵表示为 $W_{out} \in \mathbb{R}^{d_{model} \times V}$，将其偏置 (bias)表示为 $b_{out} \in \mathbb{R}^{V}$，则该操作可以描述为：


$$
\text{Logits} = H_{dec} W_{out} + b_{out}
$$


在这里，矩阵乘法独立地应用于 `target_sequence_length` 维度上每个位置的 $d_{model}$ 维向量。得到的 `Logits` 张量形状为 (batch\_size, target\_sequence\_length, $V$)。序列中特定位置 `t` 处大小为 $V$ 的每个向量都包含目标词汇表中每个可能标记 (token)的原始、未归一化 (normalization)分数（logits）。分数越高，表示该位置上该标记的可能性越大。

*实现备注：* 一种常见的方法，尤其是在源和目标词汇表共享（或紧密关联）的模型中，是在输入嵌入 (embedding)层和这个最终线性层之间共享权重。输入嵌入层将词汇表索引（实际上是独热向量）投影到 $d_{model}$ 维度，而最终线性层将 $d_{model}$ 维度投影回词汇表分数。与输入嵌入矩阵（可能转置）共享权重矩阵 $W_{out}$ 可以大幅减少模型参数 (parameter)的数量，特别是对于大型词汇表，并且已被经验证实效果良好。

### Softmax 函数

线性层生成的 logits 是实值分数，不构成概率分布。为了将这些分数转换为概率，softmax 函数独立地应用于目标序列中每个位置的 logit 向量 (vector)。

对于序列中的特定位置 $t$，令 $z_t \in \mathbb{R}^V$ 为 logit 向量。softmax 函数计算词汇表 (vocabulary)中第 $i$ 个标记 (token)的概率 $p_i$（其中 $i$ 的范围从 1 到 $V$），如下所示：


$$
p_i = \text{Softmax}(z_t)_i = \frac{e^{z_{t,i}}}{\sum_{j=1}^{V} e^{z_{t,j}}}
$$


此操作为每个位置 $t$ 产生一个概率向量 $p_t \in \mathbb{R}^V$，其中：

1. 每个元素 $p_i$ 都是非负的 ($p_i \ge 0$)。
2. 向量中所有元素的和为 1 ($\sum_{i=1}^{V} p_i = 1$)。

因此，整个Transformer解码器堆栈的输出是一个形状为 (batch\_size, target\_sequence\_length, $V$) 的张量，其中沿最后一个维度的每个向量都表示目标词汇表上的概率分布。

### 训练和推理 (inference)中的使用

- **训练：** 在训练期间，此输出概率分布用于计算损失，通常是交叉熵损失，与每个位置的实际目标标记 (token)进行对比。模型的参数 (parameter)（包括最终线性层、注意力机制 (attention mechanism)、前馈网络等中的参数）通过反向传播 (backpropagation)更新以最小化此损失，从而有效学习在给定上下文 (context)的情况下预测正确的下一个标记。标签平滑等技术通常在计算损失前应用于目标分布，这可以提高泛化能力。
- **推理：** 在推理期间（例如，生成文本），模型以自回归 (autoregressive)方式运行。在每个步骤 $t$，softmax层生成的概率分布 $p_t$ 用于选择下一个标记。常见的策略包括：
  - **贪婪解码：** 简单选择概率最高的标记 ($argmax_i p_i$)。
  - **束搜索：** 保持多个候选序列（束），并在每个步骤中寻找最有可能的扩展，可能获得比贪婪解码更好的结果。
  - **采样方法：** 通过从概率分布中采样（例如，温度采样、top-k采样、核采样）引入随机性，以生成更多样化的输出。

这个最终线性层和softmax函数连接了Transformer学习到的复杂内部表示与语言生成任务的离散、概率性质。它们提供了所需的机制，将模型的理解转换为词汇表 (vocabulary)上的明确预测。

## 参考资料

- [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 30; Pages: 5998-6008
  介绍Transformer架构的基础论文，详细说明解码器的最终线性层和softmax、权重共享机制以及标签平滑。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  权威的深度学习教材，涵盖深度学习的核心概念，包括线性层、softmax函数、概率分布和交叉熵损失。
