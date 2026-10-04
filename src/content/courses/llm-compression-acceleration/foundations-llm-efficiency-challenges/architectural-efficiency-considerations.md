---
course: "llm-compression-acceleration"
chapter: "foundations-llm-efficiency-challenges"
lesson: "architectural-efficiency-considerations"
sourceId: 4300
sourceUrl: "https://apxml.com/zh/courses/llm-compression-acceleration/chapter-1-foundations-llm-efficiency-challenges/architectural-efficiency-considerations"
title: "实现效率的架构考量"
description: "分析Transformer组件（如注意力机制和前馈网络）如何影响计算效率。"
order: 3
plots: ["plots/4300-0.json"]
sourceHash: "f53470d55a4bc2361188fefa8c5b09e72df1e5c2d2a3ce99e07ba0962e0eb4ef"
sourceCorrections: []
---

虽然扩展法则描述了模型大小、数据和性能之间的关系，但底层架构的*具体*设计精确决定了计算资源的消耗位置。现代大型语言模型（LLM）的主要架构是Transformer。了解其核心组件如何影响计算成本和内存占用，是制定有效优化方案的前提。架构蓝图中的选择直接影响推理 (inference)延迟和硬件需求。

## Transformer组件分析

Transformer架构通常以层叠方式构建，它依赖于一些重要的构造模块。每个模块都有独特的计算特性，并为提高效率提供了独特的机会和挑战。

### 自注意力 (self-attention)机制 (attention mechanism)

自注意力机制允许模型在处理特定token时，衡量输入序列中不同token的重要性。这项能力对LLM理解上下文 (context)起着主要作用。然而，它也带来了显著的计算成本。

标准的点积注意力计算涉及三个学习到的矩阵（查询 $W_Q$、键 $W_K$、值 $W_V$），它们将输入嵌入 (embedding)投影为Q、K和V向量 (vector)。注意力分数计算如下：


$$
\text{注意力}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$


令 $n$ 为序列长度，$d$ 为模型的隐藏维度（嵌入大小）。

1. **$QK^T$ 计算：** 计算所有查询和键向量之间的点积，需要将一个 $(n \times d_k)$ 大小的矩阵乘以一个 $(d_k \times n)$ 大小的矩阵，生成一个 $(n \times n)$ 的注意力分数矩阵。这一步骤的计算复杂度约为 $O(n^2 d_k)$。
2. **注意力分数应用：** 将 $(n \times n)$ 的注意力分数矩阵乘以值矩阵V（大小为 $n \times d_v$），需要 $O(n^2 d_v)$ 次操作。

假设 $d_k = d_v = d / h$，其中 $h$ 是注意力头的数量（将在后续讨论），单个注意力头的总复杂度主要由序列长度的平方依赖性决定，约为 $O(n^2 d)$。这种平方级增长使得处理长序列的计算要求很高，是一个主要的瓶颈。

此外，中间的 $(n \times n)$ 注意力分数矩阵需要 $O(n^2)$ 的内存存储。对于长序列（例如 $n > 4096$），这种内存需求会变得很可观，经常超出处理单元的缓存容量，并导致内存带宽限制。

> 缩放点积自注意力机制的基本流程，着重说明了矩阵乘法。

### 多头注意力 (multi-head attention)（MHA）

多头注意力（MHA）并非执行单次注意力计算，而是将Q、K和V向量 (vector)通过 $h$ 次不同的学习投影进行线性投影。注意力为每个“头”独立并行计算，然后将结果拼接并再次投影。

- **并行性：** MHA允许模型共同关注来自不同表示子空间、不同位置的信息。在计算上，它将 $d$ 维空间拆分为 $h$ 个更小的子空间（通常 $d_k = d_v = d/h$），可能通过并行性提高硬件利用率。
- **成本：** 尽管总计算量与维度为 $d$ 的单头注意力大致相似（因为操作在各头之间分摊），但MHA由于多次投影和最终输出投影而涉及更多矩阵乘法。这些投影矩阵导致参数 (parameter)数量增加，但通常*每个头*的参数更少。对于注意力分数， $O(n^2 d)$ 的复杂度和 $O(n^2)$ 的内存需求仍然是主要因素。

### 前馈网络（FFN）

在每个Transformer块的注意力机制 (attention mechanism)之后，会有一个逐位置的前馈网络（FFN）。这通常由两个线性变换组成，中间夹有一个非线性激活函数 (activation function)（如ReLU、GeLU或SwiGLU）。


$$
FFN(x) = \text{激活}(xW_1 + b_1)W_2 + b_2
$$


中间（“扩展”）层维度 $d_{ff}$ 通常大于模型维度 $d$，通常为 $d_{ff} = 4d$。

- **参数 (parameter)：** FFN层占据了LLM总参数的很大一部分。例如，在标准的Transformer块中，FFN参数（第一层为 $d \times d_{ff}$，第二层为 $d_{ff} \times d$）显著超过注意力参数（主要由Q、K、V投影的 $3 \times (d \times d)$ 和一个输出投影组成）。FFN通常占LLM总参数的大约三分之二。
- **计算：** FFN的计算成本约为 $O(n \cdot d \cdot d_{ff})$。虽然其计算复杂度与序列长度 $n$ 呈线性关系，但较大的维度 $d$ 和 $d_{ff}$ 使FFN成为计算密集度高的部分。这些密集的矩阵乘法在现代硬件上通常是计算受限的。



![Transformer块中参数的大致分布](plots/4300-0.json)



> 典型Transformer块中注意力投影层和前馈网络层之间参数的示意分布。FFN通常占主要部分。

### 层归一化 (normalization)和残差连接

- **层归一化（LayerNorm）：** 通常在注意力模块和FFN模块之前或之后应用（前归一化 vs. 后归一化），LayerNorm在特征维度上对激活进行归一化。它的计算涉及均值和方差的计算，然后进行缩放和偏移。尽管其成本低于注意力或FFN（$O(n d)$），但它包含规约和逐元素操作，会增加总体延迟并需要内存访问。
- **残差连接：** 这些跳跃连接（$x + \text{子层}(x)$）通过使梯度更容易流动，对训练深度网络非常重要。它们涉及简单的逐元素相加，引入最小的计算开销，但会影响数据流和内存访问模式。

## 架构选择及其效率影响

这些组件的具体配置影响整体效率：

- **隐藏维度 ($d$)：** 增加 $d$ 通常会提升模型容量，但会使FFN中的计算呈平方级增长（每个token $O(d^2)$），并显著影响注意力计算 ($O(n^2 d)$)。参数 (parameter)和激活的内存也相应地扩展。
- **层数：** 更多层会增加深度和表示能力，但会线性增加顺序计算时间和总参数数量。
- **注意力头数量 ($h$)：** 影响注意力的粒度和并行潜力。
- **FFN扩展因子：** 比例 $d_{ff}/d$ 直接影响FFN的参数数量和计算成本。使用SwiGLU等变体可能涉及FFN中的三个矩阵而非两个，进一步增加了参数。
- **序列长度 ($n$)：** 注意力机制 (attention mechanism)中 $O(n^2)$ 的平方级复杂度使得处理长序列成为一个主要的性能挑战。

理解这些架构瓶颈是迈向有针对性优化的第一步。量化 (quantization)和剪枝等技术通常会因参数数量的原因而重点关注大型FFN层。旨在改善长序列延迟的方法必须处理注意力机制的 $O(n^2)$ 复杂度，通常通过专门的核（如FlashAttention）或近似注意力算法来实现。内存节省技术对于管理大型激活张量，特别是注意力分数，不可或缺。这些组件及其资源需求之间的关系，促使我们将在本课程中研究多样化的压缩和加速技术。

## 参考资料

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Aware Operators](https://arxiv.org/abs/2205.14135) — Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré (2022)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2205.14135](https://doi.org/10.48550/arXiv.2205.14135)
  提出了一种解决二次复杂度的内存高效注意力算法。
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei (2020)
  Journal: arXiv preprint arXiv:2001.08361; DOI: [10.48550/arXiv.2001.08361](https://doi.org/10.48550/arXiv.2001.08361)
  研究了模型性能如何随模型大小、数据集大小和计算量而变化。
