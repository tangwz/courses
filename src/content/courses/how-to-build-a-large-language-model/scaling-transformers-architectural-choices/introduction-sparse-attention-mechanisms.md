---
course: "how-to-build-a-large-language-model"
chapter: "scaling-transformers-architectural-choices"
lesson: "introduction-sparse-attention-mechanisms"
sourceId: 6005
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-11-scaling-transformers-architectural-choices/introduction-sparse-attention-mechanisms"
title: "稀疏注意力机制简介"
description: "旨在降低自注意力二次方复杂度的技术概览"
order: 5
plots: []
sourceHash: "a4ead080003f5e2722cb4a7c4802f8e8929e4e3f54e22ddeccc43e9860f0ccf7"
sourceCorrections: []
---

随着Transformer模型的规模扩大，标准自注意力 (self-attention)机制 (attention mechanism)的计算需求很快成为一个主要瓶颈。自注意力计算序列中所有token之间的成对交互。如果序列长度为$N$，计算注意力分数矩阵($Q K^T$)的复杂度是$O(N^2 d)$，存储此矩阵和中间激活所需的内存也是$O(N^2)$，其中$d$是模型维度。对于几百甚至几千个token的序列来说尚可管理，但这种二次方扩展阻止了将标准Transformer应用于很长的序列，例如整个文档、视为补丁序列的高分辨率图像或扩展的音频流。处理长度为64,000的序列，其注意力计算所需的计算量将是长度为512的序列的1600万倍以上。

稀疏注意力机制提供了一种实用的办法，它通过修改自注意力层来仅计算所有可能的成对交互中的一个子集，有效地用稀疏矩阵替换了稠密的$N \times N$注意力矩阵。其主要假设是，对于许多任务，一个token不需要来自*所有*其他token的信息；相反，相关上下文 (context)可能是局部的，或者特定的全局token可能充当信息枢纽。通过选择哪些token对进行交互，我们的目标是将计算复杂度从$O(N^2)$降低到更易于处理的程度，通常是$O(N \log N)$甚至$O(N)$，同时保留了模型大部分的表达能力。

### 常见的稀疏模式

存在几种策略用于定义哪些token对应该相互关注。选择通常取决于对数据性质和任务的假设。

1. **滑动窗口（局部）注意力**：这是最简单的模式之一。每个token只关注固定数量$w$的前后token（其局部窗口）。复杂度变为$O(N \cdot w)$，如果$w$是常数，则相对于$N$是线性的。当局部上下文 (context)最重要时，例如在因果语言建模或图像处理中，这种模式很有效。
2. **膨胀滑动窗口**：为了捕获更长距离的依赖关系，同时不过度增加窗口大小$w$，可以引入膨胀机制。一个token可能会关注其窗口内距离为1、2、4、8等的邻居，类似于膨胀卷积。这使得感受野可以随层数呈指数增长，同时保持计算的线性。
3. **全局注意力**：某些token可能需要访问整个序列上下文，或作为信息的集成点。在此模式中，少量预先选择的token（例如，BERT类模型中的`[CLS]` token，或根据任务指定为重要的token）关注所有其他token，并且所有其他token也关注这些全局token。这通常与其他模式结合使用，例如滑动窗口注意力。
4. **随机注意力**：每个token除了关注其局部窗口外，还会关注固定数量的随机选择的token。这有助于信息以概率方式在序列中传播。
5. **分解注意力**：这涉及将完整注意力分解为多个成本较低的步骤。例如，注意力可能首先在固定的token块内计算，然后第二个注意力步骤可能发生在这些块的摘要表示之间。

### 示例：Longformer模式

Longformer架构提供了一个著名的例子，它结合了其中的几个思路。它主要使用滑动窗口注意力机制 (attention mechanism)。然而，为了实现信息在整个序列中的流动，它增加了全局注意力。根据任务确定的特定token（例如，用于分类的`[CLS]` token，用于问答的问题token）被允许全局关注，并且所有token都关注它们。

> 组合注意力模式的简化示意图。蓝色节点代表具有局部（滑动窗口）注意力的token。黄色节点（G）具有全局注意力，与所有其他token交互（橙色边）。局部连接显示为灰色。

这种组合使Longformer能够处理数千个token长度的序列，同时保持局部上下文 (context)感知和全局信息整合的能力，所有这些都伴随着计算复杂度随序列长度$N$线性增长的特点。

### 实现上的挑战

高效实现稀疏注意力通常不仅仅是在softmax之前应用一个掩码。标准的深度学习 (deep learning)库实现针对稠密矩阵乘法进行了高度优化。通过稀疏性获得性能提升通常需要：

- **定制核心（Kernel）**：使用专门的GPU核心（例如，用CUDA编写或使用Triton等库），可以高效地计算稀疏模式上的注意力，避免对零值项进行计算和内存访问。
- **块稀疏格式**：以适合硬件加速的格式表示稀疏注意力矩阵，例如块稀疏矩阵。OpenAI的Triton等库为此提供了工具。
- **管理稀疏模式**：代码需要高效地生成或检索定义每个token稀疏连接的索引。

这是一个在PyTorch中高度简化的草图，展示了如何为一个滑动窗口模式创建稀疏掩码。请注意，这*不*代表一个高效的实现，但它说明了掩码的原理。

```python
import torch
import torch.nn.functional as F

def simple_sliding_window_mask(sequence_length, window_size):
    """
创建滑动窗口注意力掩码。
    注意：仅用于说明，对于大型序列效率不高。
    """
    mask = torch.ones(sequence_length, sequence_length, dtype=torch.bool)
    half_window = window_size // 2
    for i in range(sequence_length):
        # 确定窗口边界，处理边缘情况
        start = max(0, i - half_window)
        end = min(sequence_length, i + half_window + 1) # Python切片需要+1

        # 允许窗口外的注意力
        mask[i, :start] = 0
        mask[i, end:] = 0

        # 可选：用于因果掩码（只关注过去和自身）
        # mask[i, i+1:] = 0

    return mask

# 示例用法：
seq_len = 10
window = 3
attention_scores = torch.randn(1, seq_len, seq_len) # 示例注意力分数（批次大小=1）

# 生成掩码
# 实际中，掩码会更高效地生成
# 并且通常会集成到定制核心中。
sparse_mask = simple_sliding_window_mask(seq_len, window)

# 应用掩码（在softmax之前将不允许的位置设置为-inf）
# 如果需要，为掩码添加批次维度
attention_scores.masked_fill_(~sparse_mask.unsqueeze(0), float('-inf'))

# 计算概率
attention_probs = F.softmax(attention_scores, dim=-1)

print("注意力掩码（True=允许）：\n", sparse_mask)
print(
    "\n掩码后的注意力概率（第0行）：\n",
    attention_probs[0, 0].detach().numpy().round(2)
)
```

"这段代码生成一个布尔掩码，其中`True`表示允许的注意力连接。然后，此掩码用于在softmax之前将不允许连接的分数设置为负无穷，以确保它们获得零概率。稀疏注意力实现绕过了完整稠密矩阵的创建，直接计算稀疏交互。"

稀疏注意力是一个活跃的研究方面，各种模式和高效实现不断出现。尽管与标准注意力相比它们增加了复杂性，但它们是促使Transformer架构应用于涉及超长序列问题的关键因素，拓展了大型模型能够处理的界限。权衡在于计算效率和通过限制token交互可能导致的信息损失之间。评估这种权衡通常需要在目标任务上进行经验性测试。

## 参考资料

- [Attention Is All You Need](https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems; Publisher: Neural Information Processing Systems; Volume: 30; DOI: [10.5555/3295222.3295349](https://doi.org/10.5555/3295222.3295349)
  介绍Transformer架构和自注意力机制的基础论文，强调其相对于序列长度的二次复杂度。
- [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) — Iz Beltagy, Matthew E. Peters, and Arman Cohan (2020)
  Journal: arXiv preprint arXiv:2004.05150; DOI: [10.48550/arXiv.2004.05150](https://doi.org/10.48550/arXiv.2004.05150)
  介绍Longformer模型，该模型结合滑动窗口和全局注意力模式，高效处理超长序列。
- [Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) — Manzil Zaheer, Guru Guruganesh, Kumar Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, Amr Ahmed (2020)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 33; Pages: 17283-17297; DOI: [10.5555/3455793.3455928](https://doi.org/10.5555/3455793.3455928)
  提出BigBird，一种结合全局、局部和随机注意力的稀疏注意力机制，实现线性复杂度，同时在长序列任务上保持良好性能。
