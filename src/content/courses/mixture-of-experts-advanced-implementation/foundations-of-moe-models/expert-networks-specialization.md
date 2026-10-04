---
course: "mixture-of-experts-advanced-implementation"
chapter: "foundations-of-moe-models"
lesson: "expert-networks-specialization"
sourceId: 7046
sourceUrl: "https://apxml.com/zh/courses/mixture-of-experts-advanced-implementation/chapter-1-foundations-of-moe-models/expert-networks-specialization"
title: "专家网络：专精与容量"
description: "介绍MoE模型中的专家网络如何形成专门的功能并提升模型的整体容量。"
order: 3
plots: []
sourceHash: "eb8720b552b7660624044c9e0c2baa2de90447f8ad1de86fa17ac9c4c4559c21"
sourceCorrections: []
---

专家混合模型的核心是专家网络本身。门控网络像一个流量控制器，而专家则是进行实际复杂计算的专门“目的地”。一个单一的大型前馈网络 (FFN) 在标准Transformer中负责处理所有标记 (token)。MoE层用一组更小、并行的FFN取代这个单一的FFN，每个都称为一个“专家”。

### 专家网络的结构

单个专家的结构很简单。在大多数现代MoE实现中，例如大型语言模型中使用的那些，一个专家只是一个标准的双层前馈网络。它由以下部分构成：

1. 一个上投影层，它是一个线性变换，用于扩展输入标记 (token)嵌入 (embedding)的维度（例如，从`d_model`到`d_ff`）。
2. 一个非线性激活函数 (activation function)，例如GELU或SwiGLU。
3. 一个下投影层，它是另一个线性变换，将维度降低回原始的`d_model`。

这个结构与标准Transformer层中的FFN块相同。区别不在于单个专家的复杂性，而在于它们的数量。

> 一个典型专家网络的内部架构。它与Transformer模型中标准FFN块的结构相同。

### 扩展模型容量，而非计算成本

使用专家的主要动机是将模型参数 (parameter)的增长与计算成本（以FLOPs衡量）的增长分离。考虑一个带有FFN块的标准Transformer。如果想增加其容量，必须使FFN的隐藏层更大。这会同时增加通过该层的每个标记 (token)的参数数量和FLOPs。

MoE模型提供了一种更有效的扩展方式。不使用一个巨大的FFN，你可以使用8个、16个甚至64个专家，每个专家的大小都与原始FFN相同。

让我们通过一个具体例子来分析这种权衡。假设一个标记在总共`N=8`个专家中只被路由到`k=2`个专家：

- **参数数量：** 模型存储所有8个专家的权重 (weight)。与密集模型相比，这导致FFN层的参数增加了8倍。
- **计算成本 (FLOPs)：** 由于每个标记只由2个专家处理，计算成本仅为密集模型FFN的2倍，而不是8倍。

这就是稀疏模型的核心优势：你可以构建一个拥有大量参数的模型，但每个输入只激活其中一小部分，且计算上易于管理。该模型在存储知识方面“大”，但在任何给定前向传递的活跃计算方面“小”。

### 专门化的形成

专家没有预先编程特定的功能。它们的专门化是训练过程中的一种自发属性，完全由门控网络的路由决策驱动。随着模型训练，门控网络学着将具有相似特点的标记 (token)路由到相同的专家组。优化过程在整体任务损失和辅助负载均衡损失的指导下，促进了这种行为。

随着时间推移，可以观察到明显的模式：

- 一个专家可能擅长处理句法结构，例如标点符号或语法模板。
- 另一个专家可能专门处理与特定方面相关的词汇，例如科学术语或编程语言关键词。
- 第三个可能处理更抽象的语义概念。

这种学习得来的分工让模型能够将能力分配到数据的不同方面。不再需要单个FFN样样精通，每个专家都可以成为特定方面的行家。

> 门控网络学着将标记路由到那些已专门处理特定类型数据（例如编程语法或科学术语）的专家。

### PyTorch中的一个简单专家实现

为了更具体地说明，以下是单个专家在PyTorch中的样子。它是一个简单的`nn.Sequential`模块，包含我们描述的线性层和激活函数 (activation function)。

```python
import torch.nn as nn

class Expert(nn.Module):
    """
    一个简单的前馈网络，用作MoE层中的专家。
    """
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)

# 示例实例化
# d_model = 512 (嵌入维度)
# d_ff = 2048 (中间前馈维度)
expert_network = Expert(d_model=512, d_ff=2048)
print(expert_network)
```

实例化这个类的输出是：

```
Expert(
  (net): Sequential(
    (0): Linear(in_features=512, out_features=2048, bias=True)
    (1): GELU(approximate='none')
    (2): Linear(in_features=2048, out_features=512, bias=True)
  )
)
```

这个简单、可重用模块构成了MoE架构强大扩展能力的构建块。在接下来的章节中，我们将了解这个`Expert`模块如何与门控网络结合，形成一个完整的MoE层。

## 参考资料

- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean (2017)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1701.06538](https://doi.org/10.48550/arXiv.1701.06538)
  介绍了稀疏门控专家混合（MoE）层，用于高效扩展神经网络，是现代MoE架构的基础工作。
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/pdf/2101.03961.pdf) — William Fedus, Barret Zoph, Noam Shazeer (2022)
  Journal: Journal of Machine Learning Research; Publisher: Microtome Publishing; Volume: 23; Pages: 1-39; DOI: [10.48550/arXiv.2101.03961](https://doi.org/10.48550/arXiv.2101.03961)
  提出了一种简化且高效的稀疏MoE架构（Switch Transformer），实现了万亿参数模型的扩展，并强调了实际实现。
- [Attention Is All You Need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS) 30; Publisher: Curran Associates, Inc.; Pages: 5998-6008; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  介绍了Transformer架构，其前馈网络（FFN）模块为现代MoE模型中的各个专家网络提供了结构范本。
