---
course: "mixture-of-experts-advanced-implementation"
chapter: "integrating-moe-into-architectures"
lesson: "moe-in-vision-transformers"
sourceId: 7075
sourceUrl: "https://apxml.com/zh/courses/mixture-of-experts-advanced-implementation/chapter-5-integrating-moe-into-architectures/moe-in-vision-transformers"
title: "视觉Transformer (ViT) 中的MoE"
description: "调整并应用混合专家理论到视觉Transformer架构，以完成图像处理任务。"
order: 3
plots: []
sourceHash: "627e3db438eee51c66b65686c0608de07ada85fa38e769336a6011ffedbc2f19"
sourceCorrections: []
---

将密集前馈网络（FFN）替换为稀疏MoE层的策略可以有效地应用于计算机视觉。这种方法尤其适用于视觉Transformer (ViT) 架构。通过调整MoE以适应ViT，可以构建具有大量参数 (parameter)的模型，能够学习到丰富的视觉特征层次，同时保持推理 (inference)和训练的计算成本可控。

在标准ViT中，输入图像首先被分割成一系列固定大小的图像块。这些图像块被展平，线性投影到嵌入 (embedding)空间中，然后由一系列Transformer编码器块处理。每个编码器块包含两个主要子层：多头自注意力 (self-attention)（MHSA）机制和位置感知前馈网络（FFN），后者通常是一个多层感知机（MLP）。

FFN是该模块中参数和计算的主要消耗者。我们旨在用混合专家层替换的就是这个组件。

> 该图展示了架构调整。标准ViT模块中的密集MLP（FFN）被替换为稀疏MoE层，而自注意力机制 (attention mechanism)和残差连接保持不变。

### 专家在视觉任务中的分工

在ViT中，一个“标记 (token)”对应一个嵌入 (embedding)的图像块。MoE层中的门控网络会学习将每个图像块嵌入路由到最适合处理该图像块的专家。这形成了引人注目的学习分工。在训练期间，不同的专家可能发展出识别不同视觉内容的能力：

- **低级特征：** 一些专家可能对基本模式很敏感，例如边缘、角点或特定纹理（如毛发、金属、水）。
- **物体部件：** 其他专家可能专注于识别更复杂的结构，例如眼睛、汽车轮子或花瓣。
- **位置分工：** 某些专家可能主要针对图像特定区域的图像块激活，例如中心或角点。
- **语义内容：** 在较深层中，专家可以学习响应更抽象的语义类别，为与“动物”或“车辆”相关的图像块激活。

这种分工使模型能够将参数 (parameter)分配给大量视觉模式，而无需每个图像块都由所有参数处理。一张田野中猫的图像主要会激活处理毛发、草地和有机形状的专家，而一张摩天大楼的图像则会激活处理直线、玻璃和几何图案的专家。

### 实现与路由

从代码角度来看，将MoE层整合到ViT模块中是直接的。门控网络是一个简单的线性层，它接收维度为 $d_{model}$ 的图像块嵌入 (embedding)，并为 $N$ 个专家输出对数几率（logits）。


$$
\text{对数几率} = \text{门控网络}(\text{图像块嵌入})
$$


其中 `GatingNetwork` 通常是 `torch.nn.Linear(d_model, N)`。`TopK` 路由机制随后选择专家，最终输出是选定专家输出的加权和，与基于语言的Transformer中类似。

一个简化的 `ViTMoEBlock` PyTorch实现展示了这种替换。

```python
import torch
import torch.nn as nn

# 假设MoELayer已在前面的章节中定义
# class MoELayer(nn.Module): ...

class ViTMoEBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_experts: int,
        top_k: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)

        # 用MoE层替换标准MLP
        self.moe_layer = MoELayer(
            input_dim=dim,
            num_experts=num_experts,
            top_k=top_k,
            # 每个专家都是一个标准FFN
            expert_hidden_dim=int(dim * mlp_ratio)
        )

    def forward(self, x: torch.Tensor):
        # 多头自注意力部分
        attn_output, _ = self.attn(*[self.norm1(x)] * 3)
        x = x + attn_output

        # MoE层部分
        moe_output, aux_loss = self.moe_layer(self.norm2(x))
        x = x + moe_output

        return x, aux_loss
```

### 性能与扩展

MoE在视觉领域的应用取得了显著成效。研究表明，ViT-MoE模型在相似计算预算（FLOPs）下，能够达到或超越密集模型的性能，同时训练步骤少得多。例如，一个拥有数万亿参数 (parameter)的ViT-MoE可以在ImageNet-21k或JFT-300M等基准测试中训练到高准确度，这表明稀疏模型是扩展视觉架构的有效途径。

核心权衡依然是关键：为了计算效率高的前向传播，你需要接受模型参数存储所需内存的大幅增加。这使得ViT-MoE特别适合于需要能力强的模型但又必须控制推理 (inference)延迟和成本的场景。

## 参考资料

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) — Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby (2020)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2010.11929](https://doi.org/10.48550/arXiv.2010.11929)
  介绍了视觉Transformer (ViT) 架构，它是将混合专家模型应用于图像数据的基础。
- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean (2017)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1701.06538](https://doi.org/10.48550/arXiv.1701.06538)
  提出了稀疏门控混合专家层的基础概念，是现代混合专家模型架构的核心组成部分。
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) — William Fedus, Barret Zoph, Noam Shazeer (2021)
  Journal: arXiv preprint; Volume: 23; Pages: 1-39; DOI: [10.48550/arXiv.2101.03961](https://doi.org/10.48550/arXiv.2101.03961)
  描述了Switch Transformer架构，展示了混合专家模型如何在大规模参数模型中保持计算效率。
- [Vision MoE: An Empirical Study of Scaling Laws for MoE in Vision](https://arxiv.org/abs/2203.05605) — William Fedus, Jeff Dean, Zhifeng Chen, Yuanzhong Xu, Anna Goldie, Basil Mustafa, Anushan Fernando, George Tucker, Yonghui Wu, David So, Blake Hechtman, Barret Zoph, David R. So, Aditya Sharma, Hieu Pham, Quoc V. Le, Paul Barham, Daniel N. Freeman, Albin Cassirer, Jiantao Jiao, Shibo Wang, Claire Cui, Ewa Dominowska, H. Yang, A. Mirhoseini (2022)
  Journal: International Conference on Machine Learning (ICML); DOI: [10.48550/arXiv.2203.05605](https://doi.org/10.48550/arXiv.2203.05605)
  研究了混合专家模型在视觉Transformer中的应用，详细说明了大型视觉模型的扩展性与性能。
