---
course: "synthetic-data-gans-diffusion"
chapter: "advanced-gan-architectures-techniques"
lesson: "attention-mechanisms-gans"
sourceId: 5657
sourceUrl: "https://apxml.com/zh/courses/synthetic-data-gans-diffusion/chapter-2-advanced-gan-architectures-techniques/attention-mechanisms-gans"
title: "GAN中的注意力机制"
description: "在GAN中整合自注意力及其他注意力模块。"
order: 5
plots: []
sourceHash: "4d95dc0e81773add60e704b2c22561f647670dbf0b88162798ca9d0e1f072051"
sourceCorrections: []
---

虽然GAN中的卷积层在捕捉局部模式方面表现出色，但其固定的局部感受野限制了它们表示图像中远距离依赖关系的能力。例如，生成一个逼真的动物可能需要确保头部朝向与肢体位置之间的一致性，这些特征在像素空间中可能相距很远。标准卷积对此类情况难以处理，需要多层才能在大距离上传播信息，这可能会稀释信号并增加计算负担，同时无法保证有效处理这些关系。

注意力机制 (attention mechanism)提供了一种有效的替代方案，使网络能够选择性地关注输入的相关部分，无论它们在空间上的距离如何。在GAN的背景下，尤其是图像生成，自注意力 (self-attention)已显示出高效。

### GAN中的自注意力 (self-attention)（SAGAN）

自注意力生成对抗网络 (GAN)（SAGAN）直接将自注意力模块引入到生成器和判别器的架构中。该模块使网络能够在计算给定位置的特征响应时，衡量所有其他位置特征的重要性。

#### 工作原理

考虑来自前一个卷积层的输入特征图 $x \in \mathbb{R}^{C \times H \times W}$。自注意力模块从该输入计算出三种表示：

1. **查询（Query）：** $q(x) = W_q x$
2. **键（Key）：** $k(x) = W_k x$
3. **值（Value）：** $v(x) = W_v x$

这里，$W_q$、$W_k$ 和 $W_v$ 是可学习的权重 (weight)矩阵，通常实现为 $1 \times 1$ 卷积。为了计算效率，维度经常被降低，例如，$W_q, W_k \in \mathbb{R}^{\bar{C} \times C}$ 和 $W_v \in \mathbb{R}^{C \times C}$，而 $\bar{C} = C/8$ 较为常见。

主要思路是根据查询和键之间的相似性来计算注意力权重。查询和键被重塑为 $\mathbb{R}^{\bar{C} \times N}$，N = H \times W 是特征图位置的数量。注意力图 $\beta \in \mathbb{R}^{N \times N}$ 计算如下：


$$
\beta_{j,i} = \frac{\exp(s_{ij})}{\sum_{k=1}^{N} \exp(s_{ik})} \quad \text{其中} \quad s_{ij} = q(x_i)^T k(x_j)
$$


此方程计算位置 $i$ 的查询与位置 $j$ 的键之间的点积相似性 $s_{ij}$。然后逐行应用 softmax 函数，将这些相似性归一化 (normalization)为注意力权重 $\beta_{j,i}$，表示模型在合成位置 $j$ 的输出时应如何关注位置 $i$。

值表示 $v(x)$ 被重塑为 $\mathbb{R}^{C \times N}$。注意力层的输出 $o \in \mathbb{R}^{C \times N}$ 然后使用注意力图计算为值特征的加权和：


$$
o_j = \sum_{i=1}^{N} \beta_{j,i} v(x_i)
$$


此输出 $o$ 被重塑回 $\mathbb{R}^{C \times H \times W}$。最后，该模块通常将此注意力导出的输出加回到原始输入特征图 $x$，通常通过一个初始化为0的可学习参数 (parameter) $\gamma$ 进行缩放：


$$
y = \gamma o + x
$$


将 $\gamma$ 初始化为0，使网络最初可以依赖局部卷积特征，并在训练过程中根据需要逐步学习整合非局部信息。

此图显示了应用于特征图的自注意力模块内部的数据流。输入特征被转换为查询、键和值表示。注意力权重通过矩阵乘法和softmax计算，然后施加到值特征上。结果被缩放并加回到原始输入。

### 在生成器和判别器中的整合

在SAGAN中，这些自注意力 (self-attention)模块通常放置于生成器和判别器的中间层。

- **生成器：** 添加注意力使生成器能够协调远距离空间位置的特征。这有助于合成具有复杂结构的对象或保持全局一致性，例如正确形成对象的形状或使背景保持一致性。
- **判别器：** 注意力使判别器能够更高效地执行对生成图像的复杂几何约束或一致性检查。它能够确认图像不同部分之间的关系根据真实数据分布是合理的。

### 优点与考虑事项

- **提升图像质量：** 注意力帮助GAN捕捉全局依赖关系，带来更逼真、更一致的图像合成，尤其适用于复杂场景或具有精细结构的对象。
- **几何一致性：** 明确表示远距离关联提升了生成样本的几何逼真度。
- **计算成本：** 主要缺点是计算成本。注意力图的矩阵乘法（$s = q^T k$）复杂度为 $O(N^2)$，N=H \times W 是像素数量。这种二次方增长使其对于极高分辨率的特征图代价高昂。因此，注意力模块通常仅应用于网络中较低或中等分辨率的特征图（例如 $32 \times 32$ 或 $64 \times 64$）。为了缓解处理更大图像的这一成本，已开发出轴向注意力或基于块的注意力等变体。
- **训练稳定性：** 尽管SAGAN旨在通过更好的表示来提升稳定性，但引入注意力有时会带来自身的训练动态，需要谨慎处理，通常结合下一章中讨论的技术，例如谱归一化 (normalization)。

“通过整合注意力机制 (attention mechanism)，像SAGAN这样的先进GAN架构克服了纯卷积方法的某些基本限制，使得能够生成具有更好全局结构和一致性的更高质量图像。这项技术是朝向捕获数据中复杂远距离关联的重要进展。”

## 参考资料

- [Attention Is All You Need](https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Publisher: Curran Associates, Inc.; Volume: 30; Pages: 5998-6008; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  一篇开创性论文，介绍了Transformer架构并推广了自注意力机制，阐释了其适用于各种深度学习任务的核心计算组件。
- [Generative Deep Learning: Teaching Machines to Paint, Write, Compose, and Play](https://oreil.ly/generative-dl) — David Foster (2023)
  Publisher: O'Reilly Media; Pages: 453
  一本内容全面的书籍，提供了生成模型的实践指导和理论基础，包括对GANs及其架构改进（如注意力机制的作用）的详细讨论。
