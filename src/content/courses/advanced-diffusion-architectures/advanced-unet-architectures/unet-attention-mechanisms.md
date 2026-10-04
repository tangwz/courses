---
course: "advanced-diffusion-architectures"
chapter: "advanced-unet-architectures"
lesson: "unet-attention-mechanisms"
sourceId: 5384
sourceUrl: "https://apxml.com/zh/courses/advanced-diffusion-architectures/chapter-2-advanced-unet-architectures/unet-attention-mechanisms"
title: "U-Net中的注意力机制（自注意力、交叉注意力）"
description: "在U-Net中实现自注意力层和交叉注意力层，以改进特征表示和条件控制。"
order: 2
plots: []
sourceHash: "9e256121791c695f23c613a544c4d168611726fe6f12c52914ddf2547897f285"
sourceCorrections: []
---

U-Net中的卷积层擅长捕捉局部模式和空间层级，但其有效感受野随网络深度增长相对缓慢。这会限制模型捕获图像远距离关联的能力，而这对于生成一致的整体结构通常很重要。此外，集成条件信息（如文本提示）需要能够将条件信号与空间特征有效对齐 (alignment)的机制。注意力机制 (attention mechanism)为这两个难题提供了有效的解决方案。

### 自注意力 (self-attention)用于全局信息

自注意力层使特征图内不同的空间位置（像素或补丁）能够直接交互，无论它们相距多远。这使得网络能够比仅依赖堆叠卷积更有效地模拟远距离关联并获取全局信息。

在U-Net块内的一个典型自注意力模块中，输入特征图 $x \in \mathbb{R}^{H \times W \times C}$ 通过学习到的线性变换被映射为查询（$Q$）、键（$K$）和值（$V$）表示：

$Q = W_Q x$
$K = W_K x$
$V = W_V x$

这里，$W_Q, W_K, W_V$是可学习的权重 (weight)矩阵。注意力权重通过对查询和键的点积进行缩放，然后应用softmax函数来计算：

$注意力权重 = softmax(\frac{QK^T}{\sqrt{d_k}})$

其中$d_k$是键的维度，用于缩放。这些权重决定了每个空间位置对所有其他位置给予的“注意力”程度。自注意力层的最终输出是值向量 (vector)的加权和：

$输出 = 注意力权重 \cdot V$

这个富含全局信息的输出特征图，通常会通过残差连接加回到原始输入特征图，并由后续层进一步处理。

自注意力常被集成到U-Net的低分辨率（瓶颈）块中，那里特征图较小，使得注意力的二次方复杂度更容易处理。然而，线性注意力或局部注意力窗口等变体也允许其在更高分辨率下使用。

> 图示自注意力模块在U-Net残差块内的集成方式。该模块作用于特征图，以引入全局信息。

### 交叉注意力用于条件控制

自注意力 (self-attention)关联图像不同部分彼此，而交叉注意力关联图像部分与外部条件信号，如文本嵌入 (embedding)、类别标签，甚至来自另一图像的特征。这对于根据特定要求引导扩散过程具有基础作用。

在U-Net中用于条件控制的交叉注意力层（在Stable Diffusion等模型中很常见），查询向量 (vector)（$Q$）源自U-Net的空间特征图，而键（$K$）和值（$V$）向量源自条件上下文 (context)嵌入（例如，文本编码器中的token嵌入）：

$Q = W_Q x_{图像}$
$K = W_K c_{上下文}$
$V = W_V c_{上下文}$

这里，$x_{图像}$是U-Net的中间特征图，$c_{上下文}$是条件向量或序列。注意力机制 (attention mechanism)随后计算：

$输出 = softmax(\frac{QK^T}{\sqrt{d_k}})V$

输出表示受条件信息调节的图像特征。U-Net特征图（$Q$）中的每个空间位置都关注条件上下文（$K, V$）的元素，有效地将相关条件信号注入空间表示中。这使得模型能够生成与提供的文本提示或其他条件高度一致的图像。

交叉注意力层通常插入到U-Net架构内的多个分辨率级别，常与自注意力块并用，使条件信息能够影响整个网络的特征生成。

> 图示交叉注意力如何将条件信息（键、值）与U-Net的图像特征（查询）集成。

### 优点和权衡

集成注意力机制 (attention mechanism)显著提升了U-Net架构在扩散模型中的能力：

1. **改进的全局一致性：** 自注意力 (self-attention)有助于生成具有更好远距离一致性和全局结构理解的图像。
2. **有效的条件控制：** 交叉注意力提供了一种有效的方式，将条件信号（文本、类别标签等）注入并与空间生成过程对齐 (alignment)。
3. **增强的特征表示：** 这两种注意力类型都能优化特征图，突出相关信息并抑制噪声。

然而，这些优点也伴随着计算复杂度的增加。标准自注意力的复杂度与输入token（像素或补丁）数量呈平方关系，为$O(N^2)$，其中$N = H \times W$。交叉注意力的复杂度为$O(N \cdot M)$，其中$N$是图像特征序列长度，$M$是上下文 (context)序列长度。这使得注意力计算量大，尤其是在高分辨率时。多头注意力 (multi-head attention)（并行运行多个小维度注意力计算）等技术是标准做法，目前的研究也在寻找更高效的注意力变体（例如，稀疏注意力、线性注意力）来降低这些成本。

通过策略性地融入自注意力与交叉注意力，U-Net成为先进扩散模型更为强大的主干网络，能够生成与复杂条件输入高度一致的高质量图像。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Volume: 30; Pages: 5998-6008; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  提出了Transformer架构和多头自注意力机制，这是现代注意力机制的基本组成部分。
- [Non-local Neural Networks](https://arxiv.org/abs/1711.07971) — Xiaolong Wang, Ross Girshick, Abhinav Gupta, Kaiming He (2018)
  Journal: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR); Pages: 7794-7803; DOI: [10.48550/arXiv.1711.07971](https://doi.org/10.48550/arXiv.1711.07971)
  提出了非局部操作，这是一种用于捕获卷积神经网络中长距离依赖关系的自注意力泛化方法，对增强U-Net的上下文理解能力非常重要。
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer (2022)
  Journal: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR); Pages: 10684-10696; DOI: [10.48550/arXiv.2112.10752](https://doi.org/10.48550/arXiv.2112.10752)
  描述了潜在扩散模型，该模型使用增强了交叉注意力机制的U-Net骨干网络，能够有效地集成文本提示等条件信号以生成高质量图像。
