---
course: "cnns-for-computer-vision"
chapter: "attention-transformers-vision"
lesson: "vit-architecture-details"
sourceId: 2591
sourceUrl: "https://apxml.com/zh/courses/cnns-for-computer-vision/chapter-5-attention-transformers-vision/vit-architecture-details"
title: "ViT 架构：图像块、嵌入和 Transformer 编码器"
description: "对 ViT 架构的详细分析：图像分块、线性嵌入、位置嵌入以及 Transformer 编码器。"
order: 4
plots: []
sourceHash: "6290af4d1cf3e82554d85899233d4295ba4063fda676718b07508c26e667a55d"
sourceCorrections: []
---

Transformer 模型能处理远距离关联。视觉 Transformer (ViT) 将这种架构直接用于图像分类，大体上放弃了卷积神经网络 (neural network) (CNN) 中固有的特定归纳偏置 (bias)，例如局部性和平移等变性。该模型由 Dosovitskiy 等人在《一幅图像相当于 16x16 个单词：用于大规模图像识别的 Transformer》 (2020) 中提出，ViT 处理图像的方式是将其视为一系列较小的图像块，这与 Transformer 处理自然语言中词语序列的方式类似。

### 图像分块

ViT 的第一步是将输入图像分解为一系列固定大小的图像块。ViT 没有单独处理像素或使用滑动卷积窗口，而是将图像重塑为一系列扁平化的图像块。

设输入图像 $x$ 的尺寸为 $H \times W \times C$（高、宽、通道数）。ViT 将此图像划分为 $N$ 个不重叠的图像块，每个块的大小为 $P \times P$。图像块的总数是 $N = (H \times W) / P^2$。每个图像块随后被展平为一个大小为 $P^2 \cdot C$ 的向量 (vector)。

例如，一个 $224 \times 224$ 的 RGB 图像（$C=3$），如果使用 $P=16$ 的图像块大小进行处理，将得到 $N = (224 \times 224) / 16^2 = 14 \times 14 = 196$ 个图像块。每个图像块展平后为 $16 \times 16 \times 3 = 768$ 维。因此，图像从空间网格转换为一个包含 196 个向量的序列，每个向量的长度为 768。这种序列化对于与 Transformer 架构的适配极为重要。图像块大小 $P$ 的选择涉及到权衡：较小的图像块会产生更长的序列（$N$ 增加）和更细致的空间粒度，但会增加计算成本，而较大的图像块会得到较短的序列，但可能丢失细节信息。

### 图像块嵌入 (embedding)：线性投影

Transformer 处理的是特定维度（通常表示为 $D$）的嵌入向量 (vector)序列。扁平化的图像块，其维度为 $P^2 \cdot C$，需要被投影到这个 $D$ 维的嵌入空间。这通过可训练的线性投影实现，通常作为一个单一的线性层（或一个核大小和步长都等于 $P$ 的一维卷积）来完成。

令 $E \in \mathbb{R}^{(P^2 \cdot C) \times D}$ 为可学习的嵌入矩阵（线性层的权重 (weight)）。每个展平的图像块向量 $x_p^i \in \mathbb{R}^{P^2 \cdot C}$（其中 $i$ 的范围从 1 到 $N$）与 $E$ 相乘，生成图像块嵌入 $z_p^i \in \mathbb{R}^{D}$:

$z_p^i = x_p^i E$

这种投影使得模型能够学习图像块的合适维度表示，供后续的 Transformer 层使用。

### 可学习的类别标记 (token)

受 BERT 中 `[CLS]` 标记的启发，ViT 在图像块嵌入 (embedding)序列前添加了一个可学习的嵌入向量 (vector) $z_{class} \in \mathbb{R}^{D}$。这个类别标记与任何特定图像块没有直接关联，但它作为一个全局表示聚合器。它通过自注意力 (self-attention)机制 (attention mechanism)在整个 Transformer 编码器层中与图像块嵌入交互。经过整个编码器后，此类别标记的最终输出状态被用作聚合图像表示，用于分类。

### 位置嵌入 (embedding)

标准的 Transformer 架构是排列不变的；它不能天然地理解序列中标记 (token)的顺序或空间位置。然而，图像块的空间排列对图像理解显然很重要。为此，ViT 将位置嵌入添加到图像块嵌入中（包括类别标记）。

这些通常是标准的、可学习的一维位置嵌入 $E_{pos} \in \mathbb{R}^{(N+1) \times D}$。$E_{pos}$ 的每一行对应序列中的一个特定位置（0 代表类别标记，1 到 $N$ 代表图像块），整个矩阵在训练过程中学习。对应于每个序列元素的位置嵌入会逐元素地添加到图像块嵌入（或类别标记嵌入）中。

最终的嵌入向量 (vector)序列，作为 Transformer 编码器 ($z_0$) 的输入，其构成方式如下：

$z_0 = [z_{class}; z_p^1; z_p^2; \dots; z_p^N] + E_{pos}$

其中 $[;]$ 表示沿着序列维度进行拼接。该序列 $z_0$ 中的每个元素都是一个维度为 $D$ 的向量。

### Transformer 编码器

ViT 的核心是由 $L$ 个相同的 Transformer 编码器块堆叠而成。这些块负责处理嵌入 (embedding)序列，通过自注意力 (self-attention)机制 (attention mechanism)使得信息能在不同图像块表示之间传递。每个块通常包含两个主要子层：

1. **多头自注意力 (MHSA)：** 该机制允许序列中的每个元素（类别标记 (token)或图像块嵌入）关注所有其他元素，根据学习到的查询、键和值来衡量它们的重要性。“多头”方面指并行运行多次注意力机制，每次使用不同的学习投影，以捕捉不同类型的关联。它使模型能够通过关联远距离的图像块来理解全局信息。
2. **前馈网络 (MLP)：** 这通常是一个简单的多层感知机，由两个线性层组成，中间带有一个非线性激活函数 (activation function)（通常是 GELU - 高斯误差线性单元）。它独立应用于序列中的每个元素，提供额外的计算能力，并转换注意力层学习到的表示。

层归一化 (normalization) (LN) 应用于每个子层（MHSA 和 MLP）*之前*，并且残差连接应用于每个子层*周围*。单个编码器块 $l$（$l$ 从 1 到 $L$）中的处理可以概括为：

1. 归一化输入 $z_{l-1}$。
2. 对归一化后的输入计算 MHSA。
3. 将 MHSA 输出添加到原始输入（残差连接）：$z'_l = \text{MHSA}(\text{LN}(z_{l-1})) + z_{l-1}$
4. 归一化中间结果 $z'_l$。
5. 对归一化后的结果应用 MLP。
6. 将 MLP 输出添加到其输入（残差连接）：$z_l = \text{MLP}(\text{LN}(z'_l)) + z'_l$

最终块的输出 $z_L$ 包含类别标记和所有图像块的处理表示。

> 视觉 Transformer (ViT) 的整体架构。输入图像被分割成图像块，进行线性投影，并用位置嵌入和类别标记进行增强，经过一系列 Transformer 编码器层的处理，最后使用类别标记的输出表示进行分类。

### 分类头

经过 $L$ 个 Transformer 编码器层的处理后，对应于初始 `[class]` 标记 (token)的输出状态，表示为 $z_L^0 \in \mathbb{R}^{D}$，被认为是最终的图像表示。这个向量 (vector)随后被送入分类头，以生成最终的类别预测。通常，该分类头包含一个层归一化 (normalization)步骤，随后是一个将 $D$ 维表示映射到输出类别数量的线性层。

$y = \text{线性}(\text{层归一化}(z_L^0))$

这种方法与 CNNs 不同，后者在分类前通常对最终特征图使用全局平均池化。尽管对输出图像块嵌入 (embedding)进行全局池化是 ViT 中一种可能的替代方案，但使用专门的类别标记是原始论文中描述的标准方法。

总之，ViT 架构通过将图像转换为带有位置信息的嵌入式图像块序列，直接调整 Transformer 模型用于图像识别。它的主要优势在于 Transformer 编码器能够通过自注意力 (self-attention)机制 (attention mechanism)模拟图像块之间的全局关联，这与 CNN 的局部处理特点有显著不同。然而，这种缺乏强大空间归纳偏置 (bias)的情况通常意味着 ViT 相较于 CNN 需要更大的数据集或更广泛的预训练 (pre-training)才能达到相似的性能。

## 参考资料

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) — Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby (2020)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2010.11929](https://doi.org/10.48550/arXiv.2010.11929)
  引入 Vision Transformer (ViT) 架构的开创性论文，详细介绍了图像分块、线性嵌入、类别令牌和用于图像分类的 Transformer 编码器等组件。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Volume: 30; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  引入 Transformer 架构的奠基性论文，ViT 大量借鉴了该架构，特别是其带有多头自注意力机制和前馈网络的编码器结构。
- [A Survey of Vision Transformers](https://arxiv.org/abs/2012.12556) — Kai Han, Yunhe Wang, Hanting Chen, Xinghao Chen, Jianyuan Guo, Zhenhua Liu, Yehui Tang, An Xiao, Chunjing Xu, Yixing Xu, Zhaohui Yang, Yiman Zhang, Dacheng Tao (2022)
  Journal: IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI); Publisher: IEEE; Volume: 45; Pages: 10186-10204; DOI: [10.1109/TPAMI.2022.3152247](https://doi.org/10.1109/TPAMI.2022.3152247)
  一份全面的综述，提供了 Vision Transformer 架构、其变体、应用和挑战的概览，补充了原始 ViT 论文之外的背景信息。
- [Vision Transformer](https://d2l.ai/chapter_attention-mechanisms-and-transformers/vision-transformer.html) — Aston Zhang, Zachary C. Lipton, Mu Li, Alex Smola (2024)
  Journal: Dive into Deep Learning
  一本广受好评的开源深度学习教材中的章节，提供了 Vision Transformer 架构及其组件的结构化和易于理解的解释。
