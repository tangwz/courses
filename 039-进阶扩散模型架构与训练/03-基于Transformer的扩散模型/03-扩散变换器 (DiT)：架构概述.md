# 扩散变换器 (DiT)：架构概述

来源：[原文](https://apxml.com/zh/courses/advanced-diffusion-architectures/chapter-3-transformer-diffusion-models/diffusion-transformers-dit)

[返回章节目录](README.md) · [返回课程目录](../README.md)

扩散变换器 (DiT) 架构代表了设计扩散模型主干网络方面的一个重要转变。它将变换器在图像数据上的成功应用（例如视觉变换器 (ViT)）的原理引入模型设计中。由 Peebles 和 Xie (2022) 提出，DiT 用纯变换器架构替代了常用的卷积 U-Net，展现出强大的性能和可扩展性，尤其在图像生成任务中。

其基本想法是，将时间步 $t$ 对含噪图像 $x_t$ 进行操作的扩散过程，视为一个适合变换器的序列建模问题。DiT 不依赖卷积层归纳偏置 (bias)（如局部性和平移等变性），而是运用变换器对图像中远距离依赖性进行建模的能力。

### 输入处理：为变换器准备图像

类似 ViT，DiT 无法直接处理原始像素网格。输入含噪图像 $x_t \in \mathbb{R}^{H \times W \times C}$ 必须首先转换成一系列令牌：

1. **图像分块：** 输入图像 $x_t$ 被划分为不重叠的图像块网格。对于高为 $H$、宽为 $W$ 的图像，使用 $P \times P$ 的图像块尺寸会产生 $N = (H \times W) / P^2$ 个图像块。
2. **线性嵌入 (embedding)：** 每个图像块被展平为一个向量 (vector)，然后线性投影成维度为 $D$ 的令牌嵌入。这就形成了一个由 $N$ 个令牌组成的初始序列：$z_0 \in \mathbb{R}^{N \times D}$。
3. **位置嵌入：** 由于变换器具有排列不变性，必须显式地添加位置信息。标准的、学习得到的 1D 或 2D 位置嵌入 $E_{pos} \in \mathbb{R}^{N \times D}$ 会被添加到图像块嵌入中：$z_0' = z_0 + E_{pos}$。这个序列 $z_0'$ 构成了变换器块的输入。

### 核心变换器块

DiT 的主体由 $L$ 个变换器块组成。每个块处理令牌序列，细化表示。一个标准 DiT 块通常包括：

1. **层归一化 (normalization) (LN)：** 应用于注意力层或 MLP 层之前，以稳定激活。
2. **多头自注意力 (self-attention) (MHSA)：** 允许序列中的每个令牌关注所有其他令牌，捕获图像块之间的全局关联。
3. **层归一化 (LN)：** 再次应用于前馈网络之前。
4. **前馈网络 (FFN)：** 通常是一个简单的多层感知机 (MLP)，包含两个线性层和一个非线性激活函数 (activation function)（例如 GeLU），独立应用于每个令牌。

MHSA 和 FFN 子层周围都使用了残差连接，保证梯度平滑流动，并使深度变换器的训练成为可能。


$$
z_l'' = \text{MHSA}(\text{LN}(z_{l-1}')) + z_{l-1}' \\
z_l' = \text{FFN}(\text{LN}(z_l'')) + z_l''
$$


在此，$z_{l-1}'$ 是第 $l$ 个块的输入，$z_l'$ 是其输出。

### 条件化：引入时间步和上下文 (context)

扩散模型必须以当前时间步 $t$ 为条件，并且可能以其他上下文信息 $c$（如类别标签、文本嵌入 (embedding)等）为条件。DiT 需要机制将这些条件信息注入到变换器块中。

1. **时间和条件嵌入：** 时间步 $t$ 首先被转换成一个向量 (vector)嵌入 $e_t$，通常使用正弦嵌入后接一个 MLP。类似地，上下文 $c$ 被映射到一个嵌入 $e_c$。
2. **自适应层归一化 (normalization) (adaLN / adaLN-Zero)：** 这是 DiT 中一种常见技术。不同于标准的层归一化，自适应归一化层会根据嵌入 $e_t$ 和 $e_c$ 动态计算尺度 ($\gamma$) 和偏移 ($\beta$) 参数 (parameter)。这些参数调节每个变换器块内的归一化激活，通常在 MHSA 和 FFN 层之前。
   
   $$
   \text{自适应层归一化}(h, e_t, e_c) = (\gamma) \cdot \text{层归一化}(h) + (\beta)
   $$
   
   其中 $\gamma$ 和 $\beta$ 是通过线性层投影 $e_t$ 和 $e_c$（通常是拼接或求和）得到的。“adaLN-Zero”变体初始化生成 $\gamma$ 和 $\beta$ 的投影层，使得初始输出接近于恒等映射，这可以通过保证条件化块在开始时表现得像残差连接来帮助训练稳定性。
3. **其他方法：** 尽管 adaLN 很常用，但也有其他选择：
   - 将 $e_t$ 和 $e_c$ 直接添加到令牌序列中（例如，将它们作为额外的令牌拼接，或添加到每个令牌嵌入中）。
   - 如果条件 $c$ 本身是一个序列（例如文本嵌入），则使用交叉注意力机制 (attention mechanism)。

### 输出处理：预测扩散目标

经过 $L$ 个变换器块处理后，输出令牌序列 $z_L'$ 需要转换回扩散目标所需的格式（通常是预测在步骤 $t$ 添加的噪声 $\epsilon$，或预测原始图像 $x_0$）。

1. **最终调节：** 条件嵌入 (embedding) $(e_t, e_c)$ 通常会使用一个最终的 adaLN 层或类似机制，对 $z_L'$ 进行最后一次输出调节。
2. **线性投影：** 一个最终的线性层将每个输出令牌嵌入投影回展平图像块的维度（例如，$P \times P \times C$）。
3. **反分块 / 重塑：** 输出令牌被重新排列（“反分块”）以重建最终的输出张量，该张量与输入图像具有相同的维度，表示预测的噪声 $\epsilon_\theta(x_t, t, c)$ 或 $x_0$。

### 总体架构图

下图阐明了扩散变换器内部信息的高层次流程：

> 扩散变换器 (DiT) 中的整体数据流。输入图像 $x_t$ 被分块并嵌入 (embedding)。变换器块处理这些令牌，并通过 adaLN 由时间 $t$ 和上下文 (context) $c$ 嵌入进行调节。最终令牌被投影并重塑，以生成扩散目标（例如，噪声 $\epsilon_\theta$）。

DiT 为 U-Net 提供了有力的替代方案。通过操作图像块序列，它们可以有效地对全局图像结构进行建模。它们的可扩展性（通过更大模型带来的性能提升得到证实）与大型语言模型中观察到的趋势一致，使其成为扩散生成建模持续发展中的重要架构。后续章节将比较 DiT 和 U-Net，并讨论实际实现细节。

## 参考资料

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) — Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit and Neil Houlsby (2020)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2010.11929](https://doi.org/10.48550/arXiv.2010.11929)
  介绍了Vision Transformer (ViT)，展示了如何通过图像分块和序列处理将Transformer应用于图像数据。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems 30 (NeurIPS); DOI: [https://doi.org/10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  介绍了Transformer架构，为不依赖循环或卷积的自注意力机制和序列到序列模型奠定了基础。

---

[上一节](02-%E4%BD%BFTransformer%E9%80%82%E5%BA%94%E5%9B%BE%E5%83%8F%E6%95%B0%E6%8D%AE%20%28ViT%2C%20%E5%9B%BE%E5%83%8F%E5%9D%97%E5%B5%8C%E5%85%A5%29.md) · [下一节](04-%E6%89%A9%E6%95%A3Transformer%E6%A8%A1%E5%9E%8B%E4%B8%AD%E7%9A%84%E6%9D%A1%E4%BB%B6%E4%BD%9C%E7%94%A8.md)
