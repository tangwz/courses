# 使Transformer适应图像数据 (ViT, 图像块嵌入)

来源：[原文](https://apxml.com/zh/courses/advanced-diffusion-architectures/chapter-3-transformer-diffusion-models/transformers-for-images)

[返回章节目录](README.md) · [返回课程目录](../README.md)

Transformer模型的基础是处理标记 (token)序列，能有效地建立一维结构中元素间的关联。这与图像的网格状、二维（或更高维）性质形成鲜明对比，卷积神经网络 (neural network)（CNN）因其对局部性和空间层级的固有归纳偏置 (bias)，在这种数据上自然表现好。要在扩散模型中把Transformer自注意力 (self-attention)机制 (attention mechanism)的优点施加于图像数据，我们首先需要一种方式将图像表示成Transformer能处理的序列。目前主流的做法直接借鉴了使Transformer在视觉任务中普及的架构：Vision Transformer (ViT)。

### 从像素到图像块：图像数据序列化

主要做法是将输入图像划分为更小、不重叠的图像块网格。可以想象把一张图像裁剪成一系列正方形瓦片。每个瓦片，或称图像块，随后被视为输入到Transformer序列中的一个“标记 (token)”。

我们以输入图像 $x \in \mathbb{R}^{H \times W \times C}$ 为例，其中 $H$ 是高度，$W$ 是宽度，$C$ 是通道数（例如，RGB图像为3）。我们将此图像分成 $N$ 个大小为 $P \times P$ 的图像块。图像块的总数是 $N = (H/P) \times (W/P)$，这里假定 $H$ 和 $W$ 都能被 $P$ 整除。

每个图像块随后都被展平为一个向量 (vector)。单个图像块最初的维度是 $P \times P \times C$，因此其展平后的表示将是一个长度为 $P^2 \cdot C$ 的向量。

### 线性投影：生成图像块嵌入 (embedding)

这些展平后的图像块向量 (vector)，虽然是序列形式，但维度通常非常高 ($P^2 \cdot C$)。更重要的是，它们可能不处于一个对Transformer自注意力 (self-attention)机制 (attention mechanism)学习关联有利的表示空间。因此，每个展平后的图像块向量都被线性投影到一个较低维度的嵌入空间，通常以维度 $D$ 表示。这通过一个可学习的线性投影矩阵（一个嵌入层）实现：


$$
\text{嵌入} = \text{展平图像块} \times W_e + b_e
$$


其中 $W_e \in \mathbb{R}^{(P^2 \cdot C) \times D}$ 是可学习的权重 (weight)矩阵，$b_e \in \mathbb{R}^{D}$ 是偏置 (bias)项。经过此投影后，我们的图像被表示为 $N$ 个向量的序列，每个向量维度为 $D$。这个序列，$[z_1, z_2, ..., z_N]$，其中每个 $z_i \in \mathbb{R}^D$，现在是适合输入标准Transformer架构的格式。

> 将图像转换为可用于Transformer处理的图像块嵌入序列的步骤。

### 纳入空间信息：位置嵌入 (embedding)

标准的Transformer架构具有置换不变性；如果打乱输入序列，其输出（在不考虑原始位置时）会仅根据打乱后的内容而非原始顺序进行可预测的改变。然而，图像中图像块的空间排列是重要信息。左上角的图像块所包含的上下文 (context)信息与中心图像块的不同。

为了让Transformer获得这种重要的空间感知，位置嵌入被加入到图像块嵌入中。这些位置嵌入是与图像块嵌入具有相同维度 $D$ 的向量 (vector)。通常，网格中每个图像块的可能位置都会学习一个独特的位置嵌入向量。在将序列输入到第一个Transformer块之前，对应的位置嵌入会逐元素地添加到每个图像块嵌入中：


$$
z'_i = z_i + p_i
$$


其中 $z_i$ 是第 $i$ 个图像块的嵌入，$p_i$ 是对应于第 $i$ 个图像块空间位置的位置嵌入。这些位置嵌入 $p_i$ 通常随机初始化，并在训练期间与模型的其余部分一起学习。还有其他一些做法，例如使用固定的正弦位置嵌入，这类似于原始Transformer论文中使用的那些，有时也会针对二维情况进行修改。

### 与Vision Transformer (ViT) 的关联

图像块划分、展平、线性投影以及添加位置嵌入 (embedding)的整个过程，是Vision Transformer (ViT) 模型定义的标准输入处理流程。ViT表明，纯Transformer架构在以这种方式处理图像数据后，能在图像分类任务上获得当前最佳成果，动摇了CNN在该领域的主导。

接下来我们将研究Diffusion Transformers (DiTs)，它直接采用了这种ViT风格的输入处理。DiTs不是将图像块序列输入Transformer进行分类，而是将Transformer作为扩散框架中的骨干网络，以预测噪声（或原始图像）。在扩散环境中，此过程的输入通常是在给定时间步 $t$ 的带噪声图像 $x_t$。由此产生的、已包含空间信息的处理后图像块嵌入序列，作为后续Transformer块的输入，这些块负责去噪这项主要工作。时间和条件信息也会被包含进来，我们将在稍后说明。

这种调整有效地将图像生成问题变为Transformer可解决的序列建模问题，借助于Transformer在捕捉图像不同部分（图像块）之间全局关联方面的优点。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems; Volume: 30; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  介绍了Transformer架构和自注意力机制，为序列处理模型奠定了基础。
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) — Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby (2020)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2010.11929](https://doi.org/10.48550/arXiv.2010.11929)
  介绍了Vision Transformer (ViT)模型，详细说明了将图像转换为补丁序列并使用Transformer进行图像分类的方法。
- [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748) — William Peebles and Saining Xie (2023)
  Journal: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV); Pages: 18146-18158; DOI: [10.48550/arXiv.2212.09748](https://doi.org/10.48550/arXiv.2212.09748)
  引入了Diffusion Transformers (DiTs)，展示了ViT风格的输入处理和Transformer架构如何扩展应用于先进的扩散模型。

---

[上一节](01-%E7%94%9F%E6%88%90%E5%BC%8F%E5%BB%BA%E6%A8%A1%E4%B8%AD%E9%87%87%E7%94%A8Transformer%E6%A8%A1%E5%9E%8B%E7%9A%84%E7%BC%98%E7%94%B1.md) · [下一节](03-%E6%89%A9%E6%95%A3%E5%8F%98%E6%8D%A2%E5%99%A8%20%28DiT%29%EF%BC%9A%E6%9E%B6%E6%9E%84%E6%A6%82%E8%BF%B0.md)
