---
course: "synthetic-data-gans-diffusion"
chapter: "diffusion-models-theory-implementation"
lesson: "diffusion-model-architectures"
sourceId: 5706
sourceUrl: "https://apxml.com/zh/courses/synthetic-data-gans-diffusion/chapter-4-diffusion-models-theory-implementation/diffusion-model-architectures"
title: "扩散模型 (U-Net) 的结构考量"
description: "为扩散模型设计有效的神经网络骨干。"
order: 6
plots: []
sourceHash: "04e9b6d4994b2193b0a845194e8d6f7366b1085a9a93350c23644d41387b9e5d"
sourceCorrections: []
---

扩散模型的有效性很大程度上取决于用于在逆向过程的每一步预测噪声（或分数）的神经网络 (neural network)结构。给定含噪输入 $\mathbf{x}_t$ 和时间步 $t$，网络（表示为 $\boldsymbol{\epsilon}_\theta$）必须学会估计从 $\mathbf{x}_{t-1}$ 到达 $\mathbf{x}_t$ 时所添加的噪声 $\boldsymbol{\epsilon}$（或更常见地，如简化损失函数 (loss function)中所用，从原始 $\mathbf{x}_0$ 到达）。此项工作需要一种能够有效处理空间信息并以时间变量 $t$ 为条件的结构。为此目的，尤其是在图像生成中，主要使用的结构是 **U-Net**。

### 噪声预测的U-Net结构

U-Net结构最初是为生物医学图像分割而开发的，但已被证明非常适合扩散模型中的条件生成任务。其结构有助于处理图像类数据，其中输出（预测噪声）必须与输入（含噪图像）具有相同的空间尺寸。

主要特点使U-Net适合此用处：

1. **编码器-解码器结构：** U-Net由一个下采样路径（编码器）组成，它以逐渐降低的空间分辨率但增加的特征通道捕获上下文 (context)信息；以及一个上采样路径（解码器），它逐渐重建空间分辨率。
2. **跳跃连接：** 重要的一点是，跳跃连接将编码器路径中的特征图直接连接到解码器路径中的对应层。这些连接使解码器能够获取编码器中的高分辨率特征，这对于准确预测噪声模式中的精细细节并最终生成清晰、细节丰富的图像非常重要。没有它们，在下采样过程中丢失的大部分空间信息将难以恢复。
3. **多尺度处理：** U-Net的分层特性使其能够同时处理多个尺度上的信息，从而有效捕获图像数据中的局部纹理和全局结构。

扩散模型中常见的U-Net实现包含：

- **下采样路径：** 一系列卷积块（通常包含卷积、归一化 (normalization)层和激活函数 (activation function)），其间穿插下采样操作（如最大池化或步长卷积）。
- **瓶颈：** 在最低空间分辨率处的一个或多个卷积块，连接编码器和解码器。
- **上采样路径：** 一系列包含上采样操作（如转置卷积或双线性/最近邻上采样后跟卷积）、将来自对应跳跃连接的特征进行拼接，以及进一步的卷积层的块。
- **输出层：** 最后一个卷积层（例如 $1 \times 1$ 卷积）将来自最后一个解码器阶段的特征映射到所需的输出形状，即预测噪声 $\boldsymbol{\epsilon}_\theta$，其尺寸与输入 $\mathbf{x}_t$ 相同。

> 示意图展示了扩散模型中常用的U-Net结构。输入 $\mathbf{x}_t$ 经过编码器路径（蓝色）、瓶颈（绿色）和解码器路径（红色），以产生预测噪声 $\boldsymbol{\epsilon}_\theta$。跳跃连接（虚线）将编码器中的特征图传递给解码器中的对应层。H、W、C 分别是高度、宽度、通道数；F 是特征的基础数量。

### 融入时间步信息

扩散过程随时间 $t$ 演变，要预测的噪声特性也相应改变（例如，当 $t$ 较大时噪声水平更高）。因此，U-Net必须以当前时间步 $t$ 为条件。一种普遍且有效的方法是使用**时间步嵌入 (embedding)**。

受Transformer中使用的位置嵌入启发，离散时间步 $t$（范围从 0 到 $T$）首先被转换为高维向量 (vector)表示。标准方法使用正弦嵌入：
$PE_{(t, 2i)} = \sin(t / 10000^{2i/d_{emb}})$
$PE_{(t, 2i+1)} = \cos(t / 10000^{2i/d_{emb}})$
其中 $d_{emb}$ 是嵌入向量的维度，$i$ 是向量的分量索引（$0 \le i < d_{emb}/2$）。这种固定嵌入为每个时间步提供了独特的表示，网络可以学习对其进行解释。通常，这种正弦嵌入在被集成到U-Net之前，会通过一个小的多层感知器（MLP）进行进一步处理。

时间嵌入向量 $emb(t)$ 在U-Net中如何使用？

- **加法/拼接：** 一种简单的方法是在空间上广播嵌入向量，并将其添加或拼接至网络中各处的特征图，通常是在U-Net的残差块中。
- **自适应组归一化 (normalization)（AdaGN）：** 嵌入可用于预测在归一化层（如组归一化）之后应用的尺度和偏移参数 (parameter)，根据时间步有效调节激活。
- **FiLM 层：** 特征维度线性调制层使用时间嵌入来预测每个特征通道的尺度（$\gamma$）和偏置 (bias)（$\beta$）向量。然后将这些向量逐元素应用于特征图 $\mathbf{h}$：$\gamma(emb(t)) \odot \mathbf{h} + \beta(emb(t))$。这使得时间步能够对网络的处理施加精细控制。

### 注意力机制 (attention mechanism)

虽然卷积层擅长捕获局部模式，但它们可能难以处理长距离依赖，尤其是在高分辨率图像中。为了解决这个问题，自注意力 (self-attention)机制常被整合到扩散模型的U-Net结构中。

- **自注意力层：** 这些层允许特征图中的不同空间位置（像素或块）相互关注，使模型能够捕获图像遥远部分之间的关系。它们通常被插入到U-Net的较低分辨率阶段（例如 $16 \times 16$ 或 $8 \times 8$ 特征图分辨率），那里的计算成本更易管理。加入注意力有助于模型生成更具全局一致性的结构。
- **交叉注意力：** 对于条件扩散模型（例如文本到图像），使用交叉注意力层。在这里，图像特征图（查询）关注条件信息，如文本嵌入 (embedding)（键、值）。这使得生成过程能够由条件输入引导。这种机制是诸如无分类器引导等技术的根本，其中条件向量 (vector)通过交叉注意力进行整合。

### 其他结构细节

其他一些设计选择也有助于U-Net在扩散模型中的表现：

- **激活函数 (activation function)：** 标准ReLU激活函数常被更平滑的替代品（如Swish或SiLU和GeLU）取代，这些替代品在深度生成模型中表现良好。
- **归一化 (normalization)：** 组归一化通常比批量归一化更受欢迎。批量归一化的统计数据依赖于批次，这可能会导致问题，因为含噪图像 $\mathbf{x}_t$ 的统计数据随时间步 $t$ 显著变化。组归一化为每个样本的通道组独立计算统计数据，使其在此背景下更稳定。
- **残差连接：** 在U-Net的每个卷积块中（除了主要的跳跃连接），通常使用残差连接。这些连接有助于稳定训练并通过减轻梯度消失来支持更深的网络。
- **网络大小：** U-Net的深度、每个级别的特征通道数以及注意力头的数量都是重要的超参数 (parameter) (hyperparameter)。更大的模型通常能产生更好的样本质量，但需要显著更多的计算资源和训练数据。

总而言之，U-Net结构，通过时间步嵌入 (embedding)、跳跃连接、残差块、合适的归一化和激活函数，以及常有的自注意力 (self-attention)机制 (attention mechanism)得到增强，为扩散模型中的核心噪声（或分数）预测任务提供了一个强大而灵活的框架。尽管变体和替代结构是当前研究的课题，但这种改进的U-Net仍然是许多先进扩散模型的基础，尤其是在图像方面。本章后面的实践部分将指导您实现这种结构的基本版本。

## 参考资料

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) — Olaf Ronneberger, Philipp Fischer, Thomas Brox (2015)
  Journal: Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015; Volume: 9351; Pages: 234-241; DOI: [10.48550/arXiv.1505.04597](https://doi.org/10.48550/arXiv.1505.04597)
  介绍了带有编码器-解码器结构和跳跃连接的U-Net架构，该架构成为图像到图像任务（包括扩散模型中的噪声预测）的基础。
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — Jonathan Ho, Ajay Jain, Pieter Abbeel (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Volume: 33; Pages: 6840-6851; DOI: [10.48550/arXiv.2006.11239](https://doi.org/10.48550/arXiv.2006.11239)
  提出了开创性的去噪扩散概率模型（DDPM）框架，确立了在逆向扩散过程中使用U-Net进行噪声预测，并采用简化的损失函数。
- [Diffusion Models Beat GANs on Image Synthesis](https://doi.org/10.48550/arXiv.2105.05233) — Prafulla Dhariwal, Alex Nichol (2021)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Volume: 34; Pages: 8780-8794; DOI: [10.48550/arXiv.2105.05233](https://doi.org/10.48550/arXiv.2105.05233)
  详细介绍了扩散模型中U-Net的架构选择，包括自注意力层、组归一化和有效的时间步条件作用，从而显著提高了图像合成质量。
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer (2022)
  Journal: CVPR 2022; Pages: 10684-10695; DOI: [10.48550/arXiv.2112.10752](https://doi.org/10.48550/arXiv.2112.10752)
  介绍了潜在扩散模型，该模型将U-Net架构应用于潜在空间，并结合自注意力与交叉注意力机制实现高分辨率、条件式图像生成。
