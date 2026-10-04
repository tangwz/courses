---
course: "advanced-diffusion-architectures"
chapter: "advanced-unet-architectures"
lesson: "unet-time-embeddings"
sourceId: 5386
sourceUrl: "https://apxml.com/zh/courses/advanced-diffusion-architectures/chapter-2-advanced-unet-architectures/unet-time-embeddings"
title: "在U-Net中整合时间嵌入"
description: "时间嵌入在扩散模型中的重要作用，解释了如何有效告知U-Net架构当前时间步，以获得卓越的去噪性能。"
order: 3
plots: []
sourceHash: "defcee0ffc2ee12ab1eaca61dbb455e8705904c30911a7017901d76ea54d5e1f"
sourceCorrections: []
---

整合时间嵌入 (embedding)。扩散模型在连续的噪声级别范围内工作，由时间步 $t$ 表示。模型的目标，无论是预测添加的噪声 $\epsilon$ 还是原始数据 $x_0$，都基本取决于 $t$。一个处理噪声极少（$t \approx 0$）图像的模型，其行为与处理接近纯噪声（$t \approx T$）的模型大相径庭。因此，有效告知神经网络 (neural network)（通常是U-Net）当前时间步 $t$，是成功构建扩散模型的核心要素。简单地将标量值 $t$ 直接作为输入是不够的，因为神经网络在这种情况下难以理解原始标量值的量级意义。

### 时间表示：从标量到嵌入 (embedding)

早期的尝试可能包括将 $t$ 归一化 (normalization)到特定范围（例如 [0, 1]），并按通道维度与输入特征图拼接。然而，这通常无法提供足够有表现力的信号。一种更有效的方法，受Transformer模型中位置编码 (positional encoding)的启发，是使用正弦时间嵌入。

给定一个时间步 $t$（通常是从 $0$ 到 $T$ 的整数），我们首先使用不同频率的函数将其映射到高维向量 (vector)：

$PE(t)_{2i} = \sin(t / 10000^{2i/d})$
$PE(t)_{2i+1} = \cos(t / 10000^{2i/d})$

这里，$i$ 是嵌入向量的维度索引，范围从 $0$ 到 $d/2 - 1$，其中 $d$ 是期望的嵌入维度（例如128、256、512）。这种表达形式为每个时间步提供了独特的向量表示。正弦特性在相邻时间步的嵌入之间创建了平滑过渡，并可能使网络能够泛化到训练中未见的时间步，尽管扩散模型通常在固定的离散时间步集上运行。

### 使用MLP处理时间嵌入 (embedding)

尽管正弦嵌入 $PE(t)$ 提供了丰富的表示，但它通常在注入U-Net之前会经过进一步处理。一个小型多层感知机（MLP），通常由两个线性层和一个非线性激活函数 (activation function)（如SiLU，即Sigmoid线性单元，也称Swish）组成，将固定的正弦嵌入转换为根据模型需要定制的学习到的特征表示。

设 $e = PE(t)$ 为正弦嵌入。MLP对其进行如下处理：

$t_{\text{emb}} = \text{Linear}_2(\text{SiLU}(\text{Linear}_1(e)))$

这个结果向量 (vector) $t_{\text{emb}}$ 以一种可以有效整合到U-Net卷积块中的格式，捕获了时间步信息。

### 将时间信息注入U-Net模块

一旦我们有了处理后的时间嵌入 (embedding) $t_{\text{emb}}$，我们需要将其整合到U-Net的架构中。存在几种策略：

#### 1. 加法整合

一种常见且直接的方法是将时间嵌入添加到U-Net残差模块中的中间特征图。通常，时间嵌入 $t_{\text{emb}}$ 首先被投影，以匹配将被添加到的特征图 $h$ 的通道数 $C$，这通常使用另一个线性层来完成。

$h' = \text{Conv}(h) + \text{Linear}_{\text{proj}}(t_{\text{emb}})[:, \text{None}, \text{None}, :]$

时间嵌入投影被重塑以进行广播（为空间高度和宽度添加维度，通常表示为 `[:, None, None, :]` 或等效的框架函数）之后进行元素级加法。这种加法通常发生在残差模块内的第一次卷积之后。

#### 2. 自适应归一化 (normalization) (AdaLN / AdaGN)

一种更精巧且通常更有效的方法是使用时间嵌入来调节归一化层的参数 (parameter)。时间嵌入并非仅仅添加信息，而是动态地控制组归一化（AdaGN）或层归一化（AdaLN）的缩放和平移参数。

回想一下，像GroupNorm这样的归一化层计算方式如下：

$\text{GroupNorm}(h) = \gamma \cdot \frac{h - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$

在自适应归一化中，缩放因子 $\gamma$ 和平移因子 $\beta$ 不再是固定的学习参数，而是由时间嵌入 $t_{\text{emb}}$ *生成*的。一个线性层将 $t_{\text{emb}}$ 映射以生成单独的 $\gamma(t)$ 和 $\beta(t)$ 向量 (vector)。

$[\gamma(t), \beta(t)] = \text{Linear}_{\text{AdaNorm}}(t_{\text{emb}})$

这些生成的参数随后用于对归一化特征图进行仿射变换：

$\text{AdaGN}(h, t) = \gamma(t) \cdot \text{GroupNorm}(h) + \beta(t)$

这使得时间步 $t$ 能够对每个模块内的特征统计数据进行精细控制，从而根据噪声水平有效调整计算的“风格”。这类似于StyleGAN等生成模型中的风格调制技术，并且在扩散模型中已被证明非常有效。AdaLN（自适应层归一化）工作方式类似，但以层归一化为基准。

以下图表说明了典型U-Net残差模块中的AdaGN机制：

> 此图说明了如何使用处理后的时间嵌入 $t_{\text{emb}}$ 来生成缩放参数 $\gamma(t)$ 和平移参数 $\beta(t)$，这些参数随后调节U-Net残差模块内组归一化层的输出（AdaGN）。

### 放置与考量

时间嵌入 (embedding)，无论是通过加法还是用于自适应归一化 (normalization)，通常会注入到U-Net的多个层中，尤其是在编码器和解码器路径中的每个残差模块内部。这确保了在整个网络计算过程中，时间步上下文 (context)信息都可用。

与简单的加法相比，AdaLN或AdaGN等自适应归一化方法通常能提供更优越的性能，因为它们为时间步提供了更具表达力的方式来影响特征分布。然而，它们也通过生成 $\gamma(t)$ 和 $\beta(t)$ 的投影层引入了稍多的参数 (parameter)。选择通常取决于具体的模型规模和性能要求。

总而言之，将标量时间步 $t$ 转换为使用正弦函数的高维嵌入，通过MLP处理它，并通过加法或更有效地说，自适应归一化将其整合到U-Net模块中，这些都是构建高性能扩散模型的标准且重要的方法。这确保了网络始终了解其在扩散轨迹上的位置，使其能够对任何给定的噪声水平执行正确的去噪操作。

## 参考资料

- [Denoising Diffusion Probabilistic Models](https://proceedings.neurips.cc/paper_files/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf) — Jonathan Ho, Ajay Jain, Pieter Abbeel (2020)
  Journal: Advances in Neural Information Processing Systems; DOI: [10.55917/b3b0-0b61](https://doi.org/10.55917/b3b0-0b61)
  介绍了采用正弦时间嵌入和自适应调制进行条件化的U-Net架构，是现代扩散模型的基础。
- [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c49a626a-Paper.pdf) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems; Publisher: Neural Information Processing Systems Foundation (NeurIPS); Volume: 30; Pages: 5998-6008; DOI: [10.55917/gh73-9a37](https://doi.org/10.55917/gh73-9a37)
  提出了Transformer架构并引入了正弦位置编码，为本文讨论的时间嵌入机制提供了思路。
- [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) — Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole (2020)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2011.13456](https://doi.org/10.48550/arXiv.2011.13456)
  为基于分数的生成模型（包括扩散模型）提供了一个统一框架，并详细阐述了使用类似嵌入策略对噪声水平进行条件化神经网络的方法。
