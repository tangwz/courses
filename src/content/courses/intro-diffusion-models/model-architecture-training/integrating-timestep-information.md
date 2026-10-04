---
course: "intro-diffusion-models"
chapter: "model-architecture-training"
lesson: "integrating-timestep-information"
sourceId: 5406
sourceUrl: "https://apxml.com/zh/courses/intro-diffusion-models/chapter-4-model-architecture-training/integrating-timestep-information"
title: "整合时间步信息"
description: "使U-Net适应当前扩散时间步t的方法 (例如，正弦编码)。"
order: 2
plots: ["plots/5406-0.json"]
sourceHash: "5aa108188cd78ecd692e77f68c0086c90d75c02259b99c8fb12d1acb231f2ed8"
sourceCorrections: []
---

逆向扩散过程的核心是一个神经网络 (neural network)，通常是U-Net，它的任务是预测添加到图像 $x_0$ 中以在特定时间步 $t$ 产生噪声版本 $x_t$ 的噪声 $\epsilon$。这个过程的一个主要方面是，*相同* 的网络参数 (parameter) $\theta$ 必须用于 *所有* 可能的时间步 $t$ (从 $0$ 到 $T$)。

单个网络如何根据当前时间步调整其行为？它需要被“告知”当前正在处理哪个时间步 $t$。简单地将整数 $t$ 直接输入网络通常不是最佳选择。神经网络通常更适合处理归一化 (normalization)输入，并且原始整数无法轻易传达扩散过程中的“位置”，以供网络充分利用。此外，不同的时间步需要非常不同的去噪行为；$t=50$ 时的噪声模式和幅度与 $t=800$ 时的有显著差异。

因此，我们需要一种有效的方法来表示时间步 $t$ 并将这些信息注入模型。最常见且有效的方法，受Transformer架构 (Vaswani et al., 2017) 启发，是使用 **正弦位置编码 (positional encoding)**。

### 正弦时间步编码

并非使用原始整数 $t$，我们将其转换为一个高维向量 (vector) $e(t)$。这个编码向量提供了时间步更丰富、更结构化的表示，网络可以更好地使用它。标准方法是使用不同频率的正弦和余弦函数：

令 $d$ 为所需编码向量的维度。对于从 $0$ 到 $d/2 - 1$ 的每个维度 $i$，编码分量计算如下：


$$
e(t)_{2i} = \sin\left( \frac{t}{10000^{2i/d}} \right)
$$


$$
e(t)_{2i+1} = \cos\left( \frac{t}{10000^{2i/d}} \right)
$$


这里：

- $t$ 是当前时间步。
- $i$ 是编码向量中的维度索引 (最高到 $d/2$)。
- $d$ 是编码的总维度 (例如，128、256 或 512)。

这种公式构建了一个向量，其中每对维度 $(2i, 2i+1)$ 对应一个特定频率的正弦波。较低维度 (小的 $i$) 随 $t$ 缓慢变化，而较高维度 (大的 $i$) 则快速变化。这种多频率表示使得网络可以轻易区分不同时间步，并在需要时更好地泛化到未见过的时间步。



![正弦时间步编码 (选定维度)](plots/5406-0.json)



> 正弦编码在时间步 0 到 1000 之间不同维度的示例值，假设编码维度 $d$ 大于 51。较低维度变化缓慢，而较高维度快速振荡。

### 将编码注入U-Net

一旦我们有了时间步编码向量 (vector) $e(t)$ (通常维度为256或512)，我们需要将其整合到U-Net架构中。U-Net通过一系列卷积层、下采样和上采样块处理噪声图像 $x_t$。时间步信息需要影响这些计算。

一个常见的方法包含：

1. **投影：** 编码向量 $e(t)$ 通常会通过一个小型多层感知机 (MLP) — 通常是一到两个线性层，并带有 SiLU (Sigmoid 线性单元) 或 ReLU 等激活函数 (activation function)。这个MLP学习将固定的正弦编码转换为最适合网络内部运作的表示。
2. **广播和相加：** 这个MLP的输出 (我们称之为处理后的编码) 通常被广播以匹配U-Net内部不同位置的特征图的空间维度 (高度和宽度)。然后将其逐元素添加到这些特征图中。这通常发生在构成U-Net骨架的残差块内。
3. **多分辨率注入：** 这种处理后的时间步编码的添加通常在U-Net内的多个分辨率级别进行，通常在下采样和上采样路径中的每个残差块内。这确保了时间步信息影响不同空间尺度的处理。

> 图示：处理后的时间编码如何典型地添加到U-Net块内的特征图。正弦编码在注入前首先通过MLP进行转换。

通过这种方式整合时间步信息，单一U-Net模型 $\epsilon_\theta(x_t, t)$ 学习根据 $t$ 所指示的当前噪声水平来调整其噪声预测。这种基于时间的条件化是扩散模型操作的重要组成部分，使网络能够在逆向过程的每一步执行适当的去噪操作。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  这篇奠基性论文介绍了 Transformer 架构和正弦位置嵌入的概念，这些概念被直接用于扩散模型中的时间步条件化。
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — Jonathan Ho, Ajay Jain, Pieter Abbeel (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2006.11239](https://doi.org/10.48550/arXiv.2006.11239)
  这篇开创性论文介绍了去噪扩散概率模型 (DDPM) 框架，详细阐述了用于噪声预测的 U-Net 架构以及通过嵌入整合时间步信息的方法。
- [U-Net Architecture and Timestep Embeddings in Diffusers](https://huggingface.co/docs/diffusers/conceptual/unet#timestep-embeddings) — Hugging Face (2023)
  Publisher: Hugging Face
  此文档部分详细介绍了扩散模型中使用的 U-Net 架构，并具体阐述了如何创建时间步嵌入以及将其集成以条件化 U-Net 的噪声预测。
