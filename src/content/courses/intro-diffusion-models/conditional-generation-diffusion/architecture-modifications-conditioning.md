---
course: "intro-diffusion-models"
chapter: "conditional-generation-diffusion"
lesson: "architecture-modifications-conditioning"
sourceId: 5448
sourceUrl: "https://apxml.com/zh/courses/intro-diffusion-models/chapter-6-conditional-generation-diffusion/architecture-modifications-conditioning"
title: "用于条件生成的架构修改"
description: "说明如何调整U-Net（例如，使用交叉注意力）以接收条件输入。"
order: 6
plots: []
sourceHash: "8895ecc94228f8126647133e1e67ceb6e6f0505fc75a8dc31cc78b4ab970dd6b"
sourceCorrections: []
---

为了有效引导扩散模型的生成过程，我们常常需要调整底层神经网络 (neural network)的结构，通过条件信息 $y$。标准的U-Net，旨在基于带噪声的输入 $x_t$ 和时间步 $t$ 预测噪声 $\epsilon_\theta(x_t, t)$，必须修改以整合 $y$。目标是使噪声预测变为条件预测，从而成为 $\epsilon_\theta(x_t, t, y)$。

存在几种整合这些条件信息的方法，选择常常取决于 $y$ 的性质和维度。

### 基于低维信息（例如，类别标签）的条件生成

当条件信息 $y$ 相对简单时，例如图像生成的类别标签（如“猫”、“狗”，对应整数标签 0、1），我们可以使用简单直接的技术。

1. **嵌入 (embedding)并添加到时间步嵌入：**
   一种常用方法，特别是在无分类器引导（CFG）下有效，是类似于处理时间步 $t$ 来处理类别标签 $y$。

   - 首先，离散类别标签 $y$ 被转换为向量 (vector)嵌入，通常使用一个标准嵌入层（在PyTorch中为 `torch.nn.Embedding`）。我们称之为 $e_y$。
   - 此类别嵌入 $e_y$ 通常会直接添加到时间步嵌入 $e_t$ 中。
   - 然后，组合嵌入（$e_t + e_y$）会被投影并添加到U-Net的各个模块中，就像原始的时间步嵌入一样。

   这种方法有效地在多个特征处理层面告知网络所需的类别。

   > 图示将时间步和类别标签嵌入组合，然后注入U-Net模块的过程。
2. **拼接：**
   另一种方法是沿着通道维度将嵌入的条件信息 $e_y$ 与输入 $x_t$ 拼接。然后，这个增强的输入被送入U-Net。此外，$e_y$ 也可以进行空间广播（平铺以匹配中间特征图的空间维度），并与网络更深层的特征图拼接。尽管简单，但与加性嵌入方法相比，这种方法可能无法总是有效地将条件信号传播到复杂的U-Net结构中。

### 基于高维信息（例如，文本嵌入 (embedding)）的条件生成

当基于更丰富、更高维度的信息（如文本描述）进行条件生成时，简单的加法或拼接通常不足。文本需要捕捉序列依赖和详细含义。为此，**交叉注意力**已成为标准机制，构成了像 Stable Diffusion 这样的现代文本到图像模型的基础。

**交叉注意力机制 (attention mechanism)**

回顾一下，U-Net（通常在Transformer风格的模块中）中的*自注意力 (self-attention)*层允许图像表示中不同空间位置互相注意。*交叉注意力*层的工作方式类似，但允许图像表示注意条件信息 $y$。

以下是它在专为条件生成设计的U-Net模块中（例如，文本到图像）通常的工作方式：

1. **输入：** 该层接收两个主要输入：

   - 来自U-Net的中间图像表示（我们将其特征表示为 $z$）。
   - 条件信息，通常是源自 $y$ 的嵌入向量 (vector)序列（例如，来自像 CLIP 这样的预训练 (pre-training)编码器的文本嵌入）。我们称此序列为 $c$。
2. **查询、键、值：**

   - **查询（Q）：** 通过线性投影从图像特征 $z$ 获得。这些表示图像特征图中每个空间位置“需要什么信息”。
   - **键（K）：** 通过线性投影从条件嵌入 $c$ 获得。这些表示条件序列中“有什么信息可用”。
   - **值（V）：** 也通过线性投影从条件嵌入 $c$ 获得。这些表示条件信息提供的实际内容或特征。
3. **注意力计算：** 核心操作是根据查询（来自图像）和键（来自条件）之间的相似性计算注意力分数。常用方式是缩放点积注意力：

   
   $$
   \text{注意力}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
   $$
   

   此处，$d_k$ 是向量的维度。softmax 确保权重 (weight)之和为1。
4. **输出：** 结果是值向量（源自条件 $c$）的加权和，其中权重取决于条件序列的每个部分与图像表示的每个部分的相关程度。此输出通常会加回到原始图像特征 $z$（通常通过残差连接），或在U-Net模块内进一步处理。

**整合到U-Net中**

交叉注意力层通常插入到U-Net的多个模块中，特别是在下采样、瓶颈和上采样路径中。这允许条件信息 $y$ 在不同级别的特征抽象中影响去噪过程。

> 图示如何使用U-Net模块中的交叉注意力层将条件嵌入 ($c$) 整合到图像特征 ($z$) 中。查询来自 $z$，而键和值来自 $c$。

**示例：文本到图像生成**

在文本到图像模型中：

1. 文本提示（例如，“一个逼真的宇航员骑着马”）由文本编码器（如 CLIP）编码成嵌入向量序列 $c$。
2. 在逆向扩散过程中，在每个时间步 $t$，U-Net 接收当前的带噪声图像 $x_t$、时间步 $t$ 和文本嵌入 $c$。
3. U-Net 中的交叉注意力层使用 $x_t$ 的特征查询文本嵌入 $c$，将文本引导整合到预测噪声 $\epsilon_\theta(x_t, t, c)$ 中。
4. 这引导去噪过程生成与输入文本提示一致的图像。

通过修改U-Net结构以整合条件信息 $y$，特别是使用诸如加性嵌入或交叉注意力等机制，扩散模型获得了执行受控生成的能力，从而生成根据类别标签或详细文本描述等特定要求定制的输出。

## 参考资料

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — Jonathan Ho, Ajay Jain, Pieter Abbeel (2020)
  Journal: arXiv; DOI: [10.48550/arXiv.2006.11239](https://doi.org/10.48550/arXiv.2006.11239)
  介绍了去噪扩散概率模型 (DDPM) 框架，概述了用于噪声预测的 U-Net 架构。该论文奠定了条件扩散模型所改编的基础架构。
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://doi.org/10.48550/arXiv.2112.10752) — Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer (2022)
  Journal: CVPR 2022; DOI: [10.48550/arXiv.2112.10752](https://doi.org/10.48550/arXiv.2112.10752)
  介绍了潜在扩散模型（LDM），该模型通过在潜在空间中操作显著降低了计算成本，并利用交叉注意力机制将高维条件信息（如文本嵌入）集成到 U-Net 架构中。
