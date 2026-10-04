---
course: "advanced-diffusion-architectures"
chapter: "advanced-unet-architectures"
lesson: "unet-conditioning-integration"
sourceId: 5389
sourceUrl: "https://apxml.com/zh/courses/advanced-diffusion-architectures/chapter-2-advanced-unet-architectures/unet-conditioning-integration"
title: "高级条件输入整合"
description: "将复杂条件信号（文本、图像、掩码）集成到U-Net中的方法，通过交叉注意力与自适应归一化。"
order: 4
plots: []
sourceHash: "8c9ce823f2d0d99704d6ef420cbf35c0ccdaee4f20d0ee3f0f8de79f70a7ad8a"
sourceCorrections: []
---

尽管标准U-Net能有效处理空间信息和时间步嵌入 (embedding)，但要使用更复杂的条件引导生成过程，需要特定的架构调整。简单的类别标签，通常通过基础嵌入层进行集成，不足以满足需要精细控制的任务，例如文本到图像生成、图像编辑或风格迁移。将丰富的条件信息直接注入U-Net结构中的高级技术得到呈现。

### 通过交叉注意力集成条件信息

交叉注意力已成为一种强大的机制，用于在神经网络 (neural network)中融合来自不同模态的信息，并且在扩散模型的条件化方面尤为有效。在U-Net中，交叉注意力层允许模型在不同空间位置和分辨率生成特征时，有选择地关注条件信号的相关部分（例如文本提示中的特定词语）。

通常，在U-Net模块内（通常与自注意力 (self-attention)层并列），会引入一个交叉注意力层。核心组成部分包括：

1. **查询（Q）：** 源自U-Net在特定层中的中间特征图。这些代表了请求引导的空间上下文 (context)。
2. **键（K）和值（V）：** 源自条件信号 $c$。对于文本条件化， $c$ 通常是一系列嵌入 (embedding)（例如来自CLIP文本编码器）。通常会对条件嵌入应用线性投影，以生成K和V矩阵。

注意力机制 (attention mechanism)随后计算空间查询（Q）和条件键（K）之间的相关性得分，利用这些得分来加权条件值（V）。所得的加权值会被添加回U-Net的特征图，从而有效地注入条件信息。


$$
\text{注意力}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$


此处，$d_k$ 是键向量 (vector)的维度。此操作允许U-Net特征图中的每个空间位置关注条件输入中最相关的部分。

> 图示了一个U-Net内的交叉注意力模块中的信息流，整合了条件嵌入。查询来自U-Net特征，而键和值则源自外部条件信号。

此机制是当前最先进的文本到图像模型（如Stable Diffusion）的核心，其中文本嵌入通过交叉注意力在U-Net内的多个分辨率层级被注入。

### 用于条件化的自适应归一化 (normalization)

另一种有效技术涉及根据条件信号调整归一化层的参数 (parameter)。标准归一化层，如组归一化（Group Normalization）或层归一化（Layer Normalization），在一个层内标准化特征。自适应归一化技术，例如自适应层归一化（AdaLN）或其变体（如AdaLN-Zero），使标准化过程本身具有条件性。

通常的思路是预测归一化层的缩放（$\gamma$）和偏移（$\beta$）参数，基于条件信息 $c$（通常也包括时间步嵌入 (embedding) $t$）。

给定输入特征图 $x$ 和归一化层 $Norm(\cdot)$，自适应归一化过程如下：

1. **计算 $\gamma$ 和 $\beta$：** 一个小型神经网络 (neural network)（通常是一个简单的MLP）接收条件嵌入 $c$（可能还有时间步嵌入 $t$）作为输入，并输出缩放和偏移参数。
   
   $$
   (\gamma, \beta) = \text{MLP}(c, t)
   $$
   
2. **应用归一化：** 对输入特征 $x$ 应用标准归一化。
   
   $$
   h = Norm(x)
   $$
   
3. **应用调制：** 使用预测的缩放和偏移来调制归一化特征。
   
   $$
   y = \gamma \odot h + \beta
   $$
   
   其中 $\odot$ 表示逐元素乘法。

这使得条件信号能够动态影响整个网络中特征的统计数据。AdaLN-Zero是一种特定的初始化策略，其中MLP在初始时输出 $\gamma=1$ 和 $\beta=0$，确保条件块在训练开始时作为恒等函数作用，这可以提高稳定性。

自适应归一化在计算上比交叉注意力更轻量，因为它不涉及基于序列长度的昂贵矩阵乘法。它通常与交叉注意力结合使用或作为其替代方案，特别是对于可以通过单个向量 (vector)表示的条件信号（例如类别标签或全局风格嵌入）。

### 更简单的集成方法：拼接与相加

集成条件信息还有一些复杂度较低的方法，尽管与注意力或自适应归一化 (normalization)相比，它们的表达能力可能较低。

1. **拼接：** 条件嵌入 (embedding)可以直接拼接到U-Net的特征图上。如果条件是非空间性的（例如单个向量 (vector)），它需要被广播或平铺以匹配特征图的空间维度（$H \times W$），然后沿通道维度进行拼接。随后的卷积层可以合并拼接后的特征。
2. **相加：** 类似于拼接，条件嵌入可以被投影以匹配U-Net特征的通道维度，然后逐元素相加。这通常是在空间上广播条件向量之后完成的。

这些方法实现起来很简单，但与注意力机制 (attention mechanism)相比，可能难以将条件信息与特定空间特征精确对齐 (alignment)。它们有时用于较简单的模型中，或用于集成全局条件信号。

### 处理多模态 (multimodal)条件信息

具体的集成技术通常取决于条件信号的性质：

- **文本：** 由于文本嵌入 (embedding)的序列性质，交叉注意力非常有效。
- **图像（例如，用于图像修复、风格迁移）：** 条件图像可以通过单独的编码器进行处理，它们的特征可以通过拼接、相加、交叉注意力或自适应归一化 (normalization)在相应的U-Net层中集成。对于图像修复，掩码本身通常作为U-Net的额外输入通道提供。
- **语义图/布局：** 这些可以类似于图像进行处理，通常进行编码并空间集成。
- **音频/其他非视觉数据：** 来自专用编码器的嵌入可以集成，通常通过交叉注意力或自适应归一化。

### 组合条件化

高级应用可能需要同时结合多个条件信号（例如，根据文本提示*和*风格图像生成图像）。架构可以通过以下方式处理此情况：

- 为不同的条件来源使用单独的交叉注意力模块。
- 在将不同的条件嵌入 (embedding)输入到用于注意力或自适应归一化 (normalization)的投影层之前进行拼接。
- 在集成之前对条件嵌入本身执行算术运算。

有效集成条件信息对于控制扩散模型的输出非常重要。虽然交叉注意力提供精细的控制，特别是对于序列或空间条件化，但自适应归一化提供了一种有效方式来调制网络特征，基于全局或基于向量 (vector)的条件。方法的选择取决于具体任务、条件信号的性质，以及计算成本和生成控制之间期望的权衡。

## 参考资料

- [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems 30; Publisher: Curran Associates, Inc.; Pages: 5998-6008; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  介绍了Transformer架构，包含多头自注意力与交叉注意力机制。
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf) — Jonathan Ho, Ajay N. Jain, Pieter Abbeel (2020)
  Journal: Advances in Neural Information Processing Systems 33 (NeurIPS 2020); Publisher: Neural Information Processing Systems Foundation, Inc.; Volume: 33; Pages: 6840-6851; DOI: [10.48550/arXiv.2006.11239](https://doi.org/10.48550/arXiv.2006.11239)
  关于去噪扩散概率模型的开创性工作，为许多现代基于扩散的生成模型奠定了基础。
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf) — Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Bjorn Ommer (2022)
  Journal: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR); Pages: 10684-10695; DOI: [10.48550/arXiv.2112.10752](https://doi.org/10.48550/arXiv.2112.10752)
  介绍了潜在扩散模型架构，其利用交叉注意力整合文本条件进行图像生成。
