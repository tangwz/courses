# U-Net与Transformer在扩散模型中的比较

来源：[原文](https://apxml.com/zh/courses/advanced-diffusion-architectures/chapter-3-transformer-diffusion-models/unet-vs-transformer-diffusion)

[返回章节目录](README.md) · [返回课程目录](../README.md)

U-Net和Transformer架构都被证明可作为扩散模型的有效骨干网络，但它们因底层设计（卷积式处理与注意力机制 (attention mechanism)处理）而具有显著特点。弄清楚它们的相对优势、不足和权衡点，对于特定生成任务的架构选择或设计非常重要。接下来将从几个方面进行比较。

### 架构基本原理

- **U-Net：** 依赖于以编码器-解码器结构组织的卷积层，并带有跳跃连接。卷积凭借其固有的局部偏差和权重 (weight)共享，擅长捕获局部模式和空间层次。跳跃连接有助于保留不同分辨率级别间的细粒度细节。其归纳偏置 (bias)非常适合图像类数据，在这些数据中，局部结构和平移等变性是重要属性。
- **Transformer (DiT)：** 将输入图像视为图像块序列（类似于自然语言处理中的标记 (token)）。它主要在Transformer块内部使用自注意力 (self-attention)机制 (attention mechanism)。自注意力允许每个图像块直接关注所有其他图像块，从而从第一层开始就提供全局感受野。这使其对于建模远距离空间依赖很有力，但缺乏CNN固有的空间偏置。

### 建模能力

- **局部特征：** U-Net及其卷积核，天然擅长在网络早期提取和处理局部特征和纹理。
- **全局依赖：** Transformer在此方面表现出色。自注意力 (self-attention)机制 (attention mechanism)使其能够比标准CNN更直接地捕获图像不同部分之间的复杂关系，而标准CNN需要非常深的层或附加的注意力层才能实现类似的全局上下文 (context)。DiT表明，纯粹的Transformer能够在扩散模型框架内有效建模图像的全局结构。

### 扩展能力与性能

Transformer的一个显著特点是，在不同应用中，它们展现出卓越的扩展能力。

- **模型扩展：** 对扩散Transformer（DiT）的研究表明，增加模型大小（深度、宽度）和计算预算通常会带来显著改进的生成质量（例如，通过FID分数衡量），尤其是在大型数据集上，有可能超越高度优化的U-Net。U-Net也能扩展，但其性能可能更早达到饱和，或者需要更多的架构调整才能有效发挥增加的容量。
- **数据扩展：** Transformer通常需要大量数据才能有效学习，部分原因是它们缺乏CNN的强归纳偏置 (bias)。U-Net由于内置了关于图像结构的先验知识，通常能在较小数据集上达到合理表现。



[交互图表：FID与计算量扩展对比](https://apxml.com/zh/courses/advanced-diffusion-architectures/chapter-3-transformer-diffusion-models/unet-vs-transformer-diffusion#plot-1gv9rp2)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "FID与计算量扩展对比",
    "xaxis": {
      "title": "计算量（相对单位）",
      "type": "log"
    },
    "yaxis": {
      "title": "FID分数（越低越好）",
      "range": [
        0,
        30
      ]
    },
    "legend": {
      "title": "架构"
    },
    "template": "plotly_white"
  },
  "data": [
    {
      "type": "scatter",
      "mode": "lines+markers",
      "name": "U-Net",
      "x": [
        1,
        4,
        16,
        64
      ],
      "y": [
        25,
        12,
        8,
        7
      ],
      "line": {
        "color": "#339af0"
      },
      "marker": {
        "color": "#339af0"
      }
    },
    {
      "type": "scatter",
      "mode": "lines+markers",
      "name": "Transformer (DiT)",
      "x": [
        1,
        4,
        16,
        64
      ],
      "y": [
        28,
        15,
        7,
        4
      ],
      "line": {
        "color": "#f76707"
      },
      "marker": {
        "color": "#f76707"
      }
    }
  ]
}
```

</details>



> 此图说明了一种趋势：在更高的计算预算下（假设数据充足），Transformer可能比U-Net表现出更强的性能扩展能力（FID更低）。实际结果在很大程度上取决于实现、数据和具体配置。

### 计算需求

- **注意力复杂度：** Transformer中的核心自注意力 (self-attention)机制 (attention mechanism)的计算复杂度为 $O(N^2)$，其中 $N$ 是序列长度（图像块数量）。对于图像而言，$N$ 随图像分辨率平方增长。这使得标准Transformer对于高分辨率图像而言计算密集且内存消耗大。
- **卷积复杂度：** 标准卷积的复杂度大致与像素数量呈线性关系（如果卷积核大小视为常数，则为 $O(N)$），这使得U-Net通常更高效，尤其是在更高分辨率下，前提是参数 (parameter)数量或模型深度可比。
- **训练：** 训练大型DiT通常需要大量的GPU资源（内存和计算力 (compute)），与训练性能相似的U-Net相比，尤其因为注意力计算以及可能更大的参数量是最佳扩展所需。
- **推理 (inference)：** 尽管两种架构都受益于优化技术（第6章讨论），但注意力的平方成本可能使Transformer推理比U-Net推理慢，尤其当图像块数量较大时。

### 条件输入整合

两种架构都支持条件输入，但机制可能有所不同：

- **U-Nets：** 通常通过自适应归一化 (normalization)层（如AdaLN或FiLM）、连接到输入通道，或通过添加交叉注意力层（在U-Net内部的特定特征图上关注条件嵌入 (embedding)）来整合条件输入（例如，时间 $t$、类别标签、文本嵌入）。
- **Transformers (DiTs)：** 提供灵活的条件输入整合方式。方法包括将条件标记 (token)（例如，类别嵌入、时间嵌入）添加到输入序列，在层归一化或基于嵌入条件的MLP中使用自适应归一化/增益/偏置 (bias)，或在图像块标记和条件标记之间使用交叉注意力。这感觉与核心Transformer块结构更统一。

### 归纳偏置 (bias)

- **U-Nets：** 拥有强的图像特定归纳偏置（局部性、平移等变性），源于卷积和池化/上采样操作。这有助于它们高效学习图像数据。
- **Transformers：** 具有较弱的内置空间偏置。它们主要通过数据和位置嵌入 (embedding)学习空间关系。这种偏置的缺乏导致它们需要更多数据，但也赋予了它们灵活性以及发现不那么明显、远距离模式的可能。

### 权衡总结

| 特点 | U-Net（卷积型） | Transformer（基于注意力） |
| --- | --- | --- |
| **主要机制** | 卷积、池化、跳跃连接 | 自注意力 (self-attention)、MLP块 |
| **归纳偏置 (bias)** | 强（局部性、平移等变） | 弱（从数据和位置嵌入 (embedding)学习） |
| **感受野** | 局部（随深度增加） | 全局（从第一层开始） |
| **远距离依赖** | 不太自然（需加深/注意力） | 强（自注意力原生） |
| **扩展能力** | 良好，可能较早饱和 | 优秀（需计算和数据） |
| **数据效率** | 通常较高 | 通常较低（需更多数据） |
| **计算（高分辨率）** | 通常更高效（$O(N)$ 卷积） | 昂贵（$O(N^2)$ 注意力） |
| **条件输入** | 交叉注意力、AdaLN、拼接 | 条件标记 (token)、交叉注意力、AdaLN |
| **成熟度（扩散模型）** | 更成熟，变体多 | 较新，快速发展（例如DiT） |

### 何时选择哪种？

- **在以下情况下选择U-Net：**
  - 计算资源（GPU内存、训练时间）有限时。
  - 处理小型数据集，且强归纳偏置 (bias)有利时。
  - 任务主要涉及生成高保真局部细节和纹理时。
  - 希望利用大量现有基于U-Net的预训练 (pre-training)扩散模型时。
- **在以下情况下选择Transformer (DiT)：**
  - 拥有大型数据集和充足计算资源时。
  - 任务中需要对超远距离依赖进行建模时。
  - 目标是通过大规模扩展来提升生成质量的上限时。
  - 倾向于使用一种灵活的架构，能以统一方式加入多种条件类型时。

研究仍在继续考察混合架构，将卷积层（可能用于早期特征提取）与Transformer块（用于全局上下文 (context)建模）结合，旨在利用这两种方法的优点。然而，纯粹的U-Net和纯粹的Transformer骨干网络之间的选择，仍然是一个基本决定，基于上述因素。

## 参考资料

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) — Olaf Ronneberger, Philipp Fischer, Thomas Brox (2015)
  Journal: Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015; Pages: 234-241; DOI: [10.48550/arXiv.1505.04597](https://doi.org/10.48550/arXiv.1505.04597)
  这篇基础论文介绍了 U-Net 架构，它是许多图像生成模型（包括扩散模型）的通用骨干网络，因为它能够捕获分层特征和空间细节。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS 2017); Volume: 30; Pages: 5998-6008; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  这篇论文介绍了 Transformer 架构和自注意力机制，它们是扩散 Transformer（DiT）中用于建模全局依赖的核心。

---

[上一节](04-%E6%89%A9%E6%95%A3Transformer%E6%A8%A1%E5%9E%8B%E4%B8%AD%E7%9A%84%E6%9D%A1%E4%BB%B6%E4%BD%9C%E7%94%A8.md) · [下一节](06-DiT%20%E7%9A%84%E5%AE%9E%E7%8E%B0%E8%80%83%E9%87%8F.md)
