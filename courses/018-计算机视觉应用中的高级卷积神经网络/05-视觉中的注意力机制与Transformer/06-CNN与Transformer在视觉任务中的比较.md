# CNN与Transformer在视觉任务中的比较

来源：[原文](https://apxml.com/zh/courses/cnns-for-computer-vision/chapter-5-attention-transformers-vision/comparing-cnns-transformers)

[返回章节目录](README.md) · [返回课程目录](../README.md)

CNN中的注意力机制 (attention mechanism)和视觉Transformer（ViT）的架构是计算机视觉任务的两种主要模型系列。对这些模型进行比较，可以展示它们各自的优势、劣势和运行特点。理解这些方面对于为特定问题和数据集选择合适的架构很重要。

它们最根本的区别在于其**归纳偏置 (bias)**。CNN对图像数据具有强大的内在假设：

1. **局部性：** 它们假设重要模式是局部的。卷积操作作用于小的空间邻域。
2. **平移等变性：** 在图像一部分检测到的模式也能在另一部分检测到。这是通过卷积层中的权重 (weight)共享实现的。

这些偏置使得CNN在学习特征的空间层次结构（从边缘、纹理到复杂的物体部分）方面，在数据效率上表现出色。即使在中等大小的数据集上，它们也能有效学习，因为架构引导它们形成有用的基于图像的假设。

视觉Transformer从自然语言处理借鉴而来，对空间结构具有显著较弱的归纳偏置。通过将图像分割成图像块，并使用自注意力 (self-attention)机制将其作为序列处理，ViT对局部空间关系做出较少的假设。主要机制——自注意力，允许每个图像块直接与所有其他图像块交互，使模型能够从一开始就捕获**长距离依赖**和全局上下文 (context)。

归纳偏置的这种差异导致了若干实际影响：

### 数据需求

与CNN相比，ViT通常需要大得多的训练数据集才能达到有竞争力的性能。如果没有强大的空间偏置 (bias)，ViT需要几乎完全从数据中学习图像的特性，包括局部性和空间关系等基本特性。当在较小数据集（如没有外部预训练 (pre-training)的ImageNet-1k）上训练时，标准ViT的性能通常不如大小相近的CNN。然而，当在海量数据集（例如ImageNet-21k、JFT-300M）上进行预训练，或使用先进的自监督预训练方法时，ViT常常达到或超越目前最先进的CNN的性能，特别是随着模型规模的增长。CNN得益于它们的偏置，在数据量较少的情况下表现更好。

### 计算考量

- **训练：** 在海量数据集上训练大型ViT计算量很大，需要大量的GPU/TPU资源。然而，对于超大型模型，训练过程有时会更稳定，而极深层的CNN可能面临优化上的挑战。
- **推断：** Transformer中的核心自注意力 (self-attention)机制 (attention mechanism)的计算复杂度为 $O(N^2 \cdot D)$，其中 $N$ 是序列长度（图像块的数量），$D$ 是嵌入 (embedding)维度。这种与图像块数量呈二次方缩放的关系使得标准ViT对于高分辨率图像计算成本高昂，因为 $N$ 随图像维度呈二次方增长。CNN的计算成本通常与像素数量呈更线性的关系。对高效Transformer变体（例如，使用线性注意力、池化或Swin Transformer中的平移窗口）的研究旨在缓解这种 $N^2$ 瓶颈。

### 性能和特征表示

- **全局与局部上下文 (context)：** 由于自注意力 (self-attention)机制 (attention mechanism)的全连接特性，ViT天生擅长处理需要理解全局上下文以及图像远距离区域之间关系的任务。CNN通过逐层增加感受野来分层构建全局理解，这有时会稀释细粒度的局部信息，或难以处理非常远距离的交互。相反，CNN由于其卷积操作，天生擅长捕获细粒度的局部模式和纹理。
- **泛化能力：** 有证据表明ViT在某些分布外基准测试中可能表现出更好的泛化性能。这可能归因于它们学习到较不固定的空间特征，这可能使它们对涉及纹理或风格变化的域偏移不太敏感，同时保留形状信息。然而，这是一个活跃的研究方向。
- **规模扩展：** ViT的性能似乎随着模型大小和数据量的增加而很好地扩展，有时超越了大型CNN中观察到的饱和点。

### 架构设计与灵活性

- **CNNs：** 提供了一个成熟的生态系统，设计原则已充分理解，并有众多高效架构（例如MobileNets、EfficientNets），为特定资源限制量身定制。深度可分离卷积是实现效率的标准方法。
- **ViTs：** 核心Transformer块高度通用。主要的架构选择包括图像块嵌入 (embedding)策略、位置编码 (positional encoding)、使用的特定Transformer变体，以及如何处理分类或其他下游任务（例如，使用类别令牌或全局平均池化）。混合架构，将卷积主干与Transformer块结合（如CoAtNet、CvT），尝试结合两者的优势：利用CNN早期的空间特征提取效率和ViT的全局上下文 (context)建模能力。

### 实际要点

选择CNN还是ViT，很大程度上取决于具体的应用、可用的数据和计算预算：

- **大型数据集与高性能：** 如果您拥有非常大的数据集（或优秀的预训练 (pre-training)模型）和计算资源，ViT（或大型混合模型）是强有力的竞争者，并且通常在许多基准测试中获得顶尖性能。
- **中小型数据集：** CNN通常是更实用的选择，由于其有益的归纳偏置 (bias)，它们无需海量预训练数据集即可提供强大的性能。在此情况下，使用预训练CNN进行高级迁移学习 (transfer learning)非常有效。
- **高分辨率图像：** 标准ViT面临计算挑战。除非二次方复杂度可管理，否则高效ViT变体或CNN可能更适合。
- **需要全局推理 (inference)的任务：** ViT在很大程度上依赖于理解长距离空间关系的任务上具有天然的架构优势。
- **边缘部署：** 高效CNN架构目前更成熟，并在资源受限设备上得到广泛部署，尽管高效ViT的研究正在迅速进展。

下表总结了主要比较点：

| 特征 | CNNs | 视觉Transformer (ViTs) |
| --- | --- | --- |
| **归纳偏置** | 强 (局部性, 平移等变性) | 弱 (依赖于学习到的关系) |
| **数据需求** | 中等到大 | 很大 (或强预训练) |
| **全局上下文 (context)** | 分层构建 | 通过自注意力 (self-attention)直接捕获 |
| **局部特征** | 优秀 (卷积固有的) | 从图像块交互中学习 |
| **计算扩展性** | 大致与像素呈线性 | 与图像块数量呈二次方 ($N^2$) |
| **预训练需求** | 有益，但无需海量数据集也能表现良好 | 通常对良好性能重要 |
| **架构成熟度** | 非常成熟，许多高效变体 | 快速演进，效率提升中 |
| **最佳应用场景 (通用)** | 局部模式强、数据适中的任务 | 需要全局上下文、大规模数据的任务 |

最终，CNN和Transformer都是计算机视觉中强大的工具。结合卷积层（常用于初始特征提取）和Transformer块（用于全局推理）的混合方法代表着一个有前景的方向，可能兼具两者的优势。随着研究的进展，这些架构之间的界限可能会继续模糊，从而产生能力更强的视觉模型。

## 参考资料

- [An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) — Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby (2020)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2010.11929](https://doi.org/10.48550/arXiv.2010.11929)
  本文介绍了视觉Transformer (ViT) 架构，展示了以前在自然语言处理中占据主导地位的Transformer如何有效地应用于图像分类，为ViT与CNN的比较提供了基础。
- [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) — Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo (2021)
  Journal: IEEE International Conference on Computer Vision (ICCV); Pages: 10012-10022; DOI: [10.48550/arXiv.2103.14030](https://doi.org/10.48550/arXiv.2103.14030)
  这项工作引入了一种分层视觉Transformer，通过在局部窗口内计算自注意力解决了标准ViT的二次计算复杂度问题，使其在各种视觉任务和高分辨率图像处理中更有效率。
- [Transformers in Vision: A Survey](https://arxiv.org/abs/2101.01169) — Salman Khan, Muzammal Naseer, Munawar Hayat, Syed Waqas Zamir, Fahad Shahbaz Khan, Mubarak Shah (2021)
  Journal: ACM Computing Surveys; Volume: 54; Pages: 1-37; DOI: [10.1145/3505244](https://doi.org/10.1145/3505244)
  这项调查全面概述了视觉Transformer，包括与卷积神经网络的比较，讨论了它们的架构差异、性能和应用。

---

[上一节](05-%E6%B7%B7%E5%90%88CNN-Transformer%E6%A8%A1%E5%9E%8B.md) · [下一节](07-%E5%9C%A8CNN%E4%B8%AD%E5%AE%9E%E7%8E%B0%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%A8%A1%E5%9D%97%E7%9A%84%E5%AE%9E%E8%B7%B5.md)
