---
course: "advanced-diffusion-architectures"
chapter: "transformer-diffusion-models"
lesson: "transformer-motivation-generation"
sourceId: 5401
sourceUrl: "https://apxml.com/zh/courses/advanced-diffusion-architectures/chapter-3-transformer-diffusion-models/transformer-motivation-generation"
title: "生成式建模中采用Transformer模型的缘由"
description: "为什么Transformer模型在序列建模和图像生成方面相比CNN具有优势。"
order: 1
plots: []
sourceHash: "643c33a87eeea6cc4bbd3937ea0e05dbaa462234f4d58dfb13650aba5df9ccfc"
sourceCorrections: []
---

卷积神经网络 (neural network) (CNN)（CNNs），特别是U-Net结构中的CNN，在扩散模型中发挥了核心作用，并在图像生成方面取得了显著成效。它们的优势在于通过卷积核和池化操作，利用空间局部性和平移不变性。这种归纳偏置 (bias)对于那些局部模式和纹理很重要的任务非常有效。然而，这种固有地侧重于局部邻域的特点，在生成需要理解并建立场景中远距离部分联系的图像时，有时会成为局限。仅凭堆叠的卷积层来把握全局信息、复杂的组合结构或长距离依赖，可能会有挑战，并可能需要非常深的网络。

Transformer架构应运而生。Transformer最初是为自然语言处理（NLP）中的序列处理任务而设计的，它展现出非凡的能力，可以建模序列中元素之间的依赖关系，不论它们距离远近。驱动此能力的核心机制是**自注意力 (self-attention)**。与卷积不同，卷积作用于固定的局部感受野，而自注意力允许每个元素（例如，句子中的一个词，或者图像中的一个块）在计算其表示时，直接关注并衡量所有其他元素的重要性。

> 此图比较了CNN操作的局部感受野与Transformer自注意力为单个输出元素考虑的全局信息。

为什么这种全局建模能力对生成式建模具有吸引力，特别是对负责生成高分辨率图像等复杂数据的扩散模型而言？

1. **全局一致性：** 生成真实图像通常需要确保大范围的一致性。例如，光照和阴影应在整个场景中保持一致，或者大型物体的纹理应均匀。自注意力提供了一种直接的机制，通过允许图像表示的不同部分直接交互来落实这种长距离一致性。
2. **建模复杂关联：** 图像中常包含物体或区域间错综复杂的关联。Transformer可能比CNN更适合学习这些复杂的非局部交互，因为CNN是更循序渐进地构建层次特征的。
3. **可扩展性和参数 (parameter)效率：** 尽管注意力机制 (attention mechanism)可能计算密集，特别是对于高分辨率图像（稍后会更详细讲解如何调整），但Transformer已展现出卓越的扩展能力。随着模型和数据集的增长，Transformer常表现出持续的性能提升，这表明其在大规模生成任务方面具备潜在优势。此外，诸如Vision Transformer（ViT）等架构已表明，Transformer可以实现良好性能，有时参数量少于同类CNN，尽管通常需要更大的数据集进行有效预训练 (pre-training)。
4. **不同的归纳偏置：** 与CNN相比，Transformer的空间归纳偏置较弱。CNN本身假定局部性和平移不变性很重要，而Transformer对输入结构的假设更少。当预先不知道处理数据的最佳方式或不完全适合卷积框架时，这种灵活性可能具有优势。然而，这也常意味着Transformer需要更多数据或特定的预训练策略才能有效学习。

因此，在扩散模型中研究Transformer模型的动因，源于克服CNN在建模全局信息和长距离依赖方面的潜在局限。自注意力在其他方面捕捉复杂关联的成功，表明它具备提升处理高维数据（如图像）的生成模型的质量、一致性和表现力的潜力。后续章节将审视这些强大的架构具体如何适应并集成到扩散框架中。
