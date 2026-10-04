---
course: "cnns-for-computer-vision"
chapter: "attention-transformers-vision"
lesson: "hybrid-cnn-transformer-models"
sourceId: 2593
sourceUrl: "https://apxml.com/zh/courses/cnns-for-computer-vision/chapter-5-attention-transformers-vision/hybrid-cnn-transformer-models"
title: "混合CNN-Transformer模型"
description: "介绍结合卷积层和Transformer块的模型，以发挥两者各自的优势。"
order: 5
plots: []
sourceHash: "a7f7b51941a26b96f113902b0fa87ee97632bcc1a4b73f2f886700d3a5f97cee"
sourceCorrections: []
---

纯视觉Transformer (ViT) 在捕捉全局图像上下文 (context)方面展现出强大的能力，但由于它们缺乏对空间局部性的固有归纳偏置 (bias)（卷积神经网络 (neural network) (CNN)CNNs擅长此点），通常需要大量数据集进行有效的预训练 (pre-training)。相反，标准CNNs尽管在学习局部模式和空间层级结构方面效率高，但在建模图像的显式长程依赖关系时可能会遇到困难。这一观察结果自然引出混合架构的发展，这些架构旨在结合两种方法的优点。

混合CNN-Transformer模型代表了一种结合策略，将擅长高效提取低级特征和空间层级结构的卷积层，与擅长建模特征之间全局交互的Transformer块结合起来。其主要思想是让每个组件发挥其最擅长的作用。

### 结合卷积特征提取与Transformer推理 (inference)

一种常用且有效的方法是，在网络的初始阶段，主要将CNN用作强大的特征提取器。

1. **初始卷积阶段：** 输入图像首先由若干卷积层或截断的标准CNN主干网络（如ResNet的早期阶段）处理。这些层执行初始特征提取，获取边缘、纹理和局部模式，同时逐渐降低空间分辨率并增加通道深度。这运用了卷积的空间归纳偏置 (bias)，使得模型在数据方面更有效率，尤其是在早期层中。
2. **转换层：** 在某个深度，CNN阶段生成的特征图被转换为适合Transformer输入的序列。这通常包括：
   - **分块：** 类似于ViT，特征图可以被划分为不重叠或重叠的块。
   - **展平：** 每个块被展平为一个向量 (vector)。
   - **线性投影：** 这些向量被线性投影到Transformer所需的嵌入 (embedding)维度。位置嵌入通常在此阶段添加，以保留空间信息。
3. **Transformer编码器：** 补丁嵌入序列随后由一个或多个标准Transformer编码器层处理。这些层使用多头自注意力 (self-attention)机制 (attention mechanism)，对CNN提取的特征补丁间的全局依赖关系进行建模。自注意力机制使模型在构建最终表示时，能够衡量不同特征区域的重要性。
4. **最终分类/任务头：** Transformer编码器的输出（通常使用特殊`[CLS]` token的表示或通过对序列输出进行池化）被送入最终的分类头（例如，一个简单的MLP）或特定任务头（例如，用于检测或分割）。

> 一种混合CNN-Transformer模型的常见结构，其中CNN获取特征，然后由Transformer编码器处理。

这种方法得益于CNN高效学习局部特征的能力，减轻了Transformer的负担，后者可以纯粹专注于对这些特征之间关系的推理。像CvT（卷积视觉Transformer）这样的模型在Transformer的token化和注意力机制中明确地包含卷积，而CoAtNet等其他模型则在不同的网络深度巧妙地安排了卷积块和Transformer块。

### 在卷积阶段中加入注意力机制 (attention mechanism)

另一个观点是，将类似Transformer的自注意力 (self-attention)机制更深层地嵌入 (embedding)到CNN架构本身中，而不是严格区分CNN和Transformer阶段。

- **替换卷积块：** 深度CNN的后期阶段，其在低分辨率、高维度特征图上操作，可以完全替换为Transformer块。在这些阶段，特征图的“像素”可以被视为token，从而使自注意力能够建立在更宽的空间范围内的关系。
- **增强卷积块：** 自注意力层可以与卷积层并行或并列插入。例如，一个块可能同时包含一个标准的3x3卷积层和一个多头自注意力层，并将它们的输出结合。这使网络层能够同时学习局部模式（通过卷积）和全局信息（通过注意力）。

### 混合模型的优点

- **性能提升：** 混合模型在各种计算机视觉基准测试中常能达到当前最优结果，可能表现优于纯CNN或纯ViT，特别是在训练数据量未达到海量网络抓取数据集的规模时。
- **数据效率：** 通过保持卷积的归纳偏置 (bias)，尤其是在早期层中，混合模型通常可以更快收敛，并且与从头开始训练的ViT相比，所需训练数据更少。
- **灵活性：** 这种结合在设计上更灵活，使设计者能够通过决定CNN部分的深度和Transformer部分的复杂性来权衡计算成本、参数 (parameter)数量和性能。
- **善用预训练 (pre-training)：** 成熟的预训练CNN主干网络可以方便地整合，为混合模型的特征提取部分提供良好的初始化。

### 考量点

混合模型虽然性能高，但设计它们会增加复杂性。重要的选择包括CNN和Transformer阶段之间的转换点、将特征图转换为序列的方法（块大小、步长）、所用Transformer层的特定结构，以及位置信息如何编码和保留。调整这些架构需要细致的实验并考虑目标任务和数据集的特点。

总而言之，混合CNN-Transformer模型是一种实际且有效的方法，结合了CNN处理局部特征的优点与Transformer处理全局信息的能力，从而得到性能优异的视觉系统。它们提供了一个明确的折中方案，它运用了数十年的CNN研究成果，同时加入了注意力机制 (attention mechanism)带来的进展。

## 参考资料

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) — Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby (2020)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2010.11929](https://doi.org/10.48550/arXiv.2010.11929)
  介绍了视觉Transformer，强调其能力和数据需求，为混合模型的发展奠定基础。
- [CoAtNet: Marrying Convolution and Attention for All Data Scales](https://arxiv.org/abs/2106.04803) — Zihang Dai, Hanxiao Liu, Quoc V. Le, Mingxing Tan (2021)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Volume: 34; Pages: 17203-17216; DOI: [10.48550/arXiv.2106.04803](https://doi.org/10.48550/arXiv.2106.04803)
  提出了一种通用架构，统一了卷积和自注意力，展示了它们在不同尺度和数据集上的互补优势。
