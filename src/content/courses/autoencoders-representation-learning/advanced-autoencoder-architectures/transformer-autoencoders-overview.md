---
course: "autoencoders-representation-learning"
chapter: "advanced-autoencoder-architectures"
lesson: "transformer-autoencoders-overview"
sourceId: 2882
sourceUrl: "https://apxml.com/zh/courses/autoencoders-representation-learning/chapter-5-advanced-autoencoder-architectures/transformer-autoencoders-overview"
title: "基于Transformer的自编码器概述"
description: "简要介绍Transformer架构在自编码任务中的应用（例如，MAE思路）。"
order: 5
plots: []
sourceHash: "601d419e4bbf1ec057aa5b757d20c41eec0c60a4ffaae4ad1a5b357f7b3b5c9c"
sourceCorrections: []
---

Transformer架构最初在自然语言处理（NLP）中取得了领先成果，展现了显著的多功能性和强大的性能，因此被不同方面采用，包括计算机视觉和时间序列分析。其主要机制——自注意力 (self-attention)，使模型能够衡量输入数据不同部分彼此之间的显著程度，从而非常擅长捕获长距离依赖关系和复杂模式。此特性自然地延伸到自编码器，为卷积或循环方法提供了一种强大的替代方案，特别适用于序列数据或需要对全局上下文 (context)进行建模的场景。

将Transformer架构应用于自编码通常涉及使用其标准的编码器-解码器结构。输入序列（可以是文本标记 (token)、图像块或时间序列中的点）被输入到Transformer编码器中。编码器通过多层自注意力和前馈网络处理输入，并生成一系列带有上下文信息的表示。此输出序列，或有时从中导出的池化表示（例如，特殊`[CLS]`标记的输出或平均池化），作为潜在表示。

Transformer解码器随后接收此潜在表示，并通常以自回归 (autoregressive)方式或通过并行解码机制，尝试重建原始输入序列。由于标准的Transformer架构本身不理解序列中元素的顺序（与RNN不同），因此在编码器和解码器中都向输入嵌入 (embedding)添加了位置编码 (positional encoding)，以提供每个元素位置的信息。

此方面一个重要的进展是掩蔽自编码器（MAE）方法，对视觉Transformer的自监督学习 (supervised learning) (self-supervised learning)特别有影响。MAE遵循一个不同于传统去噪自编码器的原则。MAE不通过噪声破坏输入，而是随机遮盖输入序列的很大一部分（例如，图像块）。

以下是典型的MAE工作流程：

1. **分块与遮盖**：输入（例如，一张图像）被分成规则的、不重叠的小块。这些小块中很大比例（例如75%）被随机选择并从输入序列中遮盖（移除）。
2. **编码可见小块**：重要的是，Transformer编码器*只*处理*未遮盖*（可见）小块的序列，及其位置编码。这使得编码过程的计算效率很高，因为它仅随可见小块的数量扩展，而非总数。
3. **解码被遮盖小块**：解码器接收可见小块的编码表示，以及表示被遮盖小块位置的特殊“遮盖标记”。这些遮盖标记也带有位置编码。解码器的任务是*只*对被遮盖小块预测原始像素值（或特征），使用可见小块和位置信息提供的上下文。
4. **重建损失**：损失函数 (loss function)（例如，均方误差）只在重建的被遮盖小块上计算，将它们与这些小块的原始值进行比较。

> 掩蔽自编码器（MAE）的简化流程。编码器只处理可见小块，而解码器使用编码的上下文和位置信息重建被遮盖的小块。

MAE策略促使模型学习丰富、高层次的输入表示，因为它必须从可见的上下文中推断缺失的内容。这已被证明对大型视觉Transformer的自监督预训练 (pre-training)非常有效，能产生很好地迁移到后续任务（如图像分类和分割）的表示，且只需少量微调 (fine-tuning)。

虽然MAE在视觉方面很突出，但使用Transformer编码器和解码器进行自编码的通用思路适用范围广：

- **自监督预训练**：从大型无标签数据集中学习通用表示是其主要应用。BERT等模型，虽然通常被视为语言模型，但它们使用掩蔽语言建模进行预训练，这与自编码的思路相关。
- **序列数据处理**：适用于文本摘要（序列到序列压缩）或复杂时间序列中的异常检测等任务，其中长距离模式很显著。
- **生成式建模**：虽然VAE和GAN常被用于生成，但基于Transformer的解码器可以在自编码器框架内使用，进行序列生成，有时以学到的潜在编码为条件。

然而，训练基于Transformer的自编码器带来挑战。标准自注意力机制 (attention mechanism)的计算复杂度与序列长度的平方成正比，这使得处理非常长的序列成本很高（尽管像MAE这样的方法在编码器中缓解了此问题）。它们通常需要大量数据和显著的计算资源才能进行有效的预训练。此外，调整架构（层数、头数、维度）和训练超参数 (parameter) (hyperparameter)对于获得最佳性能仍然是一个重要的考量。

总而言之，基于Transformer的自编码器代表了一类强大的模型，使用自注意力机制捕获数据中的复杂依赖关系。MAE等架构展现了特别的前景，用于高效的自监督预训练，为各类后续任务产生强大的表示。它们提供了一种引人注目的替代方案，替代卷积和循环自编码器，特别是在处理序列数据或需要对输入有全局理解时。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  介绍了Transformer架构及其自注意力机制，这些是讨论的自编码器的基础。
- [An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) — Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby (2021)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2010.11929](https://doi.org/10.48550/arXiv.2010.11929)
  介绍了Vision Transformer (ViT) 架构，为MAE在计算机视觉中的应用提供了背景。
