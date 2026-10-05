# 视觉Transformer简介

来源：[原文](https://apxml.com/zh/courses/cnns-for-computer-vision/chapter-5-attention-transformers-vision/vision-transformers-introduction)

[返回章节目录](README.md) · [返回课程目录](../README.md)

尽管卷积神经网络 (neural network) (CNN) 能通过其卷积核有效地获取局部特征，但由于卷积操作固有的局部性，跨整个图像模拟远距离关联仍是一个难题。大幅增加感受野需要很深的网络或很大的卷积核，这可能导致计算成本高昂且难以优化。将注意力机制 (attention mechanism)融入CNN的方法有所帮助，但它们仍在一个基本的卷积体系中运行。

Transformer结构最初为自然语言处理（NLP）所开发，提供了一种不同的方法。Transformer完全依赖自注意力 (self-attention)机制，使它们能够模拟序列中任意两个元素之间的关联，无论距离远近。这被证实对于机器翻译和文本生成等任务非常有效，其中理解长句的语境是必需的。

一个重要问题出现了：这种强大的序列建模结构能否应用于计算机视觉？图像与文本不同，不具备固有的序列一维结构。它们具有很强的二维空间结构，并且典型图像中的像素数量（潜在的序列元素）远多于典型句子中的单词数量，这给对序列长度具有二次复杂度的标准Transformer带来了计算难题。

视觉Transformer（ViT）模型代表了一种成功且有影响力的办法，将Transformer直接应用于图像分类。由Dosovitskiy等人于2020年提出，其核心思路出奇地直接：将图像视为一系列较小的、固定大小的图像块。

> 一张输入图像被分割成一个由不重叠图像块组成的网格。每个图像块随后被展平为一个向量 (vector)，形成一个可由标准Transformer编码器处理的序列。

其工作方式如下（高层视角）：

1. **图像分块：** 输入图像（例如，224x224像素）被分割成一个由固定大小、不重叠图像块（例如，16x16像素）组成的网格。
2. **展平与嵌入 (embedding)：** 每个图像块被展平为一个单独的向量（例如，RGB图像为16x16x3 = 768个元素）。这些向量随后被线性投影到一个期望的嵌入维度（例如，768维）。这形成了一个图像块嵌入序列。
3. **位置嵌入：** 由于Transformer结构本身不固有的理解空间顺序，可学习的位置嵌入被添加到图像块嵌入中，以保留位置信息。一个特殊的`[class]`标记 (token)嵌入通常被加到序列的开头，类似于BERT的`[CLS]`标记，其在Transformer中的对应输出用于分类。
4. **Transformer编码器：** 得到的向量序列（图像块嵌入 + 位置嵌入）被送入一个标准的Transformer编码器，该编码器由交替的多头自注意力（MSA）和多层感知机（MLP）模块构成。自注意力机制允许每个图像块关注所有其他图像块，使模型能够获取图像的整体关联。在每个模块前应用层归一化 (normalization)，并在每个模块后使用残差连接。
5. **分类头：** 最后，与`[class]`标记对应的输出通过一个小的分类头（通常是一个MLP）生成最终预测。

这种方法有效地绕过了对卷积的需求，将Transformer的序列处理能力直接应用于视觉数据。然而，与CNN相比的一个显著区别是缺乏强的*归纳偏置 (bias)*。CNN内置了关于局部性（相邻像素相关）和平移等变性（在一个位置检测到的特征可以在其他位置检测到）的假设。ViT的偏置弱得多，几乎完全通过自注意力从数据中学习关联。这使得ViT可能更具通用性，但通常需要大得多的数据集进行预训练 (pre-training)，才能达到与或超越最先进CNN相当的性能。

ViT的引入标志着计算机视觉研究的一个显著转变，表明高度依赖自注意力的结构在图像识别任务上也能获得出色成果，这些任务曾主要由CNN主导。接下来的部分将更详细地查看ViT的特定结构组件，并将其特性与CNN进行比较。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: arXiv; Pages: 5998-6008; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  提出了Transformer架构和自注意力机制，是Vision Transformers的基础。
- [An Image Is Worth 16x16 Words: Transformers For Image Recognition At Scale](https://arxiv.org/abs/2010.11929) — Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby (2020)
  Journal: ICLR; DOI: [10.48550/arXiv.2010.11929](https://doi.org/10.48550/arXiv.2010.11929)
  介绍了Vision Transformer (ViT)，通过将图像处理为图像块序列，将Transformer编码器直接应用于图像分类。

---

[上一节](02-%E9%9D%9E%E5%B1%80%E9%83%A8%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C.md) · [下一节](04-ViT%20%E6%9E%B6%E6%9E%84%EF%BC%9A%E5%9B%BE%E5%83%8F%E5%9D%97%E3%80%81%E5%B5%8C%E5%85%A5%E5%92%8C%20Transformer%20%E7%BC%96%E7%A0%81%E5%99%A8.md)
