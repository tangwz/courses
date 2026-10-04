---
course: "applied-autoencoders-feature-extraction"
chapter: "advanced-autoencoder-architectures"
lesson: "stacked-autoencoders-deep-architectures"
sourceId: 6469
sourceUrl: "https://apxml.com/zh/courses/applied-autoencoders-feature-extraction/chapter-4-advanced-autoencoder-architectures/stacked-autoencoders-deep-architectures"
title: "堆叠式自编码器：构建深度架构"
description: "理解堆叠式自编码器（深度自编码器），它们由多层自编码器构成，用于分层特征学习。"
order: 6
plots: []
sourceHash: "741024d51d2b101c44ad952062b1440992535fc0ba14d3d6bc689b10dc404c5a"
sourceCorrections: []
---

尽管具有单个隐藏层的基本自编码器可以有效地学习更简单数据集的压缩表示，但在处理更复杂的数据结构时，它们往往显得不足。正如监督学习 (supervised learning)中的深度神经网络 (neural network)可以通过堆叠层来模拟更复杂的功能一样，我们也可以通过增加自编码器的层数来构建更强大的自编码器。这就引出了**堆叠式自编码器**，也称为深度自编码器。

堆叠式自编码器本质上是一种自编码器，其编码器和解码器部分都包含多个隐藏层。数据不再是单一地从输入到潜在空间再返回的转换，而是经历了一系列转换。

堆叠式自编码器的**编码器**部分通常由多个层组成，这些层逐步降低输入的维度。每一层都学习将其输入转换为更抽象且通常更压缩的表示。
**瓶颈**层仍是中心的、最压缩的层，包含最终的潜在表示。
**解码器**部分则镜像编码器，包含多个层，这些层逐步从潜在表示中重构数据，恢复到其原始维度。

### 层次的效用：在多层级学习特征

构建更深自编码器的主要原因是它们学习**分层特征**的能力。这意味着网络中不同的层学习不同抽象程度的特征。

以图像数据为例：

- 第一个隐藏层（最接近输入）可能会学习检测简单的特征，例如边缘、角点或基本纹理。
- 第二个隐藏层以第一个隐藏层的输出作为其输入，可能会学习组合这些简单特征，以表示更复杂的模式，例如物体的一部分（例如，眼睛、轮子、叶子）。
- 后续层则可以将这些部分组合成更抽象的表示，可能对应于整个物体或重要的物体组成部分。

这种分层学习过程使堆叠式自编码器能够捕获数据中复杂的结构和依赖关系，从而产生比浅层自编码器所获得的更丰富且通常更有用的特征表示。

### 堆叠式自编码器的架构

典型的堆叠式自编码器可能具有这样一种架构：编码器中每层的神经元数量递减，而解码器中每层的神经元数量递增。例如，如果输入有784个维度，一个堆叠式自编码器可能有一个编码器结构，如784 -> 256 -> 128 -> 64（潜在空间），以及一个对称的解码器结构，如64 -> 128 -> 256 -> 784。

> 一张堆叠式自编码器的图表，其中编码器和解码器各有两个隐藏层，说明了数据流和逐步变换过程。

图中的每个“变换”都代表一个层的操作，通常是仿射变换后接非线性激活函数 (activation function)。解码器层通常旨在逆转其对应编码器层的变换。

### 训练堆叠式自编码器

堆叠式自编码器可以像任何其他深度神经网络 (neural network)一样，通过最小化输入$X$和输出$X'$之间的重构损失来进行**端到端**的训练。标准的反向传播 (backpropagation)和优化算法（如Adam或SGD）用于此目的。

然而，从零开始训练深度自编码器有时会遇到挑战，例如梯度消失或梯度爆炸问题，特别是当网络非常深或未仔细初始化时。另一种具有历史意义的方法是**贪婪逐层训练**，我们将在下一节中更详细地讨论它。此方法涉及顺序训练每个层（或一对编码器-解码器层）。

### 优势与考量

**优势：**

- **更丰富的表示**：深度架构可以学习更复杂的分层特征，从而更好地理解数据。
- **改进的压缩**：对于高度复杂的数据，与浅层模型相比，堆叠式自编码器通常可以获得更好的压缩比，同时保留更多信息。
- **潜在的更好下游性能**：堆叠式自编码器学习到的更具区分性的特征可以在分类或聚类等后续机器学习 (machine learning)任务中使用时带来性能提升。

**考量：**

- **复杂性增加**：与浅层自编码器相比，设计和调优堆叠式自编码器涉及更多超参数 (parameter) (hyperparameter)（层数、每层单元数、每层激活函数 (activation function)）。
- **计算成本**：训练更深的模型通常计算强度更高且更耗时。
- **过拟合 (overfitting)风险**：参数越多，过拟合的风险也随之增加，尤其是在数据集较小的情况下。正则化 (regularization)技术，包括稀疏自编码器或去噪自编码器中使用的技术，变得更加重要。

通过理解如何构建和训练这些深度架构，您可以提升更强大的特征提取能力。在后续章节中，我们将考察训练和优化这些模型的特定技术。

## 参考资料

- [Reducing the Dimensionality of Data with Neural Networks](https://doi.org/10.1126/science.1127647) — Geoffrey E. Hinton and Ruslan R. Salakhutdinov (2006)
  Journal: Science; Publisher: American Association for the Advancement of Science; Volume: 313; Pages: 504-507; DOI: [10.1126/science.1127647](https://doi.org/10.1126/science.1127647)
  一篇基础性论文，介绍了深度自动编码器，并展示了贪婪逐层预训练在降维中的有效性，这是训练深度网络的一个具有历史意义的方法。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本全面的教科书，涵盖了深度学习的理论和实践方面，包括对自动编码器、堆叠自动编码器及其训练方法的详细解释。
- [Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion](http://www.jmlr.org/papers/v11/vincent10a.html) — Pascal Vincent, Hugo Larochelle, Isabelle Lajoie, Yoshua Bengio, Pierre-Antoine Manzagol (2010)
  Journal: Journal of Machine Learning Research; Volume: 11; Pages: 3371-3408
  该论文介绍了堆叠去噪自动编码器，这是堆叠自动编码器的一个重要变体，为学习鲁棒的、分层特征提供了见解。
- [Unsupervised Feature Learning and Deep Learning (UFLDL) Tutorial](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFEUA3vZdXiEfUmoDgGCbWTSPi7wBU7NdUgzfkvARXuHa0WiOGcJPywnswXtu9OVCF6SL-KiFJvMZvfldNSYbaGOhnhDRhjLk7oKl0_DluK7_awAaswl4CT8Yw0Fjoh1ujOMvgSwg==) — Andrew Ng, Jiquan Ngiam, Chuan Yu Foo, Yifan Mai, Caroline Suen, Adam Coates, Andrew Maas, Awni Hannun, Brody Huval, Tao Wang, Sameep Tandon (2013)
  Publisher: Stanford University
  提供了自动编码器（包括堆叠自动编码器）及其训练的清晰实用解释和示例，是一个优秀的教育资源。
