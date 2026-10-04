---
course: "applied-autoencoders-feature-extraction"
chapter: "autoencoder-core-concepts"
lesson: "autoencoder-bottleneck-latent-space"
sourceId: 6447
sourceUrl: "https://apxml.com/zh/courses/applied-autoencoders-feature-extraction/chapter-2-autoencoder-core-concepts/autoencoder-bottleneck-latent-space"
title: "瓶颈层：潜在空间表示"
description: "了解瓶颈层，它包含输入数据的压缩表示或潜在变量。"
order: 3
plots: []
sourceHash: "ace6d44ef0d847fede31207a07fee2784a115487af0050af1594d4a5d8985a0b"
sourceCorrections: []
---

自动编码器架构的核心是**瓶颈层**。该层直接位于编码器和解码器之间，是压缩和表示学习“发生作用”的地方。编码器处理并压缩输入数据后，得到的压缩形式就存在于这个瓶颈中。之所以称为“瓶颈”，是因为它通常比输入层或输出层具有少得多的神经元，这迫使网络学习一种紧凑的表示。

> 自动编码器架构图，突显了瓶颈层（潜在空间），压缩表示 Z 在此层由输入 X 生成，并从中产生重构的 X̂。

这个瓶颈层的输出通常被称为**潜在空间表示**、**编码**或**编码值**。“潜在”表示这些表示形式捕捉了数据中隐藏或深层的结构。如果输入数据的维度是 $d$，瓶颈层通常会将其映射到维度为 $d'$ 的潜在表示 $z$，其中 $d' < d$。例如，MNIST 数据集中的一张图片可能有 $28 \times 28 = 784$ 像素（维度）。自动编码器可以设计一个瓶颈层，将其压缩到例如 $d' = 32$ 维。

这个低维向量 (vector) $z$ 不仅仅是信息的随机子集。通过训练过程，在最小化重构误差的目标指导下，自动编码器学习以这种紧凑的形式保留输入数据最显著和最有用的方面。它学习去除噪声和冗余，专注于定义数据的基本特征。这种潜在表示 $z$ 中的值是学到的特征。

**潜在空间**本身是这些表示 $z$“存在”的多维空间。潜在空间中的每个点都对应一个潜在输入的压缩版本。训练有素的自动编码器通常会以有意义的方式组织这个空间。例如，相似的输入（如相同数字的图片，或相似类型的客户行为）可能会被映射到潜在空间中相近的点，而不同的输入则映射得更远。

这个潜在空间的维度 $d'$ 是你在设计自动编码器时将选择的一个重要超参数 (parameter) (hyperparameter)。

- 如果 $d'$ 过小，自动编码器可能无法捕获足够的信息来准确重构输入，导致重构误差高。这有时被称为过于受限的“信息瓶颈”。
- 如果 $d'$ 过大（接近 $d$），网络可能只会简单地将输入复制到输出，而没有学到任何有趣的深层结构（一个恒等函数，特别是当编码器和解码器有足够容量时）。这对特征提取或降维的帮助较小。

目标是找到一个远小于 $d$ 的维度 $d'$，但同时又能让自动编码器学习到丰富且有信息量的表示。这种表示应该足以让解码器进行良好的重构，并且对本课程来说重要的是，可以作为其他机器学习 (machine learning)任务的有用特征集。

可以将瓶颈层视为创建输入的高效摘要或提炼后的要点。编码器的任务是编写这个摘要，而解码器的任务是将这个摘要扩展回与原始数据相似的内容。这个摘要（潜在空间表示）的质量和特性，决定了自动编码器能够实现的目标，从简单的降维到更复杂的任务，例如去噪，甚至生成新数据（这将在后续关于变分自动编码器的内容中看到）。

本质上，瓶颈层及其产生的潜在空间表示是我们使用自动编码器进行特征提取时所追求的主要结果。这些学到的特征（向量 $z$）随后可以输入到其他模型中，通常能带来性能的提升或下游任务中更高效的计算。

## 参考资料

- [Reducing the Dimensionality of Data with Neural Networks](https://www.science.org/doi/10.1126/science.1127647) — Geoffrey E. Hinton, Ruslan R. Salakhutdinov (2006)
  Journal: Science; Publisher: American Association for the Advancement of Science; Volume: 313; Pages: 504-507; DOI: [10.1126/science.1127647](https://doi.org/10.1126/science.1127647)
  这篇开创性论文介绍了深度自编码器，用于有效降维和表示学习，是理解瓶颈层作用的基础。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  第14章详细解释了自编码器，包括其架构、潜在空间的概念以及它们在学习数据表示方面的用途。
- [CS230: Deep Learning Lecture Notes - Unsupervised Learning: Autoencoders, PCA, GMM](http://cs230.stanford.edu/files_spring_2019/lectures/lecture_6_unsupervised_learning_slides_spring_2019.pdf) — Andrew Ng and Stanford CS230 Teaching Team (2019)
  本讲座涵盖自编码器基础知识，包括瓶颈层、潜在空间及其在学术深度学习课程背景下与降维的关系。
- [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python) — François Chollet (2017)
  Publisher: Manning Publications
  提供了自编码器的实用、以代码为中心的解释，阐明瓶颈层如何形成用于特征提取的潜在空间表示。第一版。
