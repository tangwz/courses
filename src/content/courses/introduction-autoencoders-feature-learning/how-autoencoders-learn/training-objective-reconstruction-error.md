---
course: "introduction-autoencoders-feature-learning"
chapter: "how-autoencoders-learn"
lesson: "training-objective-reconstruction-error"
sourceId: 6413
sourceUrl: "https://apxml.com/zh/courses/introduction-autoencoders-feature-learning/chapter-3-how-autoencoders-learn/training-objective-reconstruction-error"
title: "训练目标：减少重建误差"
description: "了解训练自编码器的主要目标：最小化输入与重建输出之间的差异。"
order: 1
plots: []
sourceHash: "0939f5c9088db6c7816cde76261eaa65e683e84c536686c754e296401a8d234e"
sourceCorrections: []
---

自编码器学习过程的核心是一个直接而有效的目标：使其输出尽可能地与其输入相似。想象一下，你正在尝试用几个词向某人描述一幅复杂的图片（这就像编码步骤）。然后，这个人根据你的简短描述尝试重新绘制这幅图片（这就像解码步骤）。训练目标就是让他们的画作（自编码器的输出）看起来与原始图片（自编码器的输入）几乎相同。

### 重建的要旨

当我们向自编码器输入数据时，我们将原始输入称为$X$。自编码器通过其编码器处理此输入，将其压缩成瓶颈层中的低维表示，然后解码器尝试从这种压缩形式重建原始输入。我们将这种重建输出称为$\hat{X}$（读作“X-hat”）。

原始输入$X$与重建输出$\hat{X}$之间的差异就是我们所说的**重建误差**。自编码器的整个训练过程都旨在最小化此误差。

> 自编码器处理输入$X$以生成重建输出$\hat{X}$。学习过程侧重于最小化$X$和$\hat{X}$之间的差异。

### 为何侧重于最小化误差？

你可能会好奇，这种简单的复制输入的目标为何如此有用。这里的诀窍是：自编码器不仅仅是复制。它被强制将信息通过一个“瓶颈”传输，即那个压缩的、低维度的表示。

1. **学习有效的表示**: 为了从瓶颈层中的压缩表示成功重建输入$X$，自编码器必须学会保留输入中最重要的、显著的信息，使其留在这个小的瓶颈层中。它必须丢弃噪声或冗余信息，并捕获数据的真实潜在结构。如果瓶颈表示不佳，解码器将无法准确重建输入，从而导致高重建误差。
2. **自我修正**: 通过尝试最小化重建误差，自编码器有效地学到了数据的哪些方面是重要的。如果它在重建时犯了错误，误差会告诉网络如何调整其内部参数 (parameter)（权重 (weight)和偏置 (bias)），以便下次做得更好。这种尝试重建、测量误差和调整的迭代过程，是神经网络 (neural network)学习方式的基本组成部分。
3. **特征学习的根本**: 当自编码器学会很好地重建数据时，其瓶颈层中的激活通常会形成输入数据的一个有用、压缩的表示。这些学到的表示本质上是能够捕获数据本质的“特征”，随后可用于其他机器学习 (machine learning)任务，例如分类或异常检测。我们将在后续章节中更详细地讨论特征学习。

可以把它想象成学习总结一本长书。要写一个好的摘要（瓶颈表示），你必须理解主要的主题和情节要点（重要的特征）。如果有人能从你的摘要中理解整个故事（重建原始信息）从你的摘要中理解整个故事，那么你做得很好。自编码器的工作方式类似：它学会总结（编码）然后展开（解码），衡量一个好的摘要的标准就是原始信息能够被重建得有多好。

重建误差越小，自编码器就越能胜任其工作。这一个单一目标推动着整个学习过程。在接下来的部分中，我们将了解如何使用称为损失函数 (loss function)的数学函数来实际量化 (quantization)这种“重建误差”，以及自编码器如何通过一个称为优化的过程来调整自身。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  涵盖自动编码器、其目标函数及其在学习高效数据表示中的作用的综合教材。
- [Reducing the Dimensionality of Data with Neural Networks](https://doi.org/10.1126/science.1127647) — Geoffrey E. Hinton, Ruslan R. Salakhutdinov (2006)
  Journal: Science; Publisher: American Association for the Advancement of Science; Volume: 313; Pages: 504-507; DOI: [10.1126/science.1127647](https://doi.org/10.1126/science.1127647)
  一篇开创性论文，展示了深度自动编码器在降维和特征学习方面的有效性，强调了重建目标。
- [Introduction to Deep Learning (MIT 6.S191)](https://introtodeeplearning.com/) — Alexander Amini, Ava Amini (2025)
  Publisher: MIT
  一门介绍深度学习的课程，提供有关深度学习的讲座和材料，包括自动编码器及其重建目标。
- [Extracting and Composing Robust Features with Denoising Autoencoders](http://doi.acm.org/10.1145/1390156.1390294) — Pascal Vincent, Hugo Larochelle, Yoshua Bengio, and Pierre-Antoine Manzagol (2008)
  Journal: Proceedings of the 25th International Conference on Machine Learning (ICML '08); Publisher: ACM; Pages: 1096-1103; DOI: [10.1145/1390156.1390294](https://doi.org/10.1145/1390156.1390294)
  介绍了去噪自动编码器，展示了如何从损坏输入进行重建以增强特征学习，与目标直接相关。
