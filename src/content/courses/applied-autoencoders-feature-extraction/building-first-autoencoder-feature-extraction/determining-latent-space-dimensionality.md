---
course: "applied-autoencoders-feature-extraction"
chapter: "building-first-autoencoder-feature-extraction"
lesson: "determining-latent-space-dimensionality"
sourceId: 6456
sourceUrl: "https://apxml.com/zh/courses/applied-autoencoders-feature-extraction/chapter-3-building-first-autoencoder-feature-extraction/determining-latent-space-dimensionality"
title: "确定潜在空间维度"
description: "选择自动编码器瓶颈层合适维度的方法和考量。"
order: 3
plots: ["plots/6456-0.json"]
sourceHash: "4a05fe589fb4a0e5d3b374501f250c97d75b4d1d15fd24e85aa3a422c7fedc95"
sourceCorrections: []
---

瓶颈，即潜在空间，是自动编码器进行数据压缩的地方。它是网络中最狭窄的部分，促使编码器学习输入数据的紧凑表示。这一潜在空间的维度，即瓶颈层中神经元或单元的数量，是一个重要的超参数 (parameter) (hyperparameter)，你需要做出决定。这个选择直接影响压缩程度以及自动编码器所学特征的丰富性。

### 为什么潜在空间维度很重要

为潜在空间选择合适的尺寸需要权衡。

- **过小：** 如果潜在空间过小，自动编码器可能难以捕获足够的输入数据信息。这可能导致较高的重建误差，意味着解码器无法准确重建原始输入。产生的特征可能过于粗略，缺乏下游任务所需的细节。想象一下用一句话概括一部复杂的小说；你将丢失大量细节。
- **过大（对于欠完备自动编码器）：** 如果潜在空间维度与输入维度过于接近（或者，在一些不常见的情况下，即使在*没有*我们稍后讨论的正则化 (regularization)技术时更大），自动编码器可能学不到非常有用的压缩表示。它可能只是学习将输入以最少的处理复制到输出，行为像一个恒等函数。这就违背了降维和特征学习的目的。目标是强制网络学习显著模式，而不仅仅是记忆。

对于本章中我们构建的标准自动编码器，我们的目标是“欠完备”自动编码器，其中潜在维度明显小于输入维度。这种固有的限制是学习有意义的压缩特征的动力。

### 选择合适维度的方法

没有通用的公式来选择完美的潜在空间维度。这通常取决于你数据的复杂性以及你打算如何使用提取的特征。然而，这里有几种实用的策略和考虑事项：

1. **从启发式方法和实验开始：**
   一个常见的起点是选择一个输入维度的一部分作为潜在维度。例如，如果你的输入数据有128个特征，你可能会尝试64、32、16甚至更小的潜在维度。“最优”压缩水平取决于数据。相比更简单的数据集，更复杂的数据集可能需要相对更大的潜在空间来保留重要信息。
2. **评估重建损失：**
   衡量自动编码器有效性（并间接衡量潜在维度合适性）的主要方法之一是监控重建损失。使用几种不同的潜在维度训练自动编码器，并比较它们在验证集上的性能。

   你可以绘制验证重建损失与潜在空间维度之间的关系图。通常，随着潜在维度的增加，重建损失会减小，因为模型有更大的容量来存储信息。然而，你需要寻找一个边际收益递减点，即图中一个“拐点”，在该点增加更多潜在空间维度并不能显著提高重建质量。这可以提示一个好的权衡。

   
   
   ![重建损失 vs. 潜在维度](plots/6456-0.json)
   
   

   > 一个图表，说明重建误差通常如何随着潜在空间维度增加而减小。目标通常是找到一个点，在该点之后进一步增加维度所带来的改进很小。

   请记住，如果你的主要目标是为下游任务提取特征，极低的重建误差并非总是唯一目的。有时，一个略高的重建误差，但具有更压缩和区别性的潜在空间，可以在该任务上带来更好的性能。
3. **考虑下游任务表现：**
   如果你正在为特定的监督学习 (supervised learning)任务（例如分类或回归）提取特征，最终的检验是这些特征的表现如何。

   - 使用各种潜在维度训练你的自动编码器。
   - 从瓶颈层提取每种配置下的特征。
   - 使用这些不同的特征集训练你的下游模型（例如逻辑回归、支持向量 (vector)机或小型神经网络 (neural network)）。
   - 在保留的测试集上评估下游模型的性能（例如准确率、F1分数、R平方）。

   这种方法直接衡量学习到的表示对于你特定应用的实用性。你可能会发现，一个更小的潜在维度，即使重建效果略差，也能更好地泛化或带来一个更简单、更高效的下游模型。
4. **数据复杂度和固有维度：**
   你的数据固有的复杂性起到作用。位于高维输入空间内或接近低维流形的数据通常可以有效地压缩到小的潜在空间中。例如，手写数字图像（如MNIST）有很多像素，但定义每个数字的实际变化可以在更低的维度中捕获。
   虽然存在估计固有维度的正式方法，但出于实际目的，如上所述的实验通常就足够了。如果你的数据非常复杂，具有许多独立的变异因素，你可能需要比简单、结构化数据更大的潜在维度。
5. **潜在空间可视化（针对极低维度）：**
   如果你选择2或3的潜在维度，你可以直接可视化数据点的潜在表示。我们将在本章后面介绍可视化技术。虽然这不是一种直接*选择*维度的方法，但如果你的2D或3D可视化显示出类别之间（如果你有标记 (token)数据）的良好分离或有意义的簇，这是一个积极的迹象。如果它是一团混乱，那么该维度可能太小，或者自动编码器没有有效学习。

### 迭代是核心

选择潜在空间维度很少是一次性决定。这是一个迭代过程。你会提出一个维度，构建并训练你的模型，评估其重建和/或特征质量，然后调整你的选择。不要害怕尝试多种选项。从这些实验中获得的见解也将加深你对数据和自动编码器行为的理解。

随着本课程的进展，你将遇到高级自动编码器架构，如稀疏自动编码器（第4章），其中潜在维度大小和特征质量之间的关系因正则化 (regularization)而变得更加复杂。然而，对于我们现在关注的基本自动编码器，寻找一种平衡信息保留和积极压缩的“欠完备”表示的原则是中心。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  全面解释了自动编码器，包括欠完备自动编码器、潜在空间的作用及其维度选择方法。
- [Reducing the Dimensionality of Data with Neural Networks](https://doi.org/10.1126/science.1127647) — Geoffrey E. Hinton, Ruslan R. Salakhutdinov (2006)
  Journal: Science; Publisher: American Association for the Advancement of Science; Volume: 313; Pages: 504-507; DOI: [10.1126/science.1127647](https://doi.org/10.1126/science.1127647)
  这篇奠基性论文介绍了使用深度自动编码器进行有效降维的概念，为潜在空间学习奠定了基础。
- [Autoencoders, Unsupervised Learning, and Deep Architectures](https://proceedings.mlr.press/v27/baldi12a.html) — Pierre Baldi (2012)
  Journal: Proceedings of ICML Workshop on Unsupervised and Transfer Learning; Publisher: PMLR; Volume: 27; Pages: 37-49
  一篇综述，提供了自动编码器的基础概念，讨论了瓶颈层在学习压缩表示中的作用和各种架构考量。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media, Inc.
  提供了实现自动编码器和调整潜在空间维度等超参数的实用指导，并附有代码示例。
