---
course: "autoencoders-representation-learning"
chapter: "foundations-representation-learning"
lesson: "non-linear-feature-extraction-need"
sourceId: 2799
sourceUrl: "https://apxml.com/zh/courses/autoencoders-representation-learning/chapter-1-foundations-representation-learning/non-linear-feature-extraction-need"
title: "非线性特征提取的需求"
description: "理解为何复杂数据表示需要非线性方法。"
order: 4
plots: ["plots/2799-0.json", "plots/2799-1.json"]
sourceHash: "e2357b81b00dfb9c588a28e094011314cc38b257441c9b1e39f5d08102c22dae"
sourceCorrections: []
---

尽管主成分分析（PCA）等线性方法在数据主要呈现线性相关性时，对于捕捉方差和降低维度很有效，但它们在处理许多数据集固有的复杂性时常常力不从心。这些方法的一个主要局限在于它们从根本上假定数据位于或接近高维空间 (high-dimensional space)中的线性子空间。这一假定经常失效。

### 流形假说与数据

许多高维数据，例如图像、音频信号或文本嵌入 (embedding)，用**流形假说**来描述更为恰当。这一假说认为，尽管数据点存在于一个极高维的环境空间（如图像所有可能像素值的空间）中，但它们实际上靠近内嵌于该空间中的一个低维非线性流形。

设想一张卷起来的纸（一个“瑞士卷”）在三维空间中。纸张表面的固有维度是二维的，但它存在于三维环境中。像PCA这样的线性方法，旨在找到最大方差的方向，可能会简单地将纸卷投影到二维平面上，从而有效地将其压扁并失去其内在结构。在纸卷表面上相距较远的点，在PCA投影中可能显得彼此靠近。



![非线性流形实例（“瑞士卷”）](plots/2799-0.json)



> 数据点在非线性流形上的简化三维表示，类似于“瑞士卷”。线性方法难以捕捉其内在结构。



![“瑞士卷”的PCA投影](plots/2799-1.json)



> 对“瑞士卷”数据应用PCA会将点投影到二维平面上。请注意，其固有结构是如何丢失的，以及在卷上相距较远的点在投影中可能变得靠近。

### 非线性对特征提取为何重要

表示学习的目的不仅是降维；它还关于找出能有效捕捉数据内在结构和变化的*特征*或*要素*。再次考虑图像数据。一个物体（例如，“猫”）的识别在光照、姿态、比例或平移等各种变换下保持不变。这些变换在像素空间中通常是高度非线性的。

- **组合性：** 特征通常以复杂、非线性的方式组合。识别一张脸涉及识别眼睛、鼻子和嘴巴（低级特征），并理解它们特定的空间关系（一种更高级的非线性组合）。线性方法难以模拟这种层级和组合结构。
- **复杂作用：** 高维数据中的变量很少纯粹线性相互作用。例如，在基因表达数据中，一个基因的作用可能会根据另一个基因的活动水平而被放大或抑制，这是一种非线性作用。
- **生成要素：** 我们常认为数据是由一组较小的潜在要素（例如，物体身份、位置、光照方向）生成的。从这些要素到观测数据的映射通常是复杂且非线性的。为了恢复或表示这些要素，我们需要能够反转或近似这种非线性映射。

### 神经网络 (neural network)的作用

这就是神经网络，即自编码器的构成元素，发挥作用的地方。深度学习 (deep learning)模型的强大之处主要源于它们学习复杂非线性函数的能力。这种能力来自两个主要部分：

1. **分层结构：** 深度网络学习层次化表示，其中较早的层可能检测简单模式（边缘、纹理），而较后的层将这些模式组合起来表示更复杂的事物（物体部分、物体）。
2. **非线性激活函数 (activation function)：** 像ReLU（修正线性单元）、sigmoid或tanh这样的函数在每一层的线性变换（矩阵乘法）之后逐元素应用。如果没有这些非线性，一个深层线性堆叠在数学上将退化为单个线性变换，在函数近似方面不比PCA等更简单的线性模型有任何优势。正是激活函数允许网络弯曲和扭曲特征空间，使其能够近似数据通常所处的复杂非线性流形。在数学上，神经网络$f(x; \theta)$从输入空间$\mathcal{X}$学习到特征空间$\mathcal{Z}$的一个映射$f: \mathcal{X} \rightarrow \mathcal{Z}$，其中这个映射是线性变换和非线性激活的组合。

像t-SNE和UMAP这些之前介绍过的技术，通过创建保留局部关系的低维嵌入 (embedding)，非常适合*可视化*这些非线性结构。然而，它们通常不提供将新数据点映射到嵌入空间的显式*编码器*函数，也不学习适合重建或生成任务的特征。

因此，为了学习能够捕捉复杂数据结构以用于超越可视化（如压缩、去噪、生成或迁移学习 (transfer learning)）任务的表示，我们需要能够学习强大的非线性特征提取器的方法。自编码器提供了一个灵活有效的框架，以无监督或自监督的方式直接从数据中学习此类非线性映射，构成了后续章节的主要内容。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  对深度学习进行了介绍，涵盖了表征学习、非线性特征提取和自编码器等概念。
- [Reducing the Dimensionality of Data with Neural Networks](https://doi.org/10.1126/science.1127647) — Geoffrey E. Hinton, Ruslan R. Salakhutdinov (2006)
  Journal: Science; Publisher: American Association for the Advancement of Science; Volume: 313; Pages: 504-507; DOI: [10.1126/science.1127647](https://doi.org/10.1126/science.1127647)
  提出深度自编码器以学习有效的低维非线性表示，解决了线性方法的局限性。
- [A Global Geometric Framework for Nonlinear Dimensionality Reduction](https://doi.org/10.1126/science.290.5500.2319) — Joshua B. Tenenbaum, Vin de Silva, John C. Langford (2000)
  Journal: Science; Volume: 290; Pages: 2319-2323; DOI: [10.1126/science.290.5500.2319](https://doi.org/10.1126/science.290.5500.2319)
  介绍了Isomap，一种降维算法，展示了流形假设以及线性方法遇到的挑战。
