---
course: "cnns-for-computer-vision"
chapter: "model-compression-efficient-dl"
lesson: "network-pruning-techniques"
sourceId: 2630
sourceUrl: "https://apxml.com/zh/courses/cnns-for-computer-vision/chapter-8-model-compression-efficient-dl/network-pruning-techniques"
title: "网络剪枝技术"
description: "介绍从训练好的网络中移除冗余权重或滤波器以减小模型大小和计算量的方法。"
order: 2
plots: []
sourceHash: "fa58cb5362f324878f1099a6df153f86b697798ea83e82835b0599b505fa74ea"
sourceCorrections: []
---

如前所述，部署大型卷积神经网络 (neural network) (CNN)通常需要方法，使其更紧凑、计算成本更低。网络剪枝是一种普遍使用的方法，它通过从训练好的网络中移除多余的参数 (parameter)或结构来解决这个问题，目的是减小其大小和推理 (inference)成本，同时对准确性影响很小。其基本思路是，许多深度学习 (deep learning)模型存在大量冗余参数，包含许多对最终预测贡献很小的权重 (weight)甚至整个滤波器。

### 剪枝的理由

深度神经网络 (neural network)性能高常部分原因在于其参数 (parameter)量大。然而，研究和实践表明，许多参数在训练后可以移除，而准确性损失不大。这种冗余可能源于优化过程或最初的网络设计。剪枝方法通过识别并移除不重要的部分来解决这一冗余问题。目的是在原训练好的网络中找到一个更小的子网络，其表现几乎一样好。

### 网络剪枝的种类

剪枝方法可以根据所移除元素的细致程度进行大体分类：

#### 非结构化剪枝（权重 (weight)剪枝）

这是最细致的剪枝形式。网络层（卷积层或全连接层）中的单个权重根据某种重要性标准（通常是其大小）来确定。大小低于某个阈值的权重被设置为零。

- **流程：**
  1. 训练一个密集网络直到收敛。
  2. 根据单个权重的绝对大小（或其他标准）进行排序。
  3. 设定一个目标稀疏度水平（例如，移除80%的权重）。
  4. 将大小最小的权重置为零，以达到目标稀疏度。
  5. 对剩余的非零权重进行微调 (fine-tuning)，以恢复剪枝过程中损失的准确性。
- **特点：**
  - 得到稀疏权重矩阵，其中零值不规则分布。
  - 可以实现非常高的压缩比（移除很大比例的权重）。
  - 需要专用硬件或软件库（例如，稀疏矩阵乘法程序）来实现推理 (inference)速度的明显提升，因为标准硬件通常不能有效加速任意稀疏模式的计算。
  - 微调步骤对于恢复性能很必要。

微调过程有助于网络适应权重的移除。通常，这包含用固定为零的剪枝权重重新训练网络，只让剩余的权重进行调整。

#### 结构化剪枝

结构化剪枝不是移除单个权重，而是移除网络的整个结构元素。这使得所得网络更小，在标准硬件上天生更快，无需专用稀疏计算库。常见形式有：

- **滤波器剪枝：** 从卷积层中移除整个滤波器（及其对应的特征图）。如果从第$i$层剪枝一个滤波器，那么第$i+1$层的相应输入通道也会被有效移除。
- **通道剪枝：** 类似于滤波器剪枝，移除整个通道。
- **神经元剪枝：** 从全连接层中移除整个神经元（权重矩阵中的列）。
- **流程：**

  1. 训练一个密集网络。
  2. 计算每个结构（例如，滤波器、通道）的重要性分数。常用的标准包含滤波器权重的L1或L2范数、滤波器在数据集中产生的平均激活值，或基于梯度的评定方法。
  3. 根据重要性分数对结构进行排序。
  4. 移除得分最低的结构，以达到目标削减水平。
  5. 微调更小、剪枝后的网络。
- **特点：**

  - 得到更小、更密集的网络（或滤波器/神经元更少的层）。
  - 在标准硬件（CPU、GPU）上直接转化为更少的计算量（FLOPs）和内存使用。
  - 可能无法达到非结构化剪枝的理论压缩比，但通常能带来更好的实际加速。
  - 微调仍然非常需要。

> 非结构化（权重）剪枝得到稀疏矩阵与结构化（滤波器）剪枝得到更小、密集层的对比。

### 剪枝流程和标准

一个典型的剪枝流程包含迭代地进行剪枝和微调 (fine-tuning)：

1. **训练：** 训练原始的大模型直到收敛。
2. **剪枝：** 选择一个剪枝方法（非结构化/结构化）和标准（例如，大小、范数）。根据此标准移除一部分网络组件。
3. **微调：** 重新训练剪枝后的网络若干个周期，让剩余参数 (parameter)进行调整并恢复准确性。
4. **迭代：** 重复步骤2和3，直到达到所需的稀疏度或大小削减水平，或者直到准确性降至可接受的阈值以下。

选择识别不重要组件的合适**标准**很重要：

- **基于大小：** 最简单且通常有效的方法。假设大小小的参数贡献较少。对于结构化剪枝，常用滤波器/通道内权重 (weight)的L1或L2范数。
- **基于梯度：** 在训练期间使用梯度信息来评估参数的重要性。移除后导致损失函数 (loss function)增加最小的参数可能被认为不那么重要。
- **基于激活：** 分析神经元或滤波器产生的输出激活。例如，在许多输入上持续产生接近零激活的滤波器可能被剪枝。
- **其他标准：** 更复杂的方法可能包含分析损失函数的Hessian矩阵或使用敏感度分析。

### 考虑因素和权衡

- **稀疏度水平：** 确定剪枝多少通常靠经验。剪枝过于激进会不可逆地损害性能，而剪枝太少则收效甚微。迭代剪枝可以逐步减少。
- **准确性恢复：** 微调 (fine-tuning)几乎总是必需的。学习率和微调的持续时间需要仔细调整。有时，由于正则化 (regularization)作用，剪枝后准确性甚至会略微提高，但更普遍的情况是会略有下降。
- **硬件/库支持：** 如前所述，非结构化剪枝带来的实际加速很大程度上取决于是否具备能高效处理稀疏计算的优化库或硬件加速器。结构化剪枝通常能在商用硬件上提供更可预测的加速。
- **层敏感性：** 网络中不同层对剪枝可能有不同的敏感性。早期层通常学习通用特征，可能比后面的层更敏感。一些方法包含对不同层应用不同的剪枝比例。

网络剪枝提供了一系列有效方法来降低深度学习 (deep learning)模型的复杂性。通过仔细移除多余的权重 (weight)或结构并微调剩余网络，可以显著减小模型大小和计算需求，使模型更适合在受限环境中部署。非结构化剪枝和结构化剪枝之间的选择通常取决于具体的硬件目标以及压缩比和实际加速实现难易程度之间的权衡。

## 参考资料

- [Learning both Weights and Connections for Efficient Neural Networks](https://proceedings.neurips.cc/paper/2015/file/ae0eb3eed39d2bcef846ba727407cffe-Paper.pdf) — Song Han, Jeff Pool, John Tran, William J. Dally (2015)
  Journal: Advances in Neural Information Processing Systems; Publisher: NeurIPS; Volume: 28; Pages: 1135-1143
  介绍基于幅度的非结构化剪枝和迭代训练-剪枝-微调方法的开创性工作。
- [Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710) — Hao Li, Asim Kadav, Igor Durdanovic, Hanan Samet, and Hans Peter Graf (2017)
  Journal: International Conference on Learning Representations (ICLR); Publisher: Curran Associates, Inc.; Pages: 1683; DOI: [10.48550/arXiv.1608.08710](https://doi.org/10.48550/arXiv.1608.08710)
  基于L1范数移除整个卷积核的结构化剪枝开创性工作，展示了实际加速效果。
- [Deep Learning Model Compression: A Comprehensive Survey](https://www.sciencedirect.com/science/article/pii/S092523122030806X) — Haitao Yang, Wenqi Li, Pengyu Li, Bingqiang Jin, and Haoliang Li (2020)
  Journal: Neurocomputing; Publisher: Elsevier; Volume: 420; Pages: 494-522; DOI: [10.1016/j.neucom.2020.07.067](https://doi.org/10.1016/j.neucom.2020.07.067)
  全面概述模型压缩技术，包括网络剪枝（非结构化和结构化）、量化和知识蒸馏。
