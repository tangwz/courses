---
course: "introduction-to-neural-networks"
chapter: "data-preparation-neural-networks"
lesson: "data-batching"
sourceId: 1936
sourceUrl: "https://apxml.com/zh/courses/introduction-to-neural-networks/chapter-2-data-preparation-neural-networks/data-batching"
title: "构建训练数据批次"
description: "了解梯度下降训练中为何以及如何将数据分批。"
order: 4
plots: ["plots/1936-0.json"]
sourceHash: "7b60e2d95635cd2074bb19b97a0f7565f6b118f3bf7b324e0e63760c1df9301c"
sourceCorrections: []
---

在对数值特征进行缩放和对类别特征进行编码以准备好数据后，接下来要考虑的是如何在训练期间将这些数据输入神经网络 (neural network)。训练神经网络涉及迭代调整其权重 (weight)和偏置 (bias)以使损失函数 (loss function)最小化。此调整过程通常使用梯度下降 (gradient descent)法，它计算损失相对于每个参数 (parameter)的变化方式。

为调整网络参数，一种计算梯度的方法是**一次性**根据整个训练数据集进行计算。这种方法被称为批量梯度下降 (Batch Gradient Descent)，它在对网络参数进行单次更新之前，会计算所有训练样本上的损失函数的精确梯度。尽管数学上合理，但这种方法带来明显的实际困难，特别是对于机器学习 (machine learning)中常见的大型数据集：

1. **计算开销：** 仅为了计算一次参数更新，就对数百万甚至数十亿数据点执行前向和后向传播，这极其耗时。训练将耗费不切实际的长时间。
2. **内存限制：** 加载整个数据集，以及计算所需的中间激活和梯度，可能超出可用内存 (RAM)；更重要的是，可能超出常用于加速训练的GPU显存 (VRAM)。

为克服这些限制，我们几乎总是将数据分成更小的块进行处理，称之为**批次**或**小批量**。

### 使用批次进行训练

小批量梯度下降 (gradient descent)法不是在每次参数 (parameter)更新时都使用完整数据集，而是在每一步处理训练数据的一个随机选择的小子集。以下是术语和过程的分类说明：

- **批次 (Batch)：** 全部训练数据的一个子集。例如，如果您有10,000个训练样本，一个批次可能包含32或64个样本。
- **批次大小 (Batch Size)：** 一个批次中包含的训练样本数量。这是您在训练前选择的一个超参数 (hyperparameter)。
- **迭代 (Iteration)：** 处理一个数据批次的一次完整过程。这包括执行前向传播、计算损失、执行反向传播 (backpropagation)以获取梯度，并根据该批次更新网络的参数。
- **周期 (Epoch)：** 完整地遍历*整个*训练数据集一次。如果您的数据集有$N$个样本，且批次大小为$B$，那么一个周期包含$N / B$次迭代。

典型的训练过程使用批次，如下所示：

1. **洗牌/打乱 (Shuffle)：** 在每个周期开始时，随机打乱整个训练数据集。这种随机化很重要，以确保每个批次具有代表性，并且模型不会根据数据呈现的顺序学习模式。
2. **迭代处理 (Iterate)：** 遍历打乱后的数据集，一次取一个批次。
3. **处理批次 (Process Batch)：** 对于每个批次：
   - 执行前向传播以获取预测结果。
   - 根据批次的预测结果和真实标签计算损失。
   - 执行反向传播，*仅*基于当前批次计算损失相对于参数的梯度。
   - 使用这些计算出的梯度和学习率更新网络的权重 (weight)和偏置 (bias) (这是小批量梯度下降法的核心)。
4. **重复 (Repeat)：** 继续处理批次，直到整个数据集都被遍历 (一个周期完成)。
5. **多个周期 (Multiple Epochs)：** 重复整个过程，进行设定数量的周期，或者直到满足停止标准 (例如，独立验证集上的损失不再改善)。

> 在一个周期内使用小批量进行训练的过程。完整数据集被打乱后，逐批次迭代处理。每次迭代都涉及前向/反向传播以及基于该批次的参数更新。

### 选择批次大小

批次大小 ($B$) 是一个重要的超参数 (parameter) (hyperparameter)，它影响训练动态、计算效率和模型泛化能力。这里存在一个权衡：

- **较小的批次大小 (例如，1, 8, 16, 32)：**
  - *优点：* 所需内存较少。参数更新更频繁 (每个周期内迭代次数更多)。梯度估计中固有的噪声 (因为它基于少量样本) 可以帮助优化器避开不良局部最小值，并可能找到更平坦的最小值，这通常能带来更好的泛化能力。极端情况，批次大小为1时，被称为随机梯度下降 (gradient descent) (SGD)。
  - *缺点：* 噪声梯度可能导致损失大幅波动，使得收敛性较不稳定。计算效率可能较低，因为现代硬件 (特别是GPU) 针对并行处理进行了优化，而极小的批次会使这种优化未能充分发挥。
- **较大的批次大小 (例如，128, 256, 512+)：**
  - *优点：* 提供对整个数据集真实梯度的更准确估计，从而实现更平滑的收敛。可以更有效地发挥硬件并行性，可能加快*每个周期*的计算速度。
  - *缺点：* 需要显著更多的内存 (RAM和GPU显存 (VRAM))。更新发生频率较低。研究表明，大批次有时可能使优化器趋向于“尖锐”的最小值，与小批次常找到的“平坦”最小值相比，这可能对未见过的数据泛化能力较差。

*实际考量:*
常用的批次大小是2的幂次方，例如32、64、128或256。这通常是由于硬件内存架构和库对这些大小进行了优化，从而带来更好的计算效率。然而，最佳批次大小很大程度上取决于具体数据集、模型架构和可用硬件。它通常通过实验和超参数调优来确定。



![批次大小对损失曲线的示例性影响](plots/1936-0.json)



> 较大的批次大小通常导致每次迭代的损失下降更平滑，但可能在实际运行时间上收敛稍慢，或达到次优最小值。较小的批次大小导致损失曲线更具噪声，但有时能更有效地处理损失。请注意，不同批次大小下，每个周期的迭代次数有显著差异。

总而言之，批处理是一种标准且必要的技术，用于在大型数据集上高效训练神经网络 (neural network)。它平衡了计算可行性、内存限制和训练动态。通过分批处理数据，我们能够使用可管理的少量数据进行频繁的参数更新，这构成了迭代学习过程的基础，该过程会在后续关于反向传播 (backpropagation)和梯度下降变体的章节中进行讲解。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本基础教科书，解释了各种梯度下降算法及其用于训练神经网络的批处理方法。
- [On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima](https://arxiv.org/abs/1706.03827) — Nitish Shirish Keskar, Komunte Mudigere, Jorge Nocedal, Mikhail Smelyanskiy, and Ping Tak Peter Tang (2017)
  Journal: International Conference on Learning Representations (ICLR 2017); DOI: [10.48550/arXiv.1706.03827](https://doi.org/10.48550/arXiv.1706.03827)
  这篇研究论文分析了批量大小对模型泛化能力的影响，以及所找到的最小值的特点（尖锐与平坦）。
- [Optimization: Stochastic Gradient Descent](http://cs231n.github.io/optimization-1/) — Andrej Karpathy, Justin Johnson, and Fei-Fei Li (2023)
  Journal: Stanford University CS231n Course Notes
  课程讲义，提供了对各种优化技术的清晰解释，包括神经网络中的批处理策略，来源于一门备受推崇的课程。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEgFDx_Z9PcUv0n_7zvIwbPS-_134RjAsIvt5PJsvx9jjwTzobVNVIjvFlPmKF_ajFDmbI7Uj4Foz3-S8CpC8S5UoI1S7s-3wG5uPIn87_8zhuq4uYQbSvogIb7csYYQXaM7id-GqSbEJU0DWU5cqVu5ez3CBdGH3FjXRpupDun0yuzYX1dtIW94ELeh-25urKJHMDs5s-FW-D6XiI=) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本实践指南，说明并演示了如何在现代深度学习框架中实现批处理。
