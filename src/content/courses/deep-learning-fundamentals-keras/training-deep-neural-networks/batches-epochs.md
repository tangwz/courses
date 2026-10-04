---
course: "deep-learning-fundamentals-keras"
chapter: "training-deep-neural-networks"
lesson: "batches-epochs"
sourceId: 4919
sourceUrl: "https://apxml.com/zh/courses/deep-learning-fundamentals-keras/chapter-3-training-deep-neural-networks/batches-epochs"
title: "批次与周期"
description: "解释模型训练中批次大小和周期的思想。"
order: 6
plots: []
sourceHash: "ae1994db17eb3dbd1764facb2c2fedb51260fb74d8c0b768fda4967de3b7d223"
sourceCorrections: []
---

训练神经网络 (neural network)时，特别是处理大型数据集时，一次性处理整个数据集来计算损失并更新权重 (weight)，可能耗费大量计算资源和内存。此外，每次权重更新都使用整个数据集（如同传统批次梯度下降 (gradient descent)）可能会导致收敛速度减慢或停留在局部最优解。为解决此问题，训练过程通常会划分为更小、更易于管理的分步操作，采用*批次*和*周期*的思想。

### 周期：完整的数据遍历

一个**周期**表示对整个训练数据集的一次完整遍历。如果你的数据集包含10,000张图片，一个周期就是在模型恰好一次性看过并从中学习过这10,000张图片之后完成。

训练深度学习 (deep learning)模型通常需要多个周期。为什么呢？因为仅一次遍历很少能让模型的权重 (weight)收敛到最佳值。网络需要多次查看数据，以便高效地掌握数据中蕴含的规律。这就像备考一样：你不会只看一遍课本；你会多次复习材料（多个周期）以巩固理解。

周期数是你在训练开始前设定的一个超参数 (parameter) (hyperparameter)。选择合适的数量很重要：

- **周期数过少：** 模型可能出现*欠拟合 (underfitting)*，意思是它没有充分学习数据中的规律。
- **周期数过多：** 模型可能出现*过拟合 (overfitting)*，意思是它过度适应了训练数据，包括数据中的噪声，导致在新、未见过的数据上表现不佳。我们将在下一节讨论如何使用验证数据来检测过拟合。

### 批次：分块处理数据

在一个周期内，我们不一次性处理整个数据集，而是将数据集划分为更小的子集，称之为**批次**。**批次大小**决定了每个批次中包含多少个训练样本。

在每个周期中，训练数据通常会被打乱顺序，然后分成这些批次。模型一次处理一个批次：

1. 将批次数据输入网络（前向传播）。
2. 根据该批次的预测结果计算损失。
3. 计算损失相对于模型权重 (weight)的梯度（反向传播 (backpropagation)）。
4. 根据这些梯度，使用所选优化器更新模型权重。

此过程在周期内对所有批次重复进行。模型每次处理一个批次并更新其权重时，称为一次**迭代**或**步**。

例如，如果你有一个包含2,000个样本的数据集，并将批次大小设定为100，那么一个周期将包含：
$\text{每个周期迭代次数} = \frac{\text{训练样本总数}}{\text{批次大小}} = \frac{2000}{100} = 20 \text{ 次迭代}$

在一个周期内，模型的权重将更新20次。

> 一张图解，展示了在单个周期内，完整的训练数据集如何被划分为批次。每个批次被顺序处理，从而进行一次权重更新（迭代）。

### 为何使用批次？

使用批次（通常称为小批次梯度下降 (gradient descent)）相较于一次性处理整个数据集（批次梯度下降）或一次处理一个样本（随机梯度下降，或SGD，尽管实践中SGD常指小批次SGD）具有多项优势：

1. **内存效率：** 处理整个数据集可能需要比可用内存（RAM或GPU显存 (VRAM)）更多的空间。批次使得通过每次仅加载一部分数据来处理大型数据集成为可能。
2. **计算效率：** GPU等硬件针对并行计算进行了优化。处理一批数据比顺序处理单个样本更能充分利用这种并行性。尽管一次性处理整个数据集看似可以并行，但内存限制通常使得小批次在实践中速度更快。
3. **梯度估计：** 从小批次计算的梯度是对真实梯度（即在整个数据集上计算的梯度）的*近似*。这种近似结果带有噪声，这实际上可能是有益的。这些噪声可以帮助优化器摆脱陡峭的局部最小值，并有可能找到更好、更平坦的最小值，这些最小值对于未见过的数据通常具有更好的泛化能力。
4. **更新更快：** 相较于批次梯度下降（在处理完整个数据集后更新），权重 (weight)更新的频率更高（每个批次后更新）。这通常会加速收敛，尽管每次更新所依据的信息较少。

### 批次大小的选择

批次大小是另一个重要的超参数 (parameter) (hyperparameter)。常见的批次大小是2的幂（例如，32、64、128、256），这是因为硬件内存对齐 (alignment)优化所致，但其他值也可能有效。选择时需要权衡利弊：

- **小批次大小（例如，1、8、16、32）：**
  - *优点：* 带有噪声的梯度更新有助于脱离局部最小值（具有正则化 (regularization)作用）。需要更少的内存。有时可以使每个周期内初步收敛速度更快（更新次数更多）。
  - *缺点：* 梯度近似结果噪声很大，可能导致训练不稳定，并且在实际运行时间上收敛较慢。由于硬件并行性利用不足，计算效率较低。
- **大批次大小（例如，128、256、512及以上）：**
  - *优点：* 更精确的梯度近似结果会使收敛更稳定。计算效率高，能最大限度地利用硬件并行性。
  - *缺点：* 需要更多内存。可能收敛到尖锐的最小值，其泛化能力可能不如小批次常找到的更平坦的最小值。即使每个周期所需时间更短，也可能需要更多周期才能达到与小批次相同的表现水平。

在实际操作中，批次大小设为32或64通常是一个好的起始选择。你可以在超参数调整过程中尝试不同的尺寸。

### Keras中的设定

当你调用Keras的`fit()`方法时，你会指定这些参数 (parameter)：

```python
# 假设 'model' 已编译，并且 'x_train'、'y_train' 是 NumPy 数组
history = model.fit(
    x_train,
    y_train,
    batch_size=32,  # 每次梯度更新的样本数量
    epochs=10,      # 遍历整个数据集的次数
    validation_data=(x_val, y_val) # 用于在每个周期结束时评估损失和指标的数据
)
```

此处，模型将训练10个周期。在每个周期中，它将以32个样本为一批次处理`x_train`数据，并在每个批次处理后更新权重 (weight)。每个周期的迭代次数将是`len(x_train) / 32`。

理解批次和周期对于控制训练过程非常重要。它们决定了模型如何随时间从数据中学习，影响着训练的稳定性、速度、内存使用，并最终影响模型的泛化表现。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  涵盖了深度神经网络训练的理论和实践，包括批次、周期和各种优化算法。
- [\`model.fit()\`: Train a model with given data](https://keras.io/api/models/model_training_apis/#fit-method) — Keras team (2024)
  Publisher: Keras
  Keras `fit` 方法的官方文档，详细说明了 `batch_size` 和 `epochs` 等参数及其用法。
- [Neural Networks Part 3: Learning and Evaluation](http://cs231n.github.io/neural-networks-3/) — Stanford University CS231n: Convolutional Neural Networks for Visual Recognition (2024)
  斯坦福大学著名深度学习课程的一部分，深入解释了小批量梯度下降、周期及其实际影响。
- [On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima](https://arxiv.org/abs/1609.04836) — Nitish Shirish Keskar, Dheevatsa Mudigere, Jorge Nocedal, Mikhail Smelyanskiy, Ping Tak Peter Tang (2017)
  Journal: ICLR 2017; DOI: [10.48550/arXiv.1609.04836](https://doi.org/10.48550/arXiv.1609.04836)
  探讨了批次大小对深度神经网络泛化性能的影响，讨论了“泛化差距”以及对平坦与尖锐最小值的偏好。
