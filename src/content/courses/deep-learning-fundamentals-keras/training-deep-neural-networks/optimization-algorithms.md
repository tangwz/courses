---
course: "deep-learning-fundamentals-keras"
chapter: "training-deep-neural-networks"
lesson: "optimization-algorithms"
sourceId: 4912
sourceUrl: "https://apxml.com/zh/courses/deep-learning-fundamentals-keras/chapter-3-training-deep-neural-networks/optimization-algorithms"
title: "优化算法"
description: "梯度下降及Adam、RMSprop、SGD等常见优化算法概述。"
order: 3
plots: ["plots/4912-0.json"]
sourceHash: "a8e774ed93cab5be46653d25ad1ee1cd55c967ce26740227b480fa7037a744b5"
sourceCorrections: []
---

一旦你选择了一个损失函数 (loss function)来衡量模型预测的偏差程度，接下来的问题是：你如何实际地*减小*这个损失？这时就需要用到优化算法。它们的工作是系统地调整模型的参数 (parameter)（权重 (weight)和偏置 (bias)），根据损失信号逐步提升模型的性能。

可以将训练过程想象成在寻找一个复杂的高维空间 (high-dimensional space)中的最低点，其中任何一点的“高度”代表着一组模型权重对应的损失值。优化算法就是指引你在此空间中移动的工具。

### 核心思路：梯度下降 (gradient descent)

最基本的优化技术是**梯度下降（Gradient Descent, GD）**。深度学习 (deep learning)模型在训练时需要指定一个优化器。这个优化器负责根据反向传播 (backpropagation)过程中计算得到的梯度来调整模型参数 (parameter)（梯度相关内容将在后续讨论）。

损失函数 (loss function)对模型参数的梯度告诉我们最陡峭的上升方向。为了使损失最小化，我们需要朝着相反的方向移动，也就是最陡峭的下降方向。

梯度下降的工作原理是：计算整个训练数据集的损失函数梯度，然后向下迈一步：


$$
\text{新权重} = \text{旧权重} - \text{学习率} \times \text{梯度}
$$


**学习率**（$\alpha$）是一个小的标量值（例如，0.01，0.001），它控制我们迈步的大小。学习率过小会导致收敛缓慢，而过大会导致优化过程越过最小值点，甚至发散。

虽然简单，但对于*整个*数据集（批量梯度下降，Batch Gradient Descent）计算梯度可能会计算成本非常高，特别是对于大型数据集。这使我们转向了更实用的变体。

### 随机梯度下降 (gradient descent)（SGD）和小批量梯度下降

\*\*随机梯度下降（Stochastic Gradient Descent, SGD）\*\*并非每次权重 (weight)更新都使用整个数据集，而是每次只从一个随机选择的训练样本计算梯度并更新权重。这使得每次更新快得多，但也嘈杂得多，因为单个样本的梯度可能不能代表整体损失。

一种常见且非常有效的折衷方案是**小批量梯度下降（Mini-Batch Gradient Descent）**。在这种方法中，梯度是基于训练数据的一小部分随机选择的子集（称为*小批量*）计算并更新权重的。典型的批次大小范围从32到256个样本。这种方法平衡了SGD的计算效率和批量梯度下降(Batch GD)的更稳定收敛性。在深度学习 (deep learning)实践中，“SGD”几乎总是指小批量梯度下降。



![优化器路径](plots/4912-0.json)



> 简化的二维视图，显示了不同梯度下降变体如何在损失曲面上趋近最小值 (0,0)。批量梯度下降采用直接路径，SGD有噪声，而小批量则提供了平衡。

**动量（Momentum）：** 对SGD的一个受欢迎的改进是**动量**。它有助于在相关方向上加速SGD并抑制震荡。它通过将前一个更新向量 (vector)的一部分添加到当前更新向量中来做到这一点，从而在持续梯度的方向上积累速度。

在Keras中，你可以这样使用带动量的SGD：

```python
import keras

optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
# 模型编译(optimizer=optimizer, ...)
```

### 自适应优化算法

虽然SGD（带动量）是一个可靠的算法，但调整其学习率有时会很有难度。自适应算法在训练期间自动调整学习率，通常需要较少的手动调整。

#### Adam (Adaptive Moment Estimation)

Adam可以说是目前深度学习 (deep learning)中受欢迎的优化算法。它通常是一个不错的默认选择，并在各种问题上表现良好。Adam为每个参数 (parameter)计算自适应学习率。它结合了两个主要思路：

1. **动量：** 它保持过去梯度的指数衰减平均值（类似于动量）。
2. **RMSprop：** 它保持过去*平方*梯度的指数衰减平均值。

它使用这些平均值来调整每个参数的学习率。接收到大梯度或频繁梯度的参数，其有效学习率会降低，而梯度小或不频繁的参数，其有效学习率会增加。

在Keras中使用Adam非常简单：

```python
import keras

optimizer = keras.optimizers.Adam(learning_rate=0.001) # 默认学习率通常是 0.001
# 模型编译(optimizer=optimizer, ...)
```

#### RMSprop (Root Mean Square Propagation)

RMSprop是另一种自适应学习率算法，它也通过将学习率除以平方梯度的指数衰减平均值来工作。它与Adam大约同时期开发，并有相似之处。它通常表现良好，尤其是在循环神经网络 (neural network) (RNN)上。

```python
import keras

optimizer = keras.optimizers.RMSprop(learning_rate=0.001)
# 模型编译(optimizer=optimizer, ...)
```

### 选择优化器

那么，你应该选择哪种优化器呢？

- **Adam** 通常是一个很好的起点。它的自适应特性通常会带来好的结果，且相对较少的调整（尽管学习率可能仍需要微调 (fine-tuning)）。
- **带动量的SGD** 有时能找到更好、泛化能力更强的解决方案，但它通常需要更仔细地调整学习率和动量参数 (parameter)。如果Adam没有获得满意结果或者你怀疑模型过拟合 (overfitting)，值得尝试。
- **RMSprop** 是另一个有力的选择，特别是当Adam表现不佳时。

通常需要进行实验。优化器的性能很大程度上取决于具体的问题、数据集和模型架构。请记住，例如学习率调度（在训练过程中调整学习率，通常通过稍后讨论的回调函数管理）等技术会显著影响*任何*优化器的性能。

在`compile`步骤中，你只需将你选择的优化器类的实例传递给`optimizer`参数即可：

```python
# 使用Adam的例子
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 使用SGD的例子
# model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
```

有了衡量模型偏差程度的损失函数 (loss function)，以及知晓如何调整权重 (weight)以减小该误差的优化器，现在我们需要了解计算必要调整的机制：反向传播 (backpropagation)。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  广泛介绍优化算法，涵盖梯度下降变体、动量以及Adam和RMSprop等自适应方法。
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — Diederik P. Kingma, Jimmy Ba (2015)
  Journal: International Conference for Learning Representations; DOI: [10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980)
  介绍Adam优化算法的原始论文，阐述其公式和实验优势。
- [Keras Optimizers](https://keras.io/api/optimizers/) — Keras team (2024)
  Keras优化算法的官方API文档，包含实际使用示例和参数配置。
- [Optimizers and Learning Rate Scheduling](https://cs231n.github.io/neural-networks-3/#update) — Stanford University CS231n team (2023)
  详细说明各种优化算法，包括RMSprop、动量和不同的学习率策略。
