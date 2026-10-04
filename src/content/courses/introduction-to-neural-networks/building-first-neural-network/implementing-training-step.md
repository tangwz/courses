---
course: "introduction-to-neural-networks"
chapter: "building-first-neural-network"
lesson: "implementing-training-step"
sourceId: 1988
sourceUrl: "https://apxml.com/zh/courses/introduction-to-neural-networks/chapter-5-building-first-neural-network/implementing-training-step"
title: "实现训练步骤"
description: "编写正向传播、损失计算、反向传播和参数更新的代码。"
order: 4
plots: []
sourceHash: "f812dea192db811a3dc5e8574b895ac5d56c34f113ae20fbee99a424871326a5"
sourceCorrections: []
---

现在我们已经定义了网络的结构并初始化了其参数 (parameter)，接下来是学习过程的核心部分：训练步骤。此步骤会在更大的训练循环中对每个数据批次重复执行。其目的是根据当前批次产生的误差，微调 (fine-tuning)网络的权重 (weight)和偏置 (bias)，逐步引导网络做出更好的预测。

实现单一训练步骤需要一系列精确的操作，这些操作结合了前几章的观念：正向传播、损失计算、反向传播 (backpropagation)以及通过梯度下降 (gradient descent)进行的参数更新。下面我们来分解这个序列。

### 训练步骤中的操作序列

假设你有一个训练示例的小批次（$X_{batch}$）及其对应的真实标签（$Y_{batch}$）。单一训练步骤按如下方式处理此批次：

1. **正向传播：** 输入数据（$X_{batch}$）被送入网络。它通过每个层，经历线性变换（加权和加上偏置 (bias)）后跟着激活函数 (activation function)。这会生成网络对输入批次的预测（$\hat{Y}_{batch}$）。这是第三章中详细说明的过程。

   - *输入：* 数据批次 $X_{batch}$，当前权重 (weight) $W$，当前偏置 $b$。
   - *处理：* 依次计算层输出，$A^{[l]} = g^{[l]}(Z^{[l]}) = g^{[l]}(W^{[l]}A^{[l-1]} + b^{[l]})$，其中 $A^{[0]} = X_{batch}$。
   - *输出：* 网络预测 $\hat{Y}_{batch} = A^{[L]}$（最终层 $L$ 的输出）。
2. **损失计算：** 将生成的预测（$\hat{Y}_{batch}$）与实际目标值（$Y_{batch}$）使用选定的损失函数 (loss function)（例如，回归任务的均方误差，分类任务的交叉熵损失）进行比较。此函数量化 (quantization)了网络对该特定批次的预测“错误”程度。

   - *输入：* 预测 $\hat{Y}_{batch}$，真实标签 $Y_{batch}$。
   - *处理：* 应用损失函数，$L = \text{LossFunction}(\hat{Y}_{batch}, Y_{batch})$。
   - *输出：* 一个标量值，表示批次上的平均损失，$L$。
3. **反向传播 (backpropagation)：** 在此阶段，网络从其错误中学习。从计算出的损失 $L$ 开始，我们计算损失相对于网络中*每个*权重和偏置的梯度（导数）。反向传播使用微积分的链式法则来高效计算这些梯度，从输出层开始，反向通过隐藏层直到输入层。

   - *输入：* 损失值 $L$，来自正向传播的缓存值（如激活值 $A^{[l]}$ 和中间值 $Z^{[l]}$）。
   - *处理：* 计算每个层 $l$ 的梯度 $\frac{\partial L}{\partial W^{[l]}}$ 和 $\frac{\partial L}{\partial b^{[l]}}$，从 $l=L$ 开始向下到 $l=1$。
   - *输出：* 所有参数 (parameter)的梯度（$\nabla_W L$，$\nabla_b L$）。
4. **参数更新（梯度下降 (gradient descent)）：** 有了梯度，我们调整网络的权重和偏置。梯度告诉我们损失函数最陡峭上升的方向。为了最小化损失，我们朝着其各自梯度*相反*的方向移动参数。此步骤的大小由学习率（$\alpha$）控制。

   - *输入：* 当前权重 $W$，当前偏置 $b$，计算出的梯度 $\frac{\partial L}{\partial W}$，$\frac{\partial L}{\partial b}$，学习率 $\alpha$。
   - *处理：* 对每个层中的每个参数应用梯度下降更新规则：
     $W^{[l]} = W^{[l]} - \alpha \frac{\partial L}{\partial W^{[l]}}$
     $b^{[l]} = b^{[l]} - \alpha \frac{\partial L}{\partial b^{[l]}}$
   - *输出：* 更新后的权重 $W_{new}$ 和偏置 $b_{new}$。这些更新后的参数将用于*下一个*训练步骤的正向传播。

这个四阶段循环构成了训练期间的基本工作单元。

> 单一训练步骤的循环：处理输入、计算损失、计算梯度和更新网络参数。

### 实现

虽然深度学习 (deep learning)框架处理反向传播 (backpropagation)的自动微分并提供优化的梯度下降 (gradient descent)算法，但了解其逻辑流程非常重要。你可以设想实现一个类似 `perform_training_step` 的函数：

```python
# 类似于Python的伪代码
def perform_training_step(X_batch, Y_batch, network_parameters, learning_rate):
    # network_parameters 包含所有层的当前 W 和 b

    # 1. 正向传播
    predictions, forward_cache = forward_propagate(X_batch, network_parameters)

    # 2. 损失计算
    loss = calculate_loss(predictions, Y_batch)

    # 3. 反向传播
    gradients = backward_propagate(loss, forward_cache, network_parameters)
    # gradients 包含所有层的 dL/dW, dL/db

    # 4. 参数更新
    updated_parameters = update_parameters(network_parameters, gradients, learning_rate)

    return updated_parameters, loss

# --- 辅助函数定义 ---
# def forward_propagate(X, params): ... 返回预测值, 缓存
# def calculate_loss(Y_hat, Y): ... 返回标量损失
# def backward_propagate(loss, cache, params): ... 返回梯度
# def update_parameters(params, grads, alpha): ... 返回更新后的参数
```

在此伪代码中：

- `forward_cache` 将存储反向传播所需的中间值（如激活值 $A^{[l]}$ 和预激活值 $Z^{[l]}$）。
- `gradients` 将是梯度矩阵/向量 (vector)的集合，每个权重 (weight)矩阵和偏置 (bias)向量对应一个。
- `update_parameters` 应用前面所示的简单梯度下降规则，尽管在实际应用中通常使用更高级的优化器（如第四章中讨论的 Adam）。

每次在训练循环中调用此 `perform_training_step` 函数（通常使用新的数据批次）时，网络的参数 (parameter)都会被稍微推向使损失函数 (loss function)最小化的值。通过在多个批次和多个训练周期中重复此过程，网络能够从训练数据中学习到复杂的模式。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  这本基础教科书全面涵盖了神经网络架构、前向传播和反向传播、损失函数以及各种梯度下降优化技术。
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) — Michael Nielsen (2015)
  Publisher: Determination Press; Pages: Chapter 1-2
  这本在线书籍提供了神经网络基础知识的易于理解、循序渐进的解释，包括前向传播、损失计算、反向传播和梯度下降，从基本原理构建概念。
- [Derivatives, Backpropagation, and Vectorization](https://cs231n.github.io/optimization-2/) — Andrej Karpathy, Justin Johnson, and Fei-Fei Li (2023)
  Publisher: Stanford University
  这些斯坦福大学的讲义提供了关于反向传播、其在神经网络训练中计算梯度的作用以及整体优化过程的清晰、简洁且有数学依据的解释。
