---
course: "introduction-to-deep-learning"
chapter: "training-loss-optimization"
lesson: "gradient-descent-algorithm"
sourceId: 5038
sourceUrl: "https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-3-training-loss-optimization/gradient-descent-algorithm"
title: "梯度下降算法"
description: "说明梯度下降优化算法的基本机制。"
order: 5
plots: ["plots/5038-0.json"]
sourceHash: "526b3671b42ed8450803b455fc79d389dc816a4e961d01e4df5630c6195aacb7"
sourceCorrections: []
---

损失函数 (loss function)，例如均方误差（$MSE$）或交叉熵，用于量化 (quantization)神经网络 (neural network)预测与实际值之间的差异。训练的初衷目标是使这个误差尽可能小。为了实现这一点，必须调整网络的权重 (weight)和偏置 (bias)以减少损失。这个调整过程由优化算法处理，而其中最基础的是**梯度下降 (gradient descent)**。

想象你正站在一个雾蒙蒙的山脉上，你的目标是到达最低的山谷（即最小损失）。你只能看到脚下附近的地面。你会怎么做？一个明智的做法是，感受你所在位置的地面坡度，然后朝着最陡峭的下坡方向迈一步。你重复这个过程，一步步地下坡，希望能最终到达谷底。

梯度下降的工作方式非常相似。“山脉”是损失函数的表面，由网络的参数 (parameter)（权重 $w$ 和偏置 $b$，统称为 $\theta$）决定。任何一点的“海拔”是给定参数集 $\theta$ 下的损失函数值。我们的目标是找到对应于该表面最低点的参数 $\theta$，即最小损失 $J(\theta)$。

微积分提供了一个工具来寻找“最陡峭的方向”：**梯度**。损失函数相对于参数的梯度，记作 $\nabla J(\theta)$，是一个向量 (vector)，它指向损失表面上*最陡峭的上升*方向。因为我们想*下坡*来最小化损失，所以我们朝着与梯度*相反*的方向（负梯度，$-\nabla J(\theta)$）迈步。

梯度下降的主要思想是通过在负梯度方向上迈小步来迭代更新参数。单个参数（或整个参数集 $\theta$）的更新规则如下：


$$
\theta_{new} = \theta_{old} - \alpha \nabla J(\theta_{old})
$$


让我们来分解一下：

- $\theta_{old}$: 更新前参数（权重和偏置）的当前值。
- $\theta_{new}$: 迈一步后参数的更新值。
- $\nabla J(\theta_{old})$: 使用当前参数 $\theta_{old}$ 计算的损失函数的梯度。这告诉我们损失最陡峭增加的方向。
- $\alpha$: 这是**学习率**。它是一个小的正数（例如，0.01，0.001），控制着我们在负梯度方向上迈出的步长。如果步长太大，我们可能会越过最小值。如果步长太小，过程可能需要太长时间。我们将在下一节中进一步讨论学习率。

因此，这个过程包括：

1. 随机初始化参数（$\theta$）或使用特定的初始化策略。
2. 根据当前参数和训练数据计算损失 $J(\theta)$。
3. 计算损失函数相对于每个参数的梯度（$\nabla J(\theta)$）。对于神经网络，这个计算通常涉及一个名为反向传播 (backpropagation)的算法（我们将在第4章中介绍）。
4. 使用更新规则更新每个参数：$\theta_{new} = \theta_{old} - \alpha \nabla J(\theta_{old})$。
5. 重复步骤2-4，进行设定次数的迭代（周期），或者直到损失不再明显减少（收敛）。

让我们在一个简单的一维损失曲线（例如 $J(w) = w^2$）上进行可视化。最小值显然在 $w=0$。梯度是 $\frac{dJ}{dw} = 2w$。



![简单损失曲线上的梯度下降 (J = w^2)](plots/5038-0.json)



> 一个简单的二次损失函数 $J(w) = w^2$。从 $w=4$ 开始，梯度下降迭代地沿着与负梯度（$-2w$）成比例的方向迈步，步长由学习率（$\alpha=0.2$）决定，趋向于 $w=0$ 处的最小值。

在实践中，对于神经网络，损失表面要复杂得多，并且是高维的（每个权重和偏置都有一个维度）。计算梯度 $\nabla J(\theta)$ 涉及计算损失函数相对于网络中*每一个*权重和偏置的偏导数。虽然思路简单，但实际计算需要链式法则和反向传播算法。

这里描述的基本版本（通常称为**批量梯度下降**）存在一个潜在问题，即计算 $\nabla J(\theta)$ 需要在每一步中评估*整个*训练数据集的损失及其梯度。对于大型数据集，这会非常缓慢且计算成本高昂。这促成了随机梯度下降（SGD）和迷你批量梯度下降等变体，我们即将讨论它们。

目前，重要的点是梯度下降提供了学习的机制。通过重复计算损失表面上最陡峭的下降方向（负梯度）并沿着该方向迈步，它迭代地调整网络参数，以最小化预测误差。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 8: Optimization for Training Deep Models
  一本全面的教材，涵盖深度学习的理论基础，包括对各种优化算法的详细说明。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media; Pages: Chapter 4: Training Models, Chapter 11: Training Deep Neural Networks
  一本实践指南，演示如何实现机器学习和深度学习模型，其中包含对梯度下降的清晰说明。
- [CS231n Convolutional Neural Networks for Visual Recognition, Lecture Notes: Optimization](https://cs231n.github.io/optimization-1/) — Justin Johnson, Andrej Karpathy, and Fei-Fei Li (2023)
  某知名大学课程的综合讲义，详细解释了包括梯度下降在内的各种优化技术。
