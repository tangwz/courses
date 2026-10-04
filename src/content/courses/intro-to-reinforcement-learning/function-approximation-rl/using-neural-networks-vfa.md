---
course: "intro-to-reinforcement-learning"
chapter: "function-approximation-rl"
lesson: "using-neural-networks-vfa"
sourceId: 1656
sourceUrl: "https://apxml.com/zh/courses/intro-to-reinforcement-learning/chapter-6-function-approximation-rl/using-neural-networks-vfa"
title: "使用神经网络进行价值函数近似"
description: "介绍神经网络作为强化学习中强大的非线性函数近似器。"
order: 7
plots: []
sourceHash: "e3d957cd342efcaf65d2e5e854b7991c007fab6564fe0f159b64224c3dca4fc3"
sourceCorrections: []
---

线性函数近似，如我们所见，提供了一种利用特征在不同状态间泛化价值估计的方法。然而，现实（以及许多有趣的模拟环境）常常表现出状态与其长期价值之间复杂、非线性的关联。简单的特征线性组合可能不足以表示这些复杂模式。例如，一个状态的价值可能取决于特征之间难以被线性模型表示的精细影响。

当面对复杂且非线性的数据模式时，神经网络 (neural network)（NNs）是强大的函数近似器，以其直接从数据中学习复杂非线性映射的能力而闻名。它们在计算机视觉和自然语言处理等许多应用中取得的成就源于此能力。在强化学习 (reinforcement learning)中，我们可以借助神经网络来近似价值函数，这可能在复杂环境中带来更好的表现。

### 使用神经网络 (neural network)近似价值函数

我们可以使用神经网络，而不是线性函数 $\hat{v}(s, \mathbf{w}) = \mathbf{w}^T \mathbf{x}(s)$。该网络将状态表示作为输入，并输出估计价值。现在，令 $\mathbf{w}$ 表示神经网络内部的全部权重 (weight)和偏差。

- **对于状态价值函数：** 网络接收状态 $s$（或其特征向量 (vector) $\mathbf{x}(s)$）作为输入，并输出一个单一的标量值，表示估计的状态价值 $\hat{v}(s; \mathbf{w})$。
- **对于动作价值函数：** 网络通常将状态 $s$ 作为输入，并输出多个值，每个可能动作 $a$ 对应一个值。该输出表示状态 $s$ 中所有可用动作的估计Q值 $\hat{q}(s, a; \mathbf{w})$。另外，网络也可以同时接收状态 $s$ 和特定动作 $a$ 作为输入，并输出这对组合的单一Q值。前一种方法在DQN等方法中更为常见。

> 一个图表显示神经网络接收状态表示作为输入，并输出多个动作的估计Q值。权重 $\mathbf{w}$ 对网络内的连接进行参数 (parameter)化。

### 使用神经网络 (neural network)的优势

1. **捕捉非线性：** 带有隐藏层和非线性激活函数 (activation function)（如ReLU）的神经网络可以近似任意复杂的函数。这使它们能够建立比线性方法更复杂的价值函数模型。
2. **自动特征表示：** 深度神经网络，特别是处理图像输入（如游戏屏幕）时的卷积神经网络（CNN），可以直接从原始数据中学习分层特征。这减少或消除了手动、领域特定特征工程的必要性，而这可能是应用强化学习 (reinforcement learning)的一个重要瓶颈。网络学会了对价值最具预测性的特征。

### 神经网络 (neural network)的价值函数近似训练

基本目标与线性价值函数近似（VFA）相同：调整参数 (parameter) $\mathbf{w}$（现在是网络的权重 (weight)），以使预测值与目标值之间的差异最小化。我们通常使用时序差分（TD）学习的变体。

例如，在Q学习中使用神经网络时，经验元组 $(S_t, A_t, R_{t+1}, S_{t+1})$ 的目标值通常为：


$$
Y_t = R_{t+1} + \gamma \max_{a'} \hat{q}(S_{t+1}, a', \mathbf{w})
$$


网络预测 $\hat{q}(S_t, A_t, \mathbf{w})$。目标是最小化目标与预测之间的平方误差，这通常被称为TD误差：$\delta_t = Y_t - \hat{q}(S_t, A_t, \mathbf{w})$。

我们使用随机梯度下降 (gradient descent)（SGD）或其变体（如Adam）来更新权重 $\mathbf{w}$。更新旨在使预测更接近目标：


$$
\mathbf{w} \leftarrow \mathbf{w} + \alpha \delta_t \nabla_{\mathbf{w}} \hat{q}(S_t, A_t, \mathbf{w})
$$


这里，$\nabla_{\mathbf{w}} \hat{q}(S_t, A_t, \mathbf{w})$ 是网络输出（对于特定动作 $A_t$）相对于其权重 $\mathbf{w}$ 的梯度。这个梯度是使用反向传播 (backpropagation)算法高效计算的，这是深度学习 (deep learning)中的一种标准方法。幸运的是，现代深度学习库，如TensorFlow或PyTorch，为我们处理自动微分和反向传播。我们只需要定义网络架构和损失函数 (loss function)（通常是基于TD误差的均方误差）。

请注意，这仍然是一种*半梯度*方法，因为目标 $Y_t$ 本身取决于当前的权重 $\mathbf{w}$（除非使用目标网络，这将在后面讨论），并且在计算梯度时，我们不对目标计算进行微分。

### 挑战与展望

尽管功能强大，但将神经网络 (neural network)与TD学习直接结合会在训练期间引入潜在的不稳定性。两个主要问题随之出现：

1. **关联数据：** 强化学习 (reinforcement learning)经验 $(S_t, A_t, R_{t+1}, S_{t+1})$ 是连续且高度关联的。这违反了标准SGD所依赖的独立同分布（IID）样本假设，可能导致学习效率低下或不稳定。
2. **移动目标：** TD目标 $Y_t$ 取决于网络自身的估计 $\hat{q}(S_{t+1}, a', \mathbf{w})$。由于权重 (weight) $\mathbf{w}$ 在每一步都被更新，目标值本身会持续变化。这就像追逐一个移动的目标，可能导致震荡或发散。

此外，神经网络引入了更多超参数 (parameter) (hyperparameter)（网络架构、层大小、学习率、激活函数 (activation function)），这些参数需要仔细选择和调整。

使用神经网络进行价值函数近似标志着向\*\*深度强化学习（DRL）\*\*的转变。下一章“深度Q网络（DQN）简介”将通过介绍经验回放（Experience Replay）和固定Q目标（Fixed Q-Targets）等方法直接解决上述稳定性挑战，这些方法对早期DRL算法的成功起到了重要作用。这些方法使我们能够有效地训练深度神经网络来解决复杂的强化学习任务。

## 参考资料

- [Reinforcement Learning: An Introduction](http://www.incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press
  一本强化学习的基石教材，涵盖强化学习的各个方面，包括价值函数近似及其挑战的广泛讨论。
- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) — Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra and Martin A. Riedmiller (2015)
  Journal: Nature; Volume: 518; Pages: 529-533; DOI: [10.1038/nature14236](https://doi.org/10.1038/nature14236)
  这篇开创性论文介绍了深度Q网络（DQN），展示了如何通过经验回放和目标网络等技术将深度神经网络与Q学习成功结合，以在复杂环境中实现稳定的训练。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本关于深度学习的综合资源，解释了神经网络、反向传播和优化算法的理论基础，这些对于训练价值函数近似器至关重要。
