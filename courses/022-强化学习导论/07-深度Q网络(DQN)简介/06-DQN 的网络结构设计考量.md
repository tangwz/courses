# DQN 的网络结构设计考量

来源：[原文](https://apxml.com/zh/courses/intro-to-reinforcement-learning/chapter-7-introduction-deep-q-networks-dqn/architectural-considerations-dqns)

[返回章节目录](README.md) · [返回课程目录](../README.md)

构建估算动作价值函数 $Q(s, a; \theta)$ 的神经网络 (neural network)是深度Q网络 (DQN) 的主要考量之一。DQN 运用经验回放和目标网络等核心机制。这个网络的设计很大程度上取决于环境提供的状态表示形式。

### 网络输入：状态表示

第一步是确定输入层的结构。智能体“看到”什么？

- **向量 (vector)形式状态：** 如果状态表示为数值特征向量（例如：位置、速度、传感器读数），标准**多层感知机（MLP）**（也称全连接网络）通常足够使用。输入层的神经元数量将与状态向量中的特征数量相等。
- **图像形式状态：** 对于状态由图像（例如屏幕像素）表示的环境，\*\*卷积神经网络 (neural network)（CNN）\*\*是标准选择。CNN擅长识别网格状数据中的空间层次和模式。输入层通常会接受图像的维度（高度、宽度、颜色通道）。通常会将多个连续帧堆叠作为输入，为网络提供时间信息，例如速度或移动方向。

### 网络结构：隐藏层

在输入层之后，一个或多个隐藏层会处理信息。

- **MLP：** MLP中的隐藏层通常是全连接层。一层中的每个神经元都从前一层的所有神经元接收输入。层的数量以及每层神经元的数量是您需要设定的超参数 (parameter) (hyperparameter)。先从简单的设置开始（例如，一或两个隐藏层，中等数量的神经元），如果需要可以增加复杂度。修正线性单元 (ReLU) 激活函数 (activation function)（$f(x) = max(0, x)$）是DQN中隐藏层的普遍且有效的选择。
- **CNN：** 对于图像输入，最初的隐藏层通常是卷积层，后接池化层。卷积层应用滤波器以识别局部模式（边缘、纹理），而池化层则降低空间维度，使表示更易处理，并且对细微平移具有不变性。在经过几个卷积和池化层之后，得到的特征图通常被展平为一个向量 (vector)，并输入到一个或多个全连接层，类似于MLP。同样，ReLU是这些层的标准激活函数。那个让DQN在Atari游戏中取得成功的著名论文，就采用了由多个卷积层和全连接层构成的CNN架构。

### 网络输出：动作价值

网络的最后一层非常重要：它必须输出估算的Q值。

- **结构：** 输出层通常是全连接层，环境中智能体可以采取的每个可能的离散动作对应一个神经元。
- **激活：** 由于Q值表示预期累积奖励，并且不限于特定范围（如概率），因此输出层通常使用**线性激活函数 (activation function)**（即无激活函数或恒等函数）。
- **解读：** 第$i$个输出神经元产生的值代表网络对在输入状态$s$下执行第$i$个动作的Q值的估算。也就是说，输出向量 (vector)同时估算所有可能的动作$a$对应的$Q(s, a; \theta)$。

### 示例架构

以下是展示常见结构的图示：

**1. 适用于向量 (vector)形式状态的MLP：**

> MLP接收状态向量 $s = (s_1, ..., s_n)$，并通过全连接（FC）隐藏层（通常带有ReLU激活）为每个动作 $a_1, ..., a_m$ 生成Q值估算。

**2. 适用于图像形式状态的CNN（简化版）：**

> CNN通过卷积层和池化层处理输入图像，以提取空间特征。这些特征随后被展平并通过全连接层，为每个动作输出Q值。

### 一般建议

- **从简入手：** 从相对较浅的网络（层数较少）以及较少的神经元/滤波器开始。如果性能不足，可以稍后增加复杂度。过于复杂的网络训练起来会更困难、更慢。
- **匹配问题复杂度：** 简单的环境（如CartPole）可能只需要一个小型MLP，而带有视觉输入的复杂环境（如Atari游戏）则需要更深层的CNN。
- **采用标准方法：** 隐藏层使用ReLU等常用激活函数 (activation function)，输出层使用线性激活函数。可以考虑将研究论文中已有的架构（例如最初的DQN Atari架构）作为起点。
- **迭代调整：** 网络设计常包含实验过程。训练您的智能体，评估其性能，并根据结果调整架构（层数、神经元数量、滤波器大小等）。

选择合适的架构是让DQN取得良好效果的一个环节。通过考虑状态空间的特性并借助常见的网络设计模式，您可以为您的强化学习 (reinforcement learning)智能体构建有效的函数估算器。

## 参考资料

- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) — Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller (2015)
  Journal: Nature; Volume: 518; Pages: 529-533; DOI: [10.1038/nature14236](https://doi.org/10.1038/nature14236)
  这篇基础性论文介绍了深度Q网络（DQN）架构，详细阐述了其用于图像输入卷积神经网络设计，并展示了其在Atari游戏上达到人类水平的表现。
- [Reinforcement Learning: An Introduction](https://www.mitpress.mit.edu/books/reinforcement-learning-second-edition) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press
  这本标准教材涵盖了强化学习，包括使用神经网络进行函数逼近，以及Q学习和深度Q网络的全面解释。(第二版)
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  这本被广泛引用的教材涵盖了深度学习的理论和实践方面，包括多层感知器（MLP）、卷积神经网络（CNN）、激活函数以及整体神经网络设计。
- [Convolutional Neural Networks for Visual Recognition (CS231n)](https://cs231n.github.io/) — Fei-Fei Li, Justin Johnson, and Serena Yeung (2024)
  Publisher: Stanford University
  提供了卷积神经网络（CNN）及其架构组件和设计模式的深入解释，这对于DQN中处理基于图像的状态至关重要。(在线课程笔记)

---

[上一节](05-DQN%20%E7%AE%97%E6%B3%95%E7%BB%93%E6%9E%84.md) · [下一节](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9E%84%E5%BB%BA%E4%B8%80%E4%B8%AA%E5%9F%BA%E7%A1%80DQN.md)
