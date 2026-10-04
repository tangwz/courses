---
course: "introduction-to-deep-learning"
chapter: "activation-functions-architecture"
lesson: "choosing-activation-functions"
sourceId: 5030
sourceUrl: "https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-2-activation-functions-architecture/choosing-activation-functions"
title: "选择合适的激活函数"
description: "为隐藏层和输出层选择合适的激活函数的建议。"
order: 6
plots: []
sourceHash: "4c599142139d1a77fd14ea63b2a0c1d6310817e6241a4db0d107683c6c1798c2"
sourceCorrections: []
---

Sigmoid、Tanh、ReLU及其变体等多种激活函数 (activation function)是神经网络 (neural network)中常用的组件。一个实际问题出现了：你应当在何时何地使用哪种？这个选择并非随意，它对网络训练和表现的影响非常大。选择恰当的激活函数，主要取决于层的作用（隐藏层或输出层）以及你要解决的具体问题（例如，分类或回归）。

### 隐藏层激活函数 (activation function)

隐藏层构成神经网络 (neural network)的核心计算部分。它们的主要作用是将输入数据转换为表示形式，以便输出层更容易完成最终任务。这些层使用的激活函数需要有效地引入非线性，从而使网络能够学习复杂的结构。

- **ReLU（修正线性单元）：** ReLU ($f(x) = \max(0, x)$) 已成为许多深度学习 (deep learning)应用中隐藏层的实际标准。

  - **优点：** 它的计算成本低，并且通常比Sigmoid或Tanh能更快地收敛训练。这主要是因为它对正输入不会饱和，有助于减缓困扰早期网络的梯度消失问题。
  - **缺点：** 如果通过ReLU单元的梯度过大，单元可能会“死亡”，导致权重 (weight)更新后该神经元再也无法激活（输出始终为零）。这实际上将该单元从网络中移除了。
- **Leaky ReLU、PReLU、ELU：** 这些变体是为了解决“ReLU死亡”问题而提出的。

  - **Leaky ReLU** 对负输入引入了一个小的非零斜率 ($f(x) = \max(0.01x, x)$)，确保神经元始终提供一些梯度。
  - **参数 (parameter)化ReLU (PReLU)** 在训练期间学习负输入的斜率。
  - **指数线性单元 (ELU)** 对负输入使用指数曲线，有时能带来比Leaky ReLU更好的性能和更快的收敛，尽管其计算成本略高。
  - **建议：** 从ReLU开始使用。如果遇到神经元死亡问题或希望获得可能更好的性能，可以尝试Leaky ReLU、PReLU或ELU。Leaky ReLU因其简单和效果好，通常是一个不错的次选。
- **Tanh（双曲正切）：** Tanh ($f(x) = \tanh(x)$) 的输出值在-1和1之间。与Sigmoid（0到1）的范围相比，其以零为中心的输出有时更有利，因为它有助于将激活值归一化 (normalization)到零附近。

  - **用途：** 尽管目前在标准前馈网络中不如ReLU常见，Tanh仍常用于某些结构，尤其是循环神经网络 (RNN) (RNNs)，我们稍后会提到。它与Sigmoid类似，对大范围正负输入存在饱和和梯度消失问题，因此与ReLU变体相比，它不太适合非常深的网络。
- **Sigmoid：** Sigmoid ($f(x) = \frac{1}{1 + e^{-x}}$) 的输出值在0到1之间。

  - **用途：** 由于其容易饱和并导致梯度消失，Sigmoid现在很少用于深度网络的隐藏层。其非零中心输出也可能比Tanh或ReLU导致训练速度慢。它的主要应用通常在输出层，用于特定任务。

**隐藏层通用建议：** 从ReLU开始使用。如果性能不佳或观察到神经元死亡，可以尝试Leaky ReLU或ELU。Tanh是较少使用的选择，而Sigmoid在现代深度学习中通常避免用于隐藏层。

### 输出层激活函数 (activation function)

输出层激活函数的选择，主要由网络需要进行的预测类型决定。

- **二分类：** （预测两个类别之一，例如：垃圾邮件/非垃圾邮件，猫/狗）

  - **函数：** 使用单个输出神经元，激活函数为**Sigmoid**。
  - **原因：** Sigmoid将输出压缩到 (0, 1) 范围，这可以直接解释为属于正类别的概率。然后使用一个阈值（通常为0.5）来做出最终的类别判断。这与二元交叉熵损失函数 (loss function)很搭配。
- **多分类：** （预测多个互斥类别中的一个，例如：数字识别0-9，多类别对象分类）

  - **函数：** 在输出层使用**Softmax**激活函数。输出层神经元的数量应等于类别数量。
  - **原因：** Softmax接收来自前一层的任意实值分数（logits）向量 (vector)，并将其转换为概率分布，其中每个输出介于0到1之间，并且所有输出之和为1。概率最高的输出表示预测的类别。Softmax通常与分类交叉熵损失函数一起使用。
- **多标签分类：** （预测单个输入可能对应的多个类别，例如：为一个博客文章添加多个相关主题标签）

  - **函数：** 使用多个输出神经元（每个潜在标签一个），并对每个神经元独立应用**Sigmoid**激活函数。
  - **原因：** 每个输出神经元独立预测特定标签是否存在（或不存在）的概率，与其他标签无关。对每个输出应用阈值（例如0.5）来决定是否分配相应的标签。二元交叉熵损失通常独立应用于每个输出神经元。
- **回归：** （预测连续数值，例如：预测房价、温度）

  - **函数：** 通常，单个输出神经元（或预测多个值时为多个神经元）上**不使用激活函数**（或者等同于使用**线性激活函数**，$f(x)=x$）。
  - **原因：** 回归问题需要预测可以自由变化（或在特定连续范围内变化）的值。Sigmoid、Tanh或ReLU等激活函数会限制输出，这通常对回归来说是不利的。线性输出允许网络预测任何实数。
  - **例外：** 如果已知目标值被限制在特定范围（例如，0到1之间的概率，-1到1之间的值），你可以分别使用Sigmoid或Tanh，但线性输出通常是默认的起始选择。与限制输出激活相比，通常更倾向于对目标变量进行缩放（例如，标准化）。

### 总结和提示

这里有一个快速参考：

| 层类型 | 任务 | 推荐激活函数 (activation function) | 说明 |
| --- | --- | --- | --- |
| 隐藏层 | 任何 | ReLU | 从此开始。速度快，简单，有效。 |
| 隐藏层 | 任何 | Leaky ReLU, ELU, PReLU | 如果ReLU死亡是个问题或追求潜在提升时使用。 |
| 隐藏层 | 不太常见 | Tanh | 零中心，有时用于RNN。 |
| 输出层 | 二分类 | Sigmoid (1个神经元) | 输出概率 (0到1)。 |
| 输出层 | 多分类 | Softmax (N个神经元) | 输出N个类别的概率分布。 |
| 输出层 | 多标签分类 | Sigmoid (N个神经元) | 输出N个标签的独立概率。 |
| 输出层 | 回归 | 线性 (无) | 输出无限制的连续值。 |

**重要提示：**

1. **这些是建议，而非严格规定。** 最佳激活函数有时会取决于特定数据集、网络架构和初始化策略。进行尝试通常很有用。
2. **一致性：** 通常，在单个隐藏层内的所有神经元使用相同的激活函数。你可以在不同的隐藏层中使用混合的激活函数，但这不太常见。
3. **与初始化的关联：** 激活函数的选择会影响最佳权重 (weight)初始化策略（在第5章讨论）。例如，He初始化常与ReLU变体搭配使用，而Xavier/Glorot初始化则是为Sigmoid/Tanh考虑的。
4. **框架默认设置：** 深度学习 (deep learning)库通常有默认的激活函数。请注意这些默认设置，但根据你的理解和具体问题，请不要犹豫进行更改。

选择恰当的激活函数是设计良好神经网络 (neural network)的一个基本步骤。通过了解它们的特性以及在隐藏层和输出层的常见应用，你可以做出明智的决定，从而帮助模型获得更好的训练和表现。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本标准教材，详细阐述了激活函数，包括它们的数学特性和在神经网络设计中的实际应用。
- [Deep Sparse Rectifier Networks](http://proceedings.mlr.press/v15/glorot11a/glorot11a.pdf) — Xavier Glorot, Antoine Bordes, and Yoshua Bengio (2011)
  Journal: Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics (AISTATS); Publisher: Machine Learning Research (MLR) Press; Volume: 15; Pages: 315-323
  介绍了整流线性单元（ReLU）作为激活函数，表明它能缓解梯度消失问题并加速训练。
- [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/pdf/1502.01852.pdf) — Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun (2015)
  Journal: Proceedings of the IEEE International Conference on Computer Vision (ICCV); Pages: 1026-1034; DOI: [10.1109/ICCV.2015.123](https://doi.org/10.1109/ICCV.2015.123)
  介绍了参数化整流线性单元（PReLU）和相关的He初始化方案，这对于训练具有ReLU类激活的深度网络很有帮助。
- [Neural Networks Part 1: Setting up the Architecture (Activation Functions section)](https://cs231n.github.io/neural-networks-1/#actfunc) — Andrej Karpathy, Justin Johnson, and Li Fei-Fei (2023)
  对深度学习中常见的激活函数提供了清晰实用的概述，提供了直观的解释和选择建议。
