---
course: "introduction-to-deep-learning"
chapter: "training-loss-optimization"
lesson: "learning-rate"
sourceId: 5039
sourceUrl: "https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-3-training-loss-optimization/learning-rate"
title: "学习率"
description: "理解学习率超参数的重要性及其对收敛的影响。"
order: 6
plots: ["plots/5039-0.json"]
sourceHash: "11663fc9e4e53f7d6dac1515d799ba34eae489778d2e2f9d77c5d9e6c1293e10"
sourceCorrections: []
---

在上一节中，我们介绍了梯度下降 (gradient descent)作为驱动神经网络 (neural network)学习过程的动力。它通过迭代调整模型的权重 (weight)和偏差来使损失函数 (loss function)最小化。但是，每一步我们应该调整权重多少呢？这由一个重要的超参数 (parameter) (hyperparameter)控制：**学习率**。

可以把梯度下降想象成在山谷中寻找最低点（即损失函数的最小值）。梯度会指明最陡峭的上升方向，因此你需要朝相反的方向下坡。学习率则决定了你下坡时每一步的大小。

从数学角度看，学习率通常用希腊字母 alpha ($\alpha$) 或 eta ($\eta$) 表示，是一个小的正标量值。它在从当前权重中减去梯度之前，先将梯度相乘。单个权重 $w$ 的更新规则如下所示：


$$
w_{new} = w_{current} - \alpha \frac{\partial L}{\partial w_{current}}
$$


这里，$\frac{\partial L}{\partial w_{current}}$ 是损失函数 $L$ 对当前权重 $w_{current}$ 的梯度。学习率 $\alpha$ 对这个梯度进行缩放。

### 学习率大小的影响

选择合适的学习率对模型训练的成功很重要。$\alpha$ 的值直接影响收敛速度以及算法能否收敛。

- **学习率过小：** 如果 $\alpha$ 非常小，权重 (weight)更新将非常微小。算法将以极小的步长下坡。虽然这最终可能会导致一个不错的最小值，但训练过程将极其缓慢，可能需要大量的训练周期（对训练数据的遍历）才能达到收敛。在复杂的损失函数 (loss function)曲面中，它还可能增加陷入较差局部最小值的风险，因为步长不够大，无法跳出小的凹陷。
- **学习率过大：** 如果 $\alpha$ 过大，所走的步长可能过大。算法可能完全越过最小值，而不是平稳地下降到最小值。它可能在山谷中来回震荡，无法稳定在最小值附近。在更糟的情况下，步长可能大到每次更新后损失反而 *增加*，导致算法完全发散。权重可能会变得非常大（NaN 或无穷大），训练过程失败。

### 学习率影响的可视化

考虑一个简单的抛物线损失函数 (loss function)。下面的图表展示了不同的学习率如何影响梯度下降 (gradient descent)走向最小值的路径。



![学习率对梯度下降路径的影响](plots/5039-0.json)



> 梯度下降在一个简单损失函数 $L = w^2$ 上从 $w=5$ 开始的示例路径。合适的学习率会稳定收敛。小的学习率进展缓慢。大的学习率会越过并震荡，可能导致发散。

### 寻找合适的平衡

选择一个合适的学习率最初往往更像是一门艺术而非科学，通常需要通过实验确定。常见的起始值通常在 0.1 到 0.0001 的范围内，但最优值很大程度上取决于：

- 具体的数据集。
- 模型的架构。
- 所选的优化器（一些优化器，稍后会讨论，会自动调整学习率）。
- SGD 中使用的批次大小。

像学习率调度（在训练期间逐步减小学习率）和自适应学习率方法（如 Adam 或 RMSprop，将在第 4 章中介绍）等技术已被开发出来，旨在更有效地管理这个参数 (parameter)。然而，在使用这些更高级的技术之前，理解基础学习率的基本作用很重要。在模型开发过程中，你通常需要调整这个超参数 (hyperparameter)，并监控训练损失，以查看其是否以合理的速度稳定下降。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  对神经网络优化提供了全面的学术论述，包含学习率在梯度下降中的作用和影响。
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) — Michael A. Nielsen (2015)
  Publisher: Determination Press
  一本易于理解的在线书籍，解释了神经网络基础，包含梯度下降和学习率重要性的清晰说明。
- [Optimization: Stochastic Gradient Descent](https://cs231n.github.io/optimization-1/) — Andrej Karpathy, Justin Johnson, and Li Fei-Fei (2023)
  Journal: Stanford CS231n Course Notes
  作为一门备受推崇的大学课程的一部分，这些笔记提供了像SGD这样的优化算法的实用直观解释，侧重于学习率。
