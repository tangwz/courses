---
course: "introduction-to-neural-networks"
chapter: "backpropagation-gradient-descent"
lesson: "learning-rate"
sourceId: 1974
sourceUrl: "https://apxml.com/zh/courses/introduction-to-neural-networks/chapter-4-backpropagation-gradient-descent/learning-rate"
title: "学习率及其作用"
description: "理解学习率的作用及其对收敛的影响。"
order: 5
plots: ["plots/1974-0.json"]
sourceHash: "2cbe70b76f41cfece212ceba155185bee0c81e99c2865bf1b717be51886a242c"
sourceCorrections: []
---

在之前的章节中，我们已经说明了梯度下降 (gradient descent)如何使用计算出的梯度（损失函数 (loss function)最陡峭上升的方向）来指导网络参数 (parameter)的调整。其主要思想体现在参数更新规则中：

$W_{new} = W_{old} - \eta \frac{\partial Loss}{\partial W_{old}}$
$b_{new} = b_{old} - \eta \frac{\partial Loss}{\partial b_{old}}$

这里，$W$ 表示权重 (weight)，$b$ 表示偏置 (bias)，并且 $\frac{\partial Loss}{\partial \text{parameter}}$ 是损失函数对该参数的梯度。但是符号 $\eta$（eta）具体有什么作用呢？这就是**学习率**，一个小的正数值，用作梯度的缩放因子。

可以把训练过程想象成试图找到山谷的底部（即最小损失）。梯度告诉你哪个方向是下坡。学习率 $\eta$ 决定了你在每次迭代中沿着下坡方向迈出多大一步。它是一个超参数 (hyperparameter)，也就是说，它是你在训练过程开始*之前*选择的配置设置，而不是在训练过程中学习到的参数。

### 平衡之道：步长很重要

选择合适的学习率对于模型训练的成功很重要。$\eta$ 的值直接影响收敛过程的速度和最终的成功。

- **学习率过高：** 如果 $\eta$ 过大，梯度下降 (gradient descent)过程中迈出的步子可能过大。想象一下在山谷中大步跳跃；你可能会完全越过底部，到达另一边，甚至可能比你开始的位置更高。这可能导致损失函数 (loss function)剧烈波动，无法持续下降，甚至发散（无限增加）。优化过程变得不稳定。
- **学习率过低：** 相反，如果 $\eta$ 过小，迈出的步子就会很小。虽然这保证了你谨慎地下坡，但进展可能很慢。可能需要不切实际的长时间才能达到最小损失值。此外，非常小的步子可能会使优化器更容易陷入浅层局部最小值，而无法在损失区域的其他地方找到更好、更深的最小值。

目标是找到一个“刚刚好”的学习率——足够大以在合理的时间内向最小值取得合理进展，但又足够小以避免越过目标和不稳定。

### 学习率影响的可视化

思考在不同学习率下，损失可能如何随训练迭代次数变化：



![学习率对损失收敛的影响](plots/1974-0.json)



> 损失曲线图说明了不同学习率的情况。高学习率会导致振荡和潜在的发散。低学习率会导致非常缓慢的改进。合适的学习率则显示出稳定的收敛。

### 寻找合适的学习率

不存在单一的“最佳”学习率；最优值很大程度上取决于特定的数据集、网络结构和损失函数 (loss function)。通常的起始值范围从 $0.1$ 到 $0.0001$。找到一个好的值通常需要通过实验：

1. **从常见值开始：** 尝试使用 $0.01$ 或 $0.001$ 等默认值。
2. **监控损失：** 观察训练损失曲线。
   - 如果损失发散或剧烈振荡，学习率可能过高。降低它（例如，降低10倍）。
   - 如果损失下降非常缓慢，学习率可能过低。尝试谨慎地提高它（例如，提高3倍或10倍）。
   - 如果损失稳定下降，你可能处于一个好的范围。
3. **迭代：** 重复此过程，直到找到一个能提供稳定且合理快速收敛的学习率。

### 训练期间调整学习率

尽管使用固定的学习率是最简单的方法，但更高级的技术是在训练过程中调整 $\eta$。一个常用的策略是**学习率衰减**（或称调度），即你从一个相对较高的学习率开始，以实现快速的初始进展，然后随着时间逐步降低它。这使得优化器在接近最小值时能够迈出更小、更精细的步子，从而可能获得更好的最终结果。我们在这里不会详细介绍具体的调度算法，但请记住自适应学习率是现代深度学习 (deep learning)中的标准做法。

总之，学习率 $\eta$ 是梯度下降 (gradient descent)中一个虽小但有影响的超参数 (parameter) (hyperparameter)。它决定了参数更新期间的步长。选择合适的值，通常通过实验，对于实现稳定收敛和训练有效的神经网络 (neural network)是必要的。不正确的选择可能导致训练缓慢或完全无法收敛。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  这本深度学习基础教材全面介绍了优化算法，包括学习率在梯度下降中的重要作用。
- [CS231n: Convolutional Neural Networks for Visual Recognition - Optimization](https://cs231n.github.io/optimization-2/) — Stanford University (2023)
  Publisher: Stanford University
  提供了优化算法的优秀且易懂的解释，包括对学习率影响及其在神经网络训练中调整策略的详细讨论和可视化。
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — Diederik P. Kingma and Jimmy Ba (2015)
  Journal: 3rd International Conference for Learning Representations; DOI: [10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980)
  介绍了Adam，一种广泛使用的自适应学习率优化器。本文阐述了设定固定学习率的难点，并提出了自适应方法的必要性，这些内容在章节中有所提及。
