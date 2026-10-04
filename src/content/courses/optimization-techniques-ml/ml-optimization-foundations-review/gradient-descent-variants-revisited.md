---
course: "optimization-techniques-ml"
chapter: "ml-optimization-foundations-review"
lesson: "gradient-descent-variants-revisited"
sourceId: 1225
sourceUrl: "https://apxml.com/zh/courses/optimization-techniques-ml/chapter-1-ml-optimization-foundations-review/gradient-descent-variants-revisited"
title: "重温梯度下降变体"
description: "回顾SGD、动量和Nesterov加速梯度，作为高级方法的起点。"
order: 1
plots: ["plots/1225-0.json"]
sourceHash: "5243f91530bc204540517efec030eba190aa0fbc6533a18eeeae9424d3fb85d6"
sourceCorrections: []
---

一阶优化方法是训练许多机器学习 (machine learning)模型的主力，特别是深度神经网络 (neural network)。尽管本课程侧重于高级方法，但对梯度下降 (gradient descent)的几种根本方法有扎实的理解是必需的。这些算法迭代地调整模型参数 (parameter)以最小化损失函数 (loss function)，主要依靠梯度，梯度指出最陡峭的上升方向。

### 梯度下降 (gradient descent) (GD)

梯度下降（常被称为批量梯度下降或BGD）是一种基础的优化方法。它使用*整个*训练数据集计算损失函数 (loss function) $J(\theta)$ 关于参数 (parameter) $\theta$ 的梯度。在迭代 $t$ 时的参数更新规则是：


$$
\theta_{t+1} = \theta_t - \eta \nabla_{\theta} J(\theta_t)
$$


这里，$\eta$ 是学习率，一个控制步长的超参数 (hyperparameter)。尽管GD保证非凸函数的收敛到局部最小值，以及凸函数的收敛到全局最小值（在给定合适学习率的情况下），但对于现代机器学习 (machine learning)中常见的大型数据集，计算整个数据集上的梯度可能在计算上不可行。每次更新都需要处理每一个训练样本。

### 随机梯度下降 (gradient descent) (SGD)

为解决BGD的计算负担，随机梯度下降 (SGD) 在每一步只使用*一个*随机选择的训练样本 $(x^{(i)}, y^{(i)})$ 来计算梯度并更新参数 (parameter)：


$$
\theta_{t+1} = \theta_t - \eta \nabla_{\theta} J(\theta_t; x^{(i)}, y^{(i)})
$$


此外，通常使用包含 $m$ 个样本的小批量数据（一个mini-batch），这在单样本SGD的计算效率和BGD更精确的梯度估计之间取得了折衷。这通常被称为小批量梯度下降，但在实践中常简称为SGD。


$$
\theta_{t+1} = \theta_t - \eta \frac{1}{m} \sum_{i=1}^{m} \nabla_{\theta} J(\theta_t; x^{(i)}, y^{(i)})
$$


SGD（无论是单样本还是小批量）的主要特点是梯度估计中引入了随机性或噪声。虽然这使得更新速度快得多，但趋向最小值的路径通常是不稳定的，表现出高方差。这种噪声有时可以帮助脱离浅层局部最小值，但这也意味着SGD通常会以锯齿状接近最优值，并且在不衰减学习率的情况下可能无法精确地稳定在最小值。

> 批量梯度下降与随机梯度下降的比较。

### 动量

SGD更新中的高方差可能导致振荡，特别是在损失曲面弯曲陡峭的方向，而平坦方向上的进展仍然缓慢。动量方法旨在减弱这些振荡并加速收敛，通过将前一次更新向量 (vector) $v_{t-1}$ 的一个分数 $\gamma$（通常约为0.9）加到当前梯度步长中。

更新规则是：


$$
v_t = \gamma v_{t-1} + \eta \nabla_{\theta} J(\theta_t) \\
\theta_{t+1} = \theta_t - v_t
$$


想象一个球从山上滚下来。动量 ($v_t$) 在梯度持续指向下坡的方向上积累速度，使它能够更快地穿过平坦区域，并通过时间上的梯度平均来平滑更新路径。这通常会比标准SGD更快地收敛。

### Nesterov 加速梯度 (NAG)

Nesterov加速梯度 (NAG) 是动量方法的一种改进。NAG不是在当前位置 $\theta_t$ 计算梯度然后加上动量项，而是采取“向前看”一步。它首先应用动量更新（估算下一个位置），然后在这个*未来*的估算位置计算梯度。

更新规则，尽管实现方式略有不同但都体现了向前看这一想法，是：


$$
v_t = \gamma v_{t-1} + \eta \nabla_{\theta} J(\theta_t - \gamma v_{t-1}) \\
\theta_{t+1} = \theta_t - v_t
$$


表达式 $\theta_t - \gamma v_{t-1}$ 代表估算的未来位置。通过在这个向前看的位置计算梯度，NAG可以预测参数 (parameter)的去向并更有效地修正路径。这种“更智能”的动量通常比标准动量更快地收敛并表现更佳，尤其是在损失曲面复杂的问题上。



![优化路径](plots/1225-0.json)



> SGD、动量和NAG在损失曲面上的收敛路径图示。SGD表现出更多振荡，而动量和NAG采取更直接的路径趋向最优值（红圈）。

这些一阶方法，特别是带有动量或NAG的SGD，构成了当今许多优化器的根本。然而，它们对所有参数都依赖于单一的学习率 $\eta$，并且不能本质上适应损失曲面的几何形状。了解它们的行为和局限性促使人们研究更复杂的方法，例如自适应学习率和二阶方法，我们将在后续章节中介绍。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  对梯度下降变体（包括GD、SGD、动量和NAG）进行了全面介绍，是深度学习的参考。
- [An Overview of Gradient Descent Optimization Algorithms](https://arxiv.org/abs/1609.04747) — Sebastian Ruder (2016)
  Journal: arXiv; Publisher: Cornell University Library; DOI: [arXiv:1609.04747](https://doi.org/arXiv%3A1609.04747)
  一篇被广泛引用的文章，简明总结了各种梯度下降优化算法，包括文中讨论的核心变体。
- [On the importance of initialization and momentum in deep learning](https://proceedings.mlr.press/v28/sutskever13.html) — Ilya Sutskever, James Martens, George Dahl, and Geoffrey Hinton (2013)
  Journal: Proceedings of the 30th International Conference on Machine Learning; Publisher: PMLR; Volume: 28; Pages: 1139-1147
  讨论了动量方法（包括Nesterov加速梯度）在深度学习中的实际应用和优势。
- [A method of solving a convex programming problem with convergence rate O(1/k^2)](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG_TlXvIRZfQQAJZgwaE93V09B09FDWQSJseovzBmoYSWa6eV8L4yB9qrWEDPzvd1YR3ib-dV1s_hYLqn0nEQjwp29WSGLOZWdhjI01TccTIEr47h_x4vsZeagv4aHyuyxseEeDqFsROCtPIEF0aA==) — Yurii Nesterov (1983)
  Journal: Soviet Mathematics Doklady; Publisher: American Mathematical Society; Volume: 27; Pages: 372-376
  提出Nesterov加速梯度的原始基础论文，证明其优越的收敛特性。
