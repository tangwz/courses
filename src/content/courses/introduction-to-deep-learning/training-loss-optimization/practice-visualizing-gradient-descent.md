---
course: "introduction-to-deep-learning"
chapter: "training-loss-optimization"
lesson: "practice-visualizing-gradient-descent"
sourceId: 5042
sourceUrl: "https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-3-training-loss-optimization/practice-visualizing-gradient-descent"
title: "动手实践：梯度下降可视化"
description: "创建简单的可视化以理解梯度下降如何寻找最小值。"
order: 9
plots: ["plots/5042-0.json", "plots/5042-1.json", "plots/5042-2.json"]
sourceHash: "3c0951394c06f212981f36774710eadcc6abc6759b6b78419c98e2adc7b86018"
sourceCorrections: []
---

理解梯度下降 (gradient descent)的理论是一回事；亲眼看它运作能更直观地理解它如何在损失曲面上找到最佳参数 (parameter)。在本实践部分，我们将使用简单的可视化方式来演示梯度下降算法的行为。

首先从一个非常简单的一维凸函数 $f(x) = x^2$ 开始。这个函数可以视为一个简化的损失函数 (loss function)，其中只有一个参数 $x$ 需要优化。目标是找到使 $f(x)$ 最小的 $x$ 值。根据微积分原理，最小值位于 $x=0$ 处，而梯度下降方法能够系统地找到这样的最优参数。

$f(x) = x^2$ 的梯度（在这种一维情况下是导数）是 $f'(x) = 2x$。梯度下降更新规则是：
$x_{new} = x_{old} - \alpha \cdot f'(x_{old})$
$x_{new} = x_{old} - \alpha \cdot (2x_{old})$
其中 $\alpha$ 是学习率。

我们来模拟这个过程。我们将从一个任意点开始，比如 $x_0 = 4$，并选择一个学习率，比如 $\alpha = 0.1$。

- **步骤 0:** $x_0 = 4.0$。$f(x_0) = 16.0$。梯度 $f'(x_0) = 2 \times 4.0 = 8.0$。
- **步骤 1:** $x_1 = x_0 - \alpha \cdot f'(x_0) = 4.0 - 0.1 \times 8.0 = 4.0 - 0.8 = 3.2$。$f(x_1) = 10.24$。梯度 $f'(x_1) = 2 \times 3.2 = 6.4$。
- **步骤 2:** $x_2 = x_1 - \alpha \cdot f'(x_1) = 3.2 - 0.1 \times 6.4 = 3.2 - 0.64 = 2.56$。$f(x_2) = 6.55$。梯度 $f'(x_2) = 2 \times 2.56 = 5.12$。
- **步骤 3:** $x_3 = x_2 - \alpha \cdot f'(x_2) = 2.56 - 0.1 \times 5.12 = 2.56 - 0.512 = 2.048$。$f(x_3) = 4.19$。
- ……依此类推。请注意，$x$ 的值如何逐渐接近 0，并且随着我们接近最小值，步长（由梯度决定）也随之减小。

我们来将这条路径可视化。



![f(x) = x^2 上的梯度下降 (alpha=0.1)](plots/5042-0.json)



> 梯度下降从 $x=4$ 开始，以学习率 $\alpha=0.1$ 迭代地向二次函数 $f(x) = x^2$ 的最小值移动。

### 学习率的影响

学习率 $\alpha$ 是一个重要的超参数 (parameter) (hyperparameter)。我们来看看不同值的情况，仍然从 $x_0 = 4$ 开始。

- **小学习率 ($\alpha = 0.01$):** 步长会非常小，导致向最小值移动的速度非常慢。
- **大学习率 ($\alpha = 0.95$):** 步长可能过大，导致算法越过最小值并来回振荡。它可能仍然收敛，但效率可能不高。
- **过大学习率 ($\alpha = 1.05$):** 步长如此之大，以至于每次更新都比上一个点离最小值更远。算法发散。



![学习率对梯度下降的影响 (f(x) = x^2)](plots/5042-1.json)



> 在 $f(x) = x^2$ 上不同学习率 ($\alpha$) 的梯度下降 (gradient descent)路径比较。小值收敛缓慢，合适的值有效收敛，较大的值可能越过甚至发散。

这表明了选择合适的学习率对成功训练的重要性。

### 二维可视化

神经网络 (neural network)损失函数 (loss function)通常依赖于数百万个参数 (parameter)，这使得它们的曲面无法直接可视化。然而，通过可视化二变量函数（例如 $f(x, y) = x^2 + y^2$）上的梯度下降 (gradient descent)，我们可以获得更好的直观理解。最小值显然在 $(0, 0)$。

梯度现在是一个向量 (vector)：
$\nabla f(x, y) = \left[ \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y} \right] = [2x, 2y]$

更新规则变为：
$x_{new} = x_{old} - \alpha \cdot (2x_{old})$
$y_{new} = y_{old} - \alpha \cdot (2y_{old})$

我们从 $(x_0, y_0) = (3, 4)$ 开始，学习率 $\alpha = 0.1$。

- **步骤 0:** $(x_0, y_0) = (3.0, 4.0)$。$f(x_0, y_0) = 9 + 16 = 25$。梯度 $\nabla f(3, 4) = [6, 8]$。
- **步骤 1:**
  - $x_1 = 3.0 - 0.1 \times 6 = 3.0 - 0.6 = 2.4$。
  - $y_1 = 4.0 - 0.1 \times 8 = 4.0 - 0.8 = 3.2$。
  - $(x_1, y_1) = (2.4, 3.2)$。$f(x_1, y_1) = 5.76 + 10.24 = 16.0$。梯度 $\nabla f(2.4, 3.2) = [4.8, 6.4]$。
- **步骤 2:**
  - $x_2 = 2.4 - 0.1 \times 4.8 = 2.4 - 0.48 = 1.92$。
  - $y_2 = 3.2 - 0.1 \times 6.4 = 3.2 - 0.64 = 2.56$。
  - $(x_2, y_2) = (1.92, 2.56)$。$f(x_2, y_2) = 3.6864 + 6.5536 = 10.24$。

我们可以将其可视化为函数 $f(x, y)$ 等高线图上的一条路径。任何一点的梯度都垂直于穿过该点的等高线，并指向最陡峭的上升方向。梯度下降则向相反的方向（最陡峭的下降方向）迈进。



![f(x, y) = x^2 + y^2 上的梯度下降 (alpha=0.1)](plots/5042-2.json)



> 梯度下降在 $f(x, y) = x^2 + y^2$ 的等高线图上的路径。从 (3, 4) 开始，算法沿着垂直于等高线的方向向最小值 (0, 0) 移动。

这些可视化虽然简单，但演示了梯度下降的核心机制。在深度学习 (deep learning)中，损失曲面要复杂得多，并且是高维的，可能存在许多局部最小值和鞍点（如 `gradient-descent-challenges` 中所述）。然而，基本思想保持不变：沿着负梯度方向迭代地减小损失。理解这种视觉直觉有助于诊断训练问题或调整学习率等超参数 (hyperparameter)。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  深度学习的权威教材，对梯度下降的机制和变体进行了全面的理论解释。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本实践指南，通过直观的示例、代码和可视化来解释包括梯度下降在内的机器学习算法。
- [CS231n: Convolutional Neural Networks for Visual Recognition - Optimization](https://cs231n.github.io/optimization-1/) — Stanford University (2023)
  Journal: Course Lecture Notes
  提供了关于梯度下降及其在神经网络中行为的清晰易懂的解释和视觉直觉，常包含交互式示例。
- [Pattern Recognition and Machine Learning](https://link.springer.com/book/10.1007/978-0-387-45528-0) — Christopher M. Bishop (2006)
  Publisher: Springer; DOI: [10.1007/978-0-387-45528-0](https://doi.org/10.1007/978-0-387-45528-0)
  机器学习的经典参考文献，提供了严谨的数学处理，包括基于梯度的优化算法的数学基础。
