---
course: "calculus-essentials-machine-learning"
chapter: "calculus-ml-introduction"
lesson: "calculus-for-algorithms"
sourceId: 1349
sourceUrl: "https://apxml.com/zh/courses/calculus-essentials-machine-learning/chapter-1-calculus-ml-introduction/calculus-for-algorithms"
title: "微积分：理解算法的工具"
description: "了解微积分如何为梯度下降等算法提供数学依据。"
order: 4
plots: []
sourceHash: "9ac6462f2de64639aa352939daa645df784723e1b8eac0cced1a7bc46753800d"
sourceCorrections: []
---

机器学习 (machine learning)模型是需要优化的函数，导数衡量这些函数如何变化。这如何转化为能够实际*找到*例如最小化成本函数 $J(\theta)$ 的最佳参数 (parameter)的算法？微积分成为理解许多机器学习算法*运作方式*的必要工具。

设想你正站在浓雾笼罩的山坡上，目标是抵达山谷底部（成本函数的最低点）。你只能感受到脚下地面的坡度。你应该往哪个方向走？直观来说，你会感受最陡峭的下坡方向，然后往那个方向迈出一小步。你会重复这个过程，希望每一步都让你更接近谷底。

微积分提供了“感受坡度”的数学对应方式。对于一个我们希望最小化的函数 $J(\theta)$：

1. **找到最陡方向：** 如我们所见，导数 $\frac{dJ}{d\theta}$（或对于多参数函数 $\theta = [\theta_1, \theta_2, ..., \theta_n]$ 来说，梯度 $\nabla J(\theta)$）指示了函数的变化率。重要的一点是，梯度向量 (vector) $\nabla J(\theta)$ 指向函数在点 $\theta$ 处*增长最快*的方向。
2. **趋向最小值：** 既然我们希望*减小*函数值（最小化成本），我们就应该沿着与梯度*相反*的方向移动。负梯度 $-\nabla J(\theta)$ 指向*下降最快*的方向。

这一核心思想构成了机器学习中最基本的优化算法之一：**梯度下降 (gradient descent)**。该算法通过沿着负梯度的方向迈出小步来迭代地更新模型参数 $\theta$。

基本更新规则如下所示：
$\theta_{new} = \theta_{old} - \alpha \nabla J(\theta_{old})$
在此：

- $\theta_{old}$ 表示当前的参数值。
- $\nabla J(\theta_{old})$ 是在当前参数处计算的成本函数的梯度。它指示了最陡峭的上升方向。
- $\alpha$ 是**学习率**，一个小的正数，控制着我们迈出步子的大小。选择合适的学习率很重要——如果太大，我们可能会越过最小值；如果太小，收敛可能会非常慢。
- $\theta_{new}$ 表示迈出一步后更新的参数值。

这个过程会重复进行：在新点计算梯度，再迈出一步下坡，依此类推，直到函数值收敛到一个最小值（或者至少不再显著下降）。

> 一个由梯度引导以找到函数最小值的迭代过程。

因此，微积分不仅仅是理论上的要求；它是推动优化过程的引擎。梯度在每一步都提供了必要的方向信息，指导算法获得更好的参数值。

尽管梯度下降是一个基本例子，但微积分的作用更广。理解偏导数和链式法则（我们稍后会讲到）这类内容对理解像神经网络 (neural network)中使用的**反向传播 (backpropagation)**这类更复杂的算法如何高效计算跨多层函数优化所需的梯度很有用。如果没有微积分，设计甚至理解这些强大的学习算法将困难得多。它为系统地处理成本函数复杂“地形”以找到机器学习问题的解提供了框架。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本全面学术教材，介绍了梯度下降等优化技术作为深度学习的核心组成部分。
- [CS229 Lecture Notes: Supervised Learning](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGaobcYL54c7kUZBIs0GIqQ-J4Qv89-VXiGpWkG6UkdTzvfuQqmv3fUfjp0nyB01o-Qw8ISDNiAGMlPCC3cciNxdIZ0qE4_mn48zVjdxVCHi9NJQf79P8kwzqsKXSBlC1exb60=) — Andrew Ng, Tengyu Ma (2023)
  Journal: Stanford University CS229 Course Materials
  提供了在监督式机器学习背景下对梯度下降清晰的学术说明。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  提供了梯度下降的实践介绍，讨论了其在机器学习项目中的应用和实践考虑。
- [Numerical Optimization](https://link.springer.com/book/10.1007/978-0-387-40065-5) — Jorge Nocedal and Stephen J. Wright (2006)
  Publisher: Springer; DOI: [10.1007/978-0-387-40065-5](https://doi.org/10.1007/978-0-387-40065-5)
  一本严谨而权威的优化方法教材，包含了基于梯度的算法的详细数学基础。
