---
course: "optimization-techniques-ml"
chapter: "second-order-optimization-methods"
lesson: "newtons-method-challenges"
sourceId: 1246
sourceUrl: "https://apxml.com/zh/courses/optimization-techniques-ml/chapter-2-second-order-optimization-methods/newtons-method-challenges"
title: "牛顿法的难点"
description: "讨论计算成本（海森矩阵求逆）以及非正定海森矩阵的问题。"
order: 3
plots: []
sourceHash: "7d6d076ca38455fb68d3503e886a60cab38ebad1584b15b47a3a948015edf4d5"
sourceCorrections: []
---

尽管牛顿法借助曲率信息，在最小值附近具有二次收敛的优点，其直接应用于大规模机器学习 (machine learning)会遇到明显的实际困难。这些困难主要源于海森矩阵 $\nabla^2 f(x)$ 的计算、存储及其性质。

### 海森矩阵的计算成本

第一个主要困难在于高昂的计算耗费。

1. **构建海森矩阵**：计算海森矩阵需要计算损失函数 (loss function) $f$ 关于参数 (parameter) $x$ 的所有二阶偏导数。如果我们的模型有 $d$ 个参数（在现代深度学习 (deep learning)中很容易达到数百万或数十亿），海森矩阵将是一个 $d \times d$ 的矩阵。计算这 $d^2$ 个元素中的每一个都可能对计算要求很高。虽然自动微分等方法可以提供帮助，但其复杂度通常随参数数量呈二次方增长，大约每次迭代需要 $O(d^2)$ 次操作来计算海森矩阵。
2. **求逆海森矩阵**：牛顿法的更新步需要求解线性系统 $\nabla^2 f(x_k) p_k = -\nabla f(x_k)$ 以得到搜索方向 $p_k$。这等同于计算 $p_k = -[\nabla^2 f(x_k)]^{-1} \nabla f(x_k)$。矩阵求逆或求解稠密线性系统的标准方法通常具有 $O(d^3)$ 次操作的计算复杂度。

对于参数数量适中的模型（例如，数万个），每次迭代 $O(d^3)$ 次操作的计算量变得不可行，远远超过一阶方法的成本，后者通常每次迭代的成本为 $O(d)$（主要由梯度计算决定）。

> 牛顿法一次迭代的计算过程，突出显示与海森矩阵相关的高昂成本（$d$ 为参数数量）。

### 存储要求

在计算上，存储海森矩阵本身就带来了一个困难。一个 $d \times d$ 矩阵需要 $O(d^2)$ 的内存。如果 $d$ 达到数百万，仅为海森矩阵存储数TB的数据通常不切实际，特别是如果计算分布在多个内存有限的设备上。相比之下，一阶方法只需存储参数 (parameter)和梯度，通常需要 $O(d)$ 的内存。

### 非正定海森矩阵

牛顿法是通过最小化函数 $f$ 在当前点 $x_k$ 附近的二次近似而得到的。这种推导隐式地假设二次近似具有唯一的最小值，这在数学上对应于海森矩阵 $\nabla^2 f(x_k)$ 是**正定的**（即其所有特征值都为正）。

在许多机器学习 (machine learning)问题中，特别是非凸问题，例如训练深度神经网络 (neural network)时，这一假设常常不成立：

- **不定海森矩阵**：海森矩阵可能同时具有正的和负的特征值。这发生在鞍点，在高维损失曲面中鞍点很常见。如果海森矩阵是不定的，牛顿步 $p_k = -[\nabla^2 f(x_k)]^{-1} \nabla f(x_k)$ 实际上可能指向函数值*增加*的方向，或者沿着鞍点侧向移动。
- **负定海森矩阵**：所有特征值都为负。这发生在局部最大值附近。牛顿步会错误地将优化引向最大值。
- **奇异海森矩阵**：海森矩阵可能是奇异的（行列式为零，意味着至少一个特征值为零）或病态的（某些特征值非常接近零）。这使得矩阵求逆在数值上不稳定或无法进行。

当海森矩阵不是正定时，纯牛顿步可能导致发散，或收敛到鞍点或局部最大值而非最小值。需要进行修改，例如在海森矩阵中加上一个单位矩阵的倍数（Levenberg-Marquardt 型调整）或使用信赖域方法来处理这些情况，这增加了算法的复杂性。

这些明显的计算、存储和理论困难意味着纯牛顿法很少直接用于训练大规模机器学习模型。然而，利用曲率信息的潜在思想很有用。这促使了拟牛顿方法（如BFGS和L-BFGS）的发展，这些方法可以有效地近似海森矩阵或其逆，而无需显式地构建、存储和求逆完整矩阵。我们将在后续章节中介绍这些更实用的替代方法。

## 参考资料

- [Numerical Optimization](https://link.springer.com/book/10.1007/978-0-387-40065-5) — Jorge Nocedal and Stephen J. Wright (2006)
  Publisher: Springer; DOI: [10.1007/978-0-387-40065-5](https://doi.org/10.1007/978-0-387-40065-5)
  经典教科书，全面介绍了牛顿法、其计算问题和理论方面。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 8
  讨论了牛顿法在大规模深度学习模型中的实际局限性，包括计算和存储负担。
- [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/) — Stephen Boyd and Lieven Vandenberghe (2004)
  Publisher: Cambridge University Press
  奠基性著作，涵盖了牛顿法的理论基础，包括正定性要求。
