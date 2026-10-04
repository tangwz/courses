---
course: "fundamentals-quantum-machine-learning"
chapter: "variational-quantum-algorithms-ml"
lesson: "quantum-natural-gradient"
sourceId: 1060
sourceUrl: "https://apxml.com/zh/courses/fundamentals-quantum-machine-learning/chapter-4-variational-quantum-algorithms-ml/quantum-natural-gradient"
title: "量子自然梯度下降"
description: "介绍量子自然梯度（QNG），用于优化VQA，并考虑量子态空间的几何结构。"
order: 6
plots: []
sourceHash: "2a0225db91626edf813fdfe18748f60e94d4c9adc3106eeac62d10480bc40219"
sourceCorrections: []
---

虽然前一节讨论的先进经典优化器中的标准梯度下降 (gradient descent)方法为变分量子算法（VQA）的训练提供了起始点，但它们基于一个隐含的假设：参数 (parameter)空间 $\theta$ 是欧几里得的。在这种标准方法中，更新规则 $\theta^{(t+1)} = \theta^{(t)} - \eta \nabla L(\theta^{(t)})$ 在*参数空间中*沿最陡峭的下降方向移动参数。然而，我们优化的实际目标通常与参数化量子线路（PQC）产生的量子态 $|\psi(\theta)\rangle$ 有关，并且从参数 $\theta$ 到状态 $|\psi(\theta)\rangle$ 的映射可以是高度非线性和非均匀的。参数的微小变化可能导致参数空间某些区域的量子态发生显著变化，而参数的较大变化在其他地方可能只引起微小的状态改变。

这一观察促使我们采用考虑量子态空间本身几何结构的方法，而不再局限于标准梯度下降。正如第1章所介绍的，信息几何提供了分析统计模型结构的方法，包括我们的PQC产生的量子态。量子自然梯度（QNG）运用这种几何观点进行优化。

### 量子态的几何学与Fubini-Study度量

QNG背后的主要思想不是在平坦的参数 (parameter)流形上进行梯度下降 (gradient descent)，而是在参数 $\theta$ 所引起的量子态的弯曲流形上进行。两个无限接近的量子态 $|\psi(\theta)\rangle$ 和 $|\psi(\theta + d\theta)\rangle$ 之间的“距离”由Fubini-Study度量衡量，在此背景下常被称为量子费雪信息矩阵（QFIM）。

令 $|\psi(\theta)\rangle$ 为由参数 $\theta = (\theta_1, \dots, \theta_M)$ 的PQC制备的状态。Fubini-Study度量张量 $g_{ij}(\theta)$ 捕获了由参数变化 $d\theta_i$ 和 $d\theta_j$ 引起的状态之间平方的无穷小距离。其分量表示为：


$$
g_{ij}(\theta) = \text{Re}\left( \langle \partial_i \psi(\theta) | \partial_j \psi(\theta) \rangle - \langle \partial_i \psi(\theta) | \psi(\theta) \rangle \langle \psi(\theta) | \partial_j \psi(\theta) \rangle \right)
$$


式中 $|\partial_i \psi(\theta)\rangle = \frac{\partial |\psi(\theta)\rangle}{\partial \theta_i}$。这个度量张量形成一个 $M \times M$ 对称正半定矩阵 $G(\theta)$，它定量表示当我们改变参数时量子态在局部如何变化。它有效地说明了量子态对参数扰动的敏感度。

### 量子自然梯度更新规则

量子自然梯度通过预乘Fubini-Study度量张量 $G(\theta)^{-1}$ 的逆矩阵来修改标准梯度 $\nabla L(\theta)$：


$$
\theta^{(t+1)} = \theta^{(t)} - \eta \, G(\theta^{(t)})^{-1} \nabla L(\theta^{(t)})
$$


这里，$\eta$ 是学习率。这个更新规则直接在量子态流形上执行最陡下降步骤。通过 $G^{-1}$ 纳入几何信息，QNG更新步骤对PQC的具体参数 (parameter)化保持不变。它根据每个参数实际改变量子态的程度，有效地重新调整梯度分量，在参数对状态影响很小的方向上迈出更大的步长，而在参数高度敏感的方向上迈出更小的步长。

### QNG的优点

1. **优化收敛性：** 通过根据状态空间的几何形状调整步长方向和大小，QNG与标准梯度下降 (gradient descent)相比，尤其是在参数 (parameter)到状态的映射高度非均匀时，通常能以更少的迭代次数收敛。
2. **参数化不变性：** 状态空间中自然梯度步长的方向与PQC的参数化方式无关，有助于实现更一致的优化行为。
3. **跨越平坦区域：** 在贫瘠高原区域，标准梯度会变得非常小，阻碍优化。尽管QNG并未从根本上解决贫瘠高原问题（该问题与梯度方差呈指数级消失有关），但QFIM中的几何信息有时可以帮助在标准梯度很小时，也能辨识状态空间中有意义的变化方向，可能有助于在某些优化环境中前进。

### 挑战与实际实施

使用QNG的主要挑战是Fubini-Study度量张量 $G(\theta)$ 的计算和求逆。

1. **计算成本：** 计算 $G(\theta)$ 的所有 $M^2$ 个分量通常需要 $O(M^2)$ 次期望值估计。有多种方法可以估计分量 $g_{ij}(\theta)$，通常涉及评估由轻微扰动参数 (parameter)产生的状态之间的重叠，或使用与线性响应理论相关的技术，有时可以利用参数平移规则等梯度计算方法。
2. **矩阵求逆：** 对 $M \times M$ 矩阵 $G(\theta)$ 求逆通常需要 $O(M^3)$ 的经典计算量，对于具有大量参数的PQC来说，这可能变得过于昂贵。
3. **奇异性与正则化 (regularization)：** 矩阵 $G(\theta)$ 有时可能是奇异的或接近奇异的，特别是当参数冗余或对状态影响很小时。实际应用中，在求逆前通常会添加一个正则化项：$(G(\theta) + \lambda I)^{-1}$，其中 $\lambda$ 是一个小的正数（阻尼因子），$I$ 是单位矩阵。这可确保数值稳定性。

### 近似方法

由于计算开销，通常使用QFIM的近似方法：

- **块对角近似：** 如果参数 (parameter)可以分组（例如，按PQC中的层分组），可以将 $G(\theta)$ 近似为块对角矩阵，假设不同块中的参数之间几何关联性可以忽略不计。这大幅降低了求逆成本。
- **对角近似：** 一种更简单的方法是只计算并使用对角元素 $g_{ii}(\theta)$，将 $G(\theta)$ 视为对角矩阵。这相当于根据每个参数自身的敏感度单独重新调整其梯度更新，忽略参数之间的关联。计算成本更低但捕捉到的几何信息较少。

### QNG的实际应用

PennyLane等量子软件库提供计算QFIM和实现QNG优化器的功能。然而，高效计算QFIM，尤其是在量子硬件上，仍然是一个活跃的研究方向。

总而言之，量子自然梯度是一种VQA优化技术，它包含了量子态空间的几何结构。尽管计算上比标准梯度下降 (gradient descent)更耗费资源，但通过采取本质上适应量子态对参数 (parameter)变化敏感度的步长，它具有更快收敛和更好优化的潜力。此方法的应用通常涉及几何精确度与计算成本之间的权衡，因此会采用各种近似方法。

## 参考资料

- [Quantum natural gradient](https://doi.org/10.22331/q-2020-05-25-269) — Stokes, James, Izaac, Josh, Killoran, Nathan, and Carleo, Giuseppe (2020)
  Journal: Quantum; Publisher: Verein zur Förderung des Open Access Publizierens in den Quantenwissenschaften; Volume: 4; Pages: 269; DOI: [10.22331/q-2020-05-25-269](https://doi.org/10.22331/q-2020-05-25-269)
  介绍了变分量子算法的量子自然梯度方法及其理论基础。
- [Optimization with the Quantum Natural Gradient](https://docs.pennylane.ai/en/stable/demos/tutorial_quantum_natural_gradient.html) — Josh Izaac, Tom Wentworth (2022)
  Publisher: PennyLane (Xanadu)
  演示了在PennyLane量子机器学习库中量子自然梯度优化的实际应用。
