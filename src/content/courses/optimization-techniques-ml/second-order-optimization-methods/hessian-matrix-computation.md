---
course: "optimization-techniques-ml"
chapter: "second-order-optimization-methods"
lesson: "hessian-matrix-computation"
sourceId: 1245
sourceUrl: "https://apxml.com/zh/courses/optimization-techniques-ml/chapter-2-second-order-optimization-methods/hessian-matrix-computation"
title: "海森矩阵：计算与性质"
description: "了解海森矩阵，它在二阶方法中的作用，以及计算或近似方法。"
order: 2
plots: []
sourceHash: "284a4acbf5b09dfe2446071eea09890fbc2f5f45b9f0c357a421d8175b39e4f7"
sourceCorrections: []
---

如前所述，二阶方法使用曲率信息，可能比一阶方法实现更快的收敛速度。表征这种曲率的主要数学对象是海森矩阵。

### 海森矩阵的定义

对于一个标量值函数 $f: \mathbb{R}^n \rightarrow \mathbb{R}$，它表示我们的损失函数 (loss function)，具有 $n$ 个参数 (parameter)（由向量 (vector) $x$ 表示），海森矩阵，通常表示为 $\nabla^2 f(x)$ 或简写为 $H$，是一个 $n \times n$ 的二阶偏导数矩阵：


$$
\nabla^2 f(x) = H =
\begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
\end{bmatrix}
$$


每个元素 $(H)_{ij}$ 包含偏导数 $\frac{\partial^2 f}{\partial x_i \partial x_j}$。该矩阵将单变量微积分中的二阶导数观念推广到多元函数。它实质上描述了损失函数在参数空间所有可能方向上的“曲率”或“斜率的斜率”。

### 海森矩阵的作用：曲率与局部几何形状

海森矩阵提供大量信息，关于损失曲面在点 $x$ 附近的局部形状。它的性质，特别是其定性（与特征值的符号相关），直接关联到局部几何形状：

- **正定海森矩阵：** 所有特征值均为正数 ($\lambda_i > 0$)。这表示函数在 $x$ 附近局部凸，形似向上开口的碗状。在点 $x$ 处，如果 $\nabla f(x) = 0$ 且 $\nabla^2 f(x)$ 为正定，则该点是严格局部最小值。
- **负定海森矩阵：** 所有特征值均为负数 ($\lambda_i < 0$)。这表示局部凹陷，形似倒扣的碗状。在点 $x$ 处，如果 $\nabla f(x) = 0$ 且 $\nabla^2 f(x)$ 为负定，则该点是严格局部最大值。
- **不定海森矩阵：** 部分特征值为正，部分为负。这出现在鞍点处，这些点在某些方向上向上弯曲，在其他方向上向下弯曲。优化算法在鞍点附近可能停滞或减速。
- **半正定海森矩阵：** 所有特征值均为非负数 ($\lambda_i \geq 0$)，且至少有一个特征值为零。这可能表示一个平坦区域或谷底，可能导致非严格最小值。
- **半负定海森矩阵：** 所有特征值均为非正数 ($\lambda_i \leq 0$)，且至少有一个特征值为零。这可能表示一个平坦区域或山脊顶点，可能导致非严格最大值。

> 海森矩阵的定性（特征值 \u03bb）与损失曲面在驻点（梯度为零处）的局部几何形状之间的关系。

在牛顿法的背景下，海森矩阵 $\nabla^2 f(x_k)$ 用于构建函数在当前迭代点 $x_k$ 附近的二次近似。该方法随后跳到此二次近似的最小值。如果海森矩阵是正定的，这种跳跃有效地使用曲率信息，相比于单独由负梯度给出的最速下降方向，能更直接地指向真实最小值。

### 海森矩阵的计算

计算海森矩阵带来显著的计算障碍，尤其是在机器学习 (machine learning)中，参数 (parameter) $n$ 的数量可能非常庞大（数百万或数十亿）。

1. **解析计算：** 如果你有损失函数 (loss function) $f(x)$ 的精确数学表达式，你可以推导出所有 $n^2$ 个二阶偏导数的公式。这对于简单模型通常可行，但对于深度神经网络 (neural network)而言，很快变得复杂或难以处理。
2. **自动微分 (AutoDiff)：** 现代深度学习 (deep learning)框架使用自动微分。虽然主要用于计算梯度（一阶导数），自动微分技术可以扩展到高效计算海森向量 (vector)积 ($Hv$)，而无需形成完整的海森矩阵。通过自动微分计算完整的海森矩阵通常涉及递归应用梯度计算或使用专门技术，但这仍然计算成本高昂。其成本大致是计算梯度成本的 $O(n)$ 倍，导致完整矩阵的总成本与 $O(n^2)$ 次运算成正比。
3. **有限差分：** 可以使用有限差分来近似二阶导数，例如：
   
   $$
   \frac{\partial^2 f}{\partial x_i \partial x_j} \approx \frac{f(x + h e_i + k e_j) - f(x + h e_i) - f(x + k e_j) + f(x)}{hk}
   $$
   
   其中 $e_i$ 和 $e_j$ 是标准基向量，且 $h, k$ 是小的步长。这需要为每个元素进行多次函数评估，并存在数值精度问题和高计算成本（$O(n^2)$ 次函数评估）。它很少用于训练大型模型。

### 性质与实际挑战

海森矩阵在实际优化中具有一些重要性质和相关挑战：

- **对称性：** 对于机器学习 (machine learning)中遇到的大多数损失函数 (loss function)（假设足够平滑），微分顺序无关紧要（施瓦茨定理或克莱罗定理），即 $\frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial^2 f}{\partial x_j \partial x_i}$。因此，海森矩阵是对称的 ($H = H^T$)。此性质将计算和存储的独特元素数量从 $n^2$ 减少到 $n(n+1)/2$。
- **计算成本：** 如前所述，计算完整的海森矩阵需要计算 $O(n^2)$ 个二阶导数。对于具有数百万参数 (parameter)的模型（$n \approx 10^6$），$n^2$ 为 $10^{12}$，这使得直接计算不可行。
- **存储成本：** 存储完整的 $n \times n$ 海森矩阵需要 $O(n^2)$ 内存。同样，对于大型 $n$，这超出了典型硬件的内存容量。一个 $n=10^6$ 参数的 float32 海森矩阵大约需要 4 太字节。
- **求逆成本：** 标准牛顿法需要计算海森矩阵的逆 $H^{-1}$，或求解线性系统 $Hp = -\nabla f$。直接求逆或使用高斯消元等标准方法求解的成本为 $O(n^3)$ 次运算，这比计算或存储海森矩阵的成本更难以接受。
- **非正定性：** 如果海森矩阵不是正定的（即它是不定或负定的），牛顿法计算出的方向 $p = -H^{-1} \nabla f$ 可能不是下降方向（它可能指向最大值或鞍点），或者逆甚至可能不存在（如果海森矩阵是奇异的）。这需要对牛顿法进行修改，例如阻尼或使用信任域，或改用不需要显式求逆或正定性的方法。

这些挑战，特别是计算和存储成本（$O(n^2)$）以及求逆成本（$O(n^3)$），使得纯牛顿法不适用于大规模机器学习。这促使了准牛顿方法的提出，我们将在下文进行讨论，这些方法仅使用一阶梯度信息来近似海森矩阵或其逆，旨在保留二阶信息的某些优点，同时避免高昂的成本。

## 参考资料

- [Numerical Optimization](https://link.springer.com/book/10.1007/978-0-387-40065-5) — Jorge Nocedal and Stephen J. Wright (2006)
  Publisher: Springer; DOI: [10.1007/978-0-387-40065-5](https://doi.org/10.1007/978-0-387-40065-5)
  全面介绍了Hessian矩阵、其性质及其在牛顿法等二阶优化方法中的作用。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Ch. 4, 8
  在深度学习优化背景下讨论了Hessian矩阵，包括其计算挑战和Hessian-向量积的使用。
- [Automatic Differentiation in Machine Learning: A Survey](http://jmlr.org/papers/v18/17-468.html) — Atilim Gunes Baydin, Barak A. Pearlmutter, Alexey Andreyevich Radul, and Jeffrey Mark Siskind (2018)
  Journal: Journal of Machine Learning Research; Volume: 18; Pages: 1-43; DOI: [10.5555/3277526.3277543](https://doi.org/10.5555/3277526.3277543)
  综述了自动微分技术，详细阐述了它们在计算梯度和Hessian-向量积方面的应用，对大规模机器学习有益。
