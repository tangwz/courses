# 有限内存BFGS (L-BFGS)

来源：[原文](https://apxml.com/zh/courses/optimization-techniques-ml/chapter-2-second-order-optimization-methods/l-bfgs-algorithm)

[返回章节目录](README.md) · [返回课程目录](../README.md)

BFGS算法作为一种优化方法，通过避免直接计算和求逆海森矩阵，相较于牛顿法等方法实现了改进。然而，BFGS算法仍需要存储和更新海森矩阵（或其逆矩阵）的一个近似，这通常是一个稠密的 $d \times d$ 矩阵，$d$ 是参数 (parameter)的数量。对于拥有数百万甚至数十亿参数的机器学习 (machine learning)模型，由于内存限制（$O(d^2)$ 内存），存储一个 $d \times d$ 矩阵是完全不可行的。在这种情况下，有限内存BFGS (L-BFGS) 算法变得必要。

### 核心思想：以内存换计算

L-BFGS巧妙地避开了标准BFGS的 $O(d^2)$ 内存需求。L-BFGS不存储稠密的 $d \times d$ 海森近似矩阵 $H_k$，它只存储少量（例如 $m$ 个）最近的校正对 $(s_i, y_i)$。回顾一下：

- $s_k = x_{k+1} - x_k$ (参数 (parameter)变化)
- $y_k = \nabla f(x_{k+1}) - \nabla f(x_k)$ (梯度变化)

通常，$m$ 是一个小的整数，常在5到20之间，这与问题维度 $d$ 无关。这使得内存需求大幅降低到 $O(md)$，其与参数数量 $d$ 成线性关系，因此即使对于非常大的模型也能应对。

其权衡之处在于，L-BFGS不直接访问矩阵 $H_k$，而是仅利用存储的 $(s_i, y_i)$ 对隐式计算矩阵向量 (vector)积 $H_k \nabla f_k$（这是确定搜索方向 $p_k = -H_k \nabla f_k$ 所需的）。这种计算涉及一个通常被称为“L-BFGS两循环递归”的过程。

### L-BFGS两循环递归

L-BFGS的核心在于它如何在不显式形成矩阵 $H_k$ 的情况下重构搜索方向 $p_k = -H_k \nabla f_k$。它利用BFGS更新公式，该公式可以递归表示。该算法使用最近的 $m$ 个对 $\{ (s_i, y_i) \}_{i=k-m}^{k-1}$ 来计算乘积 $H_k \nabla f_k$。

过程包含：

1. **初始化：** 从初始海森近似 $H_k^0$ 开始。一个常见的选择是 $H_k^0 = \gamma_k I$，其中 $I$ 是单位矩阵，$\gamma_k$ 是一个缩放因子，用于近似真实海森矩阵的大小。一个常用选择是：
   
   $$
   \gamma_k = \frac{s_{k-1}^T y_{k-1}}{y_{k-1}^T y_{k-1}}
   $$
   
   这种缩放有助于确保搜索方向具有适当的大小。
2. **第一个循环（反向传递）：** 从 $i = k-1$ 向下迭代到 $k-m$。在此循环中，算法计算中间值 $\alpha_i$ 并更新梯度向量 (vector) $q = \nabla f_k$。
   - $\rho_i = 1 / (y_i^T s_i)$
   - $\alpha_i = \rho_i s_i^T q$
   - $q = q - \alpha_i y_i$
3. **缩放：** 将中间向量 $q$ 乘以初始海森近似 $H_k^0$：$z = H_k^0 q$。
4. **第二个循环（正向传递）：** 从 $i = k-m$ 向上迭代到 $k-1$。此循环使用存储的 $\alpha_i$ 值和 $(s_i, y_i)$ 对来重构最终方向。
   - $\beta_i = \rho_i y_i^T z$
   - $z = z + s_i (\alpha_i - \beta_i)$
5. **结果：** 最终向量 $z$ 是期望的乘积，$z = H_k \nabla f_k$。然后搜索方向是 $p_k = -z$。

这种两循环过程只需向量操作（点积、标量-向量乘法、向量加法），每迭代的计算成本约为 $O(md)$，远低于标准BFGS更新的 $O(d^2)$ 成本或牛顿法的 $O(d^3)$ 成本。

### L-BFGS算法概述

L-BFGS算法的一个典型迭代如下所示：

1. **选择起始点** $x_0$。
2. **设置内存大小** $m$。
3. **对于** $k = 0, 1, 2, ...$ **直到收敛：**
   a. 计算梯度 $\nabla f_k = \nabla f(x_k)$。
   b. 使用最近的 $m$ 对 $(s_i, y_i)$ 和初始海森近似 $H_k^0$，通过两循环递归计算搜索方向 $p_k = -H_k \nabla f_k$。
   c. 使用线搜索过程（例如，满足Wolfe条件的Armijo线搜索）找到合适的步长 $\alpha_k > 0$。
   d. 更新参数 (parameter)：$x_{k+1} = x_k + \alpha_k p_k$。
   e. 计算新梯度 $\nabla f_{k+1} = \nabla f(x_{k+1})$。
   f. 计算 $s_k = x_{k+1} - x_k$ 和 $y_k = \nabla f_{k+1} - \nabla f_k$。
   g. **曲率检查：** 如果 $s_k^T y_k > \epsilon$ (对于某个小的 $\epsilon > 0$)，则存储对 $(s_k, y_k)$。如果存储对的数量超过 $m$，则丢弃最旧的对 $(s_{k-m}, y_{k-m})$。
   h. 检查收敛标准（例如，$||\nabla f_{k+1}|| < \text{容差}$）。

### L-BFGS的优点

- **内存效率高：** 其主要优点。$O(md)$ 的内存使其适用于参数 (parameter)数量庞大、存储 $O(d^2)$ 不可行的问题。
- **计算效率高：** 对于大的 $d$，每次迭代 $O(md)$ 的计算速度远快于标准BFGS或牛顿法。
- **收敛性好：** 通常达到超线性收敛速度，在许多确定性问题上，其收敛速度（以迭代次数计）通常比梯度下降 (gradient descent)或共轭梯度等一阶方法快得多。

### 缺点与考量

- **超参数 (parameter) (hyperparameter) $m$：** 内存缓冲的大小 $m$ 影响性能。较大的 $m$ 存储更多曲率信息，可能带来更好的搜索方向和更快的收敛（更少迭代次数），但会增加每次迭代的内存使用和计算成本。典型值较小（例如，5-20）。
- **线搜索：** 仍需要线搜索算法来寻找步长 $\alpha_k$，这有时本身就计算量大，每次迭代需要多次函数和梯度评估。
- **曲率条件：** 更新依赖于条件 $s_k^T y_k > 0$，这意味着函数在 $s_k$ 方向上的曲率为正。在非凸区域，这可能不成立。如果此条件不满足，实现通常会跳过更新。
- **批量方法：** L-BFGS，与标准BFGS和牛顿法一样，基本是为确定性（批量）优化设计的，其中真实梯度是使用整个数据集计算的。将其有效调整到深度学习 (deep learning)中常见的随机设置（使用小批量）具有挑战性，尽管存在一些变体。

### L-BFGS在实践中

L-BFGS是许多科学计算和优化范畴中的主力算法。在机器学习 (machine learning)中，它通常对以下方面有效：

- 训练那些计算完整梯度可行（例如，中等规模数据集上的逻辑回归、线性SVM）的模型。
- 需要在最优点附近高精度的问题。
- 有时在Adam或SGD等随机方法初步训练后，用作最后的微调 (fine-tuning)步骤。

尽管像Adam这样的自适应一阶方法因其与小批量处理的兼容性以及在复杂非凸地形中寻找最优点的能力，通常更受青睐用于训练大型深度神经网络 (neural network)，但了解L-BFGS能提供有价值的认识，关于如何在内存受限下将曲率信息应用于优化。它代表了连接一阶算法简洁性与二阶算法高效性的强大桥梁。在SciPy (`scipy.optimize.minimize(method='L-BFGS-B')`) 等库中可以方便地找到其实现，并且一些机器学习框架内部也使用它。

## 参考资料

- [Updating Quasi-Newton Matrices with Limited Storage](https://doi.org/10.1090/S0025-5718-1980-0572855-7) — Jorge Nocedal (1980)
  Journal: Mathematics of Computation; Publisher: American Mathematical Society; Volume: 35; Pages: 773-782; DOI: [10.1090/S0025-5718-1980-0572855-7](https://doi.org/10.1090/S0025-5718-1980-0572855-7)
  介绍L-BFGS算法的原始学术论文。
- [Numerical Optimization](https://link.springer.com/book/10.1007/978-0-387-40065-5) — Jorge Nocedal and Stephen J. Wright (2006)
  Publisher: Springer; DOI: [10.1007/978-0-387-40065-5](https://doi.org/10.1007/978-0-387-40065-5)
  一本广受认可的教材，严谨并全面地介绍了L-BFGS及其他优化方法。
- [On the limited memory BFGS method for large scale optimization](https://link.springer.com/article/10.1007/BF01589116) — Dong C. Liu and Jorge Nocedal (1989)
  Journal: Mathematical Programming; Publisher: Springer; Volume: 45; Pages: 503-528; DOI: [10.1007/BF01589116](https://doi.org/10.1007/BF01589116)
  本文详细阐述了L-BFGS-B算法，这是一种处理有界约束的扩展，并提供了进一步的算法分析。
- [Optimization for Machine Learning (Lecture Notes)](https://people.eecs.berkeley.edu/~jrs/slides/189/ml-opt.pdf) — Jonathan Richard Shewchuk (2005)
  Publisher: University of California, Berkeley; Pages: 26
  教育性讲义，清晰地推导并解释了L-BFGS算法，包括其双循环递归。

---

[上一节](05-BFGS%E7%AE%97%E6%B3%95%E8%AF%A6%E8%A7%A3.md) · [下一节](07-%E4%BF%A1%E8%B5%96%E5%9F%9F%E6%96%B9%E6%B3%95.md)
