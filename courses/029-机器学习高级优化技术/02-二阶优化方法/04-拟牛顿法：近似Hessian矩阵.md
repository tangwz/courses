# 拟牛顿法：近似Hessian矩阵

来源：[原文](https://apxml.com/zh/courses/optimization-techniques-ml/chapter-2-second-order-optimization-methods/quasi-newton-methods-introduction)

[返回章节目录](README.md) · [返回课程目录](../README.md)

牛顿法表明，通过Hessian矩阵引入曲率信息可以显著加快收敛到最小值。然而，牛顿法在大规模机器学习 (machine learning)中的实际应用常常受阻，原因是每次迭代中构建和求逆Hessian矩阵会带来计算上的沉重负担。对于一个有$N$个参数 (parameter)的模型，Hessian是一个$N \times N$矩阵，需要$O(N^2)$的存储空间，并且通常需要$O(N^3)$的计算量来求逆。当$N$达到百万或十亿级别时，这就完全不可行了。

这时，拟牛顿法就派上用场了。它们提供了一个巧妙的折中方案：在无需明确计算、存储或求逆Hessian矩阵的情况下，保持与二阶信息相关的超线性收敛优势。其主要理念是，仅利用连续的梯度评估和参数更新信息，逐步构建Hessian矩阵的*近似*，或更常见地，其逆矩阵的*近似*。

### 主要理念：迭代近似

拟牛顿法不是直接计算$\nabla^2 f(x_k)$，而是维护一个矩阵$B_k$来近似Hessian $\nabla^2 f(x_k)$，或一个矩阵$H_k$来近似逆Hessian $(\nabla^2 f(x_k))^{-1}$。那么更新步骤看起来与牛顿法相似，但使用了近似：

- 使用Hessian近似$B_k$：求解$B_k p_k = -\nabla f(x_k)$得到搜索方向$p_k$，然后更新$x_{k+1} = x_k + \alpha_k p_k$。
- 使用逆Hessian近似$H_k$：直接计算搜索方向$p_k = -H_k \nabla f(x_k)$，然后更新$x_{k+1} = x_k + \alpha_k p_k$。

实际使用中通常更倾向于使用矩阵$H_k$，因为它避免了在每一步中求解一个线性系统，取而代之的是一个简易的矩阵-向量 (vector)乘法。步长$\alpha_k$通常通过线搜索过程确定，确保目标函数有足够的下降。

### 割线条件：连接梯度与曲率

近似$B_k$或$H_k$如何从一次迭代更新到下一次？其要点在于**割线条件**（也称为拟牛顿条件）。

我们来定义两次迭代之间参数 (parameter)和梯度的变化：


$$
s_k = x_{k+1} - x_k
$$


$$
y_k = \nabla f(x_{k+1}) - \nabla f(x_k)
$$


回顾微积分可知，对于二次函数，Hessian将梯度的变化与位置的变化关联起来：$\nabla f(x_{k+1}) - \nabla f(x_k) = \nabla^2 f (x_{k+1} - x_k)$。拟牛顿法对*下一次*近似$B_{k+1}$施加了类似的关系。我们要求新的近似能正确地将刚迈出的步长($s_k$)映射到观察到的梯度变化($y_k$)：


$$
B_{k+1} s_k = y_k \quad \text{(Hessian近似的割线条件)}
$$


等价地，如果我们近似逆Hessian $H_k$，条件变为：


$$
H_{k+1} y_k = s_k \quad \text{(逆Hessian近似的割线条件)}
$$


这个条件采用了来自最新步骤的现成梯度信息($y_k$)和参数变化($s_k$)，对更新后的近似施加一个约束。它有效结合了关于函数沿方向$s_k$的曲率信息。

### 更新近似矩阵

割线条件提供了一个约束，但它不能唯一确定整个矩阵$B_{k+1}$或$H_{k+1}$。不同的拟牛顿法源于在满足割线条件的同时如何更新矩阵的不同选择。通常，更新被表述为向当前近似添加一个低秩修正：


$$
B_{k+1} = B_k + \Delta B_k
$$


$$
H_{k+1} = H_k + \Delta H_k
$$


更新项($\Delta B_k$或$\Delta H_k$)的具体形式定义了特定的算法。常用的方法旨在：

1. 满足割线条件。
2. 保持对称性（因为真正的Hessian是对称的）。
3. 保持正定性（以确保下降方向，尤其是在近似$H_k$时）。
4. 计算成本低（通常涉及一阶或二阶更新）。
5. 在某种意义上“接近”前一个近似$B_k$或$H_k$，体现了逐步改进的思路。

源于这些原则的最成功且普遍使用的更新公式是\*\*Broyden–Fletcher–Goldfarb–Shanno (BFGS)\*\*更新，我们将在下一节中具体阐述。其内存效率高的变体，**有限内存BFGS (L-BFGS)**，是用于大规模机器学习 (machine learning)优化问题的主力算法。

从根本上说，拟牛顿法通过梯度差分逐次建立曲率信息，巧妙地避开了牛顿法的计算瓶颈。它们代表了一类效能高的算法，常常比一阶方法收敛快得多，尤其是在曲率起主要作用的问题上，同时避免了精确二阶途径所带来的过高成本。

## 参考资料

- [Numerical Optimization](https://link.springer.com/book/10.1007/978-0-387-40065-5) — Jorge Nocedal and Stephen J. Wright (2006)
  Publisher: Springer; Pages: Chapter 6: Quasi-Newton Methods, Pages 135-163; DOI: [10.1007/978-0-387-40065-5](https://doi.org/10.1007/978-0-387-40065-5)
  一本基础教科书，全面地阐述了拟牛顿方法，包括割线条件和BFGS算法的理论与实践。
- [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf) — Stephen Boyd and Lieven Vandenberghe (2004)
  Publisher: Cambridge University Press; Pages: Section 9.3.4: Quasi-Newton methods
  一本广受推崇的教科书，涵盖了基础优化算法，在凸优化背景下清晰讨论了拟牛顿方法。
- [CS229 Lecture Notes: Optimization (Newton's Method and Quasi-Newton Methods)](http://cs229.stanford.edu/notes2020fall/cs229-notes-quasi-newton.pdf) — Stanford University, CS229 Course Staff (2020)
  提供清晰简洁的讲义，介绍牛顿法和拟牛顿法，对机器学习实践者尤其有用。

---

[上一节](03-%E7%89%9B%E9%A1%BF%E6%B3%95%E7%9A%84%E9%9A%BE%E7%82%B9.md) · [下一节](05-BFGS%E7%AE%97%E6%B3%95%E8%AF%A6%E8%A7%A3.md)
