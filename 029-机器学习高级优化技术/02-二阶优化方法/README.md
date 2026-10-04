# 第 2 章：二阶优化方法

来源：[原章节](https://apxml.com/zh/courses/optimization-techniques-ml/chapter-2-second-order-optimization-methods)

[返回课程目录](../README.md)

在一阶梯度方法的基础上，本章介绍使用二阶导数信息的优化技术。这些方法考虑损失函数的曲率，通常比梯度下降收敛更快，尤其是在接近最小值时。

我们将从牛顿法开始，了解其理论原理，即使用二次模型在局部近似目标函数：
$$
f(x_k + p) \approx f(x_k) + \nabla f(x_k)^T p + \frac{1}{2} p^T \nabla^2 f(x_k) p
$$
这里的重要组成是海森矩阵 $\nabla^2 f(x_k)$。我们将讨论其性质，以及计算和求逆所面临的计算难点，尤其是在高维机器学习问题中。

为了解决这些难点，我们将主要介绍拟牛顿法，这些方法近似海森矩阵或其逆。你将学习流行的BFGS算法及其节省内存的变体L-BFGS，它们在实际中广泛使用。此外，我们还将介绍信赖域方法，它提供了一种使用曲率信息管理步长的替代策略。本章包含一个关于L-BFGS实现的实践部分。

## 小节

- 1. [牛顿法：理论与推导](01-%E7%89%9B%E9%A1%BF%E6%B3%95%EF%BC%9A%E7%90%86%E8%AE%BA%E4%B8%8E%E6%8E%A8%E5%AF%BC.md)
- 2. [海森矩阵：计算与性质](02-%E6%B5%B7%E6%A3%AE%E7%9F%A9%E9%98%B5%EF%BC%9A%E8%AE%A1%E7%AE%97%E4%B8%8E%E6%80%A7%E8%B4%A8.md)
- 3. [牛顿法的难点](03-%E7%89%9B%E9%A1%BF%E6%B3%95%E7%9A%84%E9%9A%BE%E7%82%B9.md)
- 4. [拟牛顿法：近似Hessian矩阵](04-%E6%8B%9F%E7%89%9B%E9%A1%BF%E6%B3%95%EF%BC%9A%E8%BF%91%E4%BC%BCHessian%E7%9F%A9%E9%98%B5.md)
- 5. [BFGS算法详解](05-BFGS%E7%AE%97%E6%B3%95%E8%AF%A6%E8%A7%A3.md)
- 6. [有限内存BFGS (L-BFGS)](06-%E6%9C%89%E9%99%90%E5%86%85%E5%AD%98BFGS%20%28L-BFGS%29.md)
- 7. [信赖域方法](07-%E4%BF%A1%E8%B5%96%E5%9F%9F%E6%96%B9%E6%B3%95.md)
- 8. [动手实践：L-BFGS 的实现](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9AL-BFGS%20%E7%9A%84%E5%AE%9E%E7%8E%B0.md)
