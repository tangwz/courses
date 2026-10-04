---
course: "optimization-techniques-ml"
chapter: "advanced-specialized-optimization"
lesson: "lagrangian-duality-kkt"
sourceId: 1319
sourceUrl: "https://apxml.com/zh/courses/optimization-techniques-ml/chapter-7-advanced-specialized-optimization/lagrangian-duality-kkt"
title: "拉格朗日对偶性与KKT条件"
description: "运用拉格朗日乘数和KKT条件求解受约束优化问题的理论依据。"
order: 2
plots: []
sourceHash: "c52e0c36f8349e251d54e249485f02dd655d94cab490beebebcc662635202385"
sourceCorrections: []
---

拉格朗日对偶性与卡鲁什-库恩-塔克（KKT）条件为分析和解决受约束优化问题提供了主要的理论体系。这些思想提供了一种有效方法，可以将受约束问题转化为可能更容易求解的对偶问题，并提供了一系列条件来检验潜在解是否为最优。

### 标准受约束优化问题

我们考虑一个标准形式的通用受约束优化问题：

最小化 $f(x)$
约束条件：
$g_i(x) \le 0$，对于 $i = 1, \dots, m$ (不等式约束)
$h_j(x) = 0$，对于 $j = 1, \dots, p$ (等式约束)

这里，$x \in \mathbb{R}^n$ 是优化变量，$f: \mathbb{R}^n \to \mathbb{R}$ 是目标函数，$g_i: \mathbb{R}^n \to \mathbb{R}$ 是不等式约束函数，而 $h_j: \mathbb{R}^n \to \mathbb{R}$ 是等式约束函数。我们假设这些函数可微。满足所有约束的点集称为可行域。

### 拉格朗日函数

对偶理论的第一步是将约束条件融入目标函数。我们使用拉格朗日函数 $L: \mathbb{R}^n \times \mathbb{R}^m \times \mathbb{R}^p \to \mathbb{R}$ 来完成此操作，其定义为：


$$
L(x, \lambda, \nu) = f(x) + \sum_{i=1}^m \lambda_i g_i(x) + \sum_{j=1}^p \nu_j h_j(x)
$$


变量 $\lambda = (\lambda_1, \dots, \lambda_m)$ 和 $\nu = (\nu_1, \dots, \nu_p)$ 称为拉格朗日乘数或对偶变量。我们要求与不等式约束相关的乘数 $\lambda_i$ 为非负，即 $\lambda_i \ge 0$。与等式约束相关的乘数 $\nu_j$ 的符号不受限制。

为什么是这种形式？请注意，如果对于固定的 $x$，我们对 $\lambda \ge 0$ 和 $\nu$ 最大化 $L(x, \lambda, \nu)$：

- 若 $x$ 违反约束 $g_i(x) > 0$，我们可以选择一个大的正 $\lambda_i$ 使 $L$ 任意大。
- 若 $x$ 违反约束 $h_j(x) \ne 0$，我们可以选择一个合适的 $\nu_j$ 使 $L$ 任意大（或小）。
- 若 $x$ 是可行的（满足所有约束 $g_i(x) \le 0$ 和 $h_j(x) = 0$），则为使 $L$ 最大化，当 $g_i(x) < 0$ 时，我们必须将 $\lambda_i$ 设为 $0$（因为 $\lambda_i \ge 0$）。如果 $g_i(x) = 0$，则项 $\lambda_i g_i(x)$ 无论如何都为零。对于可行的 $x$，项 $\nu_j h_j(x)$ 始终为零。

因此，对于可行的 $x$:


$$
\sup_{\lambda \ge 0, \nu} L(x, \lambda, \nu) = f(x)
$$


而对于不可行的 $x$:


$$
\sup_{\lambda \ge 0, \nu} L(x, \lambda, \nu) = \infty
$$


这意味着我们原来的受约束问题（*原始问题*）等价于：


$$
p^* = \min_{x} \sup_{\lambda \ge 0, \nu} L(x, \lambda, \nu)
$$


其中 $p^*$ 是原始问题的最优值。

### 拉格朗日对偶函数与对偶问题

我们不首先对 $x$ 进行最小化，而是调换顺序，通过对原始变量 $x$ 最小化拉格朗日函数来定义拉格朗日对偶函数 $G(\lambda, \nu)$：


$$
G(\lambda, \nu) = \inf_{x} L(x, \lambda, \nu) = \inf_{x} \left( f(x) + \sum_{i=1}^m \lambda_i g_i(x) + \sum_{j=1}^p \nu_j h_j(x) \right)
$$


对偶函数 $G(\lambda, \nu)$ 为原始问题在任意 $\lambda \ge 0$ 和任意 $\nu$ 下的最优值 $p^*$ 提供了一个下界。此性质称为**弱对偶性**：


$$
G(\lambda, \nu) \le p^* \quad \forall \lambda \ge 0, \nu
$$


证明很简单：对于任意可行解 $\tilde{x}$ 和任意 $\lambda \ge 0, \nu$，我们有 $\sum \lambda_i g_i(\tilde{x}) \le 0$ 和 $\sum \nu_j h_j(\tilde{x}) = 0$。因此，$L(\tilde{x}, \lambda, \nu) \le f(\tilde{x})$。对左侧的所有 $x$ 取下确界，对右侧的所有可行 $\tilde{x}$ 取下确界，得出 $G(\lambda, \nu) \le p^*$。

由于 $G(\lambda, \nu)$ 是一个下界，自然可以通过在可行对偶变量上最大化对偶函数来找到*最佳*下界。这引出了**拉格朗日对偶问题**：

最大化 $G(\lambda, \nu)$
约束条件： $\lambda \ge 0$

设对偶问题的最优值为 $d^*$。弱对偶性表明 $d^* \le p^*$。

### 强对偶性与对偶间隙

差值 $p^* - d^*$ 称为**对偶间隙**。在许多情况下，特别是当原始问题是凸的（即 $f$ 和 $g_i$ 是凸函数，$h_j$ 是仿射函数）并满足某些约束资格条件（如 Slater 条件：存在一个严格可行点）时，对偶间隙为零。也就是说，$p^* = d^*$。这个条件被称为**强对偶性**。

强对偶性很重要，因为它表示我们可以通过求解对偶问题来解决原始问题，这可能更简单。即使强对偶性不完全成立，对偶仍然能提供有用的界限或近似值。

### 卡鲁什-库恩-塔克（KKT）条件

当强对偶性成立，并且最优值 $p^*$ 和 $d^*$ 分别在原始最优解 $x^*$ 和对偶最优解 $(\lambda^*, \nu^*)$ 处达到时，必须满足一系列必要条件。如果原始问题是凸的，这些条件也是最优性的充分条件。这些就是卡鲁什-库恩-塔克（KKT）条件：

1. **平稳性条件：** 拉格朗日函数对 $x$ 的梯度在最优解处必须为零：

   
   $$
   \nabla_x L(x^*, \lambda^*, \nu^*) = \nabla f(x^*) + \sum_{i=1}^m \lambda_i^* \nabla g_i(x^*) + \sum_{j=1}^p \nu_j^* \nabla h_j(x^*) = 0
   $$
   

   此条件基本指出，在最优解处，目标函数的梯度必须是活跃约束梯度的一个线性组合。
2. **原始可行性：** 最优解 $x^*$ 必须满足原始约束条件：

   
   $$
   g_i(x^*) \le 0, \quad \text{对于 } i = 1, \dots, m
   $$
   
   
   $$
   h_j(x^*) = 0, \quad \text{对于 } j = 1, \dots, p
   $$
   
3. **对偶可行性：** 不等式约束的拉格朗日乘数必须是非负的：

   
   $$
   \lambda_i^* \ge 0, \quad \text{对于 } i = 1, \dots, m
   $$
   
4. **互补松弛性：** 对于每个不等式约束，要么约束是活跃的（等式成立），要么其对应的拉格朗日乘数为零（或两者都成立）：

   
   $$
   \lambda_i^* g_i(x^*) = 0, \quad \text{对于 } i = 1, \dots, m
   $$
   

   此条件源自强对偶性的推导。它表示，如果在最优解处，约束 $g_i(x^*) < 0$ 是不活跃的，则其对应的“价格”或乘数 $\lambda_i^*$ 必须为零。反之，如果乘数 $\lambda_i^*$ 为正，则约束 $g_i(x^*) \le 0$ 在最优解处必须是绑定（活跃）的，即 $g_i(x^*) = 0$。

> KKT条件提供了一系列必要要求，在适当假设（如强对偶性）下，这些要求必须在最优解 $x^*$ 及其对应的对偶变量 $\lambda^*, \nu^*$ 处成立。对于凸问题，满足这些条件可确保最优性。

### 机器学习 (machine learning)中的意义

了解拉格朗日对偶性与KKT条件对于机器学习的多个方面都很重要：

- **支持向量 (vector)机（SVM）：** 对偶SVM问题的推导，使得核技巧得以运用，它主要依赖于拉格朗日对偶性。支持向量直接对应于不等式约束活跃（$\lambda_i^* > 0$）的点。
- **受约束参数 (parameter)估计：** 在对受约束模型参数进行估计时（例如，要求不同人口群体间表现相似的公平性约束、特征选择中的预算约束、参数的非负性约束），KKT条件描述了最优解的特性。
- **算法设计：** 许多用于受约束问题的优化算法，例如内点法，旨在迭代地找到满足KKT条件的点。

总结来说，拉格朗日对偶性将一个受约束的原始问题转化为一个最大化下界的对偶问题。当强对偶性成立时，KKT条件通过方程和不等式体系，精确地描述了最优解的数学特性，连接了原始变量、对偶变量、目标函数和约束。这些工具对于分析和求解在高级机器学习情境中常见受约束优化问题很重要。

## 参考资料

- [Convex Optimization](http://stanford.edu/~boyd/cvxbook/) — Stephen Boyd and Lieven Vandenberghe (2004)
  Publisher: Cambridge University Press
  这本经典教科书为拉格朗日对偶、KKT条件和凸优化提供了严谨的理论基础，与课程内容高度相关。
- [Lagrangian Duality](http://stanford.edu/class/ee364a/lectures/duality.pdf) — Stephen Boyd (2004)
  Publisher: Stanford University
  斯坦福大学EE364A课程（凸优化I）的优质讲义，清晰简洁地阐释了拉格朗日对偶和KKT条件。
