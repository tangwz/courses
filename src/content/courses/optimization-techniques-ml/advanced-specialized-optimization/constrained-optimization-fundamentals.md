---
course: "optimization-techniques-ml"
chapter: "advanced-specialized-optimization"
lesson: "constrained-optimization-fundamentals"
sourceId: 1317
sourceUrl: "https://apxml.com/zh/courses/optimization-techniques-ml/chapter-7-advanced-specialized-optimization/constrained-optimization-fundamentals"
title: "受限优化基本原理"
description: "介绍对参数或输出有明确约束的优化问题。"
order: 1
plots: ["plots/1317-0.json"]
sourceHash: "2d79ddbf6ab2d6271902170f790643be11135c893f038b3fceac1bf73371c10e"
sourceCorrections: []
---

尽管机器学习 (machine learning)优化大多致力于在参数 (parameter)没有明确限制时最小化损失函数 (loss function)，但许多实际情况会涉及**约束**。它们是解必须满足的条件。忽略约束可能导致不切实际、无效或不符合问题特定要求的解。

本节介绍受限优化的基本思想，为本章后面讨论的方法做准备。

### 为何要有约束？

约束在各种机器学习 (machine learning)情境中自然出现：

1. **资源限制：** 在不同特征或模型之间分配固定预算。给能力有限的工人分配任务。
2. **物理或逻辑要求：** 确保概率和为1，使参数 (parameter)保持在具有物理意义的范围内（例如，正方差），或强制变量之间存在特定关系。
3. **公平性与偏差缓解：** 对模型预测施加约束，以确保不同人口群体之间的一致性。
4. **可解释性与稀疏性：** 尽管通常通过L1正则化 (regularization)来处理，但稀疏性可被视为对非零参数数量的一种约束（尽管这种特定约束是非凸的且计算上很困难）。与此相关的是，LASSO回归在权重 (weight)L1范数约束下最小化平方误差。
5. **安全保障：** 在强化学习 (reinforcement learning)或控制系统中，确保行动保持在安全操作限制内。

### 数学表述

一般的受限优化问题可以写成以下标准形式：


$$
\begin{aligned}
\min_{x \in \mathbb{R}^n} & \quad f(x) \\
\text{受限于} & \quad g_i(x) \le 0, \quad i = 1, \dots, m \\
& \quad h_j(x) = 0, \quad j = 1, \dots, p
\end{aligned}
$$


让我们分解这些组成部分：

- $x \in \mathbb{R}^n$: 这是我们希望优化的**决策变量**向量 (vector)。在机器学习 (machine learning)中，$x$通常表示模型参数 (parameter)（权重 (weight)、偏差）。
- $f(x): \mathbb{R}^n \to \mathbb{R}$: 这是我们旨在最小化的**目标函数**。通常，这是损失函数 (loss function)（例如，均方误差、交叉熵）。
- $g_i(x) \le 0$: 这是$m$个**不等式约束**。每个$g_i(x)$是一个函数$g_i: \mathbb{R}^n \to \mathbb{R}$。这些约束定义了必须满足的“小于或等于”条件。请注意，像$a(x) \ge b$这样的约束可以重写为$b - a(x) \le 0$。
- $h_j(x) = 0$: 这是$p$个**等式约束**。每个$h_j(x)$是一个函数$h_j: \mathbb{R}^n \to \mathbb{R}$。它们指定了必须成立的精确条件。

满足*所有*不等式和等式约束的所有点$x$的集合称为**可行域**，通常表示为$\mathcal{F}$：


$$
\mathcal{F} = \{ x \in \mathbb{R}^n \mid g_i(x) \le 0 \text{ 对于所有 } i=1,\dots,m, \text{ 且 } h_j(x) = 0 \text{ 对于所有 } j=1,\dots,p \}
$$


受限优化的目标是找到一个点$x^* \in \mathcal{F}$，使得对于所有$x \in \mathcal{F}$，都有$f(x^*) \le f(x)$。

### 可行域

可行域的性质极其重要。如果约束函数$g_i$和$h_j$很简单（例如，线性），可行域可能是一个几何形状简单的图形，例如多边形或多面体。如果它们复杂且非线性，可行域可以具有复杂的结构。

考虑最小化$f(x_1, x_2) = (x_1 - 2)^2 + (x_2 - 2)^2$（其无约束最小值位于$(2, 2)$）受限于$x_1 + x_2 \le 1$、$x_1 \ge 0$和$x_2 \ge 0$。可行域是一个三角形。



![受限最小值与无约束最小值对比](plots/1317-0.json)



> 可行域（灰色阴影部分）由约束条件$x_1 \ge 0$、$x_2 \ge 0$和$x_1 + x_2 \le 1$定义。蓝色等高线表示目标函数$f(x_1, x_2) = (x_1 - 2)^2 + (x_2 - 2)^2$。无约束最小值位于$(2, 2)$（红色“x”），在可行域之外。受限最小值出现在$(0.5, 0.5)$（绿色星形），位于可行域的边界上。

### 约束带来的挑战

约束从根本上改变了优化过程：

1. **最优条件：** 在无约束优化中，局部最小值（对于可微分函数）的必要条件是梯度为零（$\nabla f(x) = 0$）。对于受限问题，这不再是充分条件，甚至不是必要条件。从上图可以看出，受限最小值$(0.5, 0.5)$出现在梯度明显不为零的地方。最优解可能位于可行域的边界上。
2. **保持可行性：** 梯度下降 (gradient descent)等标准迭代方法可能会生成使参数 (parameter)*超出*可行域的步骤。算法必须包含处理约束的机制，可以通过将迭代结果投影回可行集，或者修改搜索方向。

了解这种基本设置非常重要，然后才能研究旨在解决这些问题的方法，例如基于拉格朗日乘数（下一节讨论）或投影技术的方法。

## 参考资料

- [Convex Optimization](http://stanford.edu/~boyd/cvxbook/) — Stephen Boyd and Lieven Vandenberghe (2004)
  Publisher: Cambridge University Press
  凸优化的经典参考书，介绍了约束问题、可行域和最优条件的理论基础。
- [Numerical Optimization](https://www.springer.com/book/9780387400655) — Jorge Nocedal and Stephen J. Wright (2006)
  Publisher: Springer; DOI: [10.1007/978-0-387-40065-5](https://doi.org/10.1007/978-0-387-40065-5)
  数值优化的综合性教材，涵盖了无约束和约束问题的理论及算法。
- [Nonlinear Programming](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHNP0VnZ-zmazL_b3B49qHtD8ejy2Z87jqPjjL2g5evkGb1VKz_4jk6RNhNZy6C41SjYCxJd9MC5G7CS_xTsULUYK3kUxOTKiBxdmxnjIgDMuxOYZOuH3QUOAI9hrdiwzTU) — Dimitri P. Bertsekas (2016)
  Publisher: Athena Scientific
  非线性规划的详细教材，提供了对约束优化理论和方法的严谨分析。第三版。
- [CS229 Lecture Notes: Constrained Optimization, KKT conditions, Duality](http://cs229.stanford.edu/fall2018/Notes/constrained_optimization.pdf) — Andrew Ng and Anqi Liu (2018)
  Publisher: Stanford University
  斯坦福CS229讲义，介绍了约束优化基础知识、问题公式化，并引向KKT条件和对偶性，与机器学习高度相关。
