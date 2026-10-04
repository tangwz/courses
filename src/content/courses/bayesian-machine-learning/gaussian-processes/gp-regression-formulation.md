---
course: "bayesian-machine-learning"
chapter: "gaussian-processes"
lesson: "gp-regression-formulation"
sourceId: 3592
sourceUrl: "https://apxml.com/zh/courses/bayesian-machine-learning/chapter-4-gaussian-processes/gp-regression-formulation"
title: "高斯过程回归公式"
description: "推导高斯过程回归的预测方程，包括均值和方差。"
order: 4
plots: ["plots/3592-0.json"]
sourceHash: "4a8f06ff301522025a94bca1722343f008581298aa76addcda7ffaf1c5992f0b"
sourceCorrections: []
---

让我们从作为函数先验的高斯过程的抽象理念，转向执行回归的具体方法。主要思想简洁而精巧：如果我们假设基础函数来源于高斯过程先验，并且观测值是该函数的带噪声版本，那么预测新点上的函数值就成为一个涉及多元高斯分布的典型贝叶斯推断问题。

## 设置：观测值与潜在函数

高斯过程定义了函数上的分布。它通常由均值函数 $m(\mathbf{x})$ 和协方差函数（核函数） $k(\mathbf{x}, \mathbf{x}')$ 来定义。为求简便，通常从假设均值函数为零，即 $m(\mathbf{x}) = 0$ 开始。

在回归场景中，我们有一个训练数据集，包含 $N$ 个输入点 $X = \{\mathbf{x}_1, \dots, \mathbf{x}_N\}$ 及其对应的观测目标值 $\mathbf{y} = \{y_1, \dots, y_N\}^T$。高斯过程回归的基本假设是，这些观测值 $y_i$ 是在输入 $\mathbf{x}_i$ 处计算的潜在函数 $f$ 的带噪声实现。具体来说，我们假设：


$$
y_i = f(\mathbf{x}_i) + \epsilon_i
$$


并且 $f \sim \mathcal{GP}(m(\mathbf{x}), k(\mathbf{x}, \mathbf{x}'))$，同时噪声项 $\epsilon_i$ 通常假定为独立同分布的、均值为零、方差为 $\sigma_n^2$ 的高斯噪声：


$$
\epsilon_i \sim \mathcal{N}(0, \sigma_n^2)
$$


令 $\mathbf{f} = [f(\mathbf{x}_1), \dots, f(\mathbf{x}_N)]^T$ 为训练输入处的潜在函数值向量 (vector)。根据高斯过程先验，该向量服从多元高斯分布：


$$
\mathbf{f} \sim \mathcal{N}(\mathbf{m}, K)
$$


在此 $\mathbf{m}$ 是先验均值向量 $[m(\mathbf{x}_1), \dots, m(\mathbf{x}_N)]^T$，而 $K$ 是在所有训练输入对上计算得到的 $N \times N$ 协方差矩阵（或格拉姆矩阵）：$K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$。

结合 $\mathbf{f}$ 上的高斯过程先验和高斯噪声模型，观测目标 $\mathbf{y}$ 的边际分布也变为高斯分布：


$$
\mathbf{y} | X \sim \mathcal{N}(\mathbf{m}, K + \sigma_n^2 I)
$$


在这里，$I$ 是 $N \times N$ 单位矩阵，噪声方差 $\sigma_n^2$ 被添加到协方差矩阵 $K$ 的对角线元素上，这反映了添加到每个观测值的独立噪声。该分布 $p(\mathbf{y} | X)$ 通常称为边际似然或证据。

## 预测新点

我们的目标是预测新测试输入点 $\mathbf{x}_*$ 处的函数值 $f_*$。我们可能也希望在多个测试点 $X_* = \{\mathbf{x}_{*1}, \dots, \mathbf{x}_{*M}\}$ 上进行预测。令 $\mathbf{f}_* = [f(\mathbf{x}_{*1}), \dots, f(\mathbf{x}_{*M})]^T$ 为这些测试点处的潜在函数值向量 (vector)。

高斯过程的主要特性是，从高斯过程抽取的任何有限函数值集合都服从多元高斯分布。因此，观测训练目标 $\mathbf{y}$ 和测试点处的潜在函数值 $\mathbf{f}_*$ 的联合分布，在给定输入 $X$ 和 $X_*$ 的条件下，也是高斯分布：


$$
\begin{pmatrix} \mathbf{y} \\ \mathbf{f}_* \end{pmatrix} \sim \mathcal{N} \left( \begin{pmatrix} \mathbf{m} \\ \mathbf{m}_* \end{pmatrix}, \begin{pmatrix} K(X, X) + \sigma_n^2 I & K(X, X_*) \\ K(X_*, X) & K(X_*, X_*) \end{pmatrix} \right)
$$


让我们分解协方差矩阵分量：

- $K(X, X)$: $N \times N$ 矩阵，表示所有训练输入对之间的协方差（我们之前称其为 $K$）。
- $K(X, X_*)$: $N \times M$ 矩阵，表示训练输入与测试输入之间的协方差。 $K(X, X_*)_{ij} = k(\mathbf{x}_i, \mathbf{x}_{*j})$。
- $K(X_*, X)$: $M \times N$ 矩阵，表示测试输入与训练输入之间的协方差。请注意 $K(X_*, X) = K(X, X_*)^T$。
- $K(X_*, X_*)$: $M \times M$ 矩阵，表示所有测试输入对之间的协方差。 $K(X_*, X_*)_{ij} = k(\mathbf{x}_{*i}, \mathbf{x}_{*j})$。

以及均值向量：

- $\mathbf{m}$: $N \times 1$ 向量，表示训练输入处的先验均值。
- $\mathbf{m}_*$: $M \times 1$ 向量，表示测试输入处的先验均值。

## 预测分布

我们关注的是后验预测分布 $p(\mathbf{f}_* | X, \mathbf{y}, X_*)$，它给出了在给定观测训练数据条件下，测试点处函数值的分布。由于联合分布是高斯分布，因此该条件分布也是高斯分布。

运用条件高斯分布的标准结果，我们可以推导该预测分布的均值和协方差。为求简便，首先假设先验均值函数为零 ($m(\mathbf{x}) = 0$，因此 $\mathbf{m}=\mathbf{0}$ 且 $\mathbf{m}_*=\mathbf{0}$):

后验预测均值是：


$$
\mathbb{E}[\mathbf{f}_* | X, \mathbf{y}, X_*] = K(X_*, X) [K(X, X) + \sigma_n^2 I]^{-1} \mathbf{y}
$$


后验预测协方差是：


$$
\text{Cov}[\mathbf{f}_* | X, \mathbf{y}, X_*] = K(X_*, X_*) - K(X_*, X) [K(X, X) + \sigma_n^2 I]^{-1} K(X, X_*)
$$


如果我们使用非零先验均值函数 $m(\mathbf{x})$，这些方程会变得更具一般性：

**后验预测均值：**


$$
\mathbb{E}[\mathbf{f}_* | X, \mathbf{y}, X_*] = \mathbf{m}_* + K(X_*, X) [K(X, X) + \sigma_n^2 I]^{-1} (\mathbf{y} - \mathbf{m})
$$


**后验预测协方差：**


$$
\text{Cov}[\mathbf{f}_* | X, \mathbf{y}, X_*] = K(X_*, X_*) - K(X_*, X) [K(X, X) + \sigma_n^2 I]^{-1} K(X, X_*)
$$


请注意，后验协方差不依赖于观测目标 $\mathbf{y}$，仅依赖于输入 $X$、$X_*$、核函数 $k$ 和噪声方差 $\sigma_n^2$。

### 预测方程的解释

- **预测均值：** 预测均值 $\mathbb{E}[\mathbf{f}_* | \dots]$ 以先验均值 $\mathbf{m}_*$ 开始，并添加一个校正项。该校正项是中心化观测目标 $(\mathbf{y} - \mathbf{m})$ 的线性组合。该组合的权重 (weight)取决于测试点 $X_*$ 与训练点 $X$ 之间的协方差，并通过训练数据的带噪声协方差矩阵的逆进行调整。直观上，训练点中“更接近”（由核函数 $k$ 定义）测试点 $\mathbf{x}_*$ 的点，对其预测值将有更强的影响。
- **预测协方差：** 预测协方差 $\text{Cov}[\mathbf{f}_* | \dots]$ 以先验协方差 $K(X_*, X_*)$ 开始，并减去一个半正定矩阵。这种减法表示从观测训练数据 $\mathbf{y}$ 中获得对 $\mathbf{f}_*$ 不确定性的降低。当训练点提供关于测试点的更多信息时（即当 $K(X_*, X)$ 较大时），不确定性的降低量就越大。该协方差矩阵的对角线元素 $\text{Var}[f_{*i} | \dots]$ 给出了每个单独测试点 $\mathbf{x}_{*i}$ 的预测方差。这种方差在训练数据点附近通常较小，而在远离训练数据点的地方较大，这反映了数据缺乏之处的不确定性增加。

能够获得的不仅是点预测（均值），还有完整的不确定性度量（方差/协方差），这是高斯过程回归的一个重要优点。



![高斯过程回归示例](plots/3592-0.json)



> 1D高斯过程回归示例。蓝色点是训练数据。红线是后验预测均值。红色阴影区域表示均值周围 +/- 两个标准差，显示了模型的不确定性。不确定性在训练点附近较低，而离训练点越远则越高。

## 计算方面的考虑

标准高斯过程回归的主要计算瓶颈是逆项 $[K(X, X) + \sigma_n^2 I]^{-1}$ 的计算。这涉及对 $N \times N$ 矩阵求逆，通常需要 $O(N^3)$ 的计算时间和 $O(N^2)$ 的存储内存来保存矩阵 $K$。

- **训练/推断：** 计算逆需要 $O(N^3)$ 时间。
- **预测：** 一旦计算并存储了逆（或其Cholesky分解），预测 $M$ 个测试点的均值需要 $O(NM)$ 时间，预测完整协方差需要 $O(NM^2)$ 时间。在最初的 $O(N^3)$ 投入之后，仅预测 $M$ 个测试点的方差需要 $O(NM)$ 时间。

这种 $O(N^3)$ 的复杂性限制了精确高斯过程回归对数据点通常少于几千个的数据集的适用性。我们将在本章后面讨论将高斯过程扩展到更大数据集的方法。

既然我们已经有了回归的主要公式，接下来的步骤包含理解如何选择核函数 $k(\mathbf{x}, \mathbf{x}')$ 以及如何设置其参数 (parameter)（超参数 (hyperparameter)）和噪声方差 $\sigma_n^2$。

## 参考资料

- [Gaussian Processes for Machine Learning](http://www.gaussianprocess.org/gpml/chapters/) — Carl Edward Rasmussen and Christopher K. I. Williams (2006)
  Publisher: The MIT Press
  关于高斯过程的权威性教科书，全面涵盖其理论、回归公式和实际应用。
- [Pattern Recognition and Machine Learning](https://link.springer.com/book/9780387310732) — Christopher M. Bishop (2006)
  Publisher: Springer; Pages: 738; DOI: [10.1007/978-0-387-31073-2](https://doi.org/10.1007/978-0-387-31073-2)
  一本广泛使用的机器学习教科书，在第6章中专门清晰地介绍了用于回归的高斯过程。
- [Surrogates: Gaussian Process Modeling, Design, and Optimization for Engineers](https://www.taylorfrancis.com/books/9780367815493) — Robert B. Gramacy (2020)
  Publisher: Chapman and Hall/CRC; Pages: 560; DOI: [https://doi.org/10.1201/9780367815493](https://doi.org/10.1201/9780367815493)
  提供了一个关于高斯过程建模的现代、以应用为中心视角，清晰阐述了回归公式及其实际方面。
