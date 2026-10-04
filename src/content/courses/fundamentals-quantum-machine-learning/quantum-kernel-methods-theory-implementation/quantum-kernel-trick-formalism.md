---
course: "fundamentals-quantum-machine-learning"
chapter: "quantum-kernel-methods-theory-implementation"
lesson: "quantum-kernel-trick-formalism"
sourceId: 1047
sourceUrl: "https://apxml.com/zh/courses/fundamentals-quantum-machine-learning/chapter-3-quantum-kernel-methods-theory-implementation/quantum-kernel-trick-formalism"
title: "量子核方法技巧的形式体系"
description: "量子核方法技巧的数学推导与阐述。"
order: 1
plots: []
sourceHash: "6b8cdb82edbaea49e33f6cf4f0b3563d5d2034f5cdffb38fc583cc12f1700754"
sourceCorrections: []
---

回顾经典机器学习 (machine learning)，尤其是支持向量 (vector)机（SVM）等算法，核方法技巧的效用显而易见。它让我们能隐式地进行操作，在高维特征空间中处理数据，而无需显式计算数据在该空间中的坐标。相反，我们只需一个函数，即核函数 $k(x, x')$，该函数计算特征空间中映射数据点 $\phi(x)$ 和 $\phi(x')$ 之间的内积。

量子机器学习运用了相似的思路，建立在之前介绍的量子特征映射之上。量子特征映射 $U_\phi(x)$ 将经典数据点 $x$ 编码为一个量子态 $|\phi(x)\rangle = U_\phi(x)|0...0\rangle$，存在于希尔伯特空间 $\mathcal{H}$ 中。这个希尔伯特空间的大小可以随量子比特数呈指数增长，作为我们可能非常高维的特征空间。

**量子核** $k(x, x')$ 被定义为两个量子特征态 $|\phi(x)\rangle$ 和 $|\phi(x')\rangle$ 之间内积的一个函数。一个常见且实用的定义是：


$$
k(x, x') = |\langle\phi(x)|\phi(x')\rangle|^2
$$


为什么要用平方幅值？稍后我们会看到，这种特定形式与通过测量量子电路可以估算的概率直接相关，使其适用于近期量子硬件。内积的其他函数形式也是可行的，但这种形式较为普遍。

使用幺正算符 $U_\phi$ 替换特征态的定义：


$$
k(x, x') = |\langle 0...0 | U_\phi(x)^\dagger U_\phi(x') | 0...0 \rangle|^2
$$


这个等式展示了**量子核方法技巧**的核心：

1. **隐式映射：** 我们将 $x \mapsto |\phi(x)\rangle$ 定义为一个映射，进入一个可能非常庞大的量子希尔伯特空间 $\mathcal{H}$。
2. **内积的直接计算：** 我们计算核函数 $k(x, x')$ *无需*显式构造完整的态向量 $|\phi(x)\rangle$ 和 $|\phi(x')\rangle$，因为这可能需要指数级的经典计算资源。相反，我们设计一个量子电路，其测量结果统计数据使我们能够估算所需的内积（或其平方幅值）。

### 计算方式

主要思路是在量子电路中准备态 $|\phi(x)\rangle$ 和 $|\phi(x')\rangle$（或与之相关的态），然后执行操作和测量，以获得它们的重叠度 $\langle\phi(x)|\phi(x')\rangle$。

考虑幺正变换 $V = U_\phi(x)^\dagger U_\phi(x')$。我们需要的内积是此幺正算符在初始态 $|0...0\rangle$ 中的期望值：


$$
\langle\phi(x)|\phi(x')\rangle = \langle 0...0 | V | 0...0 \rangle
$$


可以设计电路来估算诸如 $|\langle 0...0 | V | 0...0 \rangle|^2$ 的量。例如，“交换测试”电路或相关的基于干涉的电路可以测量两个态之间的保真度，这对应于它们内积的平方幅值。我们将 $U_\phi(x')$ 应用于一个初始化为 $|0...0\rangle$ 的寄存器，将 $U_\phi(x)$ 应用于另一个初始化为 $|0...0\rangle$ 的寄存器，然后使用辅助量子比特和受控操作来使这些态发生干涉。测量辅助量子比特可提供关于 $|\langle\phi(x)|\phi(x')\rangle|^2$ 的信息。

> 量子核方法技巧的流程图。经典数据点 $x$ 和 $x'$ 通过特征映射幺正算符 $U_\phi$ 隐式映射到量子态 $|\phi(x)\rangle$ 和 $|\phi(x')\rangle$。一个量子电路直接估算它们的重叠度，得到核值 $k(x, x')$，从而避免了在高维希尔伯特空间 $\mathcal{H}$ 中的显式表示。

### 与经典核方法的关系

这种形式体系的优势在于它与经典核机器的匹配性。一旦我们能计算出量子核矩阵 $K$（其中对于数据集 $\{x_1, ..., x_N\}$，$K_{ij} = k(x_i, x_j)$），我们就可以将此矩阵直接输入到经典算法中，例如：

- 支持向量 (vector)机 (SVM)
- 核岭回归
- 高斯过程
- 核主成分分析 (KPCA)

经典算法仅基于核矩阵 $K$ 进行优化或分析，实际是在量子特征空间 $\mathcal{H}$ 中进行处理，而无需直接访问该空间。假设是，量子特征映射 $U_\phi(x)$ 可能会在 $\mathcal{H}$ 中生成经典特征映射难以实现的关联或结构，这可能在某些数据集上带来更好的性能。

### 数学特性

为了使 $k(x, x')$ 对许多经典算法（如 SVM）成为一个有效的核函数，生成的核矩阵 $K$ 必须是**半正定 (PSD)** 的。这意味着对于任意向量 (vector) $c$，二次型 $c^T K c \ge 0$。

幸运的是，常用的量子核定义 $k(x, x') = |\langle\phi(x)|\phi(x')\rangle|^2$ 通常会生成一个 PSD 核矩阵。令 $G$ 为格拉姆矩阵，其元素为 $G_{ij} = \langle\phi(x_i)|\phi(x_j)\rangle$。格拉姆矩阵总是 PSD 的。我们定义的量子核矩阵 $K$ 的元素为 $K_{ij} = |G_{ij}|^2$。这个矩阵 $K$ 是 $G$ 及其复共轭 $G^*$ 的哈达玛乘积（元素级乘积）。根据舒尔乘积定理，两个 PSD 矩阵的哈达玛乘积也是 PSD 的。由于 $G$ 和 $G^*$ 都是 PSD 的，它们的哈达玛乘积 $K$ 也是 PSD 的。

这使得我们可以方便地在标准核方法框架中使用这些量子核。下一节将说明如何利用量子电路和模拟器来实际计算这些核矩阵的元素。

## 参考资料

- [Quantum Kernels in Qiskit Machine Learning](https://qiskit.org/ecosystem/machine-learning/tutorials/02_quantum_kernel_methods.html) — IBM Quantum (2024)
  Publisher: IBM Quantum
  Qiskit机器学习的官方文档和教程，演示了如何使用该库实际实现量子核方法。
