---
course: "fundamentals-quantum-machine-learning"
chapter: "quantum-kernel-methods-theory-implementation"
lesson: "qsvm-quantum-kernels"
sourceId: 1052
sourceUrl: "https://apxml.com/zh/courses/fundamentals-quantum-machine-learning/chapter-3-quantum-kernel-methods-theory-implementation/qsvm-quantum-kernels"
title: "量子核支持向量机 (QSVM)"
description: "详细说明结合量子核的量子支持向量机 (QSVM) 算法。"
order: 6
plots: []
sourceHash: "b9ad10969b81ecb0f5f01492b12402cbcb60338c36fcfcfe3b054f1fcde4fd0c"
sourceCorrections: []
---

支持向量 (vector)机 (SVM) 是用于分类和回归任务的强大监督学习 (supervised learning)模型。SVM 的主要优势在于其使用了“核技巧”。SVM 不会显式地将数据映射到高维特征空间，而是依赖于一个核函数 $k(x, x')$，该函数计算数据点 $x$ 和 $x'$ 在该特征空间中的内积。这使得 SVM 能够高效地找到复杂的非线性决策边界。

传统 SVM 使用线性核、多项式核或径向基函数 (RBF) 核等。量子支持向量机 (QSVM) 的主要思想很直接：用通过量子特征映射计算得到的量子核来替换传统核函数。如本章前面所述，量子特征映射 $\phi$ 将经典数据 $x$ 编码为量子态 $|\phi(x)\rangle$，该量子态存在于一个可能非常大的希尔伯特空间 $\mathcal{H}$ 中。量子核随后根据这些量子态之间的相似性定义，通常是作为平方重叠（忠实度）：


$$
k_q(x, x') = |\langle\phi(x)|\phi(x')\rangle|^2
$$


有时也使用其他基于内积 $\langle\phi(x)|\phi(x')\rangle$ 的定义，以确保生成的核矩阵是半正定的，这是底层 SVM 优化问题的一个要求。

## 为什么使用量子核？

QSVM 的动机源于这样的假设：量子特征映射可以将经典数据映射到希尔伯特空间，在这些空间中数据变得更容易分离，这可能为某些数据集带来超越传统核的益处。量子特征空间 $\mathcal{H}$ 的结构，受特征映射电路 $U_\phi(x)$ 产生的纠缠和量子干涉影响，或许能够捕捉到传统核难以获取的数据中复杂的关联。如果量子特征映射提供了一种表示方式，使得线性分离器（在 $\mathcal{H}$ 中）的表现明显优于传统核在其各自特征空间中可达到的效果，那么 QSVM 可能会优于传统 SVM。

然而，证明这种优势并非易事，并且很大程度上取决于特征映射的选择和具体的应用场景。这里的量子优势潜力完全在于通过核访问的量子特征空间的 *表示能力*，而非加速 SVM 优化本身。

## QSVM 算法

QSVM 算法作为一种混合量子-经典流程运行：

1. **量子核计算：** 委托给量子计算机（或模拟器）的计算密集型部分是核矩阵 $K$ 的估计，其中每个条目 $K_{ij}$ 由训练数据点对 $(x_i, x_j)$ 的 $k_q(x_i, x_j)$ 给出。
2. **经典 SVM 优化：** 一旦量子核矩阵 $K$ 被估计出来，它将被输入到一个标准的传统 SVM 求解器中。

让我们进一步细化核计算步骤。为了估计单个核条目 $k_q(x_i, x_j) = |\langle\phi(x_i)|\phi(x_j)\rangle|^2$，我们需要一个量子电路来测量这些量子态之间的重叠或忠实度 $|\phi(x_i)\rangle = U_\phi(x_i)|0\rangle^{\otimes n}$ 和 $|\phi(x_j)\rangle = U_\phi(x_j)|0\rangle^{\otimes n}$。常用方法包括：

- **Hadamard 检验：** 使用一个辅助量子比特来估计内积 $\langle\phi(x_i)|\phi(x_j)\rangle$ 的实部和虚部。
- **SWAP 检验：** 采用受控 SWAP 门直接估计平方重叠 $|\langle\phi(x_i)|\phi(x_j)\rangle|^2$，尽管与其它方法相比，它在忠实度估计方面通常不那么实用。
- **计算-反计算方法（重叠检验）：** 一种常用方法是构建一个电路，该电路制备 $|\phi(x_i)\rangle$，然后应用 $x_j$ 的特征映射的逆运算 $U_\phi(x_j)^\dagger$，并测量返回全零态 $|0\rangle^{\otimes n}$ 的概率。这个概率恰好是 $|\langle 0|^{\otimes n} U_\phi(x_j)^\dagger U_\phi(x_i)|0\rangle^{\otimes n}|^2 = |\langle\phi(x_j)|\phi(x_i)\rangle|^2$。

> 训练量子支持向量 (vector)机 (QSVM) 的工作流程。量子资源用于估计核矩阵的条目，然后由传统 SVM 算法进行处理。预测也需要在新数据点和支持向量之间进行量子核评估。

每次估计都需要在量子计算机上进行多次测量（shots），以获得概率的统计可靠估计。对于大小为 $N$ 的训练集，计算完整的 $N \times N$ 核矩阵需要对量子核估计子程序进行 $O(N^2)$ 次调用。

## 数学公式回顾

回顾用于二元分类的经典软间隔 SVM 的对偶公式：


$$
\max_{\alpha} \sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j k(x_i, x_j)
$$


约束条件：


$$
\sum_{i=1}^N \alpha_i y_i = 0
$$


$$
0 \le \alpha_i \le C, \quad \text{对于 } i=1, \dots, N
$$


在这里，$y_i \in \{-1, +1\}$ 是类别标签，$\alpha_i$ 是拉格朗日乘子（仅对支持向量 (vector)非零），$C$ 是正则化 (regularization)参数 (parameter)，$k(x_i, x_j)$ 是核函数。

在 QSVM 中，我们只需将 $k(x_i, x_j)$ 替换为使用上述量子电路方法之一估计得到的量子核 $k_q(x_i, x_j)$。该优化问题仍然是一个凸二次规划问题，可以使用标准数值优化库在经典计算机上求解。

一旦找到最优的 $\alpha_i^*$ 和偏置 (bias) $b^*$，对新数据点 $x_{new}$ 的预测由以下公式给出：


$$
y_{pred} = \text{sign} \left( \sum_{i=1}^N \alpha_i^* y_i k_q(x_i, x_{new}) + b^* \right)
$$


请注意，预测也需要计算量子核条目，具体来说是在新数据点 $x_{new}$ 与所有支持向量 $x_i$（那些 $\alpha_i^* > 0$ 的向量）之间。

## 实施考量

- **特征映射选择：** QSVM 的成功关键取决于设计一个有效的量子特征映射 $U_\phi(x)$。选择不当的映射可能导致特征空间中的数据不如使用传统核时那样容易分离，或者可能遇到核集中等问题（在上一节中讨论过）。
- **核估计精度：** 由于量子测量是概率性的，$k_q(x_i, x_j)$ 是以有限精度估计的，这取决于测量次数。实际量子硬件上的测量噪声和门错误会进一步降低这种精度。核矩阵中的不准确性对 SVM 解决方案的影响需要仔细考量。
- **计算成本：** 估计核矩阵的 $N^2$ 个条目可能要求很高，尤其是在每个条目需要大量测量以获得足够精度的情况下。这通常在近期将 QSVM 的应用限制在相对较小的数据集上。
- **经典基准：** 比较 QSVM 性能与经过良好调优的、使用标准核（如 RBF）的传统 SVM 是必要的，以评估量子方法是否能为特定问题带来任何实际益处。

QSVM 代表了一种将量子特征空间集成到成熟机器学习 (machine learning)框架中的方式。尽管实际优势仍是活跃的研究领域，但理解其机制有助于了解量子核方法的潜力和挑战。下一节将详细介绍将 QSVM 应用于复杂数据集时的实施和挑战。

## 参考资料

- [Supervised learning with quantum-enhanced feature spaces](https://www.nature.com/articles/s41586-019-0980-2) — Vojtech Havlíček, Antonio D. Córcoles, Kristan Keisler, Bradley Neeley, Pietro Bettelli, Matthew Cross, Andrew W. Cross, Jacob M. Gambetta, Jerry M. Chow, Jay M. Gambetta (2019)
  Journal: Nature; Volume: 567; Pages: 209-212; DOI: [10.1038/s41586-019-0980-2](https://doi.org/10.1038/s41586-019-0980-2)
  展示了使用量子增强特征空间进行分类的早期实验，显示了量子优势在表示学习中的前景。
- [Supervised Learning with Quantum Computers](https://doi.org/10.1007/978-3-319-96424-9) — Maria Schuld, Francesco Petruccione (2018)
  Publisher: Springer; DOI: [10.1007/978-3-319-96424-9](https://doi.org/10.1007/978-3-319-96424-9)
  一本权威教材，全面介绍了量子机器学习，其中有专门的章节讲解量子核方法和支持向量机。
- [A quantum-enhanced feature space for machine learning](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.040332) — Yunchao Liu, Srinivasan Arunachalam, and Kristan Temme (2021)
  Journal: PRX Quantum; Publisher: American Physical Society; Volume: 2; Pages: 040332; DOI: [10.1103/PRXQuantum.2.040332](https://doi.org/10.1103/PRXQuantum.2.040332)
  讨论了量子特征映射的理论基础及其表示能力，这对于理解量子核方法的潜在益处和局限性至关重要。
- [Quantum Kernel Training](https://qiskit.org/ecosystem/machine-learning/tutorials/01_quantum_kernel_training.html) — IBM Quantum (2025)
  官方Qiskit机器学习文档，提供了实现包括QSVM在内的量子核算法的实践示例和说明。
