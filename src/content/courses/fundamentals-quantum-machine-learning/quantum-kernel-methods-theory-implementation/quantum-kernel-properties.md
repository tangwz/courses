---
course: "fundamentals-quantum-machine-learning"
chapter: "quantum-kernel-methods-theory-implementation"
lesson: "quantum-kernel-properties"
sourceId: 1049
sourceUrl: "https://apxml.com/zh/courses/fundamentals-quantum-machine-learning/chapter-3-quantum-kernel-methods-theory-implementation/quantum-kernel-properties"
title: "量子核的性质"
description: "分析量子核的数学属性，包括正半定性及潜在普适性。"
order: 3
plots: []
sourceHash: "2e6164de44fee924dc2f30ff85ad859ac1b31969c796474d2fb445992c24c105"
sourceCorrections: []
---

已经明确了量子核 $k(x, x')$ 是如何在量子特征空间中由内积产生的，我们来审视它们的基本数学属性。正如它们的经典对应物一样，理解这些特性对于在支持向量 (vector)机（SVM）等机器学习 (machine learning)算法中有效应用量子核非常重要。这些性质决定了核函数的行为，并影响了所生成的特征空间的几何结构，进而影响模型性能和可训练性。

> 经典数据、量子特征态、它们的内积以及最终核值之间的关系。

### 正半定性 (PSD)

任何函数要在支持向量 (vector)机（SVM）及其他核方法中成为有效核的一个核心要求是它必须是正半定（PSD）的。此性质确保了由在任意数据集上评估核函数所形成的矩阵，即格拉姆矩阵 $K$，具有非负特征值。形式上，对于任意数据集 $\{x_1, ..., x_N\}$ 和任意复数系数 $\{c_1, ..., c_N\}$，以下条件必须成立：


$$
\sum_{i=1}^{N} \sum_{j=1}^{N} \bar{c}_i K_{ij} c_j \ge 0
$$


其中 $K_{ij} = k(x_i, x_j)$。此条件与存在一个潜在的特征映射 $\phi$ 使得 $k(x, x') = \langle \phi(x) | \phi(x') \rangle$ 密切相关。

我们来验证一下量子核的此点。如果我们将核直接定义为内积：
$k(x, x') = \langle \phi(x) | \phi(x') \rangle$
其中 $|\phi(x)\rangle$ 是与数据点 $x$ 对应的量子态。那么，此和变为：


$$
\sum_{i,j} \bar{c}_i \langle \phi(x_i) | \phi(x_j) \rangle c_j = \left\langle \sum_i c_i \phi(x_i) \;\middle|\; \sum_j c_j \phi(x_j) \right\rangle = \left\| \sum_i c_i |\phi(x_i)\rangle \right\|^2
$$


由于希尔伯特空间中任何向量的范数平方都是非负的，因此该条件得到满足。所以，直接定义为希尔伯特空间中内积的核总是正半定的。

在量子机器学习 (machine learning)中，核通常根据特征态之间的*测量概率*或*保真度*来定义，其常见形式为：
$k(x, x') = |\langle \phi(x) | \phi(x') \rangle|^2$

这个函数仍然是正半定（PSD）的吗？是的。考虑格拉姆矩阵 $G$，其中 $G_{ij} = \langle \phi(x_i) | \phi(x_j) \rangle$。我们知道 $G$ 是正半定的。我们感兴趣的核矩阵 $K$ 的元素是 $K_{ij} = |\langle \phi(x_i) | \phi(x_j) \rangle|^2 = G_{ij} \overline{G_{ij}}$。这可以看作是 $G$ 及其复共轭 $\bar{G}$ 的元素乘积（阿达玛积）。由于 $G$ 是正半定的，$\bar{G}$ 也是正半定的。舒尔乘积定理指出，两个正半定矩阵的阿达玛积也是正半定的。因此，定义为 $k(x, x') = |\langle \phi(x) | \phi(x') \rangle|^2$ 的量子核保证是有效且正半定的核。

这种正半定（PSD）性质很重要，因为它保证了支持向量机（SVM）底层的优化问题保持凸性，从而确保我们能够找到唯一的全局最小值。

### 归一化 (normalization)与有界性

量子核，特别是那些定义为 $k(x, x') = |\langle \phi(x) | \phi(x') \rangle|^2$ 的核，通常表现出源于量子态性质的自然归一化特性。
假设量子特征态 $|\phi(x)\rangle$ 是归一化态（即 $\langle \phi(x) | \phi(x) \rangle = 1$），我们有：

- **对角元素：** $k(x, x) = |\langle \phi(x) | \phi(x) \rangle|^2 = 1^2 = 1$。一个数据点与自身的核值总是 1。
- **非对角元素：** 根据柯西-施瓦茨不等式， $|\langle \phi(x) | \phi(x') \rangle| \le \sqrt{\langle \phi(x) | \phi(x) \rangle} \sqrt{\langle \phi(x') | \phi(x') \rangle} = 1$。因此， $0 \le |\langle \phi(x) | \phi(x') \rangle|^2 \le 1$。

因此，这种常见的量子核形式自然会产生介于 0 和 1 之间的值，其中 1 表示相同的特征态（除了一个全局相位），而值越接近 0 表示特征态越不相似（正交）。这种有界性在实际应用中很有帮助，可以避免数值问题，并为相似性提供一个直观的衡量尺度。然而，输入态 $|\phi(x)\rangle$ 的归一化是先决条件，这在量子计算中是标准做法。

### 普适性

在经典机器学习 (machine learning)中，如果相应的再生核希尔伯特空间（RKHS）在输入域的连续函数空间中是稠密的，则称该核为*普适*核。实际中，这意味着具有普适核的支持向量 (vector)机在数据充足的情况下可以近似任何任意决策边界。高斯（RBF）核是普适经典核的一个著名例子。

量子核可以是普适的吗？答案完全取决于量子特征映射 $\phi$ 的表现力。

- 如果特征映射 $\phi$ 受限，并且只能在希尔伯特空间中生成有限的子集态，则生成的核可能不具有普适性。
- 如果特征映射 $\phi$ 足够强大或*富有表现力*，即它能近似任何幺正变换或生成密集覆盖希尔伯特空间的态，那么生成的核就有可能具有普适性。

研究表明，某些量子特征映射，特别是那些涉及充分纠缠和数据重上传结构（如第2章所述）的映射，确实可以产生普适核。证明普适性通常涉及分析函数 $x \mapsto |\phi(x)\rangle$ 的丰富性，并将其与傅里叶分析或函数近似理论关联起来。在核函数中生成高频分量的能力通常与普适性相关。实现普适性可能需要特征映射的参数 (parameter)数量随着数据集的大小或目标函数的复杂性而增长。

### 对特征映射选择的依赖性

需要反复强调的是：**量子核的性质和性能完全由量子特征映射 $\phi$ 的选择决定。** 用于将 $x$ 编码为 $|\phi(x)\rangle$ 的不同电路将导致不同的核。
考虑受特征映射设计影响的这些因素：

- **维度：** 所用量子比特的数量决定了希尔伯特空间的标称维度。
- **纠缠：** 电路生成的纠缠量和结构显著影响内积捕获的相关性。
- **非线性：** 从 $x$ 导出的参数 (parameter)如何加载到电路门中引入了非线性依赖，从而塑造了特征空间的几何结构。使用数据重上传的电路可以创建高度复杂的非线性特征映射。

因此，设计或选择合适的特征映射是量子核方法中的一项核心任务。一个在某个数据集上表现良好的特征映射可能在另一个数据集上表现不佳。这与第2章中关于编码策略及其表现力的主题密切相关。

### 计算考量与噪声

尽管在数学上很优美，但在量子硬件上实际估算量子核 $k(x, x') = |\langle \phi(x) | \phi(x') \rangle|^2$ 会带来一些复杂性：

- **估算：** 我们无法获得精确值，而是通过重复测量（例如，使用SWAP测试或反演测试电路）得到的估算值。这需要多次电路执行和测量，从而引入统计不确定性。
- **噪声：** 当前量子处理器上的退相干、门误差和读出误差会损坏计算，导致核矩阵元素的估算值存在偏差或噪声。
- **对性质的影响：** 噪声和统计误差意味着经验估算的核矩阵可能与理想的正半定性略有偏差。这可能需要缓解技术或在算法的经典优化部分进行仔细处理（例如，通过在估算的核矩阵的对角线上添加一个小的数值）。

这些与硬件噪声和估算相关的实际方面将在第7章中进行更详细的讨论。

总而言之，量子核通过希尔伯特空间中的内积定义，因此继承了基本的正半定（PSD）性质。它们的归一化 (normalization)取决于特征态的归一化，而其潜在的普适性则取决于所选量子特征映射的表现力。特征映射设计与核性质之间的紧密关联使得特征映射工程成为成功应用这些方法的一个重要方面。

## 参考资料

- [Supervised learning with quantum-enhanced feature spaces](https://www.nature.com/articles/s41586-019-0980-2) — Vojtěch Havlíček, Antonio D. Córcoles, Kristan Temme, Aram W. Harrow, Abhinav Kandala, Jerry M. Chow, and Jay M. Gambetta (2019)
  Journal: Nature; Volume: 567; Pages: 209-212; DOI: [10.1038/s41586-019-0980-2](https://doi.org/10.1038/s41586-019-0980-2)
  这篇基础性论文介绍了用于监督学习的量子增强特征空间和量子核概念，展示了它们在支持向量机中的应用。
- [Learning with Kernels: Support Vector Machines, Regularization, Optimization, and Beyond](https://mitpress.mit.edu/books/learning-kernels) — Bernhard Schölkopf, Alexander J. Smola (2002)
  Publisher: MIT Press
  这是一本关于经典核方法的开创性教科书，为支持向量机、正半定核和再生核希尔伯特空间提供了全面的数学基础。
- [A universal quantum kernel for quantum machine learning](https://iopscience.iop.org/article/10.1088/2058-9565/ac771d) — Basil Bichsel, Thomas B. Bäck, Tobias J. Osborne, and Moritz Weber (2022)
  Journal: Quantum Science and Technology; Publisher: IOP Publishing; Volume: 7; Pages: 035020; DOI: [10.1088/2058-9565/ac771d](https://doi.org/10.1088/2058-9565/ac771d)
  本文研究了量子核中普适性的条件，将量子特征映射的表达能力与它们逼近任意连续函数的能力联系起来。
