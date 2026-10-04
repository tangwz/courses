---
course: "fundamentals-quantum-machine-learning"
chapter: "quantum-kernel-methods-theory-implementation"
lesson: "qsvm-implementation-complex-datasets"
sourceId: 1053
sourceUrl: "https://apxml.com/zh/courses/fundamentals-quantum-machine-learning/chapter-3-quantum-kernel-methods-theory-implementation/qsvm-implementation-complex-datasets"
title: "QSVM在复杂数据集上的实施"
description: "QSVM在复杂数据集上应对挑战性分类任务的实践方面。"
order: 7
plots: []
sourceHash: "748f3b3f547f390b61963d2cb28d25c3b34fcb752674bb56ce8ac6bcadbca75c"
sourceCorrections: []
---

在确定了量子核和量子支持向量 (vector)机（QSVM）的理论基础之后，我们来考察QSVM在那些带来显著挑战的数据集上的实际应用。这些“复杂数据集”可能表现为高维度、复杂的非线性可分性要求，或者仅仅是大量数据点，这对经典方法和当前量子能力的限制都构成了考验。

其核心理念不变：运用量子特征映射 $\phi(x)$ 将经典数据 $x \in \mathcal{X}$ 隐式地映射到量子希尔伯特空间 $\mathcal{H}$ 中，然后在标准的SVM框架内使用核函数 $k(x, x') = |\langle\phi(x)|\phi(x')\rangle|^2$ (或相关的保真度度量)。量子计算机的作用是估算训练数据点 $\{x_i\}$ 的核矩阵 $K_{ij} = k(x_i, x_j)$ 中的各个元素。

### QSVM实施的工作流程

实施QSVM来处理复杂的分类任务通常遵循以下步骤：

1. **数据准备：** 这与经典机器学习 (machine learning)的流程相似。复杂数据集通常需要细致的预处理。

   - **缩放：** 特征通常应进行缩放（例如，缩放到 $[0, \pi]$ 或 $[-1, 1]$），以与特征映射中使用的参数 (parameter)化量子门（如旋转门）的输入范围保持一致。像scikit-learn等库中的`MinMaxScaler`或`StandardScaler`等标准技术很常见。
   - **降维（可选）：** 尽管量子特征映射可以处理高维度数据，但像主成分分析（PCA）这样的经典技术有时可以在量子编码*之前*使用，特别是当内在维度低于表观维度时，或者为了管理量子比特需求。然而，应用经典PCA可能会恰好去除量子核可能可以发现的细微关联。这个决定需要根据具体问题进行仔细考量。
2. **量子特征映射的选择：** 这是主要的设计选择。对于复杂数据，简单的特征映射（如深度为1的`ZZFeatureMap`）可能缺乏捕获复杂关系所需的表达能力。

   - 请参阅第二章，了解高级编码策略，如高阶多项式映射或数据重上传电路。
   - 考虑权衡：更复杂的特征映射（更深的电路，更多的纠缠）可能可以模拟更复杂的函数，但更难训练（贫瘠高原，第4.7节），对噪声更敏感（第七章），并且需要更多的量子资源。
   - 理想情况下，选择应由对数据结构的一些先验知识或通过实验来指导。
3. **核矩阵计算：** 这是量子步骤。对于大小为 $N$ 的训练集，我们需要计算 $N \times N$ 的核矩阵 $K$。每个元素 $K_{ij} = k(x_i, x_j)$ 都需要执行一个量子电路。

   - **电路设计：** 标准方法包括准备 $|\phi(x_i)\rangle$，应用准备 $|\phi(x_j)\rangle$ 的电路的逆（即 $\phi(x_j)^\dagger$），并测量全零态 $|0\dots0\rangle$ 的概率。这个概率正是 $|\langle\phi(x_j)|\phi(x_i)\rangle|^2$。对于基于保真度的核函数，例如 $(\text{Tr} \sqrt{\sqrt{\rho(x_i)} \rho(x_j) \sqrt{\rho(x_i)}})^2$，其中 $\rho(x)$ 为对应于 $\phi(x)$ 的状态，可能需要更复杂的电路，如SWAP测试或其变体，尤其是在处理因噪声引起的混合态时。
   - **估计：** 在模拟器上，内积通常可以直接从状态向量 (vector)计算。在真实硬件（或带噪声的模拟器）上，概率必须通过重复测量（shots）来估计。测量次数决定了核估计的精度，并直接影响运行时间。估计 $N^2$ 个元素（或由于对称性为 $N(N+1)/2$ 个）可能非常耗时。
   - **可扩展性难题：** 核矩阵计算的 $O(N^2)$ 成本是处理大型数据集的主要瓶颈。如果 $N$ 达到数千或更多，在当前的量子设备甚至模拟器上计算完整的核矩阵将变得计算上难以承受。应对方法可能包括：
     - 仅将QSVM应用于 $N$ 相对较小但数据在其他方面复杂（例如，高维度）的问题。
     - 使用随机子采样（Nyström方法）来近似核矩阵。
     - 考察可能绕过显式核矩阵计算的替代量子算法，尽管QSVM传统上依赖于它。
4. **经典SVM训练：** 一旦核矩阵 $K$ 被计算（或近似），它就被输入到经典的SVM求解器中。

   - 许多标准ML库（如scikit-learn的`SVC`）允许使用预计算的核。您只需使用`kernel='precomputed'`选项传递 $N \times N$ 矩阵 $K$。
   - 然后，SVM优化问题（寻找支持向量和权重 (weight)）会根据 $K$ 和训练标签 $y_i$ 以经典方式求解。

```python
# QSVM的Python风格伪代码

# 假设：
# - X_train: N个训练样本的(N, d)数组，d个特征（已缩放）
# - y_train: 训练标签的(N,)数组
# - quantum_feature_map(data_point): 返回量子电路对象的函数
#                                   （例如，Qiskit QuantumCircuit，Pennylane QNode）
#                                   用于实现 phi(data_point)|0...0>
# - estimate_kernel_entry(qc_i, qc_j, backend, shots): 估算函数
#                                                     使用电路估算 k(x_i, x_j)
#                                                     来自 quantum_feature_map

import numpy as np
# 假设存在一个经典的SVM实现，例如来自sklearn.svm的SVC

N = X_train.shape[0]
kernel_matrix = np.zeros((N, N))

print("生成特征映射电路...")
# 如果特征映射结构是固定的，则预生成电路
feature_map_circuits = [quantum_feature_map(x) for x in X_train]
# 注意：对于某些核估计方法，您会组合电路 i 和 j

print("计算量子核矩阵...")
# 此循环是计算密集型部分
for i in range(N):
    for j in range(i, N): # 利用对称性 K_ij = K_ji
        # 具体实现取决于所选的核估计方法
        # 例如，|<phi(xi)|phi(xj)>|^2 通常通过电路求逆和测量来计算
        kernel_value = estimate_kernel_entry(feature_map_circuits[i],
                                             feature_map_circuits[j],
                                             quantum_backend, # 模拟器或真实设备
                                             num_shots)
        kernel_matrix[i, j] = kernel_value
        kernel_matrix[j, i] = kernel_value # 赋值对称元素

print("使用量子核训练经典SVM...")
# 使用带有预计算核的经典SVM库
svm_classifier = SVC(kernel='precomputed')
svm_classifier.fit(kernel_matrix, y_train)

print("QSVM训练完成。")

# 对于新数据 X_new（M个样本）的预测：
# 1. 计算 M x N 核矩阵 K_new，其中 K_new[k, i] = k(x_new_k, x_train_i)
# 2. 使用 svm_classifier.predict(K_new)
```

### 实践中处理复杂性

- **非线性可分性：** QSVM处理复杂非线性问题的能力完全在于量子特征映射 $\phi(x)$ 的表达能力。如果特征映射将数据转换到一个可以线性可分化的空间中，经典SVM部分将取得成功。尝试 $\phi(x)$ 的不同Ansatz（电路结构和门参数 (parameter)）十分重要。
- **高维度和集中效应：** 正如第3.5节讨论的，从典型特征映射得出的量子核可能会受到集中效应的影响，尤其是在高维度情况下。核矩阵元素可能变得指数级接近一个常数，使得SVM难以区分数据点。这可能抵消高维希尔伯特空间的潜在益处。缓解策略可能包括：
  - 设计专门旨在避免集中效应的特征映射（这是一个活跃的研究方面）。
  - 仔细的数据缩放和预处理。
  - 使用局部核，只让邻近点有显著贡献。
- **噪声：** 在NISQ硬件上运行时，噪声会明显影响核估计的准确性。每个 $K_{ij}$ 都会是一个受门错误、退相干和读出错误影响的估计。第七章讨论了误差缓解技术（如零噪声外推或概率误差消除），这些技术可以在 `estimate_kernel_entry` 步骤中应用，以在将计算出的核矩阵输入SVM之前提高其质量。

### 复杂数据集上的评估

在复杂数据集上评估QSVM需要细致的基准测试：

- **基准：** 始终与强大的经典基准进行比较，包括使用优化标准核（例如RBF、多项式）的经典SVM。QSVM是否提供性能提升（准确性、F1分数、AUC）？
- **统计显著性：** 由于核估计中的采样噪声（尤其是在硬件上），结果可能会波动。使用不同的随机种子多次运行实验，并报告平均性能和标准差或置信区间。
- **资源成本：** 在考虑预测准确性的同时，也要考虑所需的计算资源（模拟时间、量子电路执行次数、每次估计的测量次数、量子比特数量、电路深度）。即使准确性只有小幅提升，也可能无法证明计算成本的大幅增加是合理的。

“实施QSVM处理复杂的大规模数据集仍然具有挑战性，主要原因在于核矩阵计算瓶颈以及噪声和核集中效应的影响。然而，它提供了一个框架，量子特征空间可以直接融入到像SVM这样成熟的经典算法中，为考察特定机器学习 (machine learning)任务中潜在的量子优势提供了一个试验平台，这些任务的特点是数据结构复杂。目前的应用通常侧重于 $N$ 可控（数百到数千）但其底层结构可能受益于量子特征表示的数据集。”

## 参考资料

- [Supervised learning with quantum-enhanced feature spaces](https://doi.org/10.1038/s41586-019-0980-2) — Vojtěch Havlíček, Antonio D. Córcoles, Kristan Keisler, Bram W. Sheldon, Alexis Williams, Jonathan M. Chow, Jerry M. Gambetta (2019)
  Journal: Nature; Volume: 567; Pages: 209–212; DOI: [10.1038/s41586-019-0980-2](https://doi.org/10.1038/s41586-019-0980-2)
  这项开创性工作引入了用于经典数据的量子特征映射概念，并展示了在量子处理器上实现量子支持向量机（QSVM），为所讨论的量子核方法奠定了基础。
- [Supervised Learning with Quantum Computers](https://doi.org/10.1007/978-3-319-96424-9) — Maria Schuld, Francesco Petruccione (2018)
  Publisher: Springer; DOI: [10.1007/978-3-319-96424-9](https://doi.org/10.1007/978-3-319-96424-9)
  这本书对量子机器学习进行了全面的学术呈现，其中包括关于量子核方法、特征映射以及量子支持向量机（QSVM）的理论和实践考量的详细章节。
- [Quantum Kernel Methods](https://learn.qiskit.org/course/machine-learning/quantum-feature-maps-kernels) — Qiskit Community and Contributors (2024)
  Qiskit机器学习模块的实践指南，展示了量子核和QSVM的实现，包括代码示例和特征映射选择以及核矩阵计算的说明。
- [The effect of data encoding on the expressibility and trainability of quantum machine learning models](https://quantum-journal.org/papers/q-2023-05-18-1007/) — Suppawong Thanasilp, Samson Wang, Nhat A. Nguyen, M. S. Ramchander, Alessandro Santagati, Victor V. Albert, M. Cerezo, Patrick J. Coles (2023)
  Journal: Quantum; Publisher: Quantum; Volume: 7; Pages: 1007; DOI: [10.22331/q-2023-05-18-1007](https://doi.org/10.22331/q-2023-05-18-1007)
  本文研究了量子特征映射中不同数据编码策略如何影响量子机器学习模型的表达能力和可训练性，直接讨论了核浓度等问题及其对模型性能的影响。
