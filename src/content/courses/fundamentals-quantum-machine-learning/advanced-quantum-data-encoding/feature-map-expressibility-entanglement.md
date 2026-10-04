---
course: "fundamentals-quantum-machine-learning"
chapter: "advanced-quantum-data-encoding"
lesson: "feature-map-expressibility-entanglement"
sourceId: 1043
sourceUrl: "https://apxml.com/zh/courses/fundamentals-quantum-machine-learning/chapter-2-advanced-quantum-data-encoding/feature-map-expressibility-entanglement"
title: "特征映射表达能力与纠缠能力的分析"
description: "量化不同量子特征映射构造的表达能力和纠缠能力的方法。"
order: 4
plots: []
sourceHash: "ec3c03c0f586e2715f6563650f72f634205a3e305f8f62d003d2a5e3396f9fd0"
sourceCorrections: []
---

量子特征映射，$\phi: x \mapsto |\phi(x)\rangle$，将经典数据转换为量子态。鉴于存在多种构建方法，一个关键问题是这些映射的效果如何？并非所有编码策略都具有同等效力。它们将经典数据转换为具有有益特性的量子态的能力直接影响后续量子机器学习 (machine learning)算法的性能。两个基本属性决定了这种有效性：表达能力和纠缠能力。明晰并量化 (quantization)这些方面，使我们能够设计出更适合特定问题和硬件的特征映射。

### 理解表达能力

表达能力指的是，在输入数据 $x$ 变化时，量子特征映射能在希尔伯特空间中生成不同量子态的程度。可以将其视为特征空间的“覆盖范围”。一个高表达能力的特征映射，能够生成占据态空间中大体积或复杂流形的量子态，而低表达能力的映射，则可能将所有编码后的数据点限制在一个小而简单的区域内。

这为何重要？特征映射的表达能力根本上限制了后续量子模型（例如量子核机器或变分分类器）所能表示的函数复杂度。如果特征映射未能充分分离或组织希尔伯特空间中的数据点，即使是基于它的最强大的量子算法，也可能无法找到合适的决策边界，或有效刻画潜在的数据分布。

#### 量化 (quantization)表达能力

量化表达能力是一个活跃的研究方向，但存在多种方法。一个常见的思路是比较由特征映射生成的态分布 $\{|\phi(x)\rangle\}$ （针对一组代表性输入 $x$）与希尔伯特空间上的均匀分布，后者通常通过哈尔测度近似。

在数值上，这可以包含：

1. **采样：** 通过将样本数据点 $x_i$ 输入特征映射电路 $\mathcal{U}_{\phi(x)}$ 来生成多个态 $|\phi(x_i)\rangle$。
2. **态层析成像（受限）：** 对于小型系统，可以尝试重建密度矩阵 $\rho(x_i) = |\phi(x_i)\rangle\langle\phi(x_i)|$。
3. **比较分布：** 计算采样态分布的矩（例如平均纯度或高阶矩），并将其与哈尔随机分布的预期矩进行比较。显著偏差表明表达能力有限。

另一种方法涉及分析特征映射电路中门产生的代数结构，将表达能力与李代数和控制理论中的原理联系起来。高表达能力通常与特征映射电路生成一个跨越整个算符空间的李代数相关联（例如 $n$ 个量子比特的 $\mathfrak{su}(2^n)$）。

然而，最大表达能力并非总是理想的。高表达能力的电路可能难以训练，可能导致诸如贫瘠高原之类的问题，我们将在第4章讨论这些问题。表达能力和可训练性之间常常存在权衡。

### 评估纠缠能力

纠缠是子系统之间独特的量子关联，常被认为是量子机器学习 (machine learning)中潜在量子优势的一个来源。特征映射的纠缠能力衡量其生成纠缠态 $|\phi(x)\rangle$ 的倾向。例如，仅由单量子比特旋转组成的特征映射将具有零纠缠能力，因为它无法在量子比特之间产生纠缠。

生成纠缠的能力很重要，因为它使得量子态能够以经典高维难以处理的方式编码输入特征之间的关联。这对于那些复杂特征交互很重要的数据集来说，可能特别有利。

#### 测量纠缠

有多种量度可以量化 (quantization)特征映射生成的纠缠。一个实用方法通常涉及计算输出态 $|\phi(x)\rangle$ 的纠缠量，并将其在数据集 $\{x_i\}$ 上求平均。纯态的常见纠缠量度包括：

- **纠缠熵：** 对于量子比特的二分（例如，集合 A 和集合 B），计算一个子系统的约化密度矩阵的冯·诺依曼熵：$S(\rho_A) = -\text{Tr}(\rho_A \log_2 \rho_A)$，其中 $\rho_A = \text{Tr}_B(|\phi(x)\rangle\langle\phi(x)|)$。将 $S(\rho_A)$ 在不同输入 $x$ 上平均，可以感知该特定二分下的典型纠缠程度。
- **迈耶-瓦拉赫量度：** 此量度 $Q(|\psi\rangle)$ 量化了每个量子比特与其余量子比特之间的平均纠缠。它定义为 $Q(|\psi\rangle) = \frac{1}{n} \sum_{k=1}^n L_k$，其中 $L_k$ 是线性熵 $L_k = 2(1 - \text{Tr}(\rho_k^2))$，$\rho_k$ 是第 $k$ 个量子比特的约化密度矩阵。$Q$ 的范围从 0（乘积态）到 1（最大纠缠态平均）。

纠缠能力与特征映射电路中多量子比特门（如 CNOT 或 CZ）的存在和放置位置直接相关。使用更多纠缠门或以特定模式排列它们的映射，通常表现出更高的纠缠能力。

> 一个简单的2量子比特特征映射电路。单量子比特旋转 $R_x, R_y, R_z$ 编码数据 $x = (x_0, x_1)$ 以及可能的训练参数 (parameter) $p_0, p_1$。CNOT门（CX）负责产生纠缠。纠缠能力取决于输入 $x$ 和参数 $p$。

### 表达能力与纠缠的关系

尽管有区别，表达能力和纠缠能力常常交织在一起。能产生显著纠缠的特征映射通常会覆盖希尔伯特空间中更广泛的区域，从而提高表达能力。缺乏纠缠门的电路只能生成乘积态，这严重限制了其表达能力，使其局限于完整希尔伯特空间的一个小流形。

然而，高纠缠能力并不自动保证对特定任务来说具备高表达能力。有可能生成高度纠缠但却聚集成团的态，这些态无法有效区分不同类别的数据。反之，即使纠缠有限，也可以实现一定程度的表达能力，尤其是在单量子比特态的分布方式上。

理想的平衡取决于具体问题。对于需要获取复杂、非局域关联的任务，优先考虑纠缠能力可能是有利的。对于其他任务，确保特征空间相关部分的充分覆盖（表达能力）可能更具价值，或许使用更简单、纠缠更少的结构来辅助可训练性。

### 实践考量

分析这些属性有助于指导特征映射电路的设计：

- **电路深度与结构：** 具有交替数据编码旋转层和纠缠块的更深电路，倾向于提高表达能力和纠缠能力，但也会增加对噪声的敏感性和训练难度。
- **门选择：** 纠缠门（例如 CNOT 与受控相位门）的类型及其放置位置会影响生成关联的特性。
- **数据依赖性：** 请记住，实际的态 $|\phi(x)\rangle$ 及其属性取决于输入数据 $x$。分析通常涉及对预期数据分布进行平均。

PennyLane 等量子计算库中的工具提供了计算与表达能力相关的值（例如，使用 `qml.expressibility` 函数，该函数通常依赖于将态矩与哈尔随机进行比较）以及纠缠（例如，`qml.vn_entropy`、`qml.purity`）的功能。这些使得对不同特征映射设计的经验评估成为可能。

理解所选特征映射的表达能力和纠缠能力并非仅仅是理论演练。它为您的量子机器学习 (machine learning)模型的潜在能力和局限性提供了洞察，指导架构选择并协助结果阐释，特别是在将量子方法与经典基准进行比较或诊断训练问题时。在下一章转向量 (vector)子核方法时，我们将看到特征空间的几何结构，由这些属性塑形，如何直接确定量子核。

## 参考资料

- [Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms](https://quantum-journal.org/papers/q-2019-06-17-159/) — S. Sim, P. D. Johnson, J. K. Freericks (2019)
  Journal: Quantum; Publisher: Verein zur Förderung des Open Access Publizierens in den Quantenwissenschaften; Volume: 3; Pages: 1-13 (Article 159); DOI: [10.22331/q-2019-06-17-159](https://doi.org/10.22331/q-2019-06-17-159)
  这篇基础论文介绍了表达能力和纠缠能力并对其进行了量化，为变分量子电路的设计提供了重要见解。
- [Quantum Machine Learning](https://link.springer.com/book/10.1007/978-3-030-83057-9) — Maria Schuld and Francesco Petruccione (2021)
  Publisher: Springer; DOI: [10.1007/978-3-030-83057-9](https://doi.org/10.1007/978-3-030-83057-9)
  一本全面的教材，涵盖了量子机器学习算法，包括对特征映射、核及其属性的详细讨论。
- [Barren plateaus in quantum neural networks](https://doi.org/10.1038/s41467-018-07090-4) — Jarrod R. McClean, Sergio Boixo, Vadim N. Smelyanskiy, Ryan Babbush, and Hartmut Neven (2018)
  Journal: Nature Communications; Volume: 9; Pages: 4812; DOI: [10.1038/s41467-018-07090-4](https://doi.org/10.1038/s41467-018-07090-4)
  这篇开创性论文指出了量子神经网络中的贫瘠高原现象，该现象限制了高表达能力量子电路的可训练性，这是特征映射设计的一个重要考量。
- [PennyLane Documentation: qml.expressibility](https://docs.pennylane.ai/en/stable/code/api/pennylane.expressibility.html) — Xanadu Quantum Technologies (2024)
  PennyLane官方文档中量化量子电路表达能力的函数说明，为经验评估提供了实用指导。
