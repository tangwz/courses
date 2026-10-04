---
course: "fundamentals-quantum-machine-learning"
chapter: "advanced-quantum-data-encoding"
lesson: "higher-order-quantum-feature-maps"
sourceId: 1041
sourceUrl: "https://apxml.com/zh/courses/fundamentals-quantum-machine-learning/chapter-2-advanced-quantum-data-encoding/higher-order-quantum-feature-maps"
title: "高阶多项式量子特征映射"
description: "考察基于高阶关联的特征映射，与多项式核相关。"
order: 2
plots: []
sourceHash: "25bac241f7d74a6fbd3756b14f8831ac12198fe352cc2711d16f125052a11e20"
sourceCorrections: []
---

基础量子特征映射，例如将数据特征编码为单比特旋转的那些，提供了一个起点，但它们通常对应于量子特征空间中的线性结构。经典机器学习 (machine learning)常得益于输入数据的非线性变换，使模型能够捕捉复杂关系和决策边界。例如，多项式核是支持向量 (vector)机 (SVM) 等算法实现此目的的标准技术。它们隐式地将数据映射到更高维空间 (high-dimensional space)，其中特征对应于原始输入的各种多项式组合，而无需显式计算这些高维向量。

量子计算提供了一种构建特征映射的方式，可以自然地生成等效的或可能更强大的非线性关系，包括输入特征之间的高阶多项式关联。核心思想是设计量子电路 $U(x)$，将初始态 $|0\rangle^{\otimes n}$ 转换为特征态 $|\phi(x)\rangle = U(x)|0\rangle^{\otimes n}$，使得量子比特之间的关联（受输入数据 $x$ 调制）创建缠结模式，以表示这些多项式项。

### 构建多项式特征映射

考虑一个经典数据点 $x = (x_1, x_2, ..., x_d)$。一个简单的特征映射可能将每个 $x_i$ 编码为单个量子比特的旋转角度。为引入多项式项，我们需要关联。这通常通过纠缠门实现，其旋转角度取决于输入特征的组合。

一种常见结构包含单比特旋转层和多比特纠缠门层。例如，一个二阶多项式特征映射可能使用如下电路结构：

1. **初始层：** 对所有量子比特应用哈达玛门 $H$ 以产生叠加态：$H^{\otimes n} |0\rangle^{\otimes n}$。
2. **数据编码 (线性项)：** 应用依赖于单个特征的单比特旋转，例如，在量子比特 $i$ 上应用 $R_Z(f_1(x_i))$。这里，$f_1$ 可以是一个简单的缩放函数，如 $f_1(x_i) = 2 x_i$。
3. **纠缠层 (多项式项)：** 在量子比特对 $(i, j)$ 之间应用双比特纠缠门，例如受控相位门或 $ZZ$ 关联门。重要的是，这些纠缠器的旋转角度取决于相应输入特征的一个函数，例如，作用于量子比特 $i$ 和 $j$ 的 $R_{ZZ}(f_2(x_i, x_j))$。一种为 $f_2$ 引入二阶关联的常见选择是 $f_2(x_i, x_j) = (\pi - x_i)(\pi - x_j)$。这种特定形式出现在一些标准库实现中。
4. **重复：** 步骤2和3可以重复多次（常被称为“层”或“重复”）以可能增加特征映射的复杂度和表达能力。

生成的量子态 $|\phi(x)\rangle$ 存在于 $2^n$ 维希尔伯特空间中。两个此类状态之间的内积 $\langle\phi(x')|\phi(x)\rangle$ 定义了量子核 $k(x, x')$。因为电路 $U(x)$ 包含由特征乘积（如嵌入 (embedding)在 $f_2(x_i, x_j)$ 中的 $x_i x_j$）控制的纠缠操作，所以生成的核 $k(x, x')$ 将包含输入特征 $x$ 和 $x'$ 的多项式项。

例如，在 $ZZ$ 旋转中使用 $f_2(x_i, x_j) = (\pi - x_i)(\pi - x_j)$ 关联，可有效实现与 $(\pi - x_i)(\pi - x_j) Z_i Z_j$ 成比例的算子下的演化。在计算核 $\langle\phi(x')|\phi(x)\rangle$ 时，涉及 $x_i x_j x'_i x'_j$ 等乘积的项可能会产生，这是多项式核的特点。

### 数学结构与核

让我们考虑一个简单的双比特例子 ($n=2$)，数据为 $x = (x_1, x_2)$。包含 $H^{\otimes 2}$，然后在量子比特1上应用 $R_Z(2x_1)$，在量子比特2上应用 $R_Z(2x_2)$，接着是一个纠缠门，例如 $e^{-i \theta Z_1 Z_2}$ （其中 $\theta = (\pi - x_1)(\pi - x_2)$）的电路结构，将生成一个态 $|\phi(x)\rangle$。

$x$ 和 $x'$ 之间的核元素是 $k(x, x') = |\langle\phi(x')|\phi(x)\rangle|^2$。展开电路操作 $U(x) = e^{-i \theta Z_1 Z_2} R_Z(2x_2) R_Z(2x_1) H^{\otimes 2}$ 和 $U(x') = e^{-i \theta' Z_1 Z_2} R_Z(2x'_2) R_Z(2x'_1) H^{\otimes 2}$，其中 $\theta' = (\pi - x'_1)(\pi - x'_2)$，计算 $\langle 00 | U^\dagger(x') U(x) | 00 \rangle$ 将包含依赖于诸如 $(2x_1 - 2x'_1)$ 这样的差值以及关联项 $\theta, \theta'$ 的和/差的项。这自然会使得核函数 $k(x, x')$ 在 $x_1, x_2, x'_1, x'_2$ 上呈多项式形式。

具体来说，涉及 $k$ 局部关联（作用于 $k$ 个量子比特的门）且其参数 (parameter)取决于至多 $k$ 个特征乘积的特征映射可以生成与 $k$ 阶多项式相关的核。

### 优势与考量

- **捕捉非线性：** 主要优势在于能够通过量子特征空间几何直接建模数据中的非线性关系，与线性特征映射相比，这可能在复杂分类或回归任务上带来更好的表现。
- **希尔伯特空间的丰富性：** 高阶关联可以在高维希尔伯特空间中表现更复杂的纠缠结构，可能提供比同阶经典多项式核更丰富的特征表示，尽管这仍是一个活跃的研究方向。

然而，有一些重要考量：

- **电路深度与量子比特：** 实现高阶关联通常需要更深的电路或更多的量子比特（如果使用辅助量子比特），这会增加近期设备对噪声和退相干的敏感性。
- **核集中：** 随着量子比特数量或电路深度的增加，生成的核矩阵可能会出现集中效应（核值呈指数级相互靠近），这可能会阻碍QSVM等基于核的方法的训练。我们将在第3章中详细考察这一现象。
- **关联选择：** 纠缠关联的具体选择（例如 $ZZ$, $XX$, 受控门）以及对数据特征的功能依赖性（例如 $(\pi - x_i)(\pi - x_j)$, $x_i x_j$）对生成的核及其属性有重大影响。常需要仔细设计和实验。

### 电路示例可视化

以下图示说明了针对 $n$ 量子比特和特征向量 (vector) $x$ 的两层 ($d=2$) 二阶多项式特征映射的结构。

> 一个多层量子特征映射的图示，旨在引入多项式关联。每层应用数据相关的单比特旋转 ($R_Z$) 和多比特纠缠关联 ($R_{Z...Z}$)，其参数 (parameter) $f_1, f_2$ 是输入特征 $x$ 的函数。

总而言之，高阶多项式量子特征映射通过纠缠门结合多特征关联，扩展了基本编码策略。这使得量子态 $|\phi(x)\rangle$ 能够编码经典数据 $x$ 中存在的非线性关系，可能带来更具表现力的特征空间，并改进为使用量子核而设计的QML算法的性能。这些映射的设计涉及表达能力、电路复杂度和资源需求之间的权衡，这些都是实际实现中需着重考量的方面。

## 参考资料

- [Supervised learning with quantum-enhanced feature spaces](https://doi.org/10.1038/s41586-019-0980-2) — Vojtěch Havlíček, Antonio D. Córcoles, Kristan Temme, Aram W. Harrow, Abhinav Kandala, Jerry M. Chow, Jay M. Gambetta (2019)
  Journal: Nature; Publisher: Springer Nature; Volume: 567; Pages: 209-212; DOI: [10.1038/s41586-019-0980-2](https://doi.org/10.1038/s41586-019-0980-2)
  这篇基础论文介绍了量子核方法，并提出了具体的量子特征映射电路，包括那些利用ZZ相互作用将数据编码为高阶多项式特征的电路。
- [Power of data in quantum machine learning](https://www.nature.com/articles/s41467-021-22539-9) — Hsin-Yuan Huang, Michael Broughton, Masoud Mohseni, Ryan Babbush, Sergio Boixo, Hartmut Neven, Jarrod R. McClean (2021)
  Journal: Nature Communications; Volume: 12; DOI: [10.1038/s41467-021-22539-9](https://doi.org/10.1038/s41467-021-22539-9)
  这项工作研究了量子机器学习模型的训练能力，解决了“核集中”问题，即对于大量量子比特，量子核可能变得不可训练，这是高阶特征映射的一个考虑因素。
- [Quantum Machine Learning in Feature Hilbert Spaces](https://doi.org/10.1103/PhysRevLett.122.040504) — Maria Schuld, Nathan Killoran (2019)
  Journal: Physical Review Letters; Publisher: American Physical Society; Volume: 122; Pages: 040504; DOI: [10.1103/PhysRevLett.122.040504](https://doi.org/10.1103/PhysRevLett.122.040504)
  本文为量子机器学习提供了一个理论框架，通过将经典数据嵌入量子希尔伯特空间，并在此空间中基于内积定义量子核。
- [ZZFeatureMap](https://qiskit.org/documentation/stubs/qiskit.circuit.library.ZZFeatureMap.html) — Qiskit Development Team (2023)
  Publisher: IBM Corp.
  提供了广泛使用的量子特征映射的实用细节和实现，该映射通过ZZ相互作用创建二阶多项式特征，与本节讨论的示例一致。
