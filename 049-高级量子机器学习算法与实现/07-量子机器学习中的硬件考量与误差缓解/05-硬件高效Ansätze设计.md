# 硬件高效Ansätze设计

来源：[原文](https://apxml.com/zh/courses/fundamentals-quantum-machine-learning/chapter-7-hardware-error-mitigation-qml/hardware-efficient-ansatze-design)

[返回章节目录](README.md) · [返回课程目录](../README.md)

如前所述，在当前量子硬件上运行QML算法面临不少难题，原因在于噪声、有限的量子比特连接和较短的相干时间。尽管纠错技术能在错误发生后予以*修正*，但一种辅助且通常必不可少的方法是，从一开始就构建量子电路，特别是VQAs和QNNs中使用的参数 (parameter)化Ansätze，使其自身不易受到这些硬件条件的限制。这种先行做法包含创建**硬件高效Ansätze**。

与受问题启发的Ansätze（可能试图模仿物理系统）或与理论上表现强的Ansätze（在理想环境下侧重表达能力）不同，硬件高效Ansätze侧重于最小化在特定量子处理单元（QPU）上执行所需的资源。目标是构建即便在NISQ设备的限制下也能运行并产出有效结果的电路。

## 硬件高效设计的原则

硬件高效设计主要围绕减少NISQ设备上的主要错误源和额外开销展开：电路深度和非局部门操作。以下是核心原则：

### 最小化电路深度

电路深度是指对任意单个量子比特施加的最长操作（门）序列。更深的电路意味着计算耗时更长，增加了量子比特因退相干而失去量子信息的可能性。在NISQ硬件上通常更青睐浅层电路。

- **策略：** 使用更少的门层。审慎评估提议的Ansatz中的每一层是否对模型的表达能力有明显贡献。

### 考虑硬件连接性

量子硬件通常具有有限的量子比特连接性。这意味着双量子比特纠缠门（如CNOT或CZ）只能直接应用于特定物理连接的量子比特对之间。在不连接的量子比特之间施加纠缠门需要插入额外的SWAP门，这些门会分解为多个原生门，从而显著增加电路深度和错误。

- **策略：** 设计Ansatz结构以主要使用与QPU耦合图相匹配的纠缠门。查阅硬件规格或使用量子计算库提供的工具来了解设备拓扑。除非绝对必要，否则避免需要远距离纠缠的模式。

> QPU连接图的一个简化示例。硬件高效Ansätze将主要在已连接的量子比特之间（例如q0-q1，q1-q3，q3-q4）应用双量子比特门。在q0和q4之间应用CNOT可能需要SWAP操作，从而增加深度和错误。

### 使用原生门集

每个量子硬件平台都有一组可以高保真度直接执行的“原生”门。更复杂的门（如Toffoli门或任意旋转）必须分解为这些原生门的序列。这种分解增加了门数量和潜在电路深度方面的额外开销，从而引入更多出错的可能。

- **策略：** 尽可能使用目标硬件原生门集中的门来构建Ansatz。常见的原生门集包括单量子比特旋转门（如$U3$, $R_x$, $R_y$, $R_z$）以及一两种双量子比特纠缠门（如CNOT, CZ或交叉共振门）。

### 参数 (parameter)效率

尽管不严格是硬件限制，但Ansatz中的参数数量会影响经典优化开销。通常，更少的参数意味着更快的训练迭代。硬件高效设计通常通过限制电路结构来隐含地限制参数数量。

- **策略：** 选择已知能以可控参数数量提供适当表达能力的Ansatz结构。如果适用，考虑层间或对称性内的参数共享技术。

## 常见的硬件高效Ansatz结构

一种常用的方法是使用分层架构。这些架构通常交替使用单量子比特操作层和纠缠门层，同时顾及硬件连接性。

### 分层结构示例

常见模式包含：

1. **初始层：** 通常是Hadamard门或旋转门层，用于创建叠加态。
2. **旋转层：** 对所有（或部分）量子比特应用参数 (parameter)化单量子比特旋转门（例如，$R_x(\theta_1)$, $R_y(\theta_2)$, $R_z(\theta_3)$）。这些层中的参数通常是训练期间优化的参数。
3. **纠缠层：** 根据选定的模式（线性、循环、连接限制内的所有对）在已连接的量子比特之间应用固定双量子比特纠缠门（例如CNOT, CZ）。
4. **重复：** 重复步骤2和3，直至达到所需层数（深度）。深度是一个超参数 (hyperparameter)，需要在表达能力和硬件成本之间权衡。

> 针对3个量子比特的两层硬件高效Ansatz的结构。单量子比特旋转层（可训练参数$\theta$）与固定纠缠层（例如CNOTs/CX和CZs）交替出现，并考虑了潜在的硬件连接（此处为Q0-Q1和Q1-Q2的线性纠缠）。

旋转门、纠缠模式和层数的具体选择取决于问题、硬件和经验测试。

## 权衡与考量

构建硬件高效Ansätze需要权衡多方面：

- **表达能力与效率：** 高度受限的浅层电路可能缺乏捕获数据中复杂关联或有效表示所需解状态的能力（表达能力）。这会限制QML模型的最终性能。
- **可训练性：** 尽管浅层电路可以减轻与深度相关的一些荒漠高原问题，但过于简单或结构不佳的Ansätze在优化过程中仍可能面临可训练性问题或陷入糟糕的局部最优解。
- **硬件专用性：** 为一个QPU拓扑和原生门定制的Ansatz，在不同特征的另一个设备上运行时，性能可能不佳或需要大量修改（转译）。这会影响QML模型的移植性。

找到恰当的平衡通常需要实验和迭代改进，并以模拟器（包含噪声模型）以及最终在目标量子硬件上的表现为指引。

## 在QML工作流程中的位置

硬件高效Ansatz的设计并非事后补救；它是为NISQ时代构建实用QML应用的一个重要组成部分。它与电路优化和转译并行。尽管转译器*能*将任意电路映射到硬件上，但从一开始就考虑硬件限制的设计，能显著减少转译开销（更少的SWAP操作，更少的分解），并提高成功执行的可能。

此外，通过浅层电路和更少易出错的操作来减少对噪声的固有敏感性，硬件高效设计作为一种*隐式*或*被动*的纠错方式。这与前面讨论的*主动*纠错技术（如ZNE、PEC）相辅相成，后者旨在通过计算纠正仍然存在的噪声。精心构建的硬件高效Ansatz提供了一个更好的起点，使得主动纠错可能更有效，甚至不那么必要。

培养设计这些Ansätze的直觉，需要理解QML算法的需求以及目标量子硬件的具体限制和能力，这是应用量子机器学习 (machine learning)进步不可或缺的技能。

## 参考资料

- [Hardware-efficient variational quantum eigensolver for small molecules and materials](https://www.nature.com/articles/nature23879) — Abhinav Kandala, Ali Mezzacapo, Kristan Temme, Maika Takita, Markus Brink, Jerry M. Chow, Jay M. Gambetta (2017)
  Journal: Nature; Volume: 549; Pages: 242-246; DOI: [10.1038/nature23879](https://doi.org/10.1038/nature23879)
  介绍并实验性地展示了用于超导量子比特变分量子特征求解器的硬件高效变分电路，强调了电路深度、连接性和本地门集对NISQ设备的重要性。
- [Barren plateaus in quantum neural networks](https://www.nature.com/articles/s41467-018-07090-4) — Jarrod R. McClean, Sergio Boixo, Vadim N. Smelyanskiy, Ryan Babbush, Hartmut Neven (2018)
  Journal: Nature Communications; Volume: 9; Pages: 1-6; DOI: [10.1038/s41467-018-07090-4](https://doi.org/10.1038/s41467-018-07090-4)
  探讨了量子神经网络中的荒原高原现象，其中梯度随量子比特数量呈指数级消失，影响可训练性并强调了精心设计变分电路的必要性。
- [Variational Quantum Algorithms](https://www.nature.com/articles/s42254-021-00348-9) — M. Cerezo, Andrew Arrasmith, Ryan Babbush, Simon C. Benjamin, Suguru Endo, Keisuke Fujii, Jarrod R. McClean, Kosuke Mitarai, Xiao Yuan, Lukasz Cincio, Patrick J. Coles (2021)
  Journal: Nature Reviews Physics; Volume: 3; Pages: 625-644; DOI: [10.1038/s42254-021-00348-9](https://doi.org/10.1038/s42254-021-00348-9)
  全面回顾了变分量子算法，包括对变分电路设计策略、硬件效率考量、表达能力和NISQ设备可训练性挑战的讨论。
- [Quantum Machine Learning: A Crash Course](https://link.springer.com/book/10.1007/978-3-030-83098-4) — Maria Schuld, Francesco Petruccione (2021)
  Publisher: Springer; DOI: [10.1007/978-3-030-83098-4](https://doi.org/10.1007/978-3-030-83098-4)
  提供了量子机器学习的入门级但全面的概述，包含关于变分量子算法以及在当前硬件上实现QML模型的实际考量的章节。

---

[上一节](04-%E7%94%B5%E8%B7%AF%E4%BC%98%E5%8C%96%E4%B8%8E%E8%BD%AC%E8%AF%91.md) · [下一节](06-%E5%9C%A8%E5%AE%9E%E9%99%85%E9%87%8F%E5%AD%90%E8%AE%BE%E5%A4%87%E4%B8%8A%E8%BF%9B%E8%A1%8CQML%E7%AE%97%E6%B3%95%E5%9F%BA%E5%87%86%E6%B5%8B%E8%AF%95.md)
