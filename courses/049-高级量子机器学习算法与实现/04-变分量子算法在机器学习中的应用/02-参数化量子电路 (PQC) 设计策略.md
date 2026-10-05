# 参数化量子电路 (PQC) 设计策略

来源：[原文](https://apxml.com/zh/courses/fundamentals-quantum-machine-learning/chapter-4-variational-quantum-algorithms-ml/pqc-design-strategies)

[返回章节目录](README.md) · [返回课程目录](../README.md)

参数 (parameter)化量子电路 (PQCs)，常被称为 *ansätze* (源自德语，意为“方法”或“设置”)，构成变分量子算法的核心。正如本章引言所述，变分量子算法使用经典优化器来调整量子电路的参数。PQC正是这种可调整的量子电路。它的作用是将初始状态（通常通过编码经典数据 $\vec{x}$ 准备）转换为最终状态 $|\psi(\vec{x}, \vec{\theta})\rangle$。对该最终状态执行测量，随后用于计算成本函数 $C(\vec{\theta})$，经典优化器通过调整参数 $\vec{\theta}$ 来使该函数最小化。

形式上，PQC $U(\vec{\theta})$ 是一种幺正操作，由一系列量子门组成，其中一些门依赖于可调整的经典参数 $\vec{\theta} = (\theta_1, \theta_2, \dots, \theta_M)$。通常，机器学习 (machine learning)中变分量子算法的整体电路形式如下：


$$
|\psi(\vec{x}, \vec{\theta})\rangle = U(\vec{\theta}) U_{\text{encode}}(\vec{x}) |0\rangle^{\otimes n}
$$


这里，$U_{\text{encode}}(\vec{x})$ 是数据编码电路（在第2章中讨论），它将经典输入数据 $\vec{x}$ 映射到初始量子状态，通常从 $n$ 量子比特的 $|0\rangle^{\otimes n}$ 状态开始。PQC $U(\vec{\theta})$ 随后进一步处理此状态。$U(\vec{\theta})$ 的设计是影响变分量子算法性能的一个重要因素，包含多项设计考量。

### PQC 的设计考量

选择合适的PQC架构需要在多个相互竞争的因素之间进行权衡：

1. **表达能力 (Expressibility)：** PQC能多好地覆盖希尔伯特空间？原则上，一个高表达能力的PQC可以近似更广泛的类型目标函数或分布。如果PQC能够均匀地生成希尔伯特空间中的状态（接近哈尔测度），则认为它具有高表达能力。然而，高表达能力并非总是合意的。它可能使优化过程更复杂，并增加对贫瘠高原（我们将在本章后续讨论）的易受影响性。此外，所需的表达能力取决于具体任务；较简单的任务可能不需要PQC访问整个希尔伯特空间。
2. **纠缠能力 (Entangling Capability)：** PQC在量子比特之间产生纠缠的能力通常被认为对于捕获经典模型可能难以处理的数据中复杂的关联很重要。缺少足够纠缠门的Ansätze可能可以用经典方式高效模拟，从而限制其量子优势的潜力。PQC结构中纠缠门的模式和频率直接影响此能力。
3. **可训练性 (Trainability)：** 经典优化器能够多容易地找到好的参数 (parameter) $\vec{\theta}$？这与成本函数的结构紧密相关。深层电路、某些门选择、全局成本函数和高水平纠缠可能导致贫瘠高原的出现，在这种情况下，梯度随量子比特数量呈指数级衰减，使优化变得非常困难。对于近期硬件上的更好可训练性，通常更受青睐的是具有局部成本函数的浅层电路。参数化门的选择也很重要；使用参数偏移规则（例如单量子比特旋转）等方法可以高效计算梯度的门常被选用。
4. **硬件效率 (Hardware Efficiency)：** 考虑到噪声中等规模量子 (NISQ) 设备的限制，PQC理想情况下应以低电路深度实现，并使用特定硬件平台固有的量子门。这最大程度地减少了退相干和门错误的影响。减少非原生门的数量可以避免昂贵的分解为原生门的操作，这种操作会增加深度和错误累积。最小化双量子比特门尤其重要，因为它们通常比单量子比特门具有更高的错误率。

### 常见Ansatz结构

虽然最佳的ansatz取决于具体问题，但已出现几种常见的设计策略：

#### 硬件高效Ansätze

这些可以说是近期应用中最常见的PQC类型。它们在设计时考虑了硬件限制，优先考虑浅深度并使用易于实现的门，而不是直接源于问题领域。典型的结构由重复层组成，每层包含：

1. 应用于所有（或部分）量子比特的参数 (parameter)化单量子比特旋转子层。常见的选择包括 $R_x(\theta)$、$R_y(\phi)$ 和 $R_z(\gamma)$ 的组合。
2. 以特定模式排列的固定双量子比特纠缠门（例如CNOT或CZ）子层。常见的模式包括线性（相邻量子比特）、环形（相邻带环绕）或全连接（如果硬件允许，尽管通常需要分解）。

层数决定了电路深度和参数数量。增加层数通常会提高表达能力，但也会使电路更难训练且更易受噪声影响。

> 一个简单的两层3量子比特硬件高效ansatz。每层包含参数化单量子比特旋转 ($R_y, R_z$)，后跟线性CNOT纠缠。

#### 问题驱动Ansätze

在某些情况下，问题的结构可能会提示特定的PQC设计。例如，在使用VQE的量子化学模拟中，像幺正耦合簇单双激发 (UCCSD) 这样的ansätze受到物理原理的启发。虽然在一般机器学习 (machine learning)任务中不太常见，但如果机器学习问题可以有效地映射到物理系统或哈密顿量上，专门的ansätze可能会在所需参数更少或针对该特定问题具有更好的收敛特性方面带来好处。然而，这些通常不如上述分层结构“硬件高效”。

#### 张量网络Ansätze

借鉴自凝聚态物理和经典机器学习中使用的张量网络方法，可以构建基于矩阵乘积态 (MPS) 或树张量网络 (TTN) 等结构的PQC。这些ansätze通常表现出受控的纠缠增长，并且在参数数量方面可能高效，特别是对于表示有限纠缠状态的情况。这种受约束的结构在某些情况下可能有助于缓解贫瘠高原问题，但它也可能限制其在需要高度纠缠状态的问题上的表达能力。

### 平衡权衡与实用建议

设计PQC是一个迭代过程，需要仔细权衡以下因素：

- 从简单入手：从浅层硬件高效ansatz开始（例如，1-2层）。
- 评估性能：训练变分量子算法并评估其在任务中的表现。
- 迭代改进：如果性能不足，请考虑：
  - 增加深度：添加更多层以增强表达能力，但要密切关注可训练性。
  - 调整纠缠：根据硬件连接性和潜在数据结构，尝试不同的纠缠模式（例如，线性、环形或全连接）。
  - 改变门：尝试不同的单量子比特旋转组合。
  - 重新审视编码：请记住，数据编码 $U_{\text{encode}}(\vec{x})$ 和 PQC $U(\vec{\theta})$ 协同作用。低效的编码甚至会阻碍精心设计的PQC。像数据重上传（第2章）这样的技术，它将编码层和参数 (parameter)化层交错，可以被认为是整体PQC设计策略的一部分。

量子软件库（如Qiskit、Pennylane）中存在一些工具，可以帮助量化 (quantization)ansatz的表达能力和纠缠能力等属性，尽管这些分析可能计算成本高昂。最终，在对这些设计原则有所理解的基础上进行实证测试，通常是为近期硬件上的特定机器学习 (machine learning)任务找到有效PQC的最实用方法。本章的其余部分将在此基础上进行，研究如何定义成本函数、计算这些参数化电路的梯度以及执行优化步骤。

## 参考资料

- [Variational Quantum Algorithms](https://www.nature.com/articles/s42254-021-00348-9) — Cerezo, M., Arrasmith, A., Coles, R., Cincio, P., Fouquier, K., and Patrick, S. (2021)
  Journal: Nature Reviews Physics; Volume: 3; Pages: 625-645; DOI: [10.1038/s42254-021-00348-9](https://doi.org/10.1038/s42254-021-00348-9)
  一篇关于变分量子算法的全面综述，涵盖了其基础、各种ansatz设计以及在量子机器学习和量子化学中的应用。
- [Barren plateaus in quantum neural network training landscapes](https://www.nature.com/articles/s41467-018-07090-4) — McClean, J. R., Boixo, S., Smelyanskiy, V. N., Babbush, R., and Neven, H. (2018)
  Journal: Nature Communications; Volume: 9; Pages: 4812; DOI: [10.1038/s41467-018-07090-4](https://doi.org/10.1038/s41467-018-07090-4)
  这项基础工作识别并描述了“贫瘠高原”现象，这是训练深度参数化量子电路的一个重要难题，其中梯度随量子比特数量呈指数级消失。
- [Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms](https://doi.org/10.1002/qute.201900070) — Sim, Sukin, Johnson, Peter D., and Aspuru-Guzik, Alán (2019)
  Journal: Advanced Quantum Technologies; Publisher: Wiley; Volume: 2; Pages: 1900070; DOI: [10.1002/qute.201900070](https://doi.org/10.1002/qute.201900070)
  本文严格量化并分析了各种参数化量子电路架构的表达能力和纠缠能力，对其设计提供了见解。
- [Hardware-efficient variational quantum eigensolver for small molecules and materials](https://www.nature.com/articles/nature23879) — Kandala, A., Mezzacapo, A., Temme, K., Takita, M., Chow, J. M., and Gambetta, J. M. (2017)
  Journal: Nature; Volume: 549; Pages: 242-246; DOI: [10.1038/nature23879](https://doi.org/10.1038/nature23879)
  本文展示了变分量子本征求解器 (VQE) 的早期实验实现，演示了用于分子模拟的受问题启发的 (例如 UCCSD) 和硬件高效的ansatz设计。

---

[上一节](01-%E9%87%8F%E5%AD%90%E8%AE%A1%E7%AE%97%E4%B8%AD%E7%9A%84%E5%8F%98%E5%88%86%E5%8E%9F%E7%90%86.md) · [下一节](03-QML%E4%BB%BB%E5%8A%A1%E7%9A%84%E4%BB%A3%E4%BB%B7%E5%87%BD%E6%95%B0.md)
