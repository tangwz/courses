---
course: "fundamentals-quantum-machine-learning"
chapter: "quantum-generative-models"
lesson: "qcbm-distribution-learning"
sourceId: 1072
sourceUrl: "https://apxml.com/zh/courses/fundamentals-quantum-machine-learning/chapter-6-quantum-generative-models/qcbm-distribution-learning"
title: "量子电路玻恩机用于分布学习"
description: "运用量子电路玻恩机 (QCBM) 隐式学习并从复杂概率分布中采样。"
order: 2
plots: []
sourceHash: "12f4e2238dd927be74ed18899772c004b7aff4672d9f701a529c742cac493abf"
sourceCorrections: []
---

经典的生成模型通常构建显式概率分布或使用GANs或VAEs等复杂的架构。量子电路玻恩机 (QCBM) 提供了一种不同方式，运用了量子力学固有的概率特性。其核心思路直接明了：参数 (parameter)化量子电路 (PQC) 可用于制备一个量子态，其测量统计数据定义了我们希望建模的概率分布。

### 玻恩定则作为生成机制

回顾量子力学中的玻恩定则：如果一个量子系统处于态 $|\psi\rangle$，测量到特定结果 $x$ (由基态 $|x\rangle$ 表示) 的概率由 $P(x) = |\langle x | \psi \rangle|^2$ 给出。

A QCBM直接应用此原理。我们从一个易于制备的初始态开始，通常是 $n$ 量子比特的全零态 $|0\rangle^{\otimes n}$。接着，我们应用一个PQC，记作幺正算符 $U(\theta)$，这里 $\theta$ 代表可训练的电路参数 (parameter)集 (如旋转角)。这将初始态变换为一个参数化态：
$|\psi(\theta)\rangle = U(\theta) |0\rangle^{\otimes n}$

在计算基 (对应于 $|001\rangle$、 $|101\rangle$ 等经典比特串的基) 中测量此态 $|\psi(\theta)\rangle$，会得到结果 $x$ (一个特定的比特串)，其概率为：
$p_\theta(x) = |\langle x | \psi(\theta) \rangle|^2 = |\langle x | U(\theta) |0\rangle^{\otimes n}|^2$

该等式定义了QCBM *隐式* 建模的概率分布 $p_\theta(x)$。电路 $U(\theta)$ 充当生成器，分布自然产生于量子测量。训练的目标是调整参数 $\theta$，使 $p_\theta(x)$ 密切接近目标数据分布 $p_{\text{data}}(x)$。

> 量子电路玻恩机 (QCBM) 的基本流程。可训练参数 $\theta$ 控制应用于初始态 $|0\rangle^{\otimes n}$ 的参数化量子电路 (PQC) $U(\theta)$。测量所得态 $|\psi(\theta)\rangle$ 会产生服从概率分布 $p_\theta(x)$ 的样本 $x$。

### 训练QCBM

由于QCBM定义了概率分布 $p_\theta(x)$，我们可以类似经典概率模型对其进行训练。我们需要一种方法来量化 (quantization)QCBM生成的分布 $p_\theta(x)$ 与从训练数据推导出的目标分布 $p_{\text{data}}(x)$ 之间的差异。常见方法包括最小化散度或距离度量：

1. **Kullback-Leibler (KL) 散度：** 衡量两个概率分布之间的差异。目标是最小化 $KL(p_{\text{data}} || p_\theta)$。计算此值通常需要估计两个分布。对于离散数据，$p_{\text{data}}$ 可以从数据集中的频率估算。计算所有可能 $x$ 的 $p_\theta(x)$ 可能计算量很大，特别是对于大量量子比特的情况，因为它需要为可能 $2^n$ 个结果计算 $|\langle x | U(\theta) |0\rangle^{\otimes n}|^2$。通常需要近似方法或替代成本函数。
2. **最大均值差异 (MMD)：** MMD根据可再生核希尔伯特空间 (RKHS) 中样本均值嵌入 (embedding)的差异来衡量分布间的距离。高斯核是一个常见选择。MMD损失比较从 $p_{\text{data}}$ (真实数据) 中提取的样本统计量与QCBM生成的样本 ($p_\theta$) 的统计量。一个优点是，MMD通常可以通过使用来自两个分布的批量样本来有效估算，避免了为所有 $x$ 显式计算 $p_\theta(x)$ 的需要。
3. **最大似然估计 (MLE)：** 如果我们可以高效地评估训练集中特定数据点 $x_i$ 的 $p_\theta(x)$，我们可以目标最大化对数似然 $\sum_i \log p_\theta(x_i)$。然而，计算单个概率 $p_\theta(x_i)$ 仍然可能很费力。

无论选择何种成本函数 $C(\theta)$，训练都包括寻找使其最小化的参数 (parameter) $\theta^*$：
$\theta^* = \arg \min_\theta C(\theta)$
这种优化通常使用基于梯度的方法进行。梯度 $\nabla_\theta C(\theta)$ 可以使用参数位移规则 (在第4章VQAs中有所介绍) 或有限差分等技术进行估计，这通常在每个优化步骤中需要对每个梯度分量进行多次电路执行。随后使用Adam或SGD等经典优化器来更新参数 $\theta$。

### PQC架构及考量

PQC架构 $U(\theta)$ 的选择对QCBM的性能有很大影响。

- **表达能力：** PQC必须能够生成量子态 $|\psi(\theta)\rangle$，其测量统计数据能够准确代表目标分布 $p_{\text{data}}(x)$。更复杂的分布可能需要更深的电路或特定的纠缠结构。
- **可训练性：** 如同VQAs，深度或高度非结构化的PQC可能会遇到梯度随着量子比特数呈指数衰减的“荒漠高原”问题，使训练变得困难或不可能。通常优先考虑硬件高效的Ansatz或受问题启发而构建的结构。
- **隐式建模：** 重要的是要记住QCBM提供的是一个 *隐式* 模型。我们无法获得 $p_\theta(x)$ 的解析式。相反，我们得到一个从 $p_\theta(x)$ *采样* 的过程 (运行电路，进行测量)。这类似于GANs，但依赖于量子态制备和测量。

### 从已训练的QCBM采样

QCBM训练完成后 (即我们找到了合适的参数 (parameter) $\theta^*$)，生成类似于训练数据的新样本是直接的：

1. 制备初始态 $|0\rangle^{\otimes n}$。
2. 应用已训练的PQC $U(\theta^*)$。
3. 在计算基中测量所得态 $|\psi(\theta^*)\rangle$。
4. 测量结果 $x$ 是从所学分布 $p_{\theta^*}(x)$ 中抽取的样本。
   重复步骤1-3可生成多个样本。

QCBM代表了一种生成建模的基本量子方法。它们将量子电路的结构和测量的物理特性直接映射到从概率分布中学习和采样的任务。尽管在计算成本函数、估计梯度和确保可训练性方面存在挑战，但它们为经典方法提供了一种潜在的强大替代方案，特别是对于那些量子态能自然捕获复杂关联的分布。

## 参考资料

- [Quantum circuit learning](https://arxiv.org/abs/1711.04702) — Marc Benedetti, Edward Lloyd, Sophie Sack, Alessio Fiorentini (2018)
  Journal: Quantum Science and Technology; Publisher: IOP Publishing; Volume: 3; Pages: 024002; DOI: [10.1088/2058-9565/aab427](https://doi.org/10.1088/2058-9565/aab427)
  介绍了量子电路玻尔兹曼机（QCBM），用于使用参数化量子电路学习和生成经典数据分布。
- [Barren Plateaus in Quantum Neural Network Training Landscapes](https://doi.org/10.1038/s41467-018-07090-4) — Jarrod R. McClean, Sergio Boixo, Vadim N. Smelyanskiy, Ryan Babbush, Hartmut Neven (2018)
  Journal: Nature Communications; Publisher: Nature Publishing Group; Volume: 9; Pages: 4812; DOI: [10.1038/s41467-018-07090-4](https://doi.org/10.1038/s41467-018-07090-4)
  识别了荒漠高原现象，这是训练深度参数化量子电路（包括QCBM中使用的电路）的一个挑战。
- [Quantum generative models: a perspective and a survey](https://iopscience.iop.org/article/10.1088/2632-2153/ac2e80) — Banchi, Leonardo and Fingerhuth, Markus (2021)
  Journal: Machine Learning: Science and Technology; Publisher: IOP Publishing; Volume: 2; Pages: 041002; DOI: [10.1088/2632-2153/ac2e80](https://doi.org/10.1088/2632-2153/ac2e80)
  全面概述了各种量子生成模型，包括QCBM，讨论了它们的架构、训练和应用。
