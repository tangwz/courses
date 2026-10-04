---
course: "fundamentals-quantum-machine-learning"
chapter: "quantum-generative-models"
lesson: "training-qgans-challenges-architectures"
sourceId: 1074
sourceUrl: "https://apxml.com/zh/courses/fundamentals-quantum-machine-learning/chapter-6-quantum-generative-models/training-qgans-challenges-architectures"
title: "训练 QGAN 的挑战与架构"
description: "讨论训练 QGAN 的难点，包括生成器与判别器间的平衡，并审视潜在架构。"
order: 4
plots: []
sourceHash: "1d934ae5b932d236af4e6e2f5e15d282b1573ae53ff494dc42bd87586a12c8e0"
sourceCorrections: []
---

训练量子生成对抗网络 (GAN)（QGAN）面临一系列独特的难题，它结合了传统 GAN 训练中众所周知的困难与量子计算的特有约束。成功实施 QGAN 需要仔细考量其基础量子力学原理以及优化中的实际问题。

### 平衡关系：生成器与判别器

其核心是，GAN 训练是一个生成器 ($G$) 与判别器 ($D$) 之间的最小最大博弈。生成器试图产生能欺骗判别器的样本，而判别器则努力正确识别真实样本与虚假（生成）样本。


$$
\min_{G} \max_{D} V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_{z}}[\log(1 - D(G(z)))]
$$

> GAN 的标准最小最大目标函数。$p_{\text{data}}$ 是真实数据分布，$p_z$ 是输入噪声分布（通常为均匀分布或高斯分布），$G(z)$ 是生成器的输出，$D(x)$ 是判别器对 $x$ 为真实样本的概率估计。

这种对抗过程本质上不稳定。常见的失效模式有：

1. **模式崩溃：** 生成器找到一个（或少数几个）能强力欺骗判别器的输出，并停止学习数据分布，导致样本多样性极低。可以想象一个旨在生成手写数字的 QGAN 总是只生成“1”。
2. **振荡/不收敛：** 生成器和判别器参数 (parameter)可能发生振荡，互相抵消对方的进展，而无法达到稳定均衡（纳什均衡）。

在 QGAN 中，这些问题持续存在并可能加剧。生成器 $G(\theta)$ 是一个参数化量子电路（PQC），其训练涉及估计其参数 $\theta$ 的梯度。来自量子硬件或模拟器的噪声会增加这些梯度估计的方差，可能破坏对抗训练所需的精密平衡。

### 梯度估计难题

如在变分量子算法（VQA）背景下所讨论的，计算 PQC 的梯度通常涉及参数 (parameter)移位法则等方法。对于 QGAN 生成器 $G(\theta)$，梯度计算需要考虑下游判别器 $D$。如果 $D$ 是经典判别器，我们需要像 $\nabla_\theta D(G(\theta))$ 这样的梯度。如果 $D$ 也是一个量子电路 $D(\phi)$，那么 $\nabla_\theta V(D(\phi), G(\theta))$ 和 $\nabla_\phi V(D(\phi), G(\theta))$ 的梯度计算会变得更加复杂。

每个梯度分量估计都需要多次电路执行，这使得 QGAN 训练成为计算密集型任务，特别是与反向传播 (backpropagation)通常更直接的传统 GAN 相比。此外，从量子测量中估计期望值固有的统计噪声直接影响梯度精度，可能阻碍收敛或需要更小的学习率和更多的测量次数。

### 梯度消失平原现象

量子生成器，特别是那些使用深度或全局相互作用的 PQC 实现的，容易受到梯度消失平原现象的影响。正如第 4 章所述，梯度消失平原导致梯度随量子比特数呈指数级消失，从而有效阻碍优化过程。如果生成器的量子线路结构容易出现梯度消失平原，它可能无法学习复杂分布，无论判别器有多强大。如果使用量子判别器，且其 PQC 也呈现梯度消失平原，则可能出现类似问题。仔细的量子线路结构设计、参数 (parameter)初始化策略以及可能的局部成本函数成为重要的考量因素。

### QGAN 架构选择

生成器和判别器的具体设计会显著影响训练动态和性能。

#### 生成器架构 ($G$)

生成器 $G(\theta)$ 通常是一个 PQC，它接收一个标准初始态（例如 $|0\rangle^{\otimes n}$），或者一个编码某些潜在变量 $z$ 的简单输入态，并根据参数 (parameter) $\theta$ 进行演化。最终状态被测量以产生一个样本。

- **量子线路结构设计：** PQC 量子线路结构的选择很重要。硬件高效的量子线路结构（第 7 章）减少了电路深度和门数量，提高了在近期设备上的可行性，但可能限制表达能力。受问题启发的量子线路结构可能提供更好的表征能力，但实现和训练可能更难。量子线路结构生成足够纠缠的能力通常与其建模数据中复杂关联的能力相关联。
- **输出：** 测量结果构成生成的样本。对于图像等离散数据，计算基下的测量可能就足够了。对于连续数据，可能需要更复杂的测量方案或解释。

#### 判别器架构 ($D$)

一个主要的架构决策是判别器的性质：

1. **经典判别器：** 使用经典神经网络 (neural network)（例如多层感知机 MLP、卷积神经网络 (CNN) CNN）来区分真实数据和通过测量量子生成器输出状态生成的样本。这通常更易于实现和训练，可以借助于成熟的经典机器学习 (machine learning)工具。然而，经典判别器只看到测量结果，可能丢失测量前生成器输出状态中存在的量子信息或关联。
2. **量子判别器：** 使用另一个 PQC，$D(\phi)$，来执行判别任务。这通常涉及比较 $G(\theta)$ 产生的量子态与代表真实数据的量子态（这需要将经典数据编码为量子态，参见第 2 章）。

   - **比较方法：** 比较量子态通常需要像**交换测试**这样的技术，或使用量子核方法（第 3 章）计算重叠（保真度）或距离。这些方法会增加量子电路开销。
   - **潜在优势：** 量子判别器可能更能区分由 $G(\theta)$ 产生的精巧量子特性或关联，可能形成更强大的生成模型。
   - **复杂性增加：** 训练涉及使用量子梯度方法优化 $\theta$ 和 $\phi$，这会大幅增加计算成本，并且判别器中也可能出现梯度消失平原。

> 常见 QGAN 架构。左图：量子生成器与对测量结果进行操作的经典判别器配对。右图：生成器和判别器都是量子电路，需要量子态比较机制。

### 训练策略

特定训练技术有助于应对难题：

- **优化器：** 标准梯度下降 (gradient descent)算法（Adam、RMSProp）常被使用，但能处理噪声梯度的优化器，例如 SPSA（同步扰动随机近似），可能更有益，尤其是在硬件上运行时。量子感知优化器，如量子自然梯度（第 4 章），可以通过考虑量子态空间的几何结构来改善收敛，但会增加计算开销。
- **损失函数 (loss function)：** 尽管标准最小最大损失函数很常见，但为经典 GAN 开发的替代损失函数（例如 Wasserstein 损失、最小二乘损失）在适应 QGAN 语境时可能提供更好的稳定性。实现这些通常需要仔细考虑期望值如何估计。
- **训练技巧：** 经典 GAN 文献中的技术，例如特征匹配、小批量判别或仔细的超参数 (parameter) (hyperparameter)调整（学习率、批量大小），可能有用，但需要在量子框架内进行调整和测试。

训练 QGAN 仍然是一个活跃的研究领域。量子电路属性（表达能力、可训练性）、梯度估计噪声、对抗动态以及架构选择之间的相互作用，形成了一个复杂的优化问题。成功解决这些难题对构建有效的量子生成模型非常重要。

## 参考资料

- [Quantum Generative Adversarial Networks](https://link.aps.org/doi/10.1103/PhysRevLett.121.040502) — Seth Lloyd, Christian Weedbrook (2018)
  Journal: Physical Review Letters; Publisher: American Physical Society; Volume: 121; Pages: 040502; DOI: [10.1103/PhysRevLett.121.040502](https://doi.org/10.1103/PhysRevLett.121.040502)
  介绍了量子生成对抗网络的概念，为该领域奠定了基础。它提出了QGAN的理论框架。
- [Barren plateaus in quantum neural network training landscapes](https://www.nature.com/articles/s41467-018-07090-4) — Jarrod R. McClean, Sergio Boixo, Vadim N. Smelyanskiy, Ryan Babbush, Hartmut Neven (2018)
  Journal: Nature Communications; Volume: 9; Pages: 4812; DOI: [10.1038/s41467-018-07090-4](https://doi.org/10.1038/s41467-018-07090-4)
  识别并解释了荒漠高原现象，这是训练深度参数化量子电路（包括QGAN生成器）的关键挑战。
