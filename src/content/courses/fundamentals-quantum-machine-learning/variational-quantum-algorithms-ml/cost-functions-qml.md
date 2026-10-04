---
course: "fundamentals-quantum-machine-learning"
chapter: "variational-quantum-algorithms-ml"
lesson: "cost-functions-qml"
sourceId: 1057
sourceUrl: "https://apxml.com/zh/courses/fundamentals-quantum-machine-learning/chapter-4-variational-quantum-algorithms-ml/cost-functions-qml"
title: "QML任务的代价函数"
description: "基于量子测量结果为各种机器学习任务定义合适的代价函数。"
order: 3
plots: []
sourceHash: "578a839e0ad5dd579270623975171f435c1eb56743274b08faa455a1cfecabde"
sourceCorrections: []
---

在变分量子算法 (VQA) 框架中，参数 (parameter)化量子电路 (PQC) 准备量子态 $|\psi(\boldsymbol{\theta})\rangle$，其中 $\boldsymbol{\theta}$ 表示我们希望优化的经典参数。然而，为了引导这种优化，我们需要一种方法来衡量量子态 $|\psi(\boldsymbol{\theta})\rangle$ 达到目标机器学习 (machine learning)目的的程度。这就是代价函数 $C(\boldsymbol{\theta})$ 的作用。它充当量子处理单元 (QPU) 和经典优化程序之间的重要连接，将量子测量结果转换为表示性能的标量值。

### 从量子测量到经典代价

VQA 评估步骤的核心在于量子测量。通常，在量子态 $|\psi(\boldsymbol{\theta})\rangle$ 准备好之后，我们会以特定的基（通常是计算基，$Z$-基）测量一个或多个量子比特。这个测量过程本质上是概率性的，会产生经典比特串作为结果。

为了形成适合优化的可微分代价函数，我们通常不直接使用这些比特串的原始概率。相反，我们计算所选可观测量的期望值，这是一个厄米算子 $\hat{O}$。对于由 PQC 使用参数 (parameter) $\boldsymbol{\theta}$ 准备的给定量子态 $|\psi(\boldsymbol{\theta})\rangle$，期望值计算如下：


$$
\langle \hat{O} \rangle_{\boldsymbol{\theta}} = \langle \psi(\boldsymbol{\theta}) | \hat{O} | \psi(\boldsymbol{\theta}) \rangle
$$


这个期望值 $\langle \hat{O} \rangle_{\boldsymbol{\theta}}$ 提供一个平滑的实数值输出，它取决于电路参数 $\boldsymbol{\theta}$。它表示如果我们多次测量量子态 $|\psi(\boldsymbol{\theta})\rangle$ 上的可观测物 $\hat{O}$ 将会得到的平均值。这个期望值，或从中及目标数据标签导出的函数，构成我们代价函数 $C(\boldsymbol{\theta})$ 的根本。

整个过程可以如下图所示：

> VQA 循环包括使用当前参数 $\boldsymbol{\theta}$ 通过 PQC 准备一个量子态，测量可观测物 $\hat{O}$ 以估计其期望值 $\langle \hat{O} \rangle_{\boldsymbol{\theta}}$，通过将该值与目标进行比较来计算代价 $C(\boldsymbol{\theta})$，并将代价反馈给经典优化器以提出更新的参数。

### 为特定机器学习 (machine learning)任务设计代价函数

$C(\boldsymbol{\theta})$ 的具体形式在很大程度上取决于当前的机器学习任务。让我们查看一些常见情况：

#### 分类

在分类任务中，目标是将输入数据 $x_i$ 分配到几个离散类别中的一个。

- **二分类：** 对于具有两个类别（例如，标记 (token)为 +1 和 -1）的问题，一种常见方法是设计 PQC 和可观测物 $\hat{O}$，使得期望值 $\langle \hat{O} \rangle_{\boldsymbol{\theta}}^{(i)}$（为输入 $x_i$ 计算）与类别标签 $y_i$ 相关。例如，我们可以在第一个量子比特上测量泡利 Z 算子 $\hat{Z}_0$，目标是对于一个类别使其约为 $+1$，对于另一个类别约为 $-1$。

  - **均方误差 (MSE)：** 一个直接的代价函数是预测期望值与目标标签之间的 MSE：
    
    $$
    C_{\text{MSE}}(\boldsymbol{\theta}) = \frac{1}{N} \sum_{i=1}^{N} (\langle \hat{O} \rangle_{\boldsymbol{\theta}}^{(i)} - y_i)^2
    $$
    
    其中 $N$ 是训练样本的数量。尽管简单，但 MSE 会二次方地惩罚偏差，这对于分类而言可能并非总是最有效的损失。
  - **合页损失 (Hinge Loss)：** 受到支持向量 (vector)机 (SVM) 的启发，合页损失鼓励期望值具有正确的符号并高于某个裕度：
    
    $$
    C_{\text{Hinge}}(\boldsymbol{\theta}) = \frac{1}{N} \sum_{i=1}^{N} \max(0, 1 - y_i \langle \hat{O} \rangle_{\boldsymbol{\theta}}^{(i)})
    $$
    
    如果预测 $\langle \hat{O} \rangle_{\boldsymbol{\theta}}^{(i)}$ 具有正确的符号（$y_i \langle \hat{O} \rangle_{\boldsymbol{\theta}}^{(i)} \ge 1$），则此损失为零，否则线性增加。
  - **交叉熵损失：** 这种损失在经典机器学习中常用于概率分类器。如果我们将测量结果解释为概率，则可以进行调整。例如，如果我们测量量子比特 0 并估计测量 $|0\rangle$（与标签 $y_i=1$ 相关联）的概率 $p(+1|\boldsymbol{\theta}, x_i)$ 和测量 $|1\rangle$（与标签 $y_i=0$ 相关联）的概率 $p(-1|\boldsymbol{\theta}, x_i)$，则二元交叉熵为：
    
    $$
    C_{\text{XEnt}}(\boldsymbol{\theta}) = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log p(+1|\boldsymbol{\theta}, x_i) + (1-y_i) \log p(-1|\boldsymbol{\theta}, x_i)]
    $$
    
    使用交叉熵需要从测量计数中仔细估计概率，这可能是样本密集型的。它通常需要将二元标签 $\{0, 1\}$ 或 $\{-1, +1\}$ 适当映射到从特定测量结果得出的概率。
- **多分类：** 将这些想法扩展到两个以上的类别通常涉及使用多个输出量子比特、不同的测量策略（例如，测量多个泡利算子）或组合二分类器。代价函数需要相应调整，通常使用 MSE 或交叉熵的多类别版本。

#### 回归

在回归中，目标是为输入 $x_i$ 预测一个连续值 $y_i$。

- 期望值 $\langle \hat{O} \rangle_{\boldsymbol{\theta}}^{(i)}$ 本身可以作为预测的连续输出。可观测物 $\hat{O}$ 的比例和范围理想情况下应与目标值 $y_i$ 的预期范围匹配，否则输出需要重新缩放。
- **均方误差 (MSE)：** 这是 VQA 中回归最常用的代价函数：
  
  $$
  C_{\text{MSE}}(\boldsymbol{\theta}) = \frac{1}{N} \sum_{i=1}^{N} (\langle \hat{O} \rangle_{\boldsymbol{\theta}}^{(i)} - y_i)^2
  $$
  
  它直接惩罚量子模型预测与真实连续值之间的平方差。

#### 生成模型

用于生成任务的代价函数，例如使用量子电路玻恩机 (QCBM) 学习概率分布或训练量子生成对抗网络 (GAN) (QGAN)，有所不同。它们通常涉及比较量子电路生成的概率分布与目标数据分布的度量（例如，最大平均差异、库尔巴克-莱布勒散度估计）或源自判别器网络的对抗性损失。这些将在第6章中更详细地介绍。

### 选择合适的可观测物

可观测物 $\hat{O}$ 的选择并非随意；它是 VQA 设计的一个组成部分。它精确决定了最终量子态 $|\psi(\boldsymbol{\theta})\rangle$ 的哪个属性被提取以进行预测。常见选择包括：

- **单量子比特泡利算子：** 作用于特定量子比特 $k$ 的 $\hat{Z}_k$、$\hat{X}_k$ 或 $\hat{Y}_k$。测量 $\hat{Z}_k$ 很自然，因为它对应于计算基中的测量。
- **多量子比特泡利串：** 像 $\hat{Z}_0 \otimes \hat{Z}_1$ 或 $\hat{X}_0 \otimes \hat{Y}_1$ 这样的张量积。这些捕获量子比特之间的相关性。
- **泡利串的加权和（哈密顿量）：** 更复杂的观测值，通常受到问题结构的启发，例如 $\hat{H} = \sum_k c_k \hat{P}_k$，其中 $\hat{P}_k$ 是泡利串，$c_k$ 是系数。

应根据 PQC 编码信息的方式来选择可观测物。如果最终答案预期编码在特定量子比特的偏振中，则测量该量子比特上的泡利算子有道理。如果相关性很重要，则可能需要多量子比特可观测物。在将其与目标标签或值关联时，还应考虑期望值的范围（例如，泡利算子的 $[-1, +1]$）。

### 期望值的实际估计

需要记住的是，在任何真实的量子计算机（或模拟量子计算机的模拟器）上，我们都无法获得精确的期望值 $\langle \hat{O} \rangle_{\boldsymbol{\theta}}$。相反，我们通过准备量子态 $|\psi(\boldsymbol{\theta})\rangle$ 并重复测量可观测物 $\hat{O}$，例如 $N_{\text{shots}}$ 次，来*估计*它。

- 如果 $\hat{O}$ 在测量基中是对角的（例如计算基中的 $\hat{Z}$），我们计算对应于其特征值的测量结果的频率。
- 如果 $\hat{O}$ 不是对角的（例如 $\hat{X}$ 或 $\hat{Y}$），我们需要在测量前应用适当的基变换门，或将 $\hat{O}$ 分解为可以单独测量的更简单可观测物（如泡利串）的和。

此估计的准确性取决于测量次数，估计的标准差通常按 $1/\sqrt{N_{\text{shots}}}$ 缩放。这种固有的统计噪声，被称为“散粒噪声”，意味着代价函数评估本身是嘈杂的，与经典机器学习 (machine learning)中函数评估通常是确定性相比，这给优化过程增加了一层难度。

### 对优化的影响

代价函数 $C(\boldsymbol{\theta})$ 定义了优化器需要寻找最佳参数 (parameter) $\boldsymbol{\theta}$ 的优化曲面。这个曲面的结构，受 PQC 架构、数据编码、可观测物选择以及特定代价函数公式的影响，决定了训练 VQA 的可行性和效率。存在许多局部最小值或贫瘠高原现象（梯度随量子比特数量呈指数级消失的区域）等问题，都与 $C(\boldsymbol{\theta})$ 及其梯度的属性直接相关。因此，理解如何定义有效的代价函数，对于构建成功的机器学习 (machine learning) VQA 来说是不可或缺的。关于梯度计算和优化技术的后续章节将直接以此为依据。

## 参考资料

- [A variational eigenvalue solver on a photonic quantum processor](https://www.nature.com/articles/ncomms5213) — Alberto Peruzzo, Jeremy McClean, Peter Shadbolt, M. H. Yung, Xiao-Qi Zhou, Peter J. Love, Alán Aspuru-Guzik, Jeremy L. O'Brien (2014)
  Journal: Nature Communications; Volume: 5; Pages: 4213; DOI: [10.1038/ncomms5213](https://doi.org/10.1038/ncomms5213)
  介绍了变分量子本征求解器 (VQE)，这是一种利用经典优化器最小化量子期望值的基本算法，奠定了 VQA 框架。
- [Supervised Learning with Quantum Computers](https://link.springer.com/book/10.1007/978-3-319-96424-9) — Maria Schuld, Francesco Petruccione (2018)
  Publisher: Springer; DOI: [10.1007/978-3-319-96424-9](https://doi.org/10.1007/978-3-319-96424-9)
  一本关于量子机器学习的综合教材，详细介绍了如何从量子测量中构建成本函数以进行各种监督学习任务。
- [Barren plateaus in quantum neural network training landscapes](https://www.nature.com/articles/s41467-018-07090-4) — Jarrod R. McClean, Sergio Boixo, Vadim N. Smelyanskiy, Ryan Babbush, Hartmut Neven (2018)
  Journal: Nature Communications; Volume: 9; Pages: 2421; DOI: [10.1038/s41467-018-07090-4](https://doi.org/10.1038/s41467-018-07090-4)
  识别了变分量子算法中“贫瘠高原”现象，这是一个重要挑战，即成本函数梯度呈指数级消失，阻碍了优化。
- [Quantum machine learning](https://www.nature.com/articles/nature23474) — Jacob Biamonte, Peter Wittek, Nicola Pancotti, Patrick Rebentrost, Nathan Wiebe, Seth Lloyd (2017)
  Journal: Nature; Volume: 549; Pages: 195-202; DOI: [10.1038/nature23474](https://doi.org/10.1038/nature23474)
  一篇被广泛引用的综述，概述了量子机器学习范式和算法，包括成本函数在量子学习任务中的作用。
