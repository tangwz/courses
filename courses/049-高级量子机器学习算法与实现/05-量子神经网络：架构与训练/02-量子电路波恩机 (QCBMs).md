# 量子电路波恩机 (QCBMs)

来源：[原文](https://apxml.com/zh/courses/fundamentals-quantum-machine-learning/chapter-5-quantum-neural-networks-architectures-training/quantum-circuit-born-machines)

[返回章节目录](README.md) · [返回课程目录](../README.md)

在利用参数 (parameter)化量子电路 (PQC) 作为可训练组件的思路上，我们现在关注一种专为生成建模设计的特定类型量子神经网络 (neural network)：量子电路波恩机 (QCBM)。与学习决策边界的判别模型（如之前讨论的变分量子分类器）不同，QCBMs 旨在学习给定数据集背后隐藏的未知概率分布，并生成与数据相似的新样本。“波恩机”这个名称源于其依赖波恩法则来将量子态的测量结果解释为概率。

### 核心思想：从量子态采样

QCBM 的核心是一种隐式生成模型。它不明确计算概率密度函数，而是提供一种机制，根据量子电路学到的概率分布来抽取样本。这个过程包含以下主要步骤：

1. **态制备：** 将一个 PQC，表示为 $U(\theta)$，作用于标准初始态，通常是 $n$ 量子比特系统的全零态 $|0\rangle^{\otimes n}$。这会制备一个参数 (parameter)化量子态 $|\psi(\theta)\rangle = U(\theta)|0\rangle^{\otimes n}$。参数 $\theta$ 是我们量子“网络”的可调权重 (weight)。
2. **测量：** 得到的量子态 $|\psi(\theta)\rangle$ 在计算基 $\{|x\rangle\}$ 下测量，其中 $x$ 代表一个长度为 $n$ 的比特串（例如，$x = 0110...1$）。
3. **概率解释（波恩法则）：** 根据波恩法则，得到测量结果 $x$ 的概率由下式给出：
   $p_\theta(x) = |\langle x | \psi(\theta) \rangle|^2$
4. **采样：** 通过重复制备量子态 $|\psi(\theta)\rangle$ 并测量它，我们得到一组样本 $\{x_1, x_2, ..., x_M\}$，这些样本根据 $p_\theta(x)$ 分布。

训练 QCBM 的目标是调整参数 $\theta$，使电路定义的概率分布 $p_\theta(x)$ 很好地近似目标概率分布 $p_{\text{target}}(x)$，目标概率分布通常由一组训练数据样本 $S = \{s_1, s_2, ..., s_N\}$ 隐式表示。

> 训练量子电路波恩机的基本流程。PQC 生成样本，这些样本通过损失函数 (loss function)与目标数据进行比较。经典优化器更新 PQC 参数 $\theta$。

### 训练 QCBMs：匹配分布

由于我们通常只拥有来自目标分布 $p_{\text{target}}(x)$ 的样本 $S$，我们无法直接最小化散度（如库尔巴克-莱布勒 (KL) 散度），因为这需要明确知道 $p_{\text{target}}(x)$。相反，常用方法依赖于比较 QCBM 生成的样本 $\{x_i\}$ 与训练数据样本 $\{s_j\}$。

为此目的，一种广泛使用的损失函数 (loss function)是**最大均值差异 (MMD)**。MMD 是一种概率分布之间的距离度量，其思想是：当且仅当两个分布在再生核希尔伯特空间 (RKHS) 中的均值嵌入 (embedding)相同，这两个分布才是相同的。为了训练 QCBMs，我们通常计算 MMD 的基于样本的估计值：


$$
\mathcal{L}_{\text{MMD}}(\theta) = \left\| \frac{1}{N} \sum_{j=1}^{N} \phi(s_j) - \frac{1}{M} \sum_{i=1}^{M} \phi(x_i(\theta)) \right\|_{\mathcal{H}}^2
$$


在此，$\phi$ 是与经典核 $k(x, x') = \langle \phi(x), \phi(x') \rangle_{\mathcal{H}}$（例如高斯核）相关的特征映射，$\{s_j\}$ 是来自目标数据的样本，而 $\{x_i(\theta)\}$ 是由参数 (parameter)为 $\theta$ 的 QCBM 生成的样本。目标是关于 $\theta$ 最小化 $\mathcal{L}_{\text{MMD}}(\theta)$。

训练涉及一个迭代优化循环：

1. 通过制备和测量 $U(\theta)|0\rangle^{\otimes n}$，从 QCBM 生成一批 $M$ 个样本 $\{x_i(\theta)\}$。
2. 从训练数据集中取一批 $N$ 个样本 $\{s_j\}$。
3. 使用这些批次计算 MMD 损失 $\mathcal{L}_{\text{MMD}}(\theta)$（或另一个合适的损失）。
4. 估计损失的梯度 $\nabla_\theta \mathcal{L}_{\text{MMD}}(\theta)$。这通常需要诸如参数位移规则等技术，如对通用变分量子算法 (VQA) 所讨论的，并谨慎地应用于 MMD 目标函数。
5. 使用经典优化器（例如 Adam、SGD、SPSA）更新参数 $\theta$。
6. 重复直到收敛。

### 挑战与注意事项

尽管其方法优雅，训练 QCBMs 面临显著的实际挑战，其中许多与其他 VQA 和 QNN 共有：

- **Ansatz 的选择：** QCBM 的表达能力和可训练性高度依赖于 PQC $U(\theta)$ 的结构。选择不当的 Ansatz 可能无法表示目标分布，或者可能严重受到如贫瘠高原等问题的影响。
- **优化：** 由于非凸损失景观、贫瘠高原的可能性（特别是对于深度电路或全局损失函数 (loss function)），以及基于有限样本的梯度估计的随机性，寻找最佳参数 (parameter) $\theta$ 可能很困难。
- **梯度估计：** 计算梯度（通常通过参数位移规则）需要执行额外的电路，从而增加了每步优化的计算成本。由于有限的测量次数，梯度估计中的方差也会减缓收敛速度。
- **损失函数估计：** 估计如 MMD 等损失函数，每一步都需要从 QCBM 和目标数据中获取足够数量的样本（$M$ 和 $N$），以获得可靠的估计值，这增加了总测量预算。
- **评估：** 量化 (quantization)训练后的 QCBM $p_\theta(x)$ 如何良好地近似 $p_{\text{target}}(x)$ 并非易事，尤其是在高维情况下。评估指标常依赖于比较矩、评估在下游任务上的表现，或对生成样本使用其他统计检验。

尽管存在这些挑战，QCBMs 代表了一类专注于生成任务的重要 QNN。它们提供了一个框架，用于运用量子电路学习和采样复杂概率分布，这些分布可能对经典方法来说难以处理。对其的研究阐明了使用量子计算解决无监督学习 (supervised learning) (unsupervised learning)问题的可能性和困难。我们将在下一章讨论量子生成对抗网络 (GAN) (QGAN) 时遇到相关内容。

## 参考资料

- [Barren plateaus in quantum neural networks](https://www.nature.com/articles/s41467-018-07090-4) — Jarrod R. McClean, Sergio Boixo, Vadim N. Smelyanskiy, Ryan Babbush, and Hartmut Neven (2018)
  Journal: Nature Communications; Volume: 9; Pages: 4812; DOI: [10.1038/s41467-018-07090-4](https://doi.org/10.1038/s41467-018-07090-4)
  这是一篇识别和分析“贫瘠高原”现象的重要论文，该现象是QCBMs等许多变分量子算法的一个重要训练难题。

---

[上一节](01-%E9%87%8F%E5%AD%90%E7%A5%9E%E7%BB%8F%E5%85%83%E5%92%8C%E5%B1%82%E6%A8%A1%E5%9E%8B.md) · [下一节](03-%E9%87%8F%E5%AD%90%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%20%28QCNNs%29.md)
