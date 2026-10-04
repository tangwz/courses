---
course: "meta-learning-foundation-models"
chapter: "advanced-topics-theoretical-considerations"
lesson: "bayesian-meta-learning"
sourceId: 4390
sourceUrl: "https://apxml.com/zh/courses/meta-learning-foundation-models/chapter-7-advanced-topics-theoretical-considerations/bayesian-meta-learning"
title: "贝叶斯元学习方法"
description: "元学习中用于不确定性量化的概率视角。"
order: 1
plots: []
sourceHash: "c94ed71125b30d3f7d800fe993b6946fbb54ee64fc7bfce11274b49c2e0e72df"
sourceCorrections: []
---

标准元学习算法通常给出模型参数 (parameter)或适应策略的点估计。这样做虽然有效，但忽略了一个重要方面，特别是在少样本情境下：不确定性。当使用少量样本对大型基础模型进行调整时，衡量调整过程和由此产生的预测的可信度非常重要。贝叶斯元学习方法通过将概率建模引入元学习体系来解决这个问题。

贝叶斯元学习的**主要思想**是将要分析的量，例如适合调整的初始模型参数或任务专用参数本身，视作由概率分布决定的随机变量。它不是学习单个最优参数向量 (vector)$\theta$，而是旨在推断参数的*分布*，从而把握已知和未知信息。

### 从点估计到分布

回顾标准元学习设置，我们有元训练任务 $D_{meta} = \{ \mathcal{T}_1, \mathcal{T}_2, ..., \mathcal{T}_N \}$，目标是学习一个模型或学习过程，使其能良好地泛化到新的、未见过的任务 $\mathcal{T}_{new}$。每个任务 $\mathcal{T}_i$ 通常包含一个用于调整的支持集 $S_i$ 和一个用于评估的查询集 $Q_i$。

在贝叶斯框架下，元学习涉及学习参数 (parameter)（或超参数 (hyperparameter)）上的*先验*分布 $P(\theta_{prior} | D_{meta})$，该分布表示跨任务的通用知识。当遇到新任务 $\mathcal{T}_{new}$ 及其支持集 $S_{new}$ 时，我们执行贝叶斯更新以获得任务专用*后验*分布 $P(\theta_{task} | S_{new}, \theta_{prior})$。此后验反映了我们对该特定任务最优参数的更新认知，它结合了先验中的通用知识和 $S_{new}$ 中的任务专用证据。

更新通常遵循贝叶斯定理：


$$
P(\theta_{task} | S_{new}, \theta_{prior}) \propto P(S_{new} | \theta_{task}) P(\theta_{task} | \theta_{prior})
$$


这里，$P(S_{new} | \theta_{task})$ 是给定任务参数下观察到支持数据的似然，而 $P(\theta_{task} | \theta_{prior})$ 作为从元学习分布 $\theta_{prior}$ 中得到的先验。对于查询点 $x_q$ 上的预测，我们使用后验预测分布，它对参数的后验分布进行边缘化：


$$
P(y_q | x_q, S_{new}, \theta_{prior}) = \int P(y_q | x_q, \theta_{task}) P(\theta_{task} | S_{new}, \theta_{prior}) d\theta_{task}
$$


该积分自然地考虑了进行预测时的参数不确定性。

> 贝叶斯元学习的流程。元学习先验通过贝叶斯推断利用新任务的支持集进行更新，形成任务专用后验分布，然后用此分布为查询集生成预测。

### 典型贝叶斯元学习算法

几种方法实现了此贝叶斯体系：

1. **概率性 MAML 变体**：MAML 的扩展旨在学习初始参数 (parameter) $\theta_0$ 的分布，或将不确定性纳入适应过程本身。

   - **BMAML (Bayesian MAML)**: 它在初始参数 $\theta_0$ 上设置先验分布，并在内循环更新期间对任务专用参数执行近似贝叶斯推断。这通常涉及施泰因变分梯度下降 (gradient descent) (SVGD) 或拉普拉斯近似等方法。
   - **PLATIPUS (Probabilistic LATent variable model Incorporating Priors and Uncertainty)**: 它学习初始参数的分布，并使用摊还变分推断来快速近似新任务的后验分布。
     这些方法明确地对初始化中的不确定性进行建模，承认单个最优起点可能不存在或并非对所有新任务都理想。
2. **摊还变分推断**：这些方法不是为每个新任务的后验运行优化或采样过程，而是训练一个*推断网络*。该网络将支持集 $S_i$ 作为输入，并直接输出近似后验分布 $q(\theta_{task} | S_i)$ 的参数（例如，均值和方差）。这使得元测试时的适应非常快速，只需要推断网络的一次前向传播。
3. **使用贝叶斯神经网络 (neural network) (BNN) 的元学习**：此方法将神经网络本身（无论是基础学习器还是元学习器）的权重 (weight)视为随机变量。

   - **变分推断 (VI)**: 假定后验具有可处理的形式（例如，高斯分布），并优化其参数以最小化与真实后验的 KL 散度。这通常应用于大型模型中的部分参数以提高可扩展性。
   - **蒙特卡洛 Dropout**: 可被视为深度高斯过程中贝叶斯推断的一种近似。在适应和预测期间应用 dropout 可以从近似后验预测分布中进行采样。
   - **拉普拉斯近似**: 将后验的高斯近似拟合到最大后验 (MAP) 估计值周围，并使用 MAP 估计值处损失函数 (loss function)的曲率（Hessian 矩阵）来定义协方差。
4. **高斯过程 (GP) 元学习**：GP 提供了一种非参数贝叶斯方法，天生适合带有不确定性量化 (quantization)的回归和分类任务。在元学习中，GP 可以用作：

   - 直接作为任务专用模型，元学习优化跨任务的共享核超参数 (hyperparameter)。
   - 作为元学习器的一部分，例如，学习一个嵌入 (embedding)函数，使得嵌入空间中的距离与任务相似性相关联，这适合作为 GP 先验。
     **神经过程 (NP)** 和 **条件神经过程 (CNP)** 结合了神经网络的灵活性和 GP 的概率特性，为基础模型常见的复杂、高维数据提供了更好的可扩展性。它们学习将上下文 (context)点（支持集）映射到目标点（查询集）的预测分布的函数。

### 优点与应用场景

贝叶斯元学习的主要优点是**有原则的不确定性量化 (quantization)**。这对以下方面特别有益：

- **可靠性评估**: 了解模型何时对其新任务预测有信心，何时不确定。
- **主动学习**: 根据模型不确定性，在少样本任务中选择最有信息量的样本进行标注。
- **风险敏感决策**: 在医疗诊断或自动驾驶系统等应用中，了解不确定性非常重要。
- **改进的正则化 (regularization)**: 贝叶斯公式通常通过先验提供固有正则化，可能带来更好的泛化能力，尤其是在支持集非常小的情况下。

### 基础模型背景下的挑战

将贝叶斯元学习应用于大型基础模型带来了明显的障碍：

- **可扩展性**: 对于拥有数百万或数十亿参数 (parameter)的模型，完整贝叶斯推断在计算上不可行。VI、拉普拉斯或 MC Dropout 等近似技术是必需的，但即使是这些技术，在计算和内存方面也可能要求很高，尤其是在处理分布梯度或 Hessian 矩阵时。
- **近似质量**: 不确定性估计的可靠性严重依赖于所选近似的质量。拙劣的近似可能导致校准不良或误导性的不确定性度量。评估校准是一个重要且不简单的步骤。
- **先验设定**: 为与基础模型相关的高维参数空间或函数空间定义有意义且有效的先验是困难的。先验应如何把握预训练 (pre-training)期间学习到的复杂结构并有效地引导少样本适应？
- **与 PEFT 的兼容性**: 将贝叶斯原则与参数高效微调 (fine-tuning) (PEFT) 方法（如 LoRA 或适配器）相结合是一个活跃的研究方向。例如，我们能否学习 LoRA 矩阵或适配器参数的分布？这需要仔细考量先验如何与 PEFT 中使用的低维参数化相互作用。

尽管存在这些困难，不确定性量化 (quantization)的潜在益处使贝叶斯元学习成为一个引人关注的方向。研究侧重于开发更具可扩展性的近似技术、更适合大型模型的先验设定，以及将概率推理 (inference)与 PEFT 等高效适应策略结合的方法，旨在将贝叶斯推断的优点应用于基础模型的适应实践中。

## 参考资料

- [PLATIPUS: Probabilistic Latent Variable Models for Universal Adaptation](https://proceedings.mlr.press/v119/edwards20a.html) — Harrison Edwards, Van Duong Nguyen, Alexander Badenoch, Michael M. Sollars, Seth Flaxman, Thomas Lucas (2020)
  Journal: Proceedings of the 37th International Conference on Machine Learning; Publisher: PMLR; Volume: 119; Pages: 2831-2841; DOI: [10.5555/3455790.3456079](https://doi.org/10.5555/3455790.3456079)
  提出了一种利用分摊变分推断学习初始参数分布以实现快速任务适应的方法。
