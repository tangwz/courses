---
course: "intro-diffusion-models"
chapter: "model-architecture-training"
lesson: "defining-training-objective"
sourceId: 5409
sourceUrl: "https://apxml.com/zh/courses/intro-diffusion-models/chapter-4-model-architecture-training/defining-training-objective"
title: "定义训练目标"
description: "了解损失函数：通常是预测噪声与实际噪声之间的均方误差。"
order: 3
plots: []
sourceHash: "1b686f236f54d5b929e6cc9de4c63064fc4111bbc7484bf57dd9331a7e58e6c4"
sourceCorrections: []
---

训练一个神经网络 (neural network)（通常是U-Net架构，记作 $\epsilon_\theta$）以反转扩散过程是核心任务。为了进行训练，该网络需要一个明确的目标，即一个要预测的特定量。

回顾前向过程：我们从干净数据 $x_0$ 开始，并在每个时间步 $t$ 逐步添加高斯噪声 $\epsilon$ 以得到 $x_t$。训练的核心思路出乎意料地直接：我们训练网络 $\epsilon_\theta$ 来预测添加到 $x_0$ 以生成 $x_t$ 的*确切噪声* $\epsilon$，给定噪声输入 $x_t$ 和时间步 $t$。

该网络接收 $x_t$ 和 $t$ 作为输入，并输出一个预测值，我们称之为 $\epsilon_\theta(x_t, t)$。我们的目标是使这个预测值与在前向步骤计算 $x_t$ 期间从标准高斯分布 $\mathcal{N}(0, I)$ 中采样的*实际*噪声 $\epsilon$ 相匹配。

### 损失函数 (loss function)：均方误差

为量化 (quantization)预测噪声 $\epsilon_\theta(x_t, t)$ 与真实噪声 $\epsilon$ 之间的差异，一个普遍且有效的方法是均方误差 (MSE)。最小化此误差会使得网络的输出与目标噪声紧密对应。

训练目标，即损失函数 $L$，被表述为对所有可能输入的期望：初始数据 $x_0$、随机选取的时间步 $t$（在1和最大时间步 $T$ 之间均匀采样）以及随机采样的噪声 $\epsilon$。数学上，我们将其表示为：


$$
L = \mathbb{E}_{t \sim \mathcal{U}(1, T), x_0 \sim q(x_0), \epsilon \sim \mathcal{N}(0, I)} \left[ ||\epsilon - \epsilon_\theta(x_t, t)||^2 \right]
$$


我们来细分一下：

- $\mathbb{E}[...]$: 表示期望，意味着我们对许多样本的损失进行平均。
- $t \sim \mathcal{U}(1, T)$: 我们从所有可能的时间步（1到 $T$）中均匀随机地采样一个时间步 $t$。这确保网络能够在所有噪声水平下学习去噪。
- $x_0 \sim q(x_0)$: 我们从训练数据集分布 $q(x_0)$ 中采样一个真实数据点 $x_0$。
- $\epsilon \sim \mathcal{N}(0, I)$: 我们从标准高斯分布中采样一个噪声向量 (vector) $\epsilon$。
- $x_t$: 这是 $x_0$ 在时间步 $t$ 的噪声版本，使用 $x_0$ 和采样噪声 $\epsilon$ 生成。回顾前向过程章节，$x_t$ 可以通过 $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$ 直接采样得到，而 $\bar{\alpha}_t$ 则来源于噪声调度表。
- $\epsilon_\theta(x_t, t)$: 这是我们的神经网络 (neural network)（由 $\theta$ 参数 (parameter)化）在给定噪声数据 $x_t$ 和时间步 $t$ 时预测的噪声。
- $||\epsilon - \epsilon_\theta(x_t, t)||^2$: 这是L2范数的平方（欧几里得距离），它计算实际噪声与预测噪声在所有维度上（例如，图像中的所有像素）的平方差。

实际操作中，在训练期间，我们通过对训练数据的小批量进行损失平均来近似这个期望。对于批次中的每个项，我们采样一个随机的 $t$，采样一个随机的 $\epsilon$，计算 $x_t$，将 $x_t$ 和 $t$ 输入到网络 $\epsilon_\theta$，并计算网络输出 $\epsilon_\theta(x_t, t)$ 与原始 $\epsilon$ 之间的MSE。

> 此图显示了计算单个数据点训练损失的流程。网络 $\epsilon_\theta$ 尝试基于噪声数据 $x_t$ 和时间步 $t$ 预测原始噪声 $\epsilon$。MSE 损失衡量了实际噪声与预测噪声之间的差异。

### 与概率目标的关联

虽然这种噪声MSE损失看起来直观且在经验上表现良好，但它不仅仅是一种启发式方法。它自然地源自扩散模型更严格的数学目标：最大化数据对数似然的证据下界 (ELBO)。

完整的推导涉及变分推断和贝叶斯定理，我们将在下一节中进行简化。目前，只需了解最小化 $L = ||\epsilon - \epsilon_\theta(x_t, t)||^2$ 与优化ELBO直接相关。这种关联为这个看似简单的目标为何能使模型学习生成所需的复杂数据分布提供了坚实的理论依据。

通过训练网络 $\epsilon_\theta$ 使用此目标，我们赋予其估计 $x_t$ 中存在噪声的能力。这种噪声估计正是我们在反向过程（采样）中所需的，以便迭代地对随机信号去噪，使其恢复成类似我们训练数据的样本。

## 参考资料

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — Jonathan Ho, Ajay Jain, Pieter Abbeel (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2006.11239](https://doi.org/10.48550/arXiv.2006.11239)
  介绍了扩散模型的噪声预测训练目标及其与证据下界的联系。
