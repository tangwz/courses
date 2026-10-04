---
course: "calculus-essentials-machine-learning"
chapter: "gradient-descent-algorithms"
lesson: "minibatch-gradient-descent"
sourceId: 1399
sourceUrl: "https://apxml.com/zh/courses/calculus-essentials-machine-learning/chapter-4-gradient-descent-algorithms/minibatch-gradient-descent"
title: "小批量梯度下降"
description: "了解小批量梯度下降作为批量和随机方法之间的折衷方案。"
order: 6
plots: []
sourceHash: "e998fd4529825270a329702241631fb383c3ff388f6122ba7d03ebe659928849"
sourceCorrections: []
---

在前面的章节中，我们了解了两种计算梯度的极端情况：使用整个数据集（批量梯度下降 (gradient descent)）和仅使用单个样本（随机梯度下降）。批量梯度下降能提供梯度的准确估计，但对于大型数据集而言，计算成本可能高昂且效率低下。随机梯度下降更新参数 (parameter)非常频繁，导致迭代速度更快，但更新存在噪声，收敛路径可能会剧烈波动。

小批量梯度下降在这两种方法之间提供了一种实用且广泛使用的折衷方案。它不使用所有 $m$ 个训练样本或仅一个样本，而是使用训练数据的一个随机选择的小子集（称作“小批量”）来计算梯度并更新参数。

### 小批量梯度下降 (gradient descent)的工作原理

主要思路是分小批量处理数据。设小批量的大小为 $b$。常见的小批量大小是2的幂次，例如32、64、128或256，通常是出于硬件优化原因选择的，但最佳大小可能取决于具体问题和数据集。

该算法按以下步骤进行：

1. **打乱顺序：** 在每个周期（一次完整遍历训练数据集）开始时，随机打乱整个训练数据集的顺序。这有助于防止模型受样本顺序影响而产生偏差，并提高收敛性。
2. **划分：** 将打乱顺序后的数据集划分为大小为 $b$ 的小批量。如果训练样本总数 $m$ 不能被 $b$ 完全整除，则最后一个小批量可能更小。
3. **迭代：** 对于每个小批量 $B$（包含 $b$ 个样本）：
   - 计算对小批量 $B$ 中的样本求平均值得到的损失函数 (loss function) $J_{B}(\theta)$。
   - 仅使用当前小批量 $B$ 中的样本计算梯度 $\nabla J_{B}(\theta)$。
   - 使用梯度下降更新规则更新模型参数 (parameter) $\theta$：
     
     $$
     \theta := \theta - \alpha \nabla J_{B}(\theta)
     $$
     
     其中 $\alpha$ 是学习率。

请注意，梯度 $\nabla J_{B}(\theta)$ 是真实梯度 $\nabla J(\theta)$（在整个数据集上计算的梯度）的*估计值*。它噪声小于随机梯度下降梯度（基于一个样本），但不如批量梯度下降梯度（基于所有样本）准确。

### 小批量梯度下降 (gradient descent)的优点

小批量梯度下降达成了一种有利的平衡：

- **计算效率：** 计算小批量梯度比计算整个数据集的梯度明显更快，与批量梯度下降相比，能够更频繁地更新参数 (parameter)。这通常会带来更快的整体收敛速度。
- **向量 (vector)化：** 现代计算库（如NumPy、TensorFlow、PyTorch）高度优化了矩阵和向量运算。小批量处理使我们能够运用这些优化过的例程（向量化 (quantization)），通过同时对整个小批量执行计算，使得计算效率远高于像随机梯度下降那样逐个处理样本。
- **更平滑的收敛：** 小批量上的平均梯度比随机梯度下降提供了更好的真实梯度估计值。这导致更新噪声更小，收敛路径更稳定，减少了随机梯度下降中常见的波动。虽然不像批量梯度下降那样完全平滑，但它通常更稳定地趋向于最小值。

### 小批量大小的选择

小批量大小 $b$ 是一个需要选择的超参数 (parameter) (hyperparameter)。

- **小 $b$（例如，1）：** 接近随机梯度下降 (gradient descent)，导致更新存在噪声，但进展频繁。
- **大 $b$（例如，$m$）：** 变成批量梯度下降，提供准确的梯度，但更新缓慢。
- **中等 $b$（例如，32到512）：** 提供了一种平衡。它运用向量 (vector)化效率，并为稳定的更新提供了一个相当不错的梯度估计值。

最佳小批量大小通常取决于数据集大小、模型复杂度和硬件特性。它通常通过实验确定。

### 权衡总结

| 特性 | 批量梯度下降 (gradient descent) | 随机梯度下降（SGD） | 小批量梯度下降 |
| --- | --- | --- | --- |
| 每次更新的数据量 | 整个数据集 ($m$) | 单个样本 (1) | 小批量 ($b$) |
| 更新频率 | 低（每个周期一次） | 高（每个周期 $m$ 次） | 中等（每个周期 $m/b$ 次） |
| 梯度质量 | 精确（真实梯度） | 噪声大（高方差） | 估计值（方差减小） |
| 计算成本 | 每次更新成本高 | 每次更新成本低 | 每次更新成本中等 |
| 收敛性 | 平滑，可能较慢 | 噪声大，可能较快 | 相对平滑且快速 |
| 向量 (vector)化 | 是 | 效率较低 | 是（高效） |
| 内存使用 | 可能较高 | 低 | 适中 |

因为它平衡了计算效率、更新频率和收敛稳定性，小批量梯度下降是用于在大型数据集上训练机器学习 (machine learning)模型，特别是深度神经网络 (neural network)的最常用的优化算法。它提供了一种有效调整成本函数的实用方法。

## 参考资料

- [CS230: Deep Learning, Lecture Notes - Optimization Algorithms](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEnDCA_CV0TFyy-wQHV8twQsoZMinX1HL9vUj5BeVcZ5z9dKHXOj4H4rNMvTLdW_y2MCKdfJtrsugsJWMuKrl2e1Y4jD9dCVVoc6xqxspfnqTzQWPHjvZOcCon-) — Matt Deitke (2020)
  Publisher: Stanford University
  来自知名大学课程的官方讲义，解释小批量梯度下降作为深度学习中的一种核心优化方法。
- [Optimization Methods for Large-Scale Machine Learning](https://doi.org/10.1137/16M1080173) — Léon Bottou, Frank E. Curtis, and Jorge Nocedal (2018)
  Journal: SIAM Review; Publisher: Society for Industrial and Applied Mathematics; Volume: 60; Pages: 223-311; DOI: [10.1137/16M1080173](https://doi.org/10.1137/16M1080173)
  一篇综述文章，深入回顾大规模机器学习的各种优化技术，重点涵盖随机和小批量梯度方法。
