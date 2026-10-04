---
course: "lora-peft-efficient-llm-training"
chapter: "fine-tuning-efficiency-foundations"
lesson: "svd-preliminaries"
sourceId: 4112
sourceUrl: "https://apxml.com/zh/courses/lora-peft-efficient-llm-training/chapter-1-fine-tuning-efficiency-foundations/svd-preliminaries"
title: "数学预备知识：奇异值分解"
description: "回顾奇异值分解及其在深度学习中与低秩矩阵近似的相关性。"
order: 3
plots: []
sourceHash: "634bab39b12aafe8076e37b9e9ef8b231c8d030e29f18aec37a71c5d68d93954"
sourceCorrections: []
---

完全微调 (fine-tuning)会修改大型语言模型中的所有参数 (parameter)，这些参数常以大型权重 (weight)矩阵表示。这个过程需要大量资源。参数高效微调 (PEFT) 方法旨在减轻这一负担，通过仅修改一小部分参数或引入少量新参数。许多成功的 PEFT 方法，特别是低秩适配 (LoRA)，都基于这样的思想：将预训练 (pre-training)模型适配到新任务所需的*变化*，可以有效地通过低秩结构来表示。奇异值分解 (SVD) 提供了理解和运用这些低秩属性的基本数学框架。

### 奇异值分解 (SVD)

SVD 是实数或复数矩阵的一种分解，它将方阵的特征分解推广到任意 $m \times n$ 矩阵。对于任意给定矩阵 $W \in \mathbb{R}^{m \times n}$，其 SVD 如下所示：


$$
W = U \Sigma V^T
$$


其中：

- $U$ 是一个 $m \times m$ 的正交矩阵 ($UU^T = U^T U = I_m$)。$U$ 的列是 $W$ 的左奇异向量 (vector)。
- $\Sigma$ 是一个 $m \times n$ 的矩形对角矩阵。对角线上的元素 $\sigma_i = \Sigma_{ii}$ 是 $W$ 的奇异值。它们是非负的，通常按降序排列：$\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0$，其中 $r = \text{rank}(W)$。$\Sigma$ 的所有其他元素均为零。
- $V^T$ 是一个 $n \times n$ 的正交矩阵 ($VV^T = V^T V = I_n$)。$V^T$ 的行（或 $V$ 的列）是 $W$ 的右奇异向量。

SVD 实质上将 $W$ 所表示的线性变换分解为三个更简单的操作：旋转或反射 ($V^T$)、沿轴缩放 ($\Sigma$) 和再次旋转或反射 ($U$)。

### 通过截断 SVD 进行低秩近似

SVD 在我们此处的价值在于它能提供矩阵的最佳低秩近似。Eckart-Young-Mirsky 定理指出，根据 Frobenius 范数（或谱范数），矩阵 $W$ 的最佳秩 $k$ 近似（其中 $k < r$）是通过保留其前 $k$ 个最大奇异值及其对应的奇异向量 (vector)而获得的。

令 $U_k$ 为包含 $U$ 的前 $k$ 列的矩阵，$\Sigma_k$ 为包含前 $k$ 个奇异值 ($\sigma_1, \dots, \sigma_k$) 的左上角 $k \times k$ 对角矩阵，且 $V_k^T$ 为包含 $V^T$ 的前 $k$ 行的矩阵（或 $V_k$ 包含 $V$ 的前 $k$ 列）。则秩 $k$ 近似 $W_k$ 为：


$$
W_k = U_k \Sigma_k V_k^T
$$


这个 $W_k$ 使近似误差 $||W - W_k||_F$ 在所有秩至多为 $k$ 的矩阵中最小化。

> 使用截断 SVD 分量 $U_k$、$\Sigma_k$ 和 $V_k^T$ 近似矩阵 $W$ 的图示，其中 $k$ 远小于 $m$ 和 $n$。

奇异值的大小表明了它们的重要性。较大的奇异值对应于向量空间中 $W$ 变换影响最大（捕获大部分方差）的方向。通过丢弃与小奇异值相关的分量，我们通常可以大幅减少表示矩阵所需的参数 (parameter)数量，同时保留其大部分主要信息。对于 $W_k = U_k \Sigma_k V_k^T$，所需的数值数量为 $mk + k + kn = k(m+n+1)$，如果 $k \ll \min(m, n)$，这会比原始矩阵 $W$ 中的 $m \times n$ 参数少很多。

### 与参数 (parameter)高效微调 (fine-tuning)的相关性

The core idea behind LoRA is that the update matrix $\Delta W$，代表微调过程中学习到的变化的更新矩阵 ($W_{adapted} = W_{pretrained} + \Delta W$) 通常具有较低的“内在秩”。这意味着 $\Delta W$ 可以被一个低秩矩阵有效近似。虽然 LoRA 在训练期间不直接计算 $\Delta W$ 的 SVD（这会带来高昂的计算成本），但它将受 SVD 启发的低秩假设付诸实施。

LoRA 建议直接将更新 $\Delta W$ 表示为两个较小矩阵 $B \in \mathbb{R}^{m \times k}$ 和 $A \in \mathbb{R}^{k \times n}$ 的乘积，即 $\Delta W = BA$，其中秩 $k$ 远小于 $m$ 和 $n$。这种 $BA$ 结构类似于截断 SVD 形式 $U_k (\Sigma_k V_k^T)$ 或 $(U_k \Sigma_k) V_k^T$。LoRA 不通过 SVD 寻找最佳的 $U_k, \Sigma_k, V_k^T$，而是在微调过程中直接通过反向传播 (backpropagation)学习低秩因子 $B$ 和 $A$。仅训练 $B$ 和 $A$，而原始权重 (weight) $W$ 保持不变。

理解 SVD 有助于了解这种低秩近似为何可能有效。如果适配模型所需的主要信息存在于低维子空间中，那么使用明显更少的参数（对于 $BA$ 是 $k(m+n)$）来表示更新 $\Delta W$ 是可行的，且不会显著牺牲性能。SVD 提供了理论支持，表明矩阵（特别是那些表示变化或差异的矩阵）通常可以有效地压缩成较低秩的形式。

这个数学基础很重要，因为我们将研究 LoRA 和其他 PEFT 方法，这些方法运用低秩结构或类似的降维技术来实现大型模型的有效适配。SVD 本身是一种标准的、数值稳定的算法，在所有主要的数值计算库（如 NumPy、SciPy、PyTorch、TensorFlow）中都有提供，这增强了矩阵分解思路的实际可行性。

## 参考资料

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2021)
  Journal: arXiv; DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  介绍低秩适应（LoRA）的开创性论文，详细阐述了其利用低秩更新进行高效微调的方法。
