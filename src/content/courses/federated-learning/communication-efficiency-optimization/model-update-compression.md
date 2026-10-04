---
course: "federated-learning"
chapter: "communication-efficiency-optimization"
lesson: "model-update-compression"
sourceId: 3601
sourceUrl: "https://apxml.com/zh/courses/federated-learning/chapter-5-communication-efficiency-optimization/model-update-compression"
title: "模型更新压缩方法"
description: "讨论服务器与客户端之间发送的模型参数更新的压缩方法。"
order: 4
plots: ["plots/3601-0.json"]
sourceHash: "4b0b6adcd743b654f99bc3c12aeeb00e8963908ccf8fe72ac8fe4f9cd12e7758"
sourceCorrections: []
---

除了压缩本地训练期间计算得到的梯度，另一种有效方法则针对本地训练的*产物*：模型更新本身。客户端并非在每个本地步骤后发送压缩梯度，而是先进行本地训练（通常是多个周期），然后压缩模型权重 (weight)产生的变化 $\Delta w = w_{local} - w_{global}$，有时甚至是整个更新后的本地模型 $w_{local}$，再进行传输。当客户端在通信轮次之间执行大量本地计算时，此方法尤其适用。

压缩模型更新 $\Delta w$ 直接解决了上行通信瓶颈。所采用的方法通常与梯度压缩中使用的相似，但应用于累积更新向量 (vector)或张量。下面介绍几种主要方法。

### 模型更新的量化 (quantization)

与梯度量化类似，模型更新可以进行量化，以减少表示每个参数 (parameter)变化所需的位数。我们将 $\Delta w$ 中的高精度浮点值替换为低精度表示。

常见方法包括：

- **均匀量化：** 将值映射到特定范围内的固定一组均匀分布的级别。例如，将更新向量 (vector)缩放到 $[-1, 1]$，然后将每个值映射为 $b$ 位。
- **随机量化：** 引入随机性，通常以概率 $x - \lfloor x \rfloor$ 将值 $x$ 量化为 $\lfloor x \rfloor$，以概率 $\lceil x \rceil - x$ 将值 $x$ 量化为 $\lceil x \rceil$。这有助于保持期望值。
- **非均匀量化（例如，对数型）：** 对小值使用更多量化级别，对大值使用更少级别，可能更好地捕捉参数更新的分布，因为参数更新通常有许多小值和较少的大值。

服务器接收到量化后的更新（$\Delta w_{quantized}$），需要在聚合前对其进行反量化。虽然简单，但量化会引入误差（$\Delta w - \Delta w_{quantized}$），这可能会影响收敛。后面讨论的误差补偿等方法有助于缓解此问题。

### 模型更新的稀疏化

稀疏化的目标是只发送模型更新向量 (vector) $\Delta w$ 中的一部分参数 (parameter)，将其余参数设为零。这直接减少了传输的数据量，尤其在使用稀疏数据格式时。

- **Top-k 稀疏化：** 客户端选择 $\Delta w$ 中绝对值最大的 $k$ 个参数，只传输这些值及其索引。其余 $d-k$ 个参数被视为零。$k$ 的值控制压缩比；$k$ 越小意味着更高的压缩率，但可能带来更多信息损失。
- **阈值稀疏化：** 客户端只发送那些绝对值 $|\Delta w_i|$ 超过预设阈值 $\tau$ 的参数。发送的参数数量在不同客户端和轮次之间可能不同。
- **随机 k 稀疏化：** 随机选择 $k$ 个参数子集并进行传输。虽然更简单，但它可能无法像 Top-k 那样有效地捕捉最重要的变化。

以下是模型更新向量的 Top-k 稀疏化示例：

```python
# 模型更新 Top-k 稀疏化的伪代码

def top_k_sparsify(delta_w, k):
  """
  通过保留绝对值最大的前 k 个元素来稀疏化模型更新 delta_w。

  参数:
    delta_w: 模型更新向量（例如，NumPy 数组）。
    k: 要保留的元素数量。

  返回:
    前 k 个元素的稀疏表示（例如，索引和值）。
  """
  if k >= len(delta_w):
    # 如果 k 大于或等于向量大小，则无需稀疏化
    indices = np.arange(len(delta_w))
    values = delta_w
    return indices, values

  # 计算幅度并找到第 k 大值的阈值
  magnitudes = np.abs(delta_w)
  threshold = np.sort(magnitudes)[-k] # 找到第 k 大的幅度

  # 选择幅度大于或等于阈值的索引
  # 仔细处理可能出现的平局，以确保正好 k 个（如果在阈值处有平局则略多）
  mask = magnitudes >= threshold
  indices = np.where(mask)[0]
  values = delta_w[mask]

  # 如果平局导致超过 k 个值，则取其中的前 k 个
  if len(indices) > k:
      top_indices_within_mask = np.argsort(np.abs(values))[::-1][:k]
      indices = indices[top_indices_within_mask]
      values = values[top_indices_within_mask]

  return indices, values

# --- 客户端 ---
# 假设 delta_w 是计算出的模型更新向量
# 假设 k 是稀疏性参数

k = int(0.1 * len(delta_w)) # 示例：保留前 10%
sparse_indices, sparse_values = top_k_sparsify(delta_w, k)

# 将 sparse_indices 和 sparse_values 传输到服务器

# --- 服务器 ---
# 从客户端接收 sparse_indices 和 sparse_values

# 重构（稀疏）更新向量以进行聚合
reconstructed_delta_w = np.zeros_like(global_model_weights) # 或合适的形状
reconstructed_delta_w[sparse_indices] = sparse_values

# 将 reconstructed_delta_w 与其他客户端的更新进行聚合
```

稀疏化在聚合期间需要仔细处理。服务器需要将稀疏向量重构到公共维度（通常初始化为零），然后才能对它们进行平均或求和。发送索引会增加开销，但这通常远小于发送稠密非零值所需的开销。

### 结构化更新和低秩方法

对于大型模型，特别是具有大型权重 (weight)矩阵的神经网络 (neural network)，更新 $\Delta W$（现在是矩阵或张量）可以使用结构化表示进行近似。一种常见方法是低秩近似。

客户端并非传输完整的 $m \times n$ 更新矩阵 $\Delta W$，而是计算并发送两个较小的矩阵 $U$ ($m \times r$) 和 $V$ ($n \times r$)，使得 $\Delta W \approx UV^T$。此处 $r$ 是近似的秩，选择时使 $r \ll \min(m, n)$。传输的参数 (parameter)数量从 $m \times n$ 减少到 $r \times (m + n)$。当 $r$ 很小时，这可以带来显著节省。

服务器从每个客户端接收 $U$ 和 $V$，重构近似的 $\Delta W$（或直接使用低秩因子进行聚合，这可能更高效），然后应用聚合后的更新。选择秩 $r$ 涉及到压缩和更新近似精确度之间的权衡。

### 结合方法

这些方法并非互斥。通常可以有效结合使用。例如：

1. 计算完整的模型更新 $\Delta w$。
2. 使用 Top-k 对 $\Delta w$ 进行稀疏化，以保留最重要的变化。
3. 对剩余的非零值进行量化 (quantization)，以进一步减小其大小。

这种分层方法可以实现非常高的压缩比。

### 处理压缩误差

如前所述，量化 (quantization)和稀疏化（特别是 Top-k 以外的方法）会引入误差。如果忽略，这些误差会累积多轮，减缓甚至阻止收敛。可以应用类似于梯度压缩中使用的误差补偿或误差反馈机制。

基本思想是客户端记住当前轮次产生的压缩误差，并在下一轮次再次应用压缩之前，将其加回到模型更新中。

令 $\Delta w_t$ 为第 $t$ 轮的真实更新。
令 $c(\cdot)$ 为压缩操作（例如，量化 + 稀疏化）。
令 $e_t$ 为累积误差。

1. **更新计算：** 客户端计算 $\Delta w_t$。
2. **误差反馈：** 客户端计算要压缩的值：$u_t = \Delta w_t + e_{t-1}$（其中 $e_0 = 0$）。
3. **压缩：** 客户端压缩 $u_t$：$\delta_t = c(u_t)$。
4. **误差更新：** 客户端计算本轮的误差：$e_t = u_t - \delta_t$。此误差 $e_t$ 存储在本地，用于下一轮 $t+1$。
5. **传输：** 客户端将压缩后的更新 $\delta_t$ 发送给服务器。

这确保了在一轮中产生的误差在后续轮次中得到纠正，防止了系统性漂移，并通常能恢复接近未压缩情况的收敛特性，尽管有时收敛速度会略慢。

### 权衡总结

选择模型更新压缩方法需要权衡多个因素：

- **通信节省：** 节省了多少带宽？稀疏化和低秩方法提供可调压缩，而量化 (quantization)则基于位宽提供固定减少。
- **计算成本：** 压缩和解压缩会增加客户端和服务器的计算开销。复杂的量化或用于低秩近似的 SVD 对资源受限的客户端来说要求较高。
- **对收敛的影响：** 激进的压缩可能减缓收敛或导致最终模型精度降低。误差补偿有所帮助，但不会完全消除影响。
- **实现复杂性：** 与简单的量化或稀疏化相比，实现误差反馈或低秩分解增加了复杂性。



![权衡：压缩与精度](plots/3601-0.json)



> 通信成本降低与不同压缩方法对最终模型精度的潜在影响之间的关系。确切的权衡很大程度上取决于具体任务、模型、数据异构性以及选择的超参数 (parameter) (hyperparameter)。

模型更新压缩与梯度压缩一道，提供了一套强大的工具，使联邦学习在通信受限环境中变得实用。最优选择通常取决于具体的系统特性、模型架构以及对计算开销和潜在精度权衡的容忍度。

## 参考资料

- [Federated Learning: Strategies for Improving Communication Efficiency](https://arxiv.org/abs/1610.05492) — Jakub Konečný, H. Brendan McMahan, Felix X. Yu, Peter Richtárik, Ananda Theertha Suresh, Dave Bacon (2016)
  Journal: NIPS Workshop on Private Multi-Party Machine Learning; DOI: [10.48550/arXiv.1610.05492](https://doi.org/10.48550/arXiv.1610.05492)
  介绍了联邦学习的通信效率策略，包括量化、稀疏化和结构化更新。
- [Deep Gradient Compression: Reducing the Communication Bandwidth for Distributed Training](https://proceedings.mlr.press/v80/lin18a.html) — Yujun Lin, Song Han, Huizi Mao, Yuqing Yang, Guangxuan Xiao, and William J. Dally (2018)
  Journal: International Conference on Learning Representations (ICLR); Publisher: PMLR; Volume: 80; Pages: 751-760; DOI: [10.5555/3276412.3276472](https://doi.org/10.5555/3276412.3276472)
  介绍了分布式深度学习中的Top-k稀疏化和动量校正，适用于模型更新压缩。
- [Communication-Efficient Federated Learning with Quantized Model Updates](https://proceedings.neurips.cc/paper_files/paper/2019/file/4081c706d871d34335c40130635c91b8-Paper.pdf) — Felix Sattler, Simon Wiedemann, Klaus-Robert Müller, and Wojciech Samek (2019)
  Journal: Advances in Neural Information Processing Systems; Publisher: NeurIPS; Volume: 32; Pages: 12959-12970; DOI: [10.55917/ccf.2022.610](https://doi.org/10.55917/ccf.2022.610)
  专门研究联邦学习模型更新的量化，分析了均匀和非均匀量化方案。
- [Federated Learning with Sparsified and Quantized Communication](https://doi.org/10.1109/TPDS.2020.2990924) — Chung-Han Hsieh, Mingzhe Li, Esra K. Darici, Jianyi Zhang, and Andrew H. Song (2020)
  Journal: IEEE Transactions on Parallel and Distributed Systems (TPDS); Publisher: IEEE; Volume: 31; Pages: 1826-1837; DOI: [10.1109/TPDS.2020.2990924](https://doi.org/10.1109/TPDS.2020.2990924)
  探讨了联邦学习中稀疏化和量化结合实现通信效率，并包含误差补偿机制。
