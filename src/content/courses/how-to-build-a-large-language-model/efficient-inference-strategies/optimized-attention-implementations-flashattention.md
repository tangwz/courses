---
course: "how-to-build-a-large-language-model"
chapter: "efficient-inference-strategies"
lesson: "optimized-attention-implementations-flashattention"
sourceId: 6099
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-28-efficient-inference-strategies/optimized-attention-implementations-flashattention"
title: "优化的注意力实现 (FlashAttention)"
description: "利用 FlashAttention 等优化注意力算法以减少内存 I/O。"
order: 3
plots: []
sourceHash: "39c52d196c328285d72042fe5466b11689b4be32556682152d486e2196db8c87"
sourceCorrections: []
---

即使有KV缓存来减轻跨时间步的重复计算，自注意力 (self-attention)机制 (attention mechanism)本身在推理 (inference)过程中仍是一个重要的性能瓶颈，尤其对于处理长序列或大批量数据的模型。缩放点积注意力的标准实现，虽然数学上精巧，但往往受限于内存带宽，而非原始计算能力。

### 标准注意力中的内存带宽瓶颈

让我们回顾缩放点积注意力的核心计算：


$$
\text{注意力}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$


在标准实现中，计算这需要几个步骤，这些步骤需要在GPU的高带宽内存（HBM）及其处理单元之间移动大量数据：

1. **读取Q、K、V：** 查询 ($Q$)、键 ($K$) 和值 ($V$) 矩阵（通常很大）从 HBM 中读取。
2. **计算 $S = QK^T$：** 执行一次可能非常大的矩阵乘法。
3. **写入/读取 $S$：** 结果注意力得分矩阵 $S$（大小为 $N \times N$，其中 $N$ 是序列长度）常常必须写回 HBM，因为它可能无法完全放入 GPU 更快的片上 SRAM 中，尤其对于长序列。然后将其读回进行 softmax 计算。
4. **计算 Softmax：** Softmax 函数应用于 $S$。这通常涉及读取 $S$，执行计算，并将结果写回 HBM。
5. **计算 $O = \text{softmax}(S)V$：** 另一次矩阵乘法从 HBM 中读取 softmax 输出和值矩阵 $V$。
6. **写入 $O$：** 最终输出矩阵 $O$ 写回 HBM。

此处的核心瓶颈常常是第 3 步：读取和写入中间 $N \times N$ 矩阵 $S$ 的需求。对于序列长度 $N=4096$，如果使用 FP32，该矩阵需要 $4096 \times 4096 \times 4 \text{ 字节} \approx 67 \text{ MB}$，如果使用 FP16 则为 33.5 MB。尽管这看起来可管理，但这些读/写操作发生在注意力层*内部*，并且与在快得多的 SRAM 中执行的计算相比，反复访问相对较慢的 HBM 会消耗大量时间和能量。标准注意力的总内存访问量按 $O(N^2 d + Nd^2)$ 缩放，其中 $d$ 是头维度，但与中间矩阵 $S$ 相关的 $O(N^2)$ 项在大 $N$ 时主导内存访问成本。

### FlashAttention: 消除瓶颈

FlashAttention 是一种优化的注意力算法，专门设计用于解决此 I/O 瓶颈。该算法由 Dao 等人（2022 年）提出，其主要创新在于计算*精确*的注意力输出，而无需将完整的 $N \times N$ 注意力得分矩阵 $S$ 或中间 softmax 输出写入 HBM。这大大减少了 HBM 和 GPU 核心之间的数据传输量，使计算显着更快且更节省内存。

FlashAttention 通过多种技术组合实现这一点：

1. **分块：** 计算被分解成更小的块或“瓦片”。FlashAttention 不再一次性处理整个 $Q, K, V$ 矩阵，而是从 HBM 加载这些矩阵的较小块到 GPU 的快速片上 SRAM 中。
2. **结合核函数：** 多个操作（例如，计算 $S$ 的一个块，应用 softmax，乘以 $V$ 的一个块）被组合或“结合”成一个 CUDA 核函数。这最大限度地减少了单独核函数启动的数量，并避免在这些结合步骤之间将中间结果写回 HBM。
3. **在线 Softmax：** 在每个分块计算中，softmax 计算通过数值稳定技术（减去运行最大值）小心执行，以确保正确性，而无需一次性获取整个 $S$ 矩阵。在最终结果写回 HBM 之前，输出块 $O_i$ 使用分块输入 $Q_i, K_j, V_j$ 直接在 SRAM 中计算和累加。

设想将 $Q$ 矩阵按行划分，将 $K, V$ 矩阵按列划分（或反之，取决于实现细节）。FlashAttention 将 $Q$ 的一个块加载到 SRAM 中。然后，它遍历 $K$ 和 $V$ 的块，逐一将它们加载到 SRAM 中。对于 SRAM 中每一对 $Q$ 和 $K, V$ 块，它计算相应的注意力得分块，应用 softmax 操作（同时维护跨块归一化 (normalization)所需的统计量，如运行最大值和总和），并将结果累加到一个输出块中，该输出块也保留在 SRAM 中。只有初始 $Q$ 块的最终输出块才会被写回 HBM。

> FlashAttention 通过在 GPU 更快的 SRAM 中以分块方式处理计算，避免将大型中间注意力得分矩阵写入 HBM，从而显著减少了内存 I/O。

### 优势与集成

FlashAttention 在推理 (inference)时的主要优势是速度。通过减少 HBM 访问，它能带来显著的性能提升，通常报告与标准注意力实现相比，性能提升 2 到 4 倍甚至更多，尤其对于 $N^2$ 项占主导地位的长序列。

此外，由于大型中间矩阵 $S$ 未存储在 HBM 中，FlashAttention 需要更少的内存（内存复杂度为 $O(Nd + d^2)$，而标准注意力的峰值使用为 $O(N^2 + Nd)$）。这使得在相同的 GPU 内存限制下可以处理更长的序列或使用更大的批量大小，这对于处理各种请求负载的推理服务器来说非常有价值。

将 FlashAttention 集成到您的工作流程中通常很简单，尤其是在现代深度学习 (deep learning)框架中。PyTorch 2.0 及更高版本包含 `torch.nn.functional.scaled_dot_product_attention`，在硬件和输入条件允许时，它会自动尝试使用 FlashAttention 等优化核函数（或类似的内存高效实现）。

```python
import torch
import torch.nn.functional as F
from math import sqrt

# 假设输入：查询、值张量
# query: (批量大小, 头数, 查询序列长度, 头维度)
# 键 (key): (批量大小, 头数, 键值序列长度, 头维度)
# value: (批量大小, 头数, 键值序列长度, 头维度)

# 示例用假数据
batch_size, num_heads, seq_len_q, seq_len_kv, head_dim = 2, 8, 1024, 1024, 64
query = torch.randn(
    batch_size, num_heads, seq_len_q, head_dim,
    device='cuda',
    dtype=torch.float16
)
key = torch.randn(
    batch_size, num_heads, seq_len_kv, head_dim,
    device='cuda',
    dtype=torch.float16
)
value = torch.randn(
    batch_size, num_heads, seq_len_kv, head_dim,
    device='cuda',
    dtype=torch.float16
)
is_causal = True # 解码器推理的典型情况

# 使用 scaled_dot_product_attention 启用 PyTorch 的内部优化
# 它会自动选择 FlashAttention、内存高效注意力或数学核函数
# 如果满足条件（GPU 类型、PyTorch 版本、输入形状、标志等）
# 需要时可使用 torch.backends.cuda.sdp_kernel 进行精细控制或检查
# 例如，检查 FlashAttention 是否启用：
# with torch.backends.cuda.sdp_kernel(
#     enable_flash=True,
#     enable_math=False,
#     enable_mem_efficient=False
# ):
try:
    # 简单用法，依赖 PyTorch 的自动分派
    attn_output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=None, # 使用 is_causal=True 在内部处理因果掩码
        dropout_p=0.0,  # 推理时将 dropout 设置为 0
        is_causal=is_causal,
    )
    print(
        "使用了 PyTorch 的 scaled_dot_product_attention 后端 "
        "(可能为 FlashAttention)。"
    )

except RuntimeError as e:
    # 如果优化核函数失败或不受支持，则回退到手动实现
    print(
        f"优化注意力后端失败：{e}。 "
        f"使用手动实现。"
    )
    # 注意：手动实现效率低得多
    scale_factor = 1 / sqrt(query.size(-1))
    attn_bias = torch.zeros(
        seq_len_q,
        seq_len_kv,
        dtype=query.dtype,
        device=query.device
    )
    if is_causal:
        temp_mask = torch.ones(
            seq_len_q, seq_len_kv, dtype=torch.bool, device=query.device
        ).tril(diagonal=0)
        attn_bias.masked_fill_(
            temp_mask.logical_not(), float("-inf")
        )

    # Reshape for batch matrix multiply if needed BxHxNxd -> (BxH)xNxd
    b, h, n, d = query.shape
    q = query.reshape(b*h, n, d)
    k = key.reshape(b*h, n, d)
    v = value.reshape(b*h, n, d)

    attn_weight = q @ k.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias # 添加因果掩码偏置
    attn_weight = torch.softmax(attn_weight, dim=-1)
    # attn_weight = torch.dropout(attn_weight, 0.0, train=False) # Dropout 已关闭
    attn_output_manual = attn_weight @ v
    attn_output = attn_output_manual.reshape(
        b, h, n, d
    ) # 重塑回原形

# attn_output 现在包含注意力机制的结果
print(f"输出形状: {attn_output.shape}")
```

您也可以明确使用 `flash_attn` 包等库中的实现，该包提供对高度优化核函数的直接访问，并可能为默认 PyTorch 调度程序未涵盖的特定情况提供更多控制或支持。

### FlashAttention-2

在原有工作基础上，FlashAttention-2 引入了进一步的优化，尤其侧重于提高并行性以及减少与 GPU 线程块和 warp 之间工作分配相关的潜在瓶颈，尤其是在像 H100 (Hopper 架构) 这样的新型 NVIDIA GPU 上。这些改进通常会比第一版带来额外的加速。

### 注意事项

虽然非常有效，但请记住：

- **硬件支持：** FlashAttention 及其变体依赖于特定的 GPU 功能（例如，足够的 SRAM 大小，Tensor Core 能力），这些功能主要存在于 NVIDIA Ampere (A100) 和 Hopper (H100) 等较新一代中。在旧架构上，性能提升可能不明显或不可用。
- **软件兼容性：** 确保您正在使用兼容的深度学习 (deep learning)框架版本（例如 PyTorch >= 2.0）或包含优化核函数的专用库（`flash_attn`）。请查阅文档以获取有关数据类型 (FP16, BF16)、头维度和掩码选项的具体要求。
- **精确性与实现：** FlashAttention 计算*相同*的数学注意力函数。它是一种实现优化，而非近似方法（与某些其他技术不同）。与朴素实现相比的任何差异只应归因于浮点运算的变化。

通过借助 FlashAttention 等优化的注意力实现，您可以在推理 (inference)时显着减少与注意力机制 (attention mechanism)相关的延迟和内存占用，从而使得部署更大模型和更高效地处理长序列成为可能。这是构建高性能 LLM 推理系统工具包中的一个重要组成部分。

## 参考资料

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Aware Kernels](https://arxiv.org/abs/2205.14135) — Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré (2022)
  Journal: Advances in Neural Information Processing Systems (NeurIPS) 35; DOI: [10.48550/arXiv.2205.14135](https://doi.org/10.48550/arXiv.2205.14135)
  这篇是介绍FlashAttention的原始论文，该算法通过减少内存I/O和利用GPU内存层次结构来优化自注意力机制。
- [FlashAttention-2: Faster Attention with Reduced I/O](https://arxiv.org/abs/2307.08691) — Tri Dao (2023)
  Journal: arXiv preprint arXiv:2307.08691; DOI: [10.48550/arXiv.2307.08691](https://doi.org/10.48550/arXiv.2307.08691)
  这篇论文在原始FlashAttention的基础上，进一步优化了速度和效率，特别是在现代GPU架构上。
- [torch.nn.functional.scaled_dot_product_attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) — PyTorch Developers (2024)
  Publisher: PyTorch
  PyTorch官方文档，介绍了自动调用FlashAttention等优化注意力计算核的函数，对实际应用至关重要。
