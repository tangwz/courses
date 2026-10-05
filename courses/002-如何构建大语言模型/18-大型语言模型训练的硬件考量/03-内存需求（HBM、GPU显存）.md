# 内存需求（HBM、GPU显存）

来源：[原文](https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-18-hardware-considerations-llm-training/memory-requirements-hbm-gpu-ram)

[返回章节目录](README.md) · [返回课程目录](../README.md)

大型语言模型的庞大规模对原始计算能力以及内存系统都带来了很大压力。训练拥有数百亿参数 (parameter)的模型，需要认真考虑加速器硬件（如GPU和TPU）上可用内存的*容量*（可存储多少数据）和*带宽*（数据访问速度）。内存容量或带宽不足可能成为主要的瓶颈，严重限制模型大小和训练效率。

训练期间GPU显存 (VRAM)（常称为VRAM，即视频随机存取内存）的主要消耗者包括：

1. **模型参数：** 神经网络 (neural network)的权重 (weight)和偏置 (bias)。所需内存与参数数量和所用精度（例如FP32、FP16、BF16）直接相关。
2. **优化器状态：** Adam或AdamW等优化器为每个参数维护运行平均值（动量和方差估计）。这通常会使仅参数所需的内存翻倍或三倍。
3. **梯度：** 在反向传播 (backpropagation)期间，会为每个参数计算梯度。这些通常需要与参数相同数量的内存，并使用相同的精度。
4. **激活值：** 这些是每个层的中间输出，在前向传播期间计算并存储，以供反向传播期间使用。激活值的内存消耗是高度动态的，取决于批次大小、序列长度、模型隐藏层大小和层数。对于大型模型和长序列，激活值通常消耗GPU显存的最大部分。
5. **工作区内存：** GPU内核（例如cuDNN）进行中间计算所需的临时存储。

我们来考虑一个参数和优化器内存的简化计算。对于一个拥有 $N$ 个参数、使用32位浮点（FP32）精度（每个参数4字节）的模型，仅参数所需的内存就是 $4N$ 字节。如果使用AdamW等优化器（它为每个参数存储两个状态，即动量和方差，通常也以FP32存储），优化器状态还需要额外 $2 \times 4N = 8N$ 字节。梯度则额外增加 $4N$ 字节。在FP32精度下，参数、梯度和AdamW状态的总静态内存大约是 $4N + 8N + 4N = 16N$ 字节。对于一个1750亿参数的模型（如GPT-3），仅此一项就需要 $16 \times 175 \times 10^9 \approx 2.8$ TB，远远超过任何一块当前GPU的显存容量。这项计算说明了为什么混合精度训练（使用FP16或BF16等16位格式）和分布式技术是十分必要的。

使用混合精度（例如BF16，每个参数2字节）会大幅降低这种静态内存占用。参数可以存储在BF16中（$2N$ 字节），梯度在BF16中计算（$2N$ 字节），而优化器状态可能为了稳定性而保留在FP32中（$8N$ 字节），从而大约总共需要 $2N + 2N + 8N = 12N$ 字节。即使有了这种减少，内存需求仍然很大。

```python
import torch

# 示例计算 (简化版)
num_params_billion = 175
bytes_per_bf16 = 2
bytes_per_fp32 = 4

# 计算参数大小（字节）
param_memory_bf16 = num_params_billion * 1e9 * bytes_per_bf16
# 计算梯度大小（通常与参数精度相同）
grad_memory_bf16 = num_params_billion * 1e9 * bytes_per_bf16
# 计算优化器状态大小（AdamW通常使用FP32状态）
optimizer_memory_fp32 = 2 * num_params_billion * 1e9 * bytes_per_fp32

total_static_memory_mixed = (
    param_memory_bf16
    + grad_memory_bf16
    + optimizer_memory_fp32
)
total_static_memory_tb = total_static_memory_mixed / (1024**4) # 将字节转换为太字节 (Terabytes)

print(
    f"约 {num_params_billion} 亿参数的静态内存（BF16参数/梯度，FP32 AdamW状态）：{total_static_memory_tb:.2f} TB"
)

# 注意：这不包括激活值，其变化性很大。
# 它也简化了混合精度的细节。
```

激活值内存使用通常是最具有挑战性的部分。在Transformer模型中，每层存储的激活值大小大致与 $批次大小 \times 序列长度 \times 隐藏层大小 \times 层数$ 成比例。由于LLM中的序列长度（$L$）可以很长（数千个token），并且批次大小（$B$）为了效率而增加，因此激活值所需的 $O(B \cdot L \cdot d_{model} \cdot N_{layers})$ 内存很快就会占据主导地位，尤其是在使用标准反向传播时，它需要缓存前向传播中的激活值。通常采用激活检查点（或梯度检查点）等方法来缓解此问题，通过在反向传播期间重新计算激活值而不是存储它们，以此来权衡计算时间的增加和内存使用的减少。

### 高带宽内存（HBM）

鉴于庞大的数据需求，仅仅拥有足够的GPU显存 (VRAM)容量是不够的；内存还必须足够快，才能满足GPU计算核心的数据需求。HBM正是在这一点上显得重要。

HBM是一种堆叠式DRAM技术，旨在通过硅中介层与GPU计算核心非常紧密地放置。这种紧密性以及非常宽的通信总线（例如1024位或更多，相比之下，典型GDDR内存为256/384位）能实现比消费级显卡上使用的传统GDDR内存高得多的内存带宽。

用于大规模训练的现代计算GPU（如NVIDIA的A100或H100系列）完全依赖HBM（特别是HBM2e或HBM3）。高带宽（通常每块GPU达到1.5-3 TB/s）对于以下几个方面十分重要：

1. **供给计算单元：** LLM训练涉及大量的矩阵乘法和其他操作。这些操作通常受内存限制，这意味着速度受到从内存中获取数据（权重 (weight)、激活值）的速度影响。高带宽确保计算单元得到高效运用。
2. **梯度和参数 (parameter)更新：** 参数、梯度和优化器状态的频繁读写从高带宽中获得了很大益处。
3. **GPU间通信：** 在分布式设置中，GPU之间传输的数据通常源自或目标于GPU显存。更快的内存访问速度能加快这些通信步骤。



[交互图表：GPU显存带宽对比 (GB/s)](https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-18-hardware-considerations-llm-training/memory-requirements-hbm-gpu-ram#plot-1yx3rg0)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "type": "bar",
      "x": [
        "GDDR6 (典型)",
        "HBM2e (A100 80GB)",
        "HBM3 (H100 80GB)"
      ],
      "y": [
        760,
        2039,
        3350
      ],
      "marker": {
        "color": [
          "#adb5bd",
          "#4dabf7",
          "#ae3ec9"
        ]
      }
    }
  ],
  "layout": {
    "title": {
      "text": "GPU显存带宽对比 (GB/s)"
    },
    "xaxis": {
      "title": {
        "text": "内存类型 / GPU"
      }
    },
    "yaxis": {
      "title": {
        "text": "约峰值带宽 (GB/s)"
      }
    },
    "bargap": 0.3
  }
}
```

</details>



> 不同GPU显存技术的约峰值内存带宽。HBM相比GDDR提供了更高的带宽，这对LLM训练性能非常重要。

虽然HBM提供了卓越的带宽，但其制造复杂性通常导致成本更高，且相对于GDDR，每美元的容量可能更低。因此，硬件选择需要平衡高带宽需求（倾向于配备HBM的GPU）与所需的总容量和预算限制。每块GPU的显存容量（例如40GB、80GB或更高）直接决定了模型（或分布式训练中的模型分片）的大小上限，这会影响并行策略（数据并行、张量并行、流水线并行、ZeRO）的选择，以适应训练期间的模型及其相关状态。了解这些内存限制对于设计大型语言模型的高效且可行的训练设置来说非常重要。

## 参考资料

- [Mixed-Precision Training for Deep Neural Networks](https://arxiv.org/abs/1710.03740) — Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, Hao Wu (2018)
  Journal: International Conference on Learning Representations (ICLR) 2018; DOI: [10.48550/arXiv.1710.03740](https://doi.org/10.48550/arXiv.1710.03740)
  阐述了混合精度训练（FP16/FP32）在减少深度神经网络内存占用和提升训练速度方面的益处。
- [NVIDIA A100 Tensor Core GPU Architecture](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-ampere-architecture-whitepaper.pdf) — NVIDIA Corporation (2020)
  Publisher: NVIDIA Corporation
  详细介绍了A100 GPU架构，包括HBM2e在支持高性能AI工作负载中的作用和规格。

---

[上一节](02-TPU%20%E6%9E%B6%E6%9E%84%EF%BC%88Google%20TPU%EF%BC%89.md) · [下一节](04-%E4%BA%92%E8%BF%9E%E6%8A%80%E6%9C%AF%20%28NVLink%2C%20InfiniBand%29.md)
