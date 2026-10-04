---
course: "how-to-build-a-large-language-model"
chapter: "mixed-precision-training-techniques"
lesson: "using-bf16-bfloat16-format"
sourceId: 6053
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-20-mixed-precision-training-techniques/using-bf16-bfloat16-format"
title: "使用 BF16 (BFloat16) 格式"
description: "讨论 BF16 比 FP16 更宽动态范围的优势。"
order: 5
plots: []
sourceHash: "adf6bccac6a5762caf4dc821b5d83dd6589afc275ddccc322b345ccc5f0fdf29"
sourceCorrections: []
---

尽管FP16能显著节省内存并可能提升速度，但其有限的动态范围（约 $6 	imes 10^{-5}$ 到 $6.5 	imes 10^4$）常常需要通过细致的处理（比如损失缩放等方法）来应对。另一种16位格式BFloat16（脑浮点，或 $BF16$）被专门开发出来以解决这个范围限制，它牺牲了部分精度，以换取与FP32相近的更宽的动态范围。

### 了解 BF16 格式

$FP16$ 和 $BF16$ 之间的主要区别在于它们如何在指数位和尾数位（或有效数字位）之间分配16位。

- **FP16 (IEEE 754 半精度):** 1 个符号位, 5 个指数位, 10 个尾数位。
- **BF16:** 1 个符号位, 8 个指数位, 7 个尾数位。

请注意，$BF16$ 与标准32位 $FP32$ 格式使用相同数量的指数位（8位）。这使 $BF16$ 拥有与 $FP32$ 相同的动态范围（大约 $1.18 \times 10^{-38}$ 到 $3.4 \times 10^{38}$），大幅降低了训练过程中梯度或激活值溢出或下溢的风险。但这牺牲了精度，因为 $BF16$ 只有7个尾数位，而 $FP16$ 有10个，$FP32$ 有23个。

我们可以将位分配差异可视化：

> FP16和BF16格式的位分配对比。BF16优先考虑范围（指数位）而非精度（尾数位）。

### 优点与权衡

$BF16$ 的主要优势在于，与 $FP16$ 相比，它提升了大型模型的训练稳定性。由于其动态范围与 $FP32$ 匹配，中间计算（如梯度或激活值）中出现数值溢出或下溢的可能性显著降低。这意味着复杂的动态损失缩放，尽管仍可能有用，但通常不是严格必需的，或者可以比 $FP16$ 稳定训练通常所需的配置更为宽松。

权衡之处在于尾数位较少导致的精度降低。尽管深度神经网络 (neural network)常被认为对一定程度的精度降低具有容忍度，但这种差异*可能*会影响某些特定敏感任务或架构的收敛速度或最终模型精度，相比于 $FP16$ 或 $FP32$。然而，对于许多大型语言模型训练场景，$BF16$ 更宽范围带来的稳定性优势通常超过了潜在的精度问题。

另一个实际考量是硬件支持。$BF16$ 最初在Google TPUs上引入。NVIDIA从其Ampere架构（A100 GPU）及其后续代次（例如 Hopper H100）开始添加了支持。较旧的GPU可能无法高效支持 $BF16$ 操作，这使得 $FP16$ 成为唯一的16位可行选择。

### 在 PyTorch 中结合 AMP 使用 BF16

与 $FP16$ 类似，现代深度学习 (deep learning)框架提供便捷的封装，以便在自动混合精度 (AMP) 环境中使用 $BF16$。在PyTorch中，使用 `torch.autocast` 启用 $BF16$ 非常简单。

```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

# 假设 model, optimizer, data_loader 已定义
model = nn.Linear(1024, 1024).cuda() # GPU上的示例模型层
optimizer = torch.optim.AdamW(model.parameters())
# 虚拟数据
data_loader = [
    (torch.randn(64, 1024).cuda(), torch.randn(64, 1024).cuda())
    for _ in range(10)
]

# 如果可用则使用 BF16，否则回退
# (或者如果偏好/可用则使用 FP16)
# 注意: BF16 需要 CUDA >= 11.0 和 Ampere 或更新的 GPU (或 TPU)
use_bf16 = (
    torch.cuda.is_available()
    and torch.cuda.is_bf16_supported()
)

# GradScaler 对于 BF16 通常是可选的，因为它具有更宽的范围，
# 但仍可用于保持一致性或在遇到不稳定时使用。
# 如果禁用，它将不执行任何操作。对于 BF16 来说，设置 enabled=False 很有用。
# 如果需要损失缩放，请将 enabled 设置为 True
scaler = GradScaler(enabled=False)

print(f"正在使用 BF16: {use_bf16}")

for data, target in data_loader:
    optimizer.zero_grad()

    # 自动混合精度上下文管理器
    # 将 dtype 设置为 torch.bfloat16
    with autocast(
        device_type='cuda',
        dtype=torch.bfloat16,
        enabled=use_bf16
    ):
        output = model(data)
        loss = nn.functional.mse_loss(output, target)

    # 缩放损失并执行反向传播
    # 如果 scaler 被禁用，scaler.scale(loss) 不会执行任何操作
    scaler.scale(loss).backward()

    # 优化器步进（如果 scaler 启用则取消梯度缩放）
    scaler.step(optimizer)

    # 更新 scaler 以进行下一次迭代（如果 BF16 禁用则可选）
    scaler.update()

    print(f"损失: {loss.item():.4f}")
```

在此示例中：

1. 我们使用 `torch.cuda.is_bf16_supported()` 检查硬件是否支持 $BF16$。
2. 我们初始化 `GradScaler` 但将其 `enabled` 设置为 `False`。虽然损失缩放*可以*与 $BF16$ 一起使用，但由于该格式的范围较宽，通常没有必要。禁用它可简化代码并消除与缩放检查相关的开销。如果您仍然遇到稳定性问题，可以考虑启用它，尽管这种情况比 $FP16$ 少见。
3. 在训练循环内部，`torch.autocast` 与 `dtype=torch.bfloat16` 一起使用。这指示 PyTorch 自动将块内的操作在安全且有利的情况下（通常是矩阵乘法和卷积）转换为 $BF16$，同时将其他操作（如归约）保留在 $FP32$ 中以确保数值稳定性。
4. `scaler.scale(loss).backward()` 和 `scaler.step(optimizer)` 调用在 scaler 启用或禁用时都能正确执行。如果禁用，它们实质上会成为损失和优化器步进的透传操作。

### 总结：FP16 与 BF16 的选择

$FP16$ 和 $BF16$ 之间的选择取决于几个因素：

1. **硬件支持:** $BF16$ 需要较新的硬件（NVIDIA Ampere+、Google TPU）。如果使用较旧的硬件，$FP16$ 可能是16位精度的唯一选择。
2. **训练稳定性:** $BF16$ 的主要优势在于其类似 $FP32$ 的动态范围，显著降低了溢出/下溢的风险，并且通常无需细致的损失缩放调整。如果 $FP16$ 的稳定性难以管理，$BF16$（如果可用）是一个有力的替代方案。
3. **精度要求:** $FP16$ 提供略高的精度（更多的尾数位）。如果特定任务对数值精度高度敏感，$FP16$ *可能*会产生略好的结果，前提是能保持稳定性。然而，对于许多大型模型训练任务而言，与 $BF16$ 的稳定性优势相比，这种差异通常可以忽略不计。

在实践中，如果您的硬件支持 $BF16$，它通常是训练大型语言模型的首选，因为它易于使用且具有内在的稳定性，简化了混合精度训练的配置。它提供了与 $FP16$ 几乎相同的内存和速度优势，但避免了管理狭窄动态范围的重大麻烦。

## 参考资料

- [Automatic Mixed Precision (AMP)](https://pytorch.org/docs/stable/amp.html) — PyTorch Team (2025)
  Publisher: PyTorch Documentation
  PyTorch自动混合精度(AMP)功能的官方文档，包括`torch.autocast`和`GradScaler`与`bfloat16`的使用。
- [Mixed-Precision Training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) — NVIDIA Developer Documentation (2024)
  Publisher: NVIDIA
  NVIDIA关于混合精度训练技术的指南，包括FP16和BF16的详细信息、硬件支持以及NVIDIA GPU的性能考量。
- [Mixed-Precision Training of Deep Neural Networks](https://arxiv.org/abs/1710.03740) — Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, Hao Wu (2018)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1710.03740](https://doi.org/10.48550/arXiv.1710.03740)
  介绍了像损失缩放等用于稳定FP16训练的有效技术，为理解BF16更宽范围的价值提供了基础背景。
