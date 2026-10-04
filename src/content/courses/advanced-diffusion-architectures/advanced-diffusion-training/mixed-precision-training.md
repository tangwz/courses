---
course: "advanced-diffusion-architectures"
chapter: "advanced-diffusion-training"
lesson: "mixed-precision-training"
sourceId: 5440
sourceUrl: "https://apxml.com/zh/courses/advanced-diffusion-architectures/chapter-4-advanced-diffusion-training/mixed-precision-training"
title: "扩散模型的混合精度训练"
description: "运用混合精度 (FP16/BF16) 以加速训练并减少内存占用。"
order: 7
plots: []
sourceHash: "9edb4c41059d3c83a006a906b93ee73560cfcb754d624cc35251d0f88feae573"
sourceCorrections: []
---

训练大型扩散模型，以其复杂的架构（如改进的 U-Net 或 Transformer）而闻名，需要大量的计算资源和时间。随着模型复杂度的提高和数据集的扩大，优化训练过程变得必不可少，这不仅是为了速度，也是为了在硬件限制下实现可行性。混合精度训练是一种有效技术，通过策略性地使用低精度浮点数进行计算来应对这些挑战，从而带来显著的加速和内存消耗的减少。

### 理解浮点精度

深度学习 (deep learning)模型通常使用 32 位单精度浮点数 (FP32) 来存储权重 (weight)、激活和梯度。每个 FP32 数字使用 32 位内存。混合精度训练引入了低精度格式：

- **FP16 (半精度)：** 使用 16 位。它显著节省内存（比 FP32 减少 50%），并且可以利用 NVIDIA 的 Tensor Cores 等专用硬件进行更快的计算（通常有 2-8 倍的加速）。然而，与 FP32 相比，FP16 的可表示数值范围小得多，精度较低，增加了遇到梯度下溢（梯度变为零）或上溢（梯度变为无穷大/NaN）的风险。
- **BF16 (Brain 浮点数)：** 也使用 16 位。重要的是，BF16 保持与 FP32 相同的指数范围，但降低了精度（尾数位）。这使得它比 FP16 更不容易出现上溢/下溢问题，为深度学习模型的训练提供更好的数值稳定性，通常具有类似的性能优势。对 BF16 的支持在较新的加速器上很常见，例如最新的 NVIDIA GPU（Ampere 架构及后续）和 Google TPU。

> FP32、FP16 和 BF16 浮点格式的位分配、范围和精度特征的比较。

### 混合精度训练的运行机制

简单地将所有操作切换到 FP16 可能会导致数值不稳定。有效的混合精度训练将低精度计算与保持精度和稳定性的方法相结合，通常由深度学习 (deep learning)框架自动完成：

1. **FP32 主权重 (weight)：** 模型权重的主副本以 FP32 格式保存。该副本被优化器用于累积更新，保持了在许多训练步骤中进行微小梯度调整所需的精度。
2. **FP16/BF16 计算：** 在前向和反向传播 (backpropagation)过程中，权重和激活被转换为 FP16 或 BF16，以在计算上有利且数值安全。矩阵乘法和卷积等操作在兼容硬件上获得显著加速。
3. **损失缩放（主要用于 FP16）：** 为了防止在 FP16 中计算的梯度下溢（因范围有限而变为零），在反向传播开始前，损失值乘以一个因子 $S$ 进行放大。这种乘法有效地放大了整个反向传播过程中的梯度：
   
   $$
   \text{缩放后的损失} = \text{损失} \times S
   $$
   
   这些更大的、经过缩放的梯度在 FP16 中表示时更不容易变为零。
4. **梯度反缩放：** 在优化器更新 FP32 主权重之前，计算出的梯度（现在通常又回到 FP32）通过除以相同的因子 $S$ 进行反缩放：
   
   $$
   \text{原始梯度} = \frac{\text{缩放后的梯度}}{S}
   $$
   
5. **动态损失缩放：** 缩放因子 $S$ 通常是动态调整的。如果在反缩放后在梯度中检测到上溢（NaN 或 Inf 值），该批次的优化器步骤将被跳过，并且 $S$ 会被减小（例如，减半）。如果训练在一定步数内没有发生上溢，则 $S$ 可能会增加，以便更充分地使用 FP16 范围。BF16 由于其更宽的范围，通常需要较小的缩放，有时甚至不需要缩放。
6. **FP32 操作：** 某些操作，如大规模归约、批量归一化 (normalization)更新或损失计算，可能会保留在 FP32 中以保持数值精度。现代自动混合精度（AMP）实现会根据安全实践自动处理这些选择。

### 在扩散训练中实现混合精度

主流深度学习 (deep learning)框架提供了便捷的抽象来启用混合精度：

- **PyTorch：** 使用 `torch.cuda.amp`（自动混合精度）。主要组件是 `torch.autocast`，用于在其上下文 (context)中自动转换操作，以及 `torch.cuda.amp.GradScaler`，用于管理损失缩放。
- **TensorFlow：** 使用 `tf.keras.mixed_precision`。你设置一个全局策略（例如 `mixed_float16` 或 `mixed_bfloat16`），并用 `LossScaleOptimizer` 包装优化器。

以下是一个使用 PyTorch `amp` 的简化示例：

```python
import torch
# scaler 通常在训练循环外部初始化一次
scaler = torch.cuda.amp.GradScaler(enabled=True) # 启用 AMP

# 在训练循环内部：
optimizer.zero_grad()

# 对前向传播（模型执行和损失计算）使用 autocast
# 自动为符合条件的操作选择 FP16/BF16
with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
    # 假设模型接收带噪声图像和时间步
    predicted_noise = model(noisy_images, timesteps)
    loss = loss_fn(predicted_noise, target_noise)

# 缩放损失。在缩放后的损失上调用 backward() 以生成缩放后的梯度。
scaler.scale(loss).backward()

# scaler.step() 首先对优化器分配的参数的梯度进行反缩放。
# 如果梯度不是 inf/NaN，则调用 optimizer.step()。
# 否则，optimizer.step() 被跳过。
scaler.step(optimizer)

# 更新下一次迭代的缩放因子。
scaler.update()

# 继续训练循环...
```

**扩散模型的具体考量：**

- **内存节省：** 扩散模型，特别是带有注意力机制 (attention mechanism)的 U-Net 或大型 Transformer (DiTs)，具有庞大的内存占用。混合精度可以将激活和梯度所需的内存减半，从而允许使用更大的批次大小，或将更大的模型适应到现有硬件上。
- **训练速度：** 考虑到扩散训练的迭代特性（通常是数十万或数百万步），混合精度带来的 2 倍或更高加速大大减少了总体训练时间。
- **数值稳定性：** 尽管通常稳定，但仍需密切监控训练过程。如果硬件支持（例如 NVIDIA A100/H100、Google TPUs v2/v3/v4），BF16 通常是首选，因为它能缓解许多 FP16 范围问题。如果使用 FP16，仔细监控 `GradScaler` 的缩放因子很有必要。如果缩放因子频繁下降，可能表明存在数值问题。确保涉及时间嵌入 (embedding)或归一化 (normalization)层的操作能正确处理精度变化；框架通常会处理此问题，但自定义层可能需要检查。

### 优点与权衡总结

**优点：**

- **更快的训练：** 在兼容硬件（Tensor Cores，TPU）上实现显著加速（2倍以上）。
- **减少内存使用：** 激活、梯度和潜在权重 (weight)的内存需求降低，从而可以使用更大的模型或批次大小。

**权衡：**

- **硬件依赖性：** 速度优势依赖于为低精度数学设计的硬件加速器。
- **潜在的数值问题：** 主要是在 FP16 上，需要损失缩放和仔细监控。BF16 通常更可靠。
- **微小的精度差异：** 在极少数情况下，与完整的 FP32 训练相比，混合精度可能导致最终模型收敛存在非常微小的差异，但在正确实施时通常可以忽略不计。

在实践中，混合精度训练是一种标准且非常有效的技术，用于加速大型扩散模型的开发和部署。它使得研究人员和工程师能够更快地迭代和训练更强大的模型，通过高效地使用现代硬件加速器。在训练大型扩散模型时，采用混合精度通常不只是一种优化，而是一种必要。

## 参考资料

- [Mixed-Precision Training](https://arxiv.org/abs/1710.03740) — Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, Hao Wu (2018)
  Journal: International Conference on Learning Representations; DOI: [10.48550/arXiv.1710.03740](https://doi.org/10.48550/arXiv.1710.03740)
  介绍了混合精度训练的概念及FP16的损失缩放等核心技术。
- [Training with BFloat16 on NVIDIA GPUs](https://developer.nvidia.com/blog/training-with-bfloat16-on-nvidia-gpus/) — Nikolaos Markidis, Andrew P. Overman, Michael Garland, and Jan-Dirk Wegner (2020)
  Publisher: NVIDIA
  描述了BF16的设计、相对于FP16的优点以及在NVIDIA GPU上的实现。
- [Automatic Mixed Precision package - torch.cuda.amp](https://pytorch.org/docs/stable/amp.html) — PyTorch Documentation (2024)
  Publisher: PyTorch Foundation
  PyTorch中利用`torch.cuda.amp`和`GradScaler`实现混合精度训练的官方指南。
- [Mixed precision training](https://www.tensorflow.org/guide/mixed_precision) — TensorFlow Documentation (2024)
  Publisher: Google
  TensorFlow中利用`tf.keras.mixed_precision`实现混合精度训练的官方指南。
