# 使用 torch.cuda.amp 进行混合精度训练

来源：[原文](https://apxml.com/zh/courses/advanced-pytorch/chapter-3-optimization-training-strategies/mixed-precision-training)

[返回章节目录](README.md) · [返回课程目录](../README.md)

训练大型深度学习 (deep learning)模型计算量很大，常常受限于 GPU 处理速度和内存容量。虽然标准的单精度浮点数（FP32）提供了较宽的动态范围和良好的精度，但神经网络 (neural network)中的许多操作都可以使用半精度（FP16）数充分完成。采用 FP16 可以大大加快计算速度，尤其是在配备 Tensor Cores 的 NVIDIA GPU 上，并能显著减少激活值和梯度的内存占用。

然而，简单地将整个模型切换到 FP16 通常会导致数值不稳定。FP16 的可表示范围比 FP32 小得多，因此容易出现梯度下溢（即小梯度值变为零）或梯度上溢（即大值变为无穷大）。这会阻碍或完全停止训练过程。

这就是混合精度训练的作用所在。其主要思路是，在 FP16 能带来显著速度提升和内存节约的操作中（例如大型矩阵乘法和卷积）使用 FP16，同时将重要组成部分（如权重 (weight)更新和某些对数值敏感的操作）保留在 FP32 中，以保持稳定性和准确性。PyTorch 通过 `torch.cuda.amp` 模块提供了一种方便高效的方法来实现这一点，该模块代表自动混合精度（Automatic Mixed Precision）。

## 使用 `torch.cuda.amp` 进行自动混合精度训练

PyTorch 的 `amp` 模块大体上自动化了混合精度训练的过程。它识别出受益于 FP16 执行的操作，并自动将其输入转换为 FP16，同时将其他操作（如可能需要更高精度的归约或损失函数 (loss function)）保留在 FP32 中。

### `autocast` 上下文 (context)管理器

启用自动类型转换的主要工具是 `torch.cuda.amp.autocast` 上下文管理器。你只需将模型的前向传播（包括损失计算）包装在此上下文中即可。

```python
import torch
from torch.cuda.amp import autocast

# 假设模型、数据、准则已定义并移至 GPU
model = model.cuda()
data = data.cuda()
target = target.cuda()
criterion = criterion.cuda()

# 为前向传播启用自动类型转换
with autocast():
    output = model(data)
    loss = criterion(output, target)

# autocast 上下文之外的操作以默认精度（FP32）运行
# loss.backward() # 我们接下来会修改这一行
# optimizer.step()
```

在 `autocast` 上下文内部，PyTorch 自动确定每项操作的最佳精度：

- 卷积和线性层等操作，它们显著受益于 Tensor Cores，通常以 FP16 运行。
- 可能遭受精度损失或需要更宽范围的操作，例如归约或某些激活函数 (activation function)，可能会保持在 FP32 中。
- 由 `autocast` 管理的区域的输出通常是 FP32 张量，但中间操作可能大量使用了 FP16。

这种选择性的精度处理最大程度地降低了纯 FP16 训练带来的数值风险，同时获得了大部分性能提升。

### 使用 `GradScaler` 进行梯度缩放

虽然 `autocast` 管理前向传播，但将其直接与标准反向传播 (backpropagation)一起使用仍可能导致问题。从 FP16 激活值计算出的梯度有时会非常小，超出 FP16 的可表示范围并变为零（下溢）。这会阻止相应权重 (weight)得到更新。

为了解决这个问题，`torch.cuda.amp` 提供了 `GradScaler`。它的工作方式是在反向传播*之前*缩放损失值。这有效地将所有产生的梯度乘以相同的缩放因子。


$$
\text{损失}_{\text{缩放后}} = \text{损失} \times \text{缩放因子}
$$


$$
\nabla w_{\text{缩放后}} = \nabla w \times \text{缩放因子}
$$


这种缩放将小梯度值推入 FP16 的可表示范围，从而防止下溢。在优化器更新权重之前，`GradScaler` 会将梯度*反向缩放*回其原始值。


$$
\nabla w = \frac{\nabla w_{\text{缩放后}}}{\text{缩放因子}}
$$


`GradScaler` 在训练期间动态调整缩放因子。如果在一定步数内未检测到溢出，它会增加因子，试图最大化 FP16 范围的使用。如果在反向缩放后，梯度中*确实*检测到溢出（梯度变为 `inf` 或 `NaN`），`GradScaler` 会跳过该批次的优化器步骤，并降低缩放因子以防止未来的溢出。

以下是将 `GradScaler` 整合到训练循环中的方法：

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

# --- 初始化 ---
model = YourModel().cuda()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss().cuda()
# 初始化 GradScaler
scaler = GradScaler()

# 模拟数据加载器（请替换为你的实际数据加载）
dataloader = [(torch.randn(16, 3, 224, 224, device='cuda'), torch.randint(0, 10, (16,), device='cuda')) for _ in range(10)]

# --- 训练循环 ---
num_epochs = 5
for epoch in range(num_epochs):
    for data, target in dataloader:
        optimizer.zero_grad()

        # 前向传播并进行自动类型转换
        with autocast():
            output = model(data)
            loss = criterion(output, target)

        # 缩放损失并执行反向传播
        # scaler.scale(loss) 计算 loss * scale_factor
        scaler.scale(loss).backward()

        # 反向缩放梯度并调用 optimizer.step()
        # 如果梯度溢出，会自动跳过此步骤
        scaler.step(optimizer)

        # 更新下一次迭代的缩放因子
        scaler.update()

    print(f"Epoch {epoch+1} completed. Current scale factor: {scaler.get_scale()}")
```

**使用 `GradScaler` 的步骤：**

1. **初始化：** 在训练循环开始前，创建 `GradScaler` 的一个实例。
2. **前向传播：** 将前向传播（模型执行 + 损失计算）包装在 `autocast()` 上下文 (context)内。
3. **损失缩放：** 不再调用 `loss.backward()`，而是调用 `scaler.scale(loss).backward()`。这会计算 `loss * scale_factor`，然后对缩放后的损失调用 `backward()`。
4. **优化器步进：** 将 `optimizer.step()` 替换为 `scaler.step(optimizer)`。此方法会检查由缩放后损失产生的梯度中是否存在溢出（`inf`/`NaN`）。
   - 如果未发生溢出，它会首先反向缩放附加到优化器参数 (parameter)上的梯度（将它们除以 `scale_factor`），然后调用 `optimizer.step()`。
   - 如果*确实*发生了溢出，`scaler.step()` 会跳过 `optimizer.step()` 调用，以防止损坏的权重更新。
5. **更新缩放器：** 在 `scaler.step()` 后调用 `scaler.update()`。这会更新下一次迭代的缩放因子。如果 `step` 因溢出而跳过优化器更新，它会降低缩放；如果更新在一段时间内成功，则可能增加缩放。

### 优点与注意事项

- **速度：** AMP 可以在兼容硬件（NVIDIA Volta、Turing、Ampere 架构及更新版本，它们具备专用于 FP16 计算的 Tensor Cores）上提供显著的速度提升（通常是 1.5 倍到 3 倍或更多）。
- **内存：** 减少了激活值、梯度和潜在的优化器状态（如果使用 FP16 优化器，尽管标准优化器与 AMP 也能很好地配合）的内存使用，从而允许使用更大的批次大小或模型。
- **易用性：** 与手动混合精度管理相比，`torch.cuda.amp` 使实现相对简单。

**重要注意事项：**

- **硬件：** 性能提升在配备 Tensor Cores 的 GPU 上最为明显。在旧硬件上，加速效果可能不那么明显甚至可以忽略不计。
- **数值稳定性：** 尽管 `GradScaler` 有很大帮助，但与纯 FP32 训练相比，某些模型或特定操作仍可能在收敛或最终准确性上表现出微小差异。请密切监控你的训练。
- **批归一化 (normalization)：** 像 `BatchNorm` 这样的层通常需要 FP32 来稳定累积统计数据。`autocast` 通常会正确处理这一点，但如果在实现自定义归一化层时，请注意潜在的细节。
- **保存/加载：** 在保存检查点时，建议将 `GradScaler` 状态与模型和优化器状态一起保存，以便恢复训练。

```python
# 检查点保存示例
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scaler_state_dict': scaler.state_dict(),
    'loss': loss,
    # ... 其他指标或状态
}
torch.save(checkpoint, 'model_checkpoint.pth')

# 检查点加载示例
checkpoint = torch.load('model_checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scaler.load_state_dict(checkpoint['scaler_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
```

使用 `torch.cuda.amp` 进行混合精度训练是一种高效的方法，可加速深度学习 (deep learning)工作流程并训练更大模型。通过巧妙地结合 FP16 和 FP32 计算，它以最少的代码改动带来了显著的性能提升，并应对了低精度算术固有的数值稳定性问题。它已成为现代深度学习实践者工具包中的标准工具，尤其是在处理大型模型或追求更快迭代周期时。

## 参考资料

- [Mixed-Precision Training of Deep Neural Networks](https://arxiv.org/abs/1710.03740) — Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, Hao Wu (2018)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1710.03740](https://doi.org/10.48550/arXiv.1710.03740)
  这篇论文介绍了混合精度训练的技术，包括梯度缩放，这些是 `torch.cuda.amp` 的基础。
- [Automatic Mixed Precision package - torch.cuda.amp](https://pytorch.org/docs/stable/amp.html) — PyTorch Developers (2024)
  提供了关于在 PyTorch 中使用 `torch.cuda.amp`、`autocast` 和 `GradScaler` 的官方且最新的指南。
- [Floating Point and Mixed Precision](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#floating-point-and-mixed-precision) — NVIDIA Corporation (2024)
  Journal: NVIDIA CUDA C++ Programming Guide; Publisher: NVIDIA Corporation
  解释了浮点数的理论和硬件方面，包括在配备 Tensor Cores 的 NVIDIA GPU 上使用混合精度的优势和挑战。

---

[上一节](04-%E6%A2%AF%E5%BA%A6%E8%A3%81%E5%89%AA%E4%B8%8E%E7%B4%AF%E7%A7%AF.md) · [下一节](06-%E5%A4%84%E7%90%86%E5%A4%A7%E5%9E%8B%E6%95%B0%E6%8D%AE%E9%9B%86%E7%9A%84%E7%AD%96%E7%95%A5.md)
