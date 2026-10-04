# 使用 PyTorch AMP 进行混合精度训练

来源：[原文](https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-6-advanced-pytorch-features-tf-users/pytorch-mixed-precision-amp)

[返回章节目录](README.md) · [返回课程目录](../README.md)

训练现代深度学习 (deep learning)模型常常挑战硬件极限，无论是在计算时间还是内存方面。一种有效方法来缓解这些压力是混合精度训练。这种方法将模型的某些部分使用低精度浮点数（如16位浮点数），而其他部分使用高精度数字（如32位浮点数），旨在加速训练并减少内存占用，同时不显著降低模型准确性。

### 什么是混合精度训练？

传统上，大多数神经网络 (neural network)训练都使用32位浮点数（FP32或单精度）进行。虽然FP32提供宽动态范围和良好精度，但与低精度格式相比，其计算可能较慢且需要更多内存。

混合精度训练巧妙地结合使用：

- **FP16（半精度）：** 使用16位表示数字。FP16操作在现代GPU上速度明显更快，特别是那些配备专用硬件（如NVIDIA的Tensor Cores）的GPU。FP16还将存储权重 (weight)、激活和梯度所需的内存减半。
- **FP32（单精度）：** 继续用于需要更高精度以保持数值稳定性和模型准确性的操作，例如权重更新或某些敏感层。

目标是获取FP16的优势（速度、内存），同时减轻其潜在缺点，例如较小的可表示范围，这可能导致溢出（数字变得过大）或下溢（梯度变为零）。

### PyTorch AMP: `torch.cuda.amp`

PyTorch主要通过 `torch.cuda.amp` 模块提供便捷工具进行自动混合精度（AMP）训练。此模块自动化大部分过程，使其相对简单地集成到现有训练脚本中。你将主要使用的两个组成部分是 `autocast` 和 `GradScaler`。

#### `torch.cuda.amp.autocast`

`autocast` 上下文 (context)管理器是选择哪些操作在FP16中运行、哪些保留在FP32中的主要工具。当你为代码的某个部分（通常是前向传播）启用 `autocast` 时，它会自动将符合条件的PyTorch操作的输入转换为FP16。

哪些操作符合条件？

- 最能从FP16中受益的操作，例如卷积和矩阵乘法（线性层），通常在FP16中运行。
- 在FP16中可能出现数值不稳定的操作，例如归约（如求和）或归一化 (normalization)层（如 `BatchNorm`），通常保留在FP32中以保持精度。

`autocast` 动态处理这些转换。例如：

```python
# 模型和数据都在CUDA上
model = MyModel().cuda()
input_data = torch.randn(N, C, H, W, device="cuda")

# 为前向传播启用autocast
with torch.cuda.amp.autocast():
    output = model(input_data)
    loss = loss_fn(output, target) # 损失计算也在autocast下进行

# 来自此损失的梯度将是FP16
# loss.backward() # (我们将了解GradScaler如何修改此部分)
```

在 `autocast` 块内部，如果 `input_data` 是CUDA张量且PyTorch认为操作在FP16中安全高效，则 `model(input_data)` 等操作将在内部对许多计算使用FP16。生成的 `loss` 张量也可能是FP16。

#### `torch.cuda.amp.GradScaler`

虽然 `autocast` 处理前向传播，但在反向传播 (backpropagation)期间对梯度使用FP16可能导致下溢。梯度，特别是对于深度网络或小参数 (parameter)更新，会变得非常小。如果这些小值低于FP16中可表示的最小正数，它们就会变为零，有效停止这些参数的学习。

`GradScaler` 通过在反向传播前缩放损失来帮助避免这种情况。过程如下：

1. **缩放损失：** 计算出的损失（可能在 `autocast` 下为FP16）乘以一个大的缩放因子。这会增大后续梯度的量级。
2. **反向传播：** 在此缩放后的损失上执行 `backward()` 调用。生成的梯度也被缩放。因为它们更大，所以更不容易在FP16中下溢。
3. **反缩放梯度：** 在优化器更新模型权重 (weight)之前，`GradScaler` 通过除以相同的缩放因子来反缩放梯度。将其恢复到正确的量级。
4. **优化器步进：** 优化器然后使用这些反缩放（但现在数值稳定）的梯度来更新权重。
5. **更新缩放因子：** `GradScaler` 动态调整缩放因子。如果梯度在某一步中溢出（变为 `inf` 或 `NaN`），这意味着缩放因子过高，因此在下一次迭代中减小它，并跳过当前迭代的优化器步进。如果在一定步数内没有发生溢出，缩放因子可以增加以进一步提高精度。

### 将AMP集成到你的训练循环中

我们来看看如何修改标准PyTorch训练循环以使用AMP。

**典型训练循环 (FP32)：**

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 假设已定义模型、data_loader和loss_fn
# model = MyModel().cuda()
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
# loss_fn = nn.CrossEntropyLoss()

# for epoch in range(num_epochs):
#     for input_batch, target_batch in data_loader:
#         input_batch, target_batch = input_batch.cuda(), target_batch.cuda()

#         optimizer.zero_grad()

#         outputs = model(input_batch)
#         loss = loss_fn(outputs, target_batch)

#         loss.backward()
#         optimizer.step()
#     print(f"第 {epoch+1} 轮训练完成。")
```

**使用 PyTorch AMP 的训练循环：**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler # 导入AMP组件

# 假设已定义模型、data_loader和loss_fn
# model = MyModel().cuda()
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
# loss_fn = nn.CrossEntropyLoss()

scaler = GradScaler() # 初始化GradScaler

# for epoch in range(num_epochs):
#     for input_batch, target_batch in data_loader:
#         input_batch, target_batch = input_batch.cuda(), target_batch.cuda()

#         optimizer.zero_grad()

#         # 使用autocasting进行前向传播
#         with autocast():
#             outputs = model(input_batch)
#             loss = loss_fn(outputs, target_batch)

#         # 缩放损失并在缩放后的损失上调用backward()
#         scaler.scale(loss).backward()

#         # 反缩放梯度并调用optimizer.step()
#         scaler.step(optimizer)

#         # 更新下一次迭代的缩放因子
#         scaler.update()
#     print(f"第 {epoch+1} 轮训练使用AMP完成。")
```

主要变化是：

1. 初始化 `GradScaler()`。
2. 将前向传播和损失计算包装在 `with autocast():` 块中。
3. 使用 `scaler.scale(loss).backward()` 而不是仅使用 `loss.backward()`。
4. 使用 `scaler.step(optimizer)` 而不是 `optimizer.step()`。
5. 在 `scaler.step(optimizer)` 之后调用 `scaler.update()`。

`optimizer.zero_grad()` 调用可以放置在 `autocast` 块之前或像往常一样在循环开始时。PyTorch建议将梯度设置为 `None` 而不是清零，以获得微小的性能增益，这可以通过 `optimizer.zero_grad(set_to_none=True)` 完成。

### 影响的可视化

AMP 的主要益处是缩短训练时间和减少内存占用。虽然具体数字因模型、GPU和批次大小而异，但改进可能很可观。



[交互图表：AMP对训练的影响](https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-6-advanced-pytorch-features-tf-users/pytorch-mixed-precision-amp#plot-3l0zc8)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "type": "bar",
      "name": "训练时间 (小时)",
      "x": [
        "FP32",
        "AMP"
      ],
      "y": [
        10,
        6
      ],
      "marker": {
        "color": "#228be6"
      }
    },
    {
      "type": "bar",
      "name": "峰值内存 (GB)",
      "x": [
        "FP32",
        "AMP"
      ],
      "y": [
        12,
        7
      ],
      "marker": {
        "color": "#20c997"
      }
    }
  ],
  "layout": {
    "title": {
      "text": "AMP对训练的影响"
    },
    "barmode": "group",
    "xaxis": {
      "title": "精度模式"
    },
    "yaxis": {
      "title": "值"
    }
  }
}
```

</details>



> 这张图表显示了从标准FP32训练切换到自动混合精度（AMP）时，训练时间和峰值内存使用量的潜在减少。

### 与你的TensorFlow使用经验的联系

如果你使用过TensorFlow，你可能熟悉 `tf.keras.mixed_precision`。理念非常相似：

- TensorFlow 使用 `Policy`（例如 `mixed_precision.set_global_policy('mixed_float16')`）来定义层应如何处理混合精度，这与 `autocast` 隐式确定类型的方式有些类似。
- TensorFlow 的 `LossScaleOptimizer` 包装现有优化器以执行损失缩放，类似于PyTorch 的 `GradScaler`。

这两个框架都旨在通过自动化类型转换和损失缩放来简化混合精度的采用。基本原理相同，尽管具体的API调用和实现细节有所不同。PyTorch 的 `autocast` 和 `GradScaler` 提供一种灵活的方式来应用混合精度，通常只需几行代码修改。

### 重要考量

- **硬件支持：** 带有FP16的AMP在带有Tensor Core支持的NVIDIA GPU（Volta、Turing、Ampere 架构及更新版本）上提供最显著的加速。虽然它可以在其他GPU上运行，但性能增益可能不太明显。一些更新的GPU也支持BFloat16（BF16），如果指定 `dtype=torch.bfloat16`，`autocast` 也可以支持BF16。BF16与FP32范围相似但精度较低，通常提供良好的平衡，并且不需要像FP16那样频繁地进行损失缩放。
- **批归一化 (normalization)：** 像 `BatchNorm` 这样的层通常保持其权重 (weight)并在FP32中执行计算，即使在 `autocast` 块内，以确保稳定性。PyTorch 自动处理此问题。
- **梯度裁剪：** 如果你使用梯度裁剪，它应该在 `GradScaler` 反缩放梯度*之后*应用，但在 `optimizer.step()` *之前*。

  ```python
  # scaler.scale(loss).backward()
  # # 就地反缩放优化器分配的参数的梯度
  # scaler.unscale_(optimizer)
  # # 在反缩放后裁剪梯度
  # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
  # scaler.step(optimizer)
  # scaler.update()
  ```
- **保存和加载检查点：** 如果你保存和加载检查点，请记住也要保存和恢复 `GradScaler` 的状态：

  ```python
  # 保存
  checkpoint = {
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'scaler_state_dict': scaler.state_dict(),
      # ... 其他内容
  }
  torch.save(checkpoint, 'my_checkpoint.pth')

  # 加载
  # checkpoint = torch.load('my_checkpoint.pth')
  # model.load_state_dict(checkpoint['model_state_dict'])
  # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  # scaler.load_state_dict(checkpoint['scaler_state_dict'])
  ```
- **数值稳定性：** 虽然AMP旨在保持数值稳定，但在极少数情况下，某些特定操作或模型架构可能仍然出现数值问题。如果你怀疑有问题，可以尝试对模型的某些部分禁用 `autocast` 或明确将特定操作转换为FP32。

通过善用 PyTorch 的 `torch.cuda.amp`，你通常可以实现显著的训练加速和内存节省，只需极少的代码修改，使你能够训练更大的模型或更快地迭代现有模型。这对于任何转向PyTorch并希望优化其训练流程的TensorFlow开发者来说，是一个有价值的工具。

## 参考资料

- [Mixed-Precision Training](https://arxiv.org/abs/1710.03740) — Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, Hao Wu (2018)
  Journal: ICLR 2018; DOI: [10.48550/arXiv.1710.03740](https://doi.org/10.48550/arXiv.1710.03740)
  介绍了混合精度训练的基础技术和优势，包括损失缩放，这些技术已在现代深度学习框架中实现。
- [Automatic Mixed Precision (AMP)](https://pytorch.org/docs/stable/amp.html) — PyTorch Authors (2025)
  PyTorch 官方文档，提供了 `torch.cuda.amp`、`autocast` 和 `GradScaler` 的详细 API 参考和实际使用指南。
- [Mixed precision](https://www.tensorflow.org/guide/mixed_precision) — TensorFlow Authors (2024)
  TensorFlow 官方关于实现混合精度训练的指南，为从 TensorFlow 过渡到 PyTorch 的开发者提供了直接的比较参考。

---

[上一节](02-%E5%88%86%E5%B8%83%E5%BC%8F%E8%AE%AD%E7%BB%83%E6%96%B9%E6%B3%95.md) · [下一节](04-%E5%89%96%E6%9E%90%20PyTorch%20%E4%BB%A3%E7%A0%81%E4%BB%A5%E6%9F%A5%E6%89%BE%E6%80%A7%E8%83%BD%E7%93%B6%E9%A2%88.md)
