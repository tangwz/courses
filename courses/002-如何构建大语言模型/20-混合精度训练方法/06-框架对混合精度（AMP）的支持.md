# 框架对混合精度（AMP）的支持

来源：[原文](https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-20-mixed-precision-training-techniques/framework-support-mixed-precision-amp)

[返回章节目录](README.md) · [返回课程目录](../README.md)

手动实现混合精度训练，小心地在$FP32$、$FP16$或$BF16$之间进行张量类型转换并管理损失缩放，可能既复杂又容易出错。幸运的是，像PyTorch这样的现代深度学习 (deep learning)框架提供了内置工具来自动化此过程的大部分，通常被称为自动混合精度（AMP）。这些工具大大简化了混合精度技术的采用，使工程师能够将精力放在模型架构和训练动态上，而不是底层数值管理。

### PyTorch 自动混合精度 (`torch.cuda.amp`)

PyTorch 通过其 `torch.cuda.amp` 模块提供强有力的 AMP 支持（对于英特尔 GPU 是 `torch.xpu.amp`，对于苹果 M 系列芯片是 `torch.mps.amp` 等，尽管对于大型语言模型（LLM）来说，CUDA 最常用）。你将使用的两个主要组件是 `autocast` 上下文 (context)管理器和 `GradScaler`。

#### `autocast` 上下文管理器

`autocast` 上下文管理器为你代码的选定区域启用自动类型转换，通常是模型的前向传播。启用后，它会识别可以安全高效地以更低精度（$FP16$ 或 $BF16$）运行的操作，并自动相应地转换它们的输入。被认为对数值敏感或不从更低精度中获益的操作（例如通常配置为在$FP32$中运行的归约或归一化 (normalization)层）会保持完全精度。

这是你在模型前向传播周围通常如何使用 `autocast` 的方法：

```python
import torch

# 假设模型、数据、损失函数已定义
# 假设使用 CUDA 设备

scaler = torch.cuda.amp.GradScaler() # 初始化 GradScaler（下一步解释）
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 训练循环迭代内的示例
for data, target in dataloader:
    optimizer.zero_grad()

    # 为前向传播启用自动类型转换
    # 默认情况下，在 CUDA 上使用 FP16
    with torch.cuda.amp.autocast():
        output = model(data)
        loss = loss_fn(output, target)

    # 这里的损失是 FP32，但 backward() 中计算的梯度
    # 将由 GradScaler 进行缩放

    # 反向传播需要 scaler
    scaler.scale(loss).backward()

    # 优化器步骤需要 scaler
    scaler.step(optimizer)

    # 更新下一次迭代的缩放因子
    scaler.update()
```

默认情况下，在CUDA设备上，`autocast` 使用$FP16$数据类型。如果需要，你可以显式指定数据类型，例如使用$BF16$：

```python
# 启用 BFloat16 自动类型转换
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    output = model(data)
    loss = loss_fn(output, target)
# ... 循环的其余部分
```

仅使用 `autocast` 可以管理前向传播精度。然而，当使用$FP16$时，其小的可表示范围可能导致反向传播 (backpropagation)过程中梯度下溢（变为零）。这使得使用梯度缩放变得必要。

#### `GradScaler`

`torch.cuda.amp.GradScaler` 在使用$FP16$时有助于防止梯度下溢。它通过在调用 `backward()` 前将损失值乘以一个缩放因子来发挥作用。这有效地放大了整个反向传播过程中的梯度，将小值推入$FP16$的可表示范围。在优化器更新权重 (weight)之前，`GradScaler` 会对梯度进行反缩放（通过除以相同的缩放因子），使其恢复到原始值，从而确保权重更新正确。如果`inf`或`NaN`值在反缩放步骤中在梯度中检测到（如果缩放因子变得过大，可能会发生这种情况），则该批次的优化器步骤将被跳过，并且缩放因子会在后续迭代中被减小。反之，如果梯度在一段时间内保持稳定，`scaler` 会增加缩放因子，以充分使用更多的$FP16$范围。

典型使用模式包括：

1. 在训练循环开始前初始化一个 `GradScaler` 实例。
2. 在调用 `backward()` 前，使用 `scaler.scale(loss)` 对损失进行缩放。
3. 使用 `scaler.step(optimizer)` 对梯度进行反缩放并调用优化器步骤。此步骤会自动检查 `inf`/`NaN` 梯度，并在必要时跳过更新。
4. 使用 `scaler.update()` 更新下一次迭代的缩放因子。

前面的代码片段已经展示了这种集成方式。请注意，当使用 `autocast` 和$BF16$时，通常*不*需要 `GradScaler`，因为$BF16$的动态范围类似于$FP32$，使得梯度下溢的可能性大大降低。如果使用$BF16$，你通常可以省略 `GradScaler` 步骤：

```python
# BF16 示例（通常不带 GradScaler）
optimizer.zero_grad(set_to_none=True)
# 使用 set_to_none=True 可能会节省内存

# 启用 BFloat16 自动类型转换
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    output = model(data)
    loss = loss_fn(output, target)

# 不带缩放的直接反向传播
loss.backward()

# 直接优化器步骤
optimizer.step()
```

然而，即使使用$BF16$，检查梯度范数发散可能仍然有用，并且一些实践者为了程序的稳定性仍然选择使用$BF16$搭配 `GradScaler`，尽管其主要原因（防止$FP16$下溢）没那么重要。

### 将 AMP 集成到训练循环中

让我们改进典型的训练循环结构，以清楚地展示$FP16$训练的 AMP 组件：

```python
import torch
import torch.cuda.amp as amp

# --- 初始化 ---
model = YourLargeModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()
dataloader = YourDataLoader() # 假设返回数据和目标张量

# 初始化 FP16 的 GradScaler
# 设置 enabled=False 可轻松禁用 AMP
scaler = amp.GradScaler(enabled=True)

# --- 训练循环 ---
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()

        # --- 使用 Autocast 的前向传播 ---
        # 自动对符合条件的操作使用 FP16
        with amp.autocast(enabled=True):
            predictions = model(data)
            loss = loss_fn(predictions, target)
            # 这里的“损失”通常是 FP32，autocast 会转换中间操作

        # --- 带缩放的反向传播 ---
        # 缩放损失，计算梯度，并在内部处理梯度反缩放
        scaler.scale(loss).backward()

        # 可选：梯度裁剪（未缩放的梯度）
        # scaler.unscale_(optimizer) # 在裁剪前对梯度进行反缩放
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # --- 优化器步骤 ---
        # 检查 inf/NaN，如果梯度有效则执行优化器步骤
        scaler.step(optimizer)

        # --- 更新 Scaler ---
        # 调整下一次迭代的缩放因子
        scaler.update()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, "
                  f"Scale: {scaler.get_scale()}")

    # --- 验证循环（通常只使用 autocast 完成）---
    model.eval()
    with torch.no_grad(): # 禁用梯度计算
        for data_val, target_val in \
                validation_dataloader: # 假设 validation_dataloader 存在
             data_val, target_val = (data_val.cuda(),
                                     target_val.cuda())
             with amp.autocast(enabled=True):
                 val_output = model(data_val)
                 # 计算验证指标...
```

这种结构结合了$FP16$混合精度训练的必要的 `autocast` 和 `GradScaler` 步骤。`enabled=True` 标志允许你轻松地开启或关闭 AMP，以便进行比较或调试。请注意，可选的梯度裁剪步骤需要在裁剪*之前*对梯度进行反缩放，这由 `scaler.unscale_` 处理。

### 分布式训练的注意事项

AMP 旨在与 PyTorch 的分布式训练工具（`torch.distributed.DistributedDataParallel`）以及像 DeepSpeed 或 Fully Sharded Data Parallel (FSDP) 这样的更高层级库协同工作。通常，`autocast` 可以应用于每个副本的前向传播中，而 `GradScaler` 可以管理跨进程的缩放，这通常只需要对上面所示的标准 AMP 设置进行少量更改。像 DeepSpeed 这样的框架可能提供自己的集成混合精度处理，这可能会取代或封装 PyTorch 的原生 AMP，因此请查阅它们的具体文档。

通过使用框架对 AMP 的支持，你大大降低了实现混合精度训练的复杂性。这让你更容易地使用$FP16$和$BF16$等更低精度格式的速度和内存优势，使得在现有硬件上训练大型语言模型变得更可行。

## 参考资料

- [Automatic Mixed Precision examples](https://pytorch.org/docs/stable/amp.html) — PyTorch Contributors (2025)
  PyTorch 自动混合精度 (AMP) 功能的官方指南，包括 `autocast` 和 `GradScaler`。
- [Mixed-Precision Training](https://arxiv.org/abs/1710.03740) — Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, Hao Wu (2018)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1710.03740](https://doi.org/10.48550/arXiv.1710.03740)
  引入混合精度训练和梯度缩放等技术的开创性论文，对 $FP16$ 尤其重要。
- [BFloat16: A New Universal Floating Point Format for Machine Learning](https://ieeexplore.ieee.org/document/8750800) — Naveen Rao, Brian Van Essen, Andrew R. Thorson (2019)
  Journal: Proceedings of the 2019 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW); Publisher: IEEE; Pages: 806-813; DOI: [10.1109/IPDPSW.2019.00169](https://doi.org/10.1109/IPDPSW.2019.00169)
  描述 BFloat16 格式及其在机器学习应用中的优势，并与 FP16 和 FP32 进行对比。

---

[上一节](05-%E4%BD%BF%E7%94%A8%20BF16%20%28BFloat16%29%20%E6%A0%BC%E5%BC%8F.md) · [下一节](../21-%E5%86%85%E5%9C%A8%E8%AF%84%E4%BC%B0%E6%8C%87%E6%A0%87/01-%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E8%AF%84%E4%BC%B0%E6%96%B9%E6%B3%95.md)
