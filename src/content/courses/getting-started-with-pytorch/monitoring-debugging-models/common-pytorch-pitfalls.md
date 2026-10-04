---
course: "getting-started-with-pytorch"
chapter: "monitoring-debugging-models"
lesson: "common-pytorch-pitfalls"
sourceId: 2221
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-8-monitoring-debugging-models/common-pytorch-pitfalls"
title: "PyTorch开发中常见错误"
description: "识别形状不匹配、设备放置错误和静默失败等常见问题。"
order: 1
plots: []
sourceHash: "aa941a4ef73eae1597a01fa1cd19ddd32a19718e4f19d3ca866365bf140fb5c6"
sourceCorrections: []
---

随着您开始构建较复杂的PyTorch模型和训练循环，您将不可避免地遇到错误，或模型行为不符预期的情况。有些错误会给出清晰的消息，而另一些则可能更难察觉，导致性能不佳却无明显崩溃。识别常见模式是高效调试的第一步。以下是一些PyTorch开发中常出现的问题。

### 形状不匹配

PyTorch中最常见的运行时错误可能涉及张量形状不兼容。这通常发生在某层的输出形状与下一层的预期输入形状不匹配，或输入数据的形状与模型第一层不一致时。

考虑一个简单序列：一个卷积层后接一个全连接（线性）层。`nn.Conv2d`层需要形状为 (Batch Size, Input Channels, Height, Width) 的输入张量，通常简写为 $(N, C_{in}, H_{in}, W_{in})$。它会生成形状为 $(N, C_{out}, H_{out}, W_{out})$ 的输出。然而，`nn.Linear`层需要一个形状为 (Batch Size, Input Features) 的2D输入，或 $(N, \text{输入特征数})$。直接连接它们而不改变形状会导致错误。

```python
import torch
import torch.nn as nn

# 示例层
conv_layer = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
linear_layer = nn.Linear(in_features=???, out_features=10) # 问题：in_features 是什么？

# 模拟输入数据
input_data = torch.randn(64, 3, 32, 32) # (批量大小, 输入通道数, 高, 宽)

# 卷积层前向传播
conv_output = conv_layer(input_data)
print(f"Conv output shape: {conv_output.shape}")
# 输出：卷积输出形状：torch.Size([64, 16, 32, 32])

# 尝试直接传递给线性层（会失败）
# output = linear_layer(conv_output) # 这会引发一个 RuntimeError

# 正确方法需要展平
flattened_output = conv_output.view(conv_output.size(0), -1) # 展平除批量维度外的所有维度
print(f"Flattened output shape: {flattened_output.shape}")
# 输出：展平后输出形状：torch.Size([64, 16384]) # 16 * 32 * 32 = 16384

# 现在我们知道了线性层所需的 in_features
correct_linear_layer = nn.Linear(in_features=16384, out_features=10)
output = correct_linear_layer(flattened_output)
print(f"Final output shape: {output.shape}")
# 输出：最终输出形状：torch.Size([64, 10])
```

错误通常看起来像：`RuntimeError: size mismatch, m1: [64 x 16384], m2: [? x 10]`。`m1` 通常指传递给层的输入张量，`m2` 指的是层的权重 (weight)矩阵。错误消息表明PyTorch尝试进行乘法运算的形状。调试这些错误包括：

1. 使用`.shape`打印前一层输出张量的形状。
2. 计算有问题层预期的输入特征数。对于卷积后的`nn.Linear`，这通常涉及使用`tensor.view(batch_size, -1)`将 $(N, C, H, W)$ 输出展平为 $(N, C*H*W)$。
3. 确保`nn.Linear`层的`in_features`参数 (parameter)与展平后的尺寸匹配。

### 设备不匹配 (CPU/GPU)

PyTorch允许在不同设备上进行计算，主要是在CPU和NVIDIA GPU（使用CUDA）。当您尝试对位于不同设备上的张量进行操作时，会经常发生运行时错误。例如，如果您的模型被移至GPU (`model.to('cuda')`)，但您的输入数据保留在CPU上，前向传播将失败。

```python
# 假设 CUDA 可用
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

model = nn.Linear(10, 5)
input_cpu = torch.randn(1, 10) # 默认在CPU上的张量

# 将模型移至GPU（如果可用）
model.to(device)
print(f"Model device: {next(model.parameters()).device}")

# 尝试用CPU张量和GPU模型进行前向传播（如果设备是cuda则会失败）
try:
    output = model(input_cpu)
except RuntimeError as e:
    print(f"Error: {e}")
    # 输出可能是：错误：要求所有张量在同一设备上，
    # 但找到了至少两个设备，cuda:0 和 cpu！

# 正确方法：将输入张量移至与模型相同的设备
input_gpu = input_cpu.to(device)
print(f"Input tensor device: {input_gpu.device}")

output = model(input_gpu) # 这可以正常运行
print(f"Output tensor device: {output.device}")
print("Forward pass successful!")
```

> 导致设备不匹配错误的常见情况。在前向传播之前，输入张量必须移动到与模型相同的设备上（例如GPU）。

错误消息 `RuntimeError: Expected all tensors to be on the same device...` 相当明确。调试方法包括：

1. 在脚本早期确定目标`device`（检查`torch.cuda.is_available()`）。
2. 使用`model.to(device)`将模型移动到目标设备。
3. 在训练或评估循环中，使用`data = data.to(device)`和`targets = targets.to(device)`将输入数据张量明确地移动到同一目标设备。
4. 如果不确定，检查张量和模型参数 (parameter)的`.device`属性（`next(model.parameters()).device`）。

### 损失函数 (loss function)或目标形状/类型不正确

选择正确的损失函数很重要，但您还需要确保您的模型输出和目标标签具有该损失函数预期的形状和数据类型。使用错误的组合可能不会总是导致崩溃，但会导致“静默”失败，即损失减小但模型未能正确学习实际任务。

- **`nn.CrossEntropyLoss`**：常用于多类别分类。
  - 需要模型输出的原始、未归一化 (normalization)分数（logits），通常形状为 $(N, C)$，N 表示批量大小，C 表示类别数量。
  - 需要目标标签为类别索引（长整型），通常形状为 $(N)$。
  - 通常不应在此损失函数 *之前* 应用 `softmax` 函数，因为`CrossEntropyLoss`结合了`LogSoftmax`和`NLLLoss`。
- **`nn.MSELoss` (均方误差)**：常用于回归任务。
  - 需要模型输出和目标张量具有相同的形状（例如，$(N, \text{输出特征数})$）。
  - 两个张量通常都应是浮点类型。
- **`nn.BCEWithLogitsLoss`**：用于二元分类或多标签分类。
  - 需要模型输出的原始logits，形状为 $(N, *)$，这里的 $*$ 是一个或多个维度。
  - 需要目标标签为相同形状 $(N, *)$ 的概率（浮点数），通常包含0和1。
  - 在此损失函数 *之前* 使用 `sigmoid` 是不正确的，因为它已包含了 sigmoid 计算。

此处的不匹配可能导致：

- 如果形状基本不兼容，则引发 `RuntimeError`。
- 如果数据类型错误（例如，向`CrossEntropyLoss`提供浮点型目标），则梯度计算不正确。
- 模型进行训练但学习效果不佳，因为损失函数衡量的是错误的东西（例如，将`MSELoss`用于分类索引）。

调试需要仔细阅读您选择的损失函数在PyTorch文档中的说明，并验证：

1. 模型输出张量的形状。
2. 目标张量的形状和数据类型（`.dtype`）。
3. 损失函数 *之前* 是否应应用任何激活函数 (activation function)（如`softmax`或`sigmoid`）。

### 梯度流问题

有时，梯度未能如预期地反向传播 (backpropagation)通过网络，导致参数 (parameter)没有得到更新。如果不监控，这可能会静默发生。

- **忘记 `requires_grad=True`:** 尽管`nn.Module`参数会自动带有`requires_grad=True`，但如果您创建的中间张量应是计算图的一部分，请确保它们正确设置了此标志。通常，如果它们是已需要梯度的张量运算的结果，这会自动处理。
- **原地操作:** 某些原地操作（如`tensor.add_()`）可能会干扰较旧PyTorch版本或复杂的计算图中的梯度跟踪。虽然PyTorch已改进其处理方式，但在需要梯度的网络计算中，通常更安全地使用非原地操作版本（`y = x + 1`而不是`x += 1`）。
- **使用NumPy:** 将张量转换为NumPy（`.numpy()`）会将其从计算图中分离。使用该NumPy数组进行的任何后续操作，即使转换回张量，也不会有梯度流回图的原始部分。
- **`.detach()`:** 对张量调用`.detach()`会明确地将其从计算图中移除。这有时是必要的（例如，在评估期间），但如果在训练期间意外使用它将停止梯度。

这种问题的症状是发现某些参数的`.grad`属性在`loss.backward()`之后保持`None`，或者尽管训练循环正在运行，模型性能却没有改善。本章后续您将学习如何更规范地检查梯度。

### 数据加载与预处理错误

错误也可能源于您的`Dataset`实现或数据转换。

- **`__getitem__`不正确:** 返回错误类型的数据（例如，如果模型期望张量，却返回NumPy数组而非张量图像）或不正确的形状。
- **大小不一致:** 如果`__getitem__`返回大小不一的张量（例如，变长序列或不同维度的图像），并且`DataLoader`的`collate_fn`未能正确处理这种填充或堆叠，这可能在批量创建期间导致错误。
- **转换错误:** 不正确地应用转换（例如，错误的归一化 (normalization)常数，在`transforms.Compose`流程中过早或过晚地转换为张量）可以静默地破坏输入到模型的数据。

调试这些错误通常包括：

1. 单独实例化您的`Dataset`。
2. 使用`dataset[i]`手动获取少量项目。
3. 在样本进入`DataLoader` *之前*，检查其形状、数据类型和值范围。

了解这些常见问题有助于您预见潜在问题，并在问题出现时提供诊断的起点。后续章节将提供更结构化的方法，用于调试和监控模型。

## 参考资料

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) — PyTorch Contributors (2023)
  Publisher: PyTorch Foundation
  一个涵盖所有 PyTorch 功能的综合资源，包括张量操作、模型定义、自动微分和数据处理的实用函数。对于理解核心组件和解决常见问题至关重要。
- [Deep Learning with PyTorch](https://www.manning.com/books/deep-learning-with-pytorch) — Eli Stevens, Luca Antiga, and Thomas Viehmann (2020)
  Publisher: Manning Publications
  一本使用 PyTorch 构建和调试深度学习模型的实用、深入指南，涵盖张量操作、网络架构和训练循环。
- [Debugging PyTorch Models](https://www.oreilly.com/library/view/programming-pytorch-for/9781492045359/) — Ian Pointer (2019)
  Publisher: O'Reilly Media, Inc.
  一份官方指南，提供识别和解决 PyTorch 模型问题的实用技术和工具，直接针对本节中讨论的许多常见错误。
- [Autograd Mechanics](https://pytorch.org/docs/stable/notes/autograd.html) — PyTorch Team (2017)
  解释了 PyTorch 自动微分引擎 `autograd` 的内部工作原理，这对于理解梯度如何计算和传播以及诊断梯度流问题非常重要。
