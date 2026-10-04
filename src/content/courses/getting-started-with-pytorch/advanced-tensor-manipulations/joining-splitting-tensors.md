---
course: "getting-started-with-pytorch"
chapter: "advanced-tensor-manipulations"
lesson: "joining-splitting-tensors"
sourceId: 2111
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-2-advanced-tensor-manipulations/joining-splitting-tensors"
title: "张量的合并与分割"
description: "沿着指定维度组合多个张量或将单个张量分割成部分。"
order: 3
plots: []
sourceHash: "8cb458f2b7a473afbfc64b9f0c7e0653169f954ec96827756d5597b7126a0630"
sourceCorrections: []
---

许多深度学习 (deep learning)情境下，你需要将多个张量组合成一个，或将一个更大的张量拆分为更小的部分。这可能涉及汇总不同处理步骤的结果、准备数据批次或分离特征。PyTorch 提供了几个函数，用于有效合并和分割张量。

### 合并张量

组合张量是一个常见操作，尤其是在处理数据批次或合并特征表示时。PyTorch 提供了两种主要方式来合并张量：拼接 (`torch.cat`) 和堆叠 (`torch.stack`)。主要区别在于它们是沿着现有维度操作，还是引入一个新维度。

#### 使用 `torch.cat` 进行拼接

`torch.cat` 函数沿着现有维度拼接一系列张量。序列中的所有张量必须形状相同（除了拼接维度），或者为空。

```python
import torch

# 创建两个张量
tensor_a = torch.randn(2, 3)
tensor_b = torch.randn(2, 3)
print(f"Tensor A (Shape: {tensor_a.shape}):\n{tensor_a}")
print(f"Tensor B (Shape: {tensor_b.shape}):\n{tensor_b}\n")

# 沿着维度0（行）进行拼接
# 结果形状: (2+2, 3) = (4, 3)
cat_dim0 = torch.cat((tensor_a, tensor_b), dim=0)
print(f"沿着维度0拼接 (形状: {cat_dim0.shape}):\n{cat_dim0}\n")

# 沿着维度1（列）进行拼接
# 张量必须在其他维度（维度0）上匹配
# 结果形状: (2, 3+3) = (2, 6)
cat_dim1 = torch.cat((tensor_a, tensor_b), dim=1)
print(f"沿着维度1拼接 (形状: {cat_dim1.shape}):\n{cat_dim1}")

# 3D张量示例
tensor_c = torch.randn(1, 2, 3)
tensor_d = torch.randn(1, 2, 3)
# 沿着维度0（批次维度）进行拼接
# 结果形状: (1+1, 2, 3) = (2, 2, 3)
cat_3d_dim0 = torch.cat((tensor_c, tensor_d), dim=0)
print(f"\n3D张量沿着维度0拼接 (形状: {cat_3d_dim0.shape})")
```

请注意，`torch.cat` 增加了指定维度的大小，同时保持其他维度不变。张量在所有维度上都必须大小匹配，*除了*你进行拼接的那个维度。

> `torch.cat` 沿维度0和维度1对两个2x3张量进行拼接的视觉比较。

#### 使用 `torch.stack` 进行堆叠

与 `cat` 不同，`torch.stack` 沿着一个*新*维度连接一系列张量。当你希望从单个示例创建批次或将相关张量分组时，这会很有用。为了 `stack` 能够工作，输入序列中的所有张量必须具有完全相同的形状。

```python
import torch

# 创建两个形状相同的张量
tensor_e = torch.arange(6).reshape(2, 3)
tensor_f = torch.arange(6, 12).reshape(2, 3)
print(f"Tensor E (Shape: {tensor_e.shape}):\n{tensor_e}")
print(f"Tensor F (Shape: {tensor_f.shape}):\n{tensor_f}\n")

# 沿着新维度0进行堆叠
# 结果形状: (2, 2, 3)
stack_dim0 = torch.stack((tensor_e, tensor_f), dim=0)
print(f"沿着新维度0堆叠 (形状: {stack_dim0.shape}):\n{stack_dim0}\n")

# 沿着新维度1进行堆叠
# 结果形状: (2, 2, 3)
stack_dim1 = torch.stack((tensor_e, tensor_f), dim=1)
print(f"沿着新维度1堆叠 (形状: {stack_dim1.shape}):\n{stack_dim1}\n")

# 沿着新维度2（最后一个维度）进行堆叠
# 结果形状: (2, 3, 2)
stack_dim2 = torch.stack((tensor_e, tensor_f), dim=2)
print(f"沿着新维度2堆叠 (形状: {stack_dim2.shape}):\n{stack_dim2}")
```

> `torch.stack` 在 `dim=0` 和 `dim=1` 处插入新维度的视觉比较。请注意原始张量如何成为新张量中的切片。

选择 `cat` 还是 `stack` 取决于你是想沿着现有维度合并，还是创建一个新维度。`cat` 通常用于水平/垂直组合批次或特征，`stack` 则常用于从单个样本创建批次。

### 分割张量

正如你可以合并张量一样，你也经常需要将它们分开。这可能涉及将一个批次拆分回单个样本、将特征与标签分离或为并行处理划分数据。PyTorch 为这些任务提供了 `torch.split` 和 `torch.chunk` 函数。

#### 使用 `torch.split` 按特定大小分割

`torch.split` 函数沿着指定维度将张量分割成块。你可以指定每个块的大小（如果你想要等份），或者提供一个包含每个所需块大小的列表。

```python
import torch

# 创建一个要分割的张量
tensor_g = torch.arange(12).reshape(6, 2)
print(f"原始张量 (形状: {tensor_g.shape}):\n{tensor_g}\n")

# 沿着维度0（行）按大小2分割成块
# 6行 / 2行/块 = 3块
split_equal = torch.split(tensor_g, 2, dim=0)
print("分割成大小为2的等份（dim=0）:")
for i, chunk in enumerate(split_equal):
    print(f" 块 {i} (形状: {chunk.shape}):\n{chunk}")

print("-" * 20)

# 沿着维度0按大小 [1, 2, 3] 分割成块
# 总大小必须等于该维度的大小 (1 + 2 + 3 = 6)
split_unequal = torch.split(tensor_g, [1, 2, 3], dim=0)
print("\n分割成大小不等的块 [1, 2, 3]（dim=0）:")
for i, chunk in enumerate(split_unequal):
    print(f" 块 {i} (形状: {chunk.shape}):\n{chunk}")

print("-" * 20)

# 沿着维度1（列）进行分割
# 形状: (6, 2)。沿着维度1按大小1分割成块
split_dim1 = torch.split(tensor_g, 1, dim=1)
print("\n分割成大小为1的等份（dim=1）:")
for i, chunk in enumerate(split_dim1):
    # 使用 squeeze 移除大小为1的维度，以便更清晰地显示
    print(f" 块 {i} (形状: {chunk.shape}):\n{chunk.squeeze()}")
```

`torch.split` 返回一个张量元组。如果你为 `split_size_or_sections` 参数 (parameter)提供一个整数，PyTorch 会沿着指定的 `dim` 将张量分割成该大小的块。如果维度大小不能被分割大小完全整除，最后一个块会更小。如果你提供一个大小列表，它们的总和必须等于被分割维度的大小。

#### 使用 `torch.chunk` 按数量分割

另一种方法是，`torch.chunk` 沿着给定维度将张量分割成指定*数量*的块。PyTorch 会尝试使这些块的大小尽可能相等。与需要指定块大小的 `torch.split` 不同，`chunk` 只需指定所需的块数量。

```python
import torch

# 创建一个张量
tensor_h = torch.arange(10).reshape(5, 2) # 沿着维度0的大小为5
print(f"原始张量 (形状: {tensor_h.shape}):\n{tensor_h}\n")

# 沿着维度0分割成3个块
# 5行 / 3块 -> 大小将是 [2, 2, 1] （前几个块取 ceil(5/3)=2）
chunked_tensor = torch.chunk(tensor_h, 3, dim=0)
print("分割成3个部分（dim=0）:")
for i, chunk in enumerate(chunked_tensor):
    print(f" 块 {i} (形状: {chunk.shape}):\n{chunk}")

print("-" * 20)

# 创建另一个张量
tensor_i = torch.arange(12).reshape(3, 4) # 沿着维度1的大小为4
print(f"\n原始张量 (形状: {tensor_i.shape}):\n{tensor_i}\n")

# 沿着维度1分割成2个块
# 4列 / 2块 -> 大小将是 [2, 2] （ceil(4/2)=2）
chunked_tensor_dim1 = torch.chunk(tensor_i, 2, dim=1)
print("分割成2个部分（dim=1）:")
for i, chunk in enumerate(chunked_tensor_dim1):
    print(f" 块 {i} (形状: {chunk.shape}):\n{chunk}")
```

当你知道想要多少个部分，而不关心维度大小是否能被均匀整除时，`torch.chunk` 很方便。当你需要大小精确且可能变化的块时，`torch.split` 提供了更多的控制。

掌握这些合并和分割操作很重要，可以帮助你有效处理数据，因为它会流经你的深度学习 (deep learning)管线的不同阶段，从初始加载和预处理，到训练的批处理，以及模型输出的分析。

## 参考资料

- [torch.cat](https://pytorch.org/docs/stable/generated/torch.cat.html) — PyTorch Documentation (2024)
  Publisher: PyTorch Foundation
  官方文档，详细说明了torch.cat沿现有维度连接张量的用法和行为。
- [torch.stack](https://pytorch.org/docs/stable/generated/torch.stack.html) — PyTorch Documentation (2024)
  Publisher: PyTorch Foundation
  官方文档，介绍了torch.stack的用法，用于沿新维度堆叠张量。
- [torch.split](https://pytorch.org/docs/stable/generated/torch.split.html) — PyTorch Documentation (2024)
  Publisher: PyTorch Foundation
  官方文档，提供了torch.split的详细信息，该函数用于将张量分割成指定大小的块。
- [torch.chunk](https://pytorch.org/docs/stable/generated/torch.chunk.html) — PyTorch Documentation (2024)
  官方文档，解释了torch.chunk如何将张量分割成特定数量的块。
