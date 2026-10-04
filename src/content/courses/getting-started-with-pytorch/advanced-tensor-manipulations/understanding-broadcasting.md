---
course: "getting-started-with-pytorch"
chapter: "advanced-tensor-manipulations"
lesson: "understanding-broadcasting"
sourceId: 2115
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-2-advanced-tensor-manipulations/understanding-broadcasting"
title: "理解广播机制"
description: "了解 PyTorch 如何自动扩展张量维度以进行逐元素操作。"
order: 4
plots: []
sourceHash: "f9e7313d30381da0ae3965f716323cf206936e243f3d115b9b86ac3094937278"
sourceCorrections: []
---

当对张量执行逐元素操作（如加法、减法或乘法）时，它们的形状通常需要对齐 (alignment)。但是，手动调整或重复张量以匹配形状可能会很繁琐且效率低下，尤其是在处理大型数据集时。PyTorch 通过一种称为\*\*广播（broadcasting）\*\*的机制解决了这个问题。

广播提供了一套规则，允许 PyTorch 在执行操作时自动扩展张量维度，前提是它们的形状满足特定的兼容标准。这在许多常见情况下省去了显式维度扩展的需要，使得代码更简洁，内存使用更优化，因为实际数据并未重复；只有计算行为像数据重复了一样。

### 广播规则

PyTorch 通过逐元素比较两个张量的形状来判断它们是否“可广播”，比较从*末尾*（最右侧）维度开始。如果满足以下条件，则两个张量可兼容进行广播（从右到左比较每个维度对）：

1. **维度相等：** 维度大小相等。
2. **其中一个维度为 1：** 两个维度中的一个为 1。
3. **缺少维度：** 一个张量不具备该维度（在此比较中，其大小被视为 1）。

如果所有维度对都满足这些条件，则张量是可广播的。结果张量的形状将沿每个维度对取最大尺寸。如果任何维度对不满足条件（即，维度不同且都不为 1），则会引发 `RuntimeError`。

我们来分析一下这个过程：

1. **对齐 (alignment)形状：** 张量根据它们的末尾维度进行对齐。如果一个张量的维度少于另一个，那么为了对齐，会在其形状前面添加大小为 1 的维度。
2. **检查兼容性并确定结果形状：** 从最右侧维度开始，比较尺寸：
   - 如果维度相等，则结果维度大小就是该尺寸。
   - 如果一个维度为 1，则结果维度大小是另一个（较大）维度的大小。
   - 如果一个张量缺少某个维度（由于对齐），则结果维度大小是另一个张量中该维度的大小。
3. **执行操作：** 操作的执行方式，就像是沿着给定维度大小为 1 的张量，其值被复制以匹配另一个张量中对应维度的大小一样。

### 广播示例

我们用代码示例来说明。

#### 标量与张量

将标量（一个 0 维张量）添加到任何张量时，总是通过广播机制生效。标量会有效地扩展以匹配张量的形状。

```python
import torch

# 张量 A: 形状 [2, 3]
a = torch.tensor([[1, 2, 3], [4, 5, 6]])
# 标量 B: 形状 [] (0 维度)
b = torch.tensor(10)

# 将标量添加到张量
c = a + b

print(f"Shape of a: {a.shape}")
# 张量 a 的形状: torch.Size([2, 3])
print(f"Shape of b: {b.shape}")
# 标量 b 的形状: torch.Size([])
print(f"Shape of c: {c.shape}")
# 张量 c 的形状: torch.Size([2, 3])
print(f"Result c:\n{c}")
# 结果 c:
# tensor([[11, 12, 13],
#         [14, 15, 16]])
```

这里，`b`（形状 `[]`）被广播到形状 `[2, 3]` 以匹配 `a`。

#### 行向量 (vector)与矩阵

考虑将一个行向量（形状 `[3]`）添加到一个矩阵（形状 `[2, 3]`）中。

```python
# 张量 A: 形状 [2, 3]
a = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
# 张量 B: 形状 [3] (为了广播，可以视为 [1, 3])
b = torch.tensor([10, 20, 30])

# 将行向量添加到矩阵
c = a + b

print(f"Shape of a: {a.shape}") # torch.Size([2, 3])
print(f"Shape of b: {b.shape}") # torch.Size([3])
print(f"Shape of c: {c.shape}") # torch.Size([2, 3])
print(f"Result c:\n{c}")
# 结果 c:
# tensor([[11, 22, 33],
#         [14, 25, 36]])
```

- **对齐 (alignment)：** `a` 的形状为 `[2, 3]`。`b` 的形状为 `[3]`。右侧对齐结果如下：

  ```
    张量 A:   2 x 3
    张量 B:       3
  ```
- **兼容性检查：**
  - 末尾维度：`3` 等于 `3`。兼容。结果维度大小为 `3`。
  - 下一个维度：`a` 为 `2`，`b` 在此处没有维度（隐式大小为 `1`）。兼容。结果维度大小为 `2`。
- **结果形状：** `[2, 3]`。
- **扩展：** 张量 `b` 被视为形状 `[1, 3]`，并且其单行沿第一个维度复制以匹配 `a` 的形状 `[2, 3]`。

#### 列向量与矩阵

现在，我们来将一个列向量（形状 `[2, 1]`）添加至同一矩阵（形状 `[2, 3]`）。

```python
# 张量 A: 形状 [2, 3]
a = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
# 张量 B: 形状 [2, 1]
b = torch.tensor([[10], [20]])

# 将列向量添加到矩阵
c = a + b

print(f"Shape of a: {a.shape}") # torch.Size([2, 3])
print(f"Shape of b: {b.shape}") # torch.Size([2, 1])
print(f"Shape of c: {c.shape}") # torch.Size([2, 3])
print(f"Result c:\n{c}")
# 结果 c:
# tensor([[11, 12, 13],
#         [24, 25, 26]])
```

- **对齐：**

  ```
    张量 A:   2 x 3
    张量 B:   2 x 1
  ```
- **兼容性检查：**
  - 末尾维度：`a` 为 `3`，`b` 为 `1`。兼容（其中一个为 1）。结果维度大小为 `3`。
  - 下一个维度：`a` 为 `2`，`b` 为 `2`。兼容（相等）。结果维度大小为 `2`。
- **结果形状：** `[2, 3]`。
- **扩展：** 张量 `b` 中大小为 `1` 的维度（列维度）通过跨列复制值来扩展，以匹配 `a` 的形状 `[2, 3]`。

#### 可视化示例

我们来可视化张量 `A`（形状 `[3, 1]`）和 `B`（形状 `[4]`）的广播过程。

> 张量 A (形状 [3, 1]) 和张量 B (形状 [4]) 进行广播加法的示意图。张量 A 的第二个维度（大小 1）扩展到 4。张量 B 获得一个大小为 1 的前置维度（变为形状 [1, 4]），然后扩展到大小 3。两者都有效地变为形状 [3, 4] 以进行逐元素加法。

#### 不兼容的形状

如果非匹配维度不为 1，则广播会失败。

```python
# 张量 A: 形状 [2, 3]
a = torch.tensor([[1, 2, 3], [4, 5, 6]])
# 张量 B: 形状 [2]
b = torch.tensor([10, 20])

try:
    c = a + b
except RuntimeError as e:
    print(f"Error: {e}")
# 错误: 张量 a (3) 的大小必须与张量 b (2) 在非单例维度 1 处匹配
```

- **对齐：**

  ```
    张量 A:   2 x 3
    张量 B:       2
  ```
- **兼容性检查：**
  - 末尾维度：`a` 为 `3`，`b` 为 `2`。两者都不为 1。**不兼容。** 操作失败。

### 常见用途

广播在神经网络 (neural network)中经常使用：

- **添加偏置 (bias)：** 将偏置向量 (vector)（形状 `[output_features]`）添加到线性层的输出（形状 `[batch_size, output_features]`）。
- **归一化 (normalization)：** 从一批数据中减去均值（标量或按特征向量）并除以标准差（标量或按特征向量）。
- **应用掩码：** 将数据与可能具有较少维度的布尔掩码进行逐元素乘法。

理解广播对于编写简洁高效的 PyTorch 代码非常重要。它允许你自然地对不同形状的张量执行操作，只要它们遵守兼容性规则，从而简化了许多常见的数据处理和建模任务。

## 参考资料

- [Broadcasting Semantics](https://pytorch.org/docs/stable/notes/broadcasting.html) — PyTorch Authors (2021)
  解释 PyTorch 在张量运算中广播规则的官方文档。
- [Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) — NumPy Developers (2024)
  定义了数组操作的基本广播规则，PyTorch 紧密遵循这些规则。
- [2.1. NDArray (Tensor) - Operations](https://d2l.ai/chapter_introduction/ndarray.html) — Aston Zhang, Zachary C. Lipton, Mu Li, and Alexander J. Smola (2023)
  Publisher: Cambridge University Press
  一本广泛使用的开源深度学习教科书中的章节，涵盖张量运算，包括广播，并提供了 PyTorch 示例。
