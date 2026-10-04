---
course: "advanced-pytorch"
chapter: "pytorch-internals-autograd"
lesson: "memory-management"
sourceId: 2675
sourceUrl: "https://apxml.com/zh/courses/advanced-pytorch/chapter-1-pytorch-internals-autograd/memory-management"
title: "内存管理考量"
description: "Learn about PyTorch's memory allocation, caching, and strategies for efficient memory usage."
order: 7
plots: []
sourceHash: "ab2d0af2f8ebb10619b328cb4f28564d7dcbbeb903c4989eac5ebea3e69fd2b7"
sourceCorrections: []
---

随着您构建和训练日益复杂的模型，管理内存成为开发和调试的一个重要方面，尤其是在使用GPU时，因为GPU的内存通常比CPU更有限。了解PyTorch如何处理内存分配以及您的操作如何影响内存，对于提高效率和避免常见的内存不足错误很必要。PyTorch的内存管理机制，以及它们与张量结构和自动求导过程的相互作用，在此进行审视。

### 张量存储与内存布局

其核心是，PyTorch张量（`torch.Tensor`）是对由`torch.Storage`对象管理的连续内存块的视图。`Storage`对象保存实际的数值数据，而`Tensor`对象包含形状（大小）、步长和数据类型（`dtype`）等元数据，以及它在`Storage`中的位置信息。

多个张量可以共享同一个底层`Storage`。例如，对张量进行切片或使用`view()`等操作通常会创建一个新的张量对象，它指向相同的存储但具有不同的元数据。

```python
import torch

# 创建一个张量；PyTorch分配存储空间
x = torch.randn(2, 3)
print(f"x storage: {x.storage().data_ptr()}")

# 切片操作会创建一个共享存储的新张量视图
y = x[0, :]
print(f"y storage: {y.storage().data_ptr()}") # 相同的指针
print(f"Do x and y share storage? {x.storage().data_ptr() == y.storage().data_ptr()}")

# 修改y会影响x，因为它们共享存储
y.fill_(1.0)
print("修改y后x的值：\n", x)
```

这种存储共享非常高效，因为它避免了不必要的数据复制。然而，了解这一点很重要，尤其是在执行原地操作时。

张量在内存中的*布局*由其*步长*决定。如果张量的元素在内存中逐行（对于二维张量）顺序排列且没有间隙，则认为该张量是**连续的**。非连续张量可能由转置或某些类型的索引操作产生。

```python
# 连续张量
a = torch.arange(6).reshape(2, 3)
print(f"a is contiguous: {a.is_contiguous()}, Stride: {a.stride()}") # 步长: (3, 1)

# 转置会创建非连续视图
b = a.t()
print(f"b is contiguous: {b.is_contiguous()}, Stride: {b.stride()}") # 步长: (1, 3)

# 访问元素仍然正确，但内存访问模式不同
print("b:\n", b)

# 某些PyTorch函数需要连续张量
# 尝试对非连续张量进行view等操作可能会失败
try:
    b.view(-1)
except RuntimeError as e:
    print(f"\nError viewing non-contiguous tensor: {e}")

# 使用 .contiguous() 获取连续副本
c = b.contiguous()
print(f"c is contiguous: {c.is_contiguous()}, Stride: {c.stride()}") # 步长: (2, 1)
print("c (contiguous version of b):\n", c)
print(f"Does b and c share storage? {b.storage().data_ptr() == c.storage().data_ptr()}") # 否，新的存储空间
```

虽然PyTorch操作通常能正确处理非连续张量，但某些底层操作或接口（例如导出到NumPy或某些自定义扩展）可能需要连续数据。如果原始张量不是连续的，调用`.contiguous()`会创建一个带有全新、连续数据副本的新张量。这会产生内存复制开销。

数据类型（`dtype`）也直接影响内存使用。一个`torch.float32`张量每个元素使用4字节，而`torch.float16`使用2字节，`torch.int64`使用8字节。选择合适的数据类型是提高内存效率的基本要求。

### PyTorch缓存内存分配器

使用CUDA API（`cudaMalloc`、`cudaFree`）在GPU上分配和释放内存可能很慢。为了缓解这个问题，PyTorch为GPU张量采用了一个**缓存内存分配器**。当一个张量被释放时（例如，超出作用域且其引用计数降至零），它所占用的内存不一定立即返回给GPU操作系统。相反，PyTorch将此内存块保留在缓存中。

当需要分配新张量时，PyTorch首先检查其缓存中是否有大小合适的空闲块。如果找到，它会重用该块，避免了对CUDA驱动程序的昂贵调用。这显著加快了张量的创建和删除速度，而这在训练期间经常发生。

> PyTorch缓存分配器与CUDA驱动程序和张量内存交互的简化视图。

您可以查看缓存分配器的状态：

- `torch.cuda.memory_allocated()`：返回默认设备上张量当前占用的总GPU内存（以字节为单位）。
- `torch.cuda.memory_reserved()` 或 `torch.cuda.memory_cached()`（已弃用）：返回缓存分配器管理的总GPU内存（包括已分配的张量和缓存的空闲块）。
- `torch.cuda.max_memory_allocated()`：返回从开始执行或上次重置以来，在任何时间点张量占用的最大GPU内存。
- `torch.cuda.reset_peak_memory_stats()`：重置峰值内存计数器。
- `torch.cuda.memory_summary()`：提供已分配和缓存内存的详细报告，通常有助于发现碎片问题。

有时，您可能希望清除缓存内存，也许是为了使其可供其他GPU应用程序或库使用。您可以使用`torch.cuda.empty_cache()`。

```python
# 需要GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Initial allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MiB")
    print(f"Initial reserved:  {torch.cuda.memory_reserved(device) / 1024**2:.2f} MiB")

    # 分配一些张量
    t1 = torch.randn(1024, 1024, device=device)
    t2 = torch.randn(512, 512, device=device)
    print(f"\nAfter allocation:")
    print(f"Allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MiB")
    print(f"Reserved:  {torch.cuda.memory_reserved(device) / 1024**2:.2f} MiB")

    # 删除张量
    del t1
    del t2
    print(f"\nAfter deleting tensors (before empty_cache):")
    # 已分配内存减少，但由于缓存，保留内存仍然很高
    print(f"Allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MiB")
    print(f"Reserved:  {torch.cuda.memory_reserved(device) / 1024**2:.2f} MiB")

    # 清除缓存
    torch.cuda.empty_cache()
    print(f"\nAfter empty_cache:")
    # 保留内存也减少（尽管可能由于内部分配而不降为零）
    print(f"Allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MiB")
    print(f"Reserved:  {torch.cuda.memory_reserved(device) / 1024**2:.2f} MiB")
else:
    print("CUDA不可用，跳过GPU内存示例。")
```

**重要提示：** `torch.cuda.empty_cache()` **不**会释放当前被活跃张量使用的内存。它只释放未被任何张量引用的缓存块。它主要用于将内存释放回系统以供其他进程使用，而不是在张量仍然存在的情况下减少您*正在运行*的PyTorch脚本的内存占用。它还会产生性能开销，因为后续分配将需要再次请求驱动程序。

缓存分配器的一个副作用是**碎片化**。如果您分配和释放不同大小的张量，缓存最终可能持有许多小的、非连续的空闲块。即使这些缓存块的总大小很大，您也可能无法分配一个大的*连续*块，从而导致内存不足（OOM）错误。`torch.cuda.memory_summary()`可以帮助诊断碎片问题。

### 自动求导和内存

自动求导引擎显著影响内存使用。为了在反向传播 (backpropagation)期间计算梯度，自动求导通常需要存储作为计算图一部分的中间激活值（前向操作的输出）。

- **计算图：** 当对需要梯度的张量（`requires_grad=True`）执行操作时，PyTorch会构建一个图，存储这些操作以及对所涉及张量的引用。这些引用会使张量在内存中保持活跃，即使它们在您的Python代码中可能看起来已超出作用域。
- **反向传播：** 在`loss.backward()`期间，自动求导会反向遍历此图。它使用存储的中间值来计算梯度。一旦梯度被计算并且在反向传播中不再需要进行进一步计算时，持有相应中间激活值的缓冲区通常会被释放。
- **`retain_graph=True`：** 如果您调用`backward(retain_graph=True)`，即使在反向传播完成后，PyTorch也会保留图和中间激活缓冲区。这允许您多次调用`backward()`（例如，计算不同损失相对于相同参数 (parameter)的梯度），但这代价是占用可能大量的内存。仅在必要时使用它。
- **`torch.no_grad()`：** 将代码包裹在`with torch.no_grad():`块中会向PyTorch发出信号，表明此块内的操作不应被自动求导跟踪。这可以防止为这些操作创建计算图，并避免存储中间激活值，从而节省大量内存。在验证或推理 (inference)循环中使用此上下文 (context)管理器是标准做法。
- **`.detach()`：** 对张量调用`.detach()`会创建一个新张量，它共享相同的存储空间但与计算图分离。它不需要梯度，并且不涉及它的操作将不会被跟踪。如果您需要使用张量的值而不跟踪其历史记录（例如，用于日志记录或绘图），这很有用。

考虑这个简单示例：

```python
# 设置
a = torch.randn(100, 100, requires_grad=True)
b = torch.randn(100, 100, requires_grad=True)

# 被自动求导跟踪的操作
c = a * b
d = c.sin()
loss = d.mean()

# 中间张量'c'和'd'被保留在内存中
# 因为反向传播需要它们。

# 调用backward会释放缓冲区（除非retain_graph=True）
loss.backward() # 计算a和b的梯度

# 现在，让我们尝试不跟踪梯度
with torch.no_grad():
    c_no_grad = a * b # 操作已执行，但未被跟踪
    d_no_grad = c_no_grad.sin()
    loss_no_grad = d_no_grad.mean()
    # PyTorch不需要为未来的反向传播存储'c_no_grad'
    # 中间结果的内存可能更早被释放。

print(f"a的梯度：{'存在' if a.grad is not None else '无'}")
# loss_no_grad.backward() # 这将引发错误，因为历史记录未被跟踪。
```

### 高效内存使用的方法

以下是实用的方法：

1. **作用域和`del`：** 当对象不再被引用时，Python的垃圾回收器会回收内存。确保不再需要的大张量超出作用域。如果需要，可以使用`del`语句明确删除引用，尤其是在可能进行内存密集型操作（如`backward()`）或分配新的大张量之前。

def process\_data(data):
intermediate = data \* 2 # 大型中间张量
result = intermediate.sum()
# 如果不删除，'intermediate'可能会在内存中停留更长时间
del intermediate # 明确删除引用
return result
\`\`\`

2. **原地操作：** 以单个下划线（`_`）结尾的操作，如`add_()`、`relu_()`，会直接修改张量，而不是创建新张量。这样可以通过避免为结果分配新张量来节省内存。
   **注意：** 原地修改计算梯度所需的张量可能会破坏反向传播 (backpropagation)。自动求导会跟踪原地操作，如果检测到此类修改干扰梯度计算，就会引发错误。请谨慎使用它们，通常在图中是叶子节点或您确定不会影响所需梯度的张量上使用。

x = torch.randn(1000, 1000)
y = torch.randn(1000, 1000)

# 非原地操作：创建一个新张量z

z = x + y

# 原地操作：直接修改x，为结果张量节省内存

x.add\_(y) # x现在包含x + y的结果
\`\`\`

3. **梯度检查点（激活检查点）：** 对于具有非常深层结构的模型，如果存储所有中间激活值会消耗过多内存，梯度检查点提供了一种权衡。它在前向传播期间只存储一部分激活值，而不是全部。在反向传播期间，它会即时重新计算必要的激活值。这会使用更多的计算时间，但显著减少峰值内存使用。PyTorch为此提供了`torch.utils.checkpoint.checkpoint`。
4. **混合精度训练：** 使用`torch.float16`或`torch.bfloat16`等低精度数据类型，与`torch.float32`相比，存储激活值、梯度和参数 (parameter)所需的内存减少一半。`torch.cuda.amp`（自动混合精度）等库有助于有效管理这一点（第3章介绍）。
5. **数据加载和批大小：** 确保您的数据加载流程高效。如果遇到OOM错误，减小批大小通常是第一步，因为激活值及其梯度会随批大小线性增长。

### 内存问题调试

- **内存不足（OOM）错误：** 当您遇到CUDA OOM错误时，错误消息本身通常会说明请求了多少内存以及有多少可用内存。
  - 使用`torch.cuda.memory_summary()`来查看已分配块和缓存碎片的分布。即使总空闲内存看起来足够，高度碎片化也可能导致OOM。
  - 系统地减小批大小。
  - 检查模型大小和复杂度。
  - 在训练循环的不同位置插入`torch.cuda.memory_allocated()`的打印语句，以找出内存使用量激增的地方。
  - 使用PyTorch分析器（第4章介绍）获取每个操作符的内存使用详细分类。
- **内存泄漏：** 如果内存使用量在训练迭代过程中持续增长而不稳定，您可能存在内存泄漏。这通常发生在带有计算历史的张量在`torch.no_grad()`上下文 (context)之外被无意中累积到列表或字典中时。
  - 泄漏示例：在不分离的情况下将损失存储在列表中：`all_losses.append(loss)`而不是`all_losses.append(loss.item())`或`all_losses.append(loss.detach())`。存储原始的`loss`张量会使其整个计算图保持活跃。
  - 仔细检查张量在迭代之间如何存储。如果您需要张量值而不跟踪其历史记录，可以使用`.item()`从单元素张量获取Python数字，或者使用`.detach()`。

有效的内存管理通常是一个迭代的过程，它需要理解模型行为，应用适当的方法，并使用PyTorch的工具检查和调试内存使用。扎实掌握这些知识在扩展到更大的数据集和更复杂的架构时是不可或缺的。

## 参考资料

- [PyTorch Documentation: CUDA Memory Management](https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management) — PyTorch Developers (2023)
  Publisher: PyTorch Foundation
  介绍 PyTorch 的 GPU 内存分配策略，包括缓存分配器、内存碎片以及监控内存使用的工具。
- [PyTorch Documentation: Automatic Differentiation Package - torch.autograd](https://pytorch.org/docs/stable/autograd.html) — PyTorch Developers (2023)
  解释了 autograd 引擎如何跟踪操作以进行梯度计算及其对张量内存保留的影响。
- [PyTorch Documentation: Automatic Mixed Precision (AMP)](https://pytorch.org/docs/stable/amp.html) — PyTorch Developers (2025)
  Publisher: PyTorch Foundation
  详细说明了自动混合精度在神经网络训练中的应用，以减少内存并加速计算。
