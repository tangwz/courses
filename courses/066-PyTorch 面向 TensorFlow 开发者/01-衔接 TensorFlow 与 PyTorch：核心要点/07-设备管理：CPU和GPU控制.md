# 设备管理：CPU和GPU控制

来源：[原文](https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-1-pytorch-tensorflow-core-concepts/pytorch-device-management)

[返回章节目录](README.md) · [返回课程目录](../README.md)

有效管理计算的运行位置，无论是在 CPU 还是一个或多个 GPU 上，对高效训练深度学习 (deep learning)模型都很重要。如果你使用过 TensorFlow，你可能熟悉使用 `tf.device` 指定设备或依赖 Keras 处理放置。PyTorch 提供了类似的功能，但设备控制方法更明确，这让你能对硬件资源进行细致的指令控制。

### 识别和选择设备

在告诉 PyTorch 在哪里运行操作之前，你需要了解有哪些可用设备。PyTorch 为此提供了直接的函数：

- `torch.cuda.is_available()`: 如果 PyTorch 找到并能访问支持 CUDA 的 GPU，则返回 `True`，否则返回 `False`。
- `torch.cuda.device_count()`: 返回可用 GPU 的数量。
- `torch.cuda.get_device_name(i)`: 返回索引 `i` 处 GPU 的名称（例如，'NVIDIA GeForce RTX 3090'）。

常见做法是检查 GPU 是否可用，如果存在则选择它，否则退回到 CPU。

```python
import torch

if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    print(f"GPU is available. Using device: {device_name}")
    device = torch.device("cuda")
else:
    print("GPU not available, using CPU instead.")
    device = torch.device("cpu")

print(f"Selected device: {device}")
# GPU 可用时的输出示例:
# GPU 可用。使用设备: NVIDIA GeForce RTX 3090
# 已选设备: cuda
# GPU 不可用时的输出示例:
# GPU 不可用，转而使用 CPU。
# 已选设备: cpu
```

在 PyTorch 中，设备由 `torch.device` 对象表示。你可以通过指定设备类型（'cpu' 或 'cuda'）来创建这些对象，如果你有多个 GPU，还可以选择指定其索引：

- `torch.device('cpu')`: 表示系统的 CPU。
- `torch.device('cuda')`: 表示默认 GPU（等同于 `torch.device('cuda:0')`）。
- `torch.device('cuda:0')`: 表示第一个 GPU。
- `torch.device('cuda:1')`: 表示第二个 GPU，依此类推。

在没有 GPU 可用时尝试使用 `torch.device('cuda')` 会导致错误。上面所示的条件检查是处理这种情况的标准方法。

### 将张量移动到设备

一旦你有了 `torch.device` 对象，就可以使用 `.to()` 方法将张量移动到指定设备。此方法是就地操作的，这意味着它会在目标设备上返回一个 *新* 张量；除非你重新赋值，否则它不会修改原始张量。

```python
# 假设 'device' 已如上定义（例如，torch.device('cuda') 或 torch.device('cpu')）

# 创建一个张量（默认为 CPU）
x_cpu = torch.randn(3, 3)
print(f"x_cpu device: {x_cpu.device}")

# 将其移动到所选设备
x_on_device = x_cpu.to(device)
print(f"x_on_device device: {x_on_device.device}")

# 如果 'device' 是 'cuda'，x_on_device 将在 GPU 上。
# 如果 'device' 是 'cpu'，x_on_device 将保留在 CPU 上（如果它原先在 GPU 上，则会被复制）。
```

PyTorch 还提供了便捷方法：

- `tensor.cuda()`: 将张量移动到默认 GPU（等同于 `tensor.to(torch.device('cuda'))`）。
- `tensor.cpu()`: 将张量移动到 CPU（等同于 `tensor.to(torch.device('cpu'))`）。

记住，张量之间的操作通常要求它们位于 *同一设备* 上。尝试对位于不同设备上的张量（例如，一个在 CPU 上，一个在 GPU 上）执行操作会引发运行时错误。

```python
# if torch.cuda.is_available(): # 确保此代码仅在 GPU 存在时运行
#     a_cpu = torch.randn(2, 2)
#     b_gpu = torch.randn(2, 2).cuda()
#     try:
#         c = a_cpu + b_gpu # 这将导致错误
#     except RuntimeError as e:
#         print(f"Error: {e}")
#         # 要解决此问题，请将 a_cpu 移动到与 b_gpu 相同的设备：
#         a_gpu = a_cpu.cuda()
#         c_gpu = a_gpu + b_gpu
#         print(f"GPU 上的和: {c_gpu.device}")
# else:
#     print("跳过 GPU 特定的张量操作示例，因为没有 GPU 可用。")
```

这种明确的张量本地性管理与 TensorFlow 不同，在 TensorFlow 中，`tf.device('/GPU:0')` 块内定义的操作的设备放置可能会更隐式地处理张量传输，或者 Keras 层确保其内部操作使用适当放置的张量。在 PyTorch 中，你更直接地负责确保数据位于所需位置。

### 将模型移动到设备

与张量类似，PyTorch 模型（`torch.nn.Module` 的子类）也需要移动到目标设备。在模型上调用 `.to(device)` 方法会将其所有参数 (parameter)和缓冲区移动到该设备。

```python
import torch.nn as nn

# 定义一个简单模型
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

model = SimpleNet()
print(f"Model parameters device (before move): {next(model.parameters()).device}")

# 将模型移动到所选设备
model.to(device)
print(f"Model parameters device (after move): {next(model.parameters()).device}")

# 现在，传递给 model.forward() 的任何输入数据也必须在 'device' 上
# 例如：
# dummy_input = torch.randn(5, 10).to(device)
# output = model(dummy_input)
# print(f"输出设备: {output.device}")
```

当你调用 `model.to(device)` 时，模型内的所有参数（权重 (weight)和偏置 (bias)）和缓冲区都会被传输。常见做法是在模型实例化后、训练开始前，将其一次性移动到目标设备。随后，在正向传播期间馈送到模型的所有输入数据也必须位于同一设备上。

### 典型设备管理流程

PyTorch 中训练循环的标准模式涉及：

1. 在脚本开头确定 `device`（CPU 或 GPU）。
2. 使用 `model.to(device)` 将你的模型移动到此 `device`。
3. 在训练循环内部，对于你的 `DataLoader` 加载的每个数据批次，在将数据张量（输入和标签）传递给模型或损失函数 (loss function)之前，将它们移动到 `device`。

```python
# 假设 'model' 和 'device' 已经定义，并且模型在 'device' 上
# 假设 'dataloader' 提供批次数据 (inputs, labels)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# criterion = nn.MSELoss() # 损失函数

# for epoch in range(num_epochs):
#     for inputs, labels in dataloader:
#         # 1. 将数据移动到所选设备
#         inputs = inputs.to(device)
#         labels = labels.to(device)

#         # 2. 正向传播
#         outputs = model(inputs)
#         loss = criterion(outputs, labels) # 损失也在 'device' 上计算

#         # 3. 反向传播和优化
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     print(f"第 {epoch+1} 轮完成。")
```

这确保了与模型的正向传播、损失计算和反向传播 (backpropagation)相关的所有计算都在指定设备上进行。

### 与 TensorFlow 的 `tf.device` 对比

在 TensorFlow 中，你可能会使用像 `with tf.device('/GPU:0'):` 这样的上下文 (context)管理器来建议操作和变量的放置位置。TensorFlow 运行时通常会管理放置或必要的数据传输。虽然 `tf.config.set_visible_devices` 提供了对 TensorFlow 可见设备的更明确控制，但每个操作的放置通常由这些上下文管理器或 Keras API 在模型构建期间处理。

PyTorch 则要求你更明确：

- 你创建张量（默认为 CPU）。
- 你明确调用 `tensor.to(device)` 或 `model.to(device)` 来移动它们。
  操作随后将在其输入张量所在的设备上运行。这种直接控制可以非常强大，能清楚显示数据位置，但这也意味着你要负责这些传输。

### 与 NumPy 交互

NumPy 数组总是绑定在 CPU 上。如果你需要将一个位于 GPU 上的 PyTorch 张量转换为 NumPy 数组，你必须首先将其移动到 CPU：

```python
gpu_tensor = torch.randn(2, 2).to(device) # 假设 'device' 是一个 GPU 设备

if gpu_tensor.is_cuda: # 检查张量是否确实在 CUDA 上
    cpu_tensor = gpu_tensor.cpu()
    numpy_array = cpu_tensor.numpy()
    print("Converted GPU tensor to NumPy array via CPU.")
else: # 张量已经在 CPU 上
    numpy_array = gpu_tensor.numpy()
    print("Converted CPU tensor to NumPy array.")

# 要从 NumPy 转换为特定设备上的 PyTorch 张量：
# new_tensor = torch.from_numpy(numpy_array).to(device)
```

在 GPU 张量上忘记在 `.numpy()` 之前调用 `.cpu()` 会导致错误。

### 性能考量

虽然 GPU 可以加速计算，但 CPU 和 GPU 内存之间的数据传输会产生开销。为获得最佳性能：

- 尽量减少频繁的数据传输。将数据移动到 GPU 上，并尽可能让它留在那里进行更多操作。
- 以批次而非逐项方式传输数据。`DataLoaders` 有助于此。
- 在模型需要输入数据之前，将其移动到 GPU。
- 在本课程后面，我们将提及可以帮助找出数据传输瓶颈的性能分析工具。

保存和加载模型时（将在后面的章节中介绍），`torch.load()` 提供了一个 `map_location` 参数 (parameter)。此参数对于将模型加载到与保存模型时不同设备上（例如，将 GPU 训练的模型加载到只有 CPU 的机器上）非常有用。

以下是快速对比：

| 特性 | TensorFlow（常见做法） | PyTorch |
| --- | --- | --- |
| **设备指定** | 字符串（例如，`'/CPU:0'`，`'/GPU:0'`） | `torch.device` 对象（例如，`torch.device('cuda')`） |
| **张量放置** | `with tf.device(...)`，`tf.identity(t, dev)` | `tensor.to(device)`，`tensor.cuda()`，`tensor.cpu()` |
| **模型放置** | Keras 中常为隐式，或通过 `tf.distribute.Strategy` | `model.to(device)` |
| **默认张量设备** | CPU（除非在 GPU 设备范围或 GPU 是默认设备） | CPU |
| **检查 GPU 可用性** | `len(tf.config.list_physical_devices('GPU')) > 0` | `torch.cuda.is_available()` |
| **转换为 NumPy** | `tensor.numpy()`（如果即时执行，与设备无关） | `tensor.cpu().numpy()`（如果张量在 GPU 上） |

PyTorch 的设备管理方法给你直接而明确的控制。虽然这意味着你需要明确管理数据移动，但它也简化了设备相关问题的调试，并清楚显示了你的计算发生在哪里。这种明确的控制为更高级别的应用场景打下良好基础，例如跨多个 GPU 或机器的分布式训练。

## 参考资料

- [CUDA semantics](https://pytorch.org/docs/stable/notes/cuda.html) — PyTorch Authors (2024)
  官方文档，解释PyTorch的GPU支持，包括设备选择、张量移动和常见问题。
- [Dive into Deep Learning](https://d2l.ai/chapter_computational-performance/device-management.html) — Aston Zhang, Zack C. Lipton, Mu Li, Alex Smola (2024)
  Publisher: Cambridge University Press
  在线教材章节，涵盖深度学习框架（包括PyTorch）中的设备管理（CPU/GPU），并提供实用示例。
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — NVIDIA Corporation (2024)
  Publisher: NVIDIA Corporation
  NVIDIA并行计算平台的权威指南，提供关于GPU架构和CUDA编程的基础知识，与高性能深度学习相关。
- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes_fe_for_acceleration.html) — Szymon Migacz (2024)
  Publisher: PyTorch Foundation
  官方指南，详细说明优化PyTorch模型性能的策略，包括高效的设备利用和最小化数据传输开销。

---

[上一节](06-NumPy%20%E5%9C%A8%20PyTorch%20%E5%92%8C%20TensorFlow%20%E4%B8%AD%E7%9A%84%E7%BB%93%E5%90%88.md) · [下一节](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%BC%A0%E9%87%8F%E6%93%8D%E4%BD%9C%E4%B8%8E%E8%87%AA%E5%8A%A8%E6%B1%82%E5%AF%BC.md)
