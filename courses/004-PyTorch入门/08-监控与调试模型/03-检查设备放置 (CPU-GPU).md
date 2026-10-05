# 检查设备放置 (CPU/GPU)

来源：[原文](https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-8-monitoring-debugging-models/checking-device-placement)

[返回章节目录](README.md) · [返回课程目录](../README.md)

在 PyTorch 中使用 GPU 时，最常见的运行时错误之一源于尝试在位于不同设备（CPU 与 GPU）上的张量或模块之间执行操作。PyTorch 要求操作中涉及的张量以及执行操作的模型位于同一设备上。未能确保这种一致性会导致明确的错误，并停止执行。识别和纠正这些设备放置问题是主要关注点。

### 理解设备专属性

PyTorch 张量和模型参数 (parameter)有特定的关联设备：可以是 CPU 或特定的 GPU。默认情况下，张量是在 CPU 上创建的。为了使用 GPU 提供的加速，您必须将模型和数据都明确地移动到 GPU 上。

操作通常要求所有参与的张量都在同一设备上。例如，您不能直接将位于 CPU 上的张量与位于 GPU 上的张量相加。同样，位于 GPU 上的模型层（其中包含参数，参数本身也是张量）不能直接处理仍位于 CPU 上的输入张量。

### 识别设备不匹配

此问题最常见的表现是 `RuntimeError`，通常带有类似以下的消息：

```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0! (when checking argument for argument mat1 in method wrapper_addmm)
```

此错误消息很有说明性。它告诉您：

1. 存在设备不匹配。
2. 它识别出涉及的设备（例如 `cpu` 和 `cuda:0`，它表示第一个 GPU）。
3. 它通常指向发生不匹配的具体操作（例如 `addmm`，它用于线性层）。

这通常发生在模型的正向传播或计算损失时，因为在这些地方模型参数 (parameter)直接与输入数据或标签进行交互。

### 检查张量和模型的设备

为了调试这些错误，您首先需要确定您的张量和模型参数 (parameter)位于何处。

**检查张量的设备：**
每个张量都有一个 `.device` 属性，它告诉您其当前位置。

```python
import torch

# 在 CPU 上创建的张量（默认）
cpu_tensor = torch.randn(2, 2)
print(f"cpu_tensor 位于: {cpu_tensor.device}")

# 检查 GPU 是否可用并移动张量
if torch.cuda.is_available():
    gpu_tensor = cpu_tensor.to("cuda")
    print(f"gpu_tensor 位于: {gpu_tensor.device}")
else:
    print("GPU 不可用，无法创建 gpu_tensor。")

# 输出（如果 GPU 可用）：
# cpu_tensor 位于: cpu
# gpu_tensor 位于: cuda:0
```

**检查模型的设备：**
使用 `torch.nn.Module` 定义的模型也需要位于正确的设备上。由于模型由包含参数（参数本身是张量）的层组成，您可以检查任何参数的设备，以推断模型的实际设备。一个常见方法是检查第一个参数的设备：

```python
import torch
import torch.nn as nn

# 定义一个简单模型
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)

# 实例化模型（最初在 CPU 上）
model = SimpleNet()
# 参数最初在 CPU 上
print(f"模型最初位于: {next(model.parameters()).device}")

# 如果 GPU 可用，将模型移动到 GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    model.to(device)
    print(f"模型已移至: {next(model.parameters()).device}")
else:
    device = torch.device("cpu")
    print("GPU 不可用，模型仍留在 CPU 上。")

# 输出（如果 GPU 可用）：
# 模型最初位于: cpu
# 模型已移至: cuda:0
```

请注意，`model.to(device)` 会就地修改模型的参数和缓冲区，*如果*模型已在目标设备上，否则会返回一个已移动到设备上的*新*模型对象。通常的做法是重新赋值结果，例如 `model = model.to(device)`，尽管不重新赋值地调用 `model.to(device)` 通常也有效，因为它会修改内部状态。但是，明确的重新赋值更安全、更清晰。

### 纠正设备放置

一旦您识别出不匹配，解决方案是使用 `.to(device)` 方法将相关的对象（张量或模型）移动到所需的公共设备上。

**建立设备环境：**
一种标准做法是在脚本开始时定义一个 `device` 对象。此对象保存目标设备（如果 GPU 可用则为 GPU，否则为 CPU），并且可以在您的代码中重复使用。

```python
import torch

# 在开始时定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ... 定义您的模型、损失函数、优化器 ...

# 确保模型在正确的设备上
model = SimpleNet().to(device)

# 在您的训练循环中：
# 确保输入数据和标签已移至设备
for inputs, labels in data_loader:
    inputs = inputs.to(device)
    labels = labels.to(device)

    # 现在，模型和数据都在同一设备上
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
```

通过在训练开始*前*持续应用 `.to(device)` 到您的模型，以及在训练循环*内*（对于每个批次）应用于输入数据，您可以确保所有计算都在目标设备上进行，从而防止设备不匹配错误。

### 设备错误的调试流程

如果您遇到指示设备不匹配的 `RuntimeError`：

1. **识别出错的行：** 查看堆栈跟踪以精确找出导致错误的具体操作。
2. **检查操作数：** 就在出错行之前，插入打印语句或使用调试器检查该操作中涉及的所有张量和模型参数 (parameter)的 `.device` 属性。例如，如果错误发生在 `outputs = model(inputs)` 期间，检查 `inputs.device` 和 `next(model.parameters()).device`。如果错误发生在 `loss = criterion(outputs, labels)` 期间，检查 `outputs.device` 和 `labels.device`。
3. **应用 `.to(device)`：** 确保任何被识别为位于错误设备上的张量或模型都使用 `.to(device)` 明确移动到出错操作发生*之前*。请记住在数据加载循环内移动输入和标签。

检查和管理设备放置是编写 PyTorch 代码的基本方面，特别是在使用 GPU 进行加速时。采纳尽早定义 `device` 对象并相应地持续移动模型和数据的做法将帮助您避免许多常见的运行时错误。

## 参考资料

- [CUDA semantics](https://pytorch.org/docs/stable/notes/cuda.html) — PyTorch Core Team (2024)
  Publisher: PyTorch Foundation
  提供关于PyTorch与NVIDIA GPU交互的信息，包括设备管理、设备属性以及多GPU设置的考量。
- [Training a Classifier](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) — PyTorch Core Team (2024)
  Publisher: PyTorch
  一个官方PyTorch教程，实际演示了如何在训练工作流中使用`torch.device`和`.to()`将模型和输入数据移动到GPU。
- [Deep Learning with PyTorch](https://www.manning.com/books/deep-learning-with-pytorch) — Eli Stevens, Luca Antiga, Thomas Viehmann (2020)
  Publisher: Manning Publications
  一本提供PyTorch基础知识详细介绍的书，包括关于在CPU和GPU设备上管理张量和模型的章节和部分。

---

[上一节](02-%E8%B0%83%E8%AF%95%E5%BC%A0%E9%87%8F%E5%BD%A2%E7%8A%B6%E4%B8%8D%E5%8C%B9%E9%85%8D.md) · [下一节](04-%E6%A3%80%E6%9F%A5%E6%A2%AF%E5%BA%A6%E9%97%AE%E9%A2%98%EF%BC%88%E6%B6%88%E5%A4%B1-%E7%88%86%E7%82%B8%EF%BC%89.md)
