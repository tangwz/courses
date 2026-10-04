---
course: "advanced-pytorch"
chapter: "deployment-performance-optimization"
lesson: "torchscript-fundamentals"
sourceId: 2730
sourceUrl: "https://apxml.com/zh/courses/advanced-pytorch/chapter-4-deployment-performance-optimization/torchscript-fundamentals"
title: "TorchScript 基础: 追踪与脚本化"
description: "了解 TorchScript 模型序列化中的追踪和脚本的区别。"
order: 1
plots: []
sourceHash: "f21f830982add4edf8eea89d115bc93afa9f064cecf4b85b01346a7503ddefb0"
sourceCorrections: []
---

PyTorch 模型训练完成后，要将其部署到生产环境或嵌入 (embedding)到应用程序中，需要一种不依赖 Python 运行时、可序列化且针对推理 (inference)进行优化的格式。TorchScript 通过将 PyTorch 模型转换为中间表示 (IR) 来实现这一功能，该 IR 可以在 C++ 服务器或移动设备等环境中保存、加载和执行，而无需 Python 依赖。

TorchScript 在 PyTorch 灵活的即时执行模式（其中操作按照 Python 中的定义立即运行）与部署环境通常需要的静态图和性能优化之间架起了一座桥梁。它通过两种主要方法实现此目的：**追踪**和**脚本化**。了解这两种方法的区别对于有效使用 TorchScript 进行模型部署是十分重要的。

### 使用 `torch.jit.trace` 进行追踪

追踪通过使用一组示例输入执行 PyTorch 模型，并记录在此特定执行过程中执行的操作序列来运作。这个被记录的序列，或称“追踪”，随后被转换为封装在 `torch.jit.ScriptModule` 中的静态图表示。

**工作原理：**
当你调用 `torch.jit.trace(model, example_inputs)` 时，PyTorch 会使用所提供的 `example_inputs` 运行模型的 `forward` 方法。每次操作执行时，PyTorch 都会对其进行记录。生成的 `ScriptModule` 本质上包含该单次前向传播期间计算图的冻结快照。

**示例：**

```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        # 简单、直接的计算
        return torch.relu(self.linear(x))

# 实例化模型
model = SimpleModel()
model.eval() # 设置为评估模式

# 提供示例输入
example_input = torch.randn(1, 10)

# 追踪模型
traced_model = torch.jit.trace(model, example_input)

print(traced_model.code) # 打印生成的 TorchScript 代码（通常与追踪结果相似）
print(traced_model.graph) # 打印底层图表示

# 测试追踪后的模型
output = traced_model(example_input)
print("输出形状:", output.shape)
```

**追踪的优点：**

- **简便性：** 通常易于应用，仅需模型和示例输入。
- **现有代码：** 对许多现有模型运行良好，无需代码修改，只要其结构是静态的。

**追踪的局限性：**
追踪的主要局限在于它无法捕获**数据依赖的控制流**。因为追踪只记录针对*特定*示例输入执行的操作，所以任何行为依赖于输入张量*值*的条件语句 (`if`) 或循环 (`for`、`while`) 都不会在追踪图中正确表示。追踪只包含示例输入所采用的路径。

考虑这个修改后的模型：

```python
class ControlFlowModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 5)
        self.linear2 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        # 数据依赖的控制流
        if x.mean() > 0.5:
            return self.linear2(x)
        else:
            return torch.zeros_like(self.linear2(x))

model_cf = ControlFlowModel()
model_cf.eval()

# 示例输入 1（可能触发 'if' 分支）
input1 = torch.randn(1, 10) * 2
traced_model_cf1 = torch.jit.trace(model_cf, input1)

# 示例输入 2（可能触发 'else' 分支）
input2 = torch.randn(1, 10) * -2
# 注意：使用 input2 进行追踪会产生 *不同* 的追踪结果！

print(f"输入 1 均值: {input1.mean().item()}")
print(f"输入 2 均值: {input2.mean().item()}")

# 将两个输入都通过使用 input1 追踪的模型运行
output1_trace1 = traced_model_cf1(input1)
output2_trace1 = traced_model_cf1(input2) # 如果 input2 走 'else' 路径，这很可能是错误的

print(f"输入 1 的输出（用 input1 追踪）: {output1_trace1.item()}")
print(f"输入 2 的输出（用 input1 追踪）: {output2_trace1.item()}") # 无论 input2 的均值如何，都遵循追踪到的路径

# 与即时执行进行比较
output1_eager = model_cf(input1)
output2_eager = model_cf(input2)
print(f"输入 1 的输出（即时执行）: {output1_eager.item()}")
print(f"输入 2 的输出（即时执行）: {output2_eager.item()}") # 正确使用了 'else' 路径
```

在上面的示例中，`traced_model_cf1` 总是会执行使用 `input1` 追踪时记录的操作序列，无论新输入是否应该实际触发 `else` 分支。

### 使用 `torch.jit.script` 进行脚本化

脚本化采取了不同的方法。`torch.jit.script` 不会执行代码并记录操作，而是直接使用 TorchScript 编译器分析你的 Python 源代码。该编译器能够识别 Python 语言的一个子集（包括 `if`、`for`、`while` 等控制流结构），并将其转换为 TorchScript IR。

**工作原理：**
You可以通过在函数或整个 `nn.Module` 类上使用 `@torch.jit.script` 装饰器，或通过在实例或函数上调用 `torch.jit.script()` 来应用脚本化。编译器会解析 Python 代码，检查与 TorchScript 语言子集的兼容性，并生成一个 `ScriptModule` 或 `ScriptFunction`，它准确地表示了原始逻辑，包括控制流。

**示例：**
让我们对 `ControlFlowModel` 进行脚本化：

```python
# 沿用之前的 ControlFlowModel 类

model_cf = ControlFlowModel()
model_cf.eval()

# 脚本化模型实例
scripted_model = torch.jit.script(model_cf)

print(scripted_model.code) # 打印 TorchScript 代码，包括 if/else

# 使用不同输入进行测试
input1 = torch.randn(1, 10) * 2
input2 = torch.randn(1, 10) * -2

print(f"\n输入 1 均值: {input1.mean().item()}")
print(f"输入 2 均值: {input2.mean().item()}")

output1_script = scripted_model(input1)
output2_script = scripted_model(input2)

print(f"输入 1 的输出（脚本化）: {output1_script.item()}")
print(f"输入 2 的输出（脚本化）: {output2_script.item()}") # 正确处理了控制流

# 与即时执行进行比较（应该匹配）
output1_eager = model_cf(input1)
output2_eager = model_cf(input2)
print(f"输入 1 的输出（即时执行）: {output1_eager.item()}")
print(f"输入 2 的输出（即时执行）: {output2_eager.item()}")
```

正如你所见，脚本化的模型正确处理了数据依赖的控制流，因为 `if/else` 逻辑被编译器直接翻译了。

**脚本化的优点：**

- **处理控制流：** 精确捕获数据依赖的条件逻辑和循环。
- **通用性：** 生成更通用的模型表示，不与转换期间使用的特定输入形状或值绑定。
- **稳定性：** 更适合具有动态行为的复杂模型。

**脚本化的局限性：**

- **Python 子集：** 要求代码符合 TorchScript 语言子集。并非所有 Python 功能或库都受支持（例如，任意外部库调用、高度动态的元编程）。你可能需要重构部分模型代码才能使其可脚本化。
- **调试：** 编译器错误有时可能不如标准 Python 错误直观，需要仔细检查有问题的代码段。

### 追踪与脚本化的选择

追踪和脚本化之间的选择主要取决于模型 `forward` 方法的性质：

> 根据模型控制流决定使用 TorchScript 追踪还是脚本化。

- **在以下情况下使用追踪（`torch.jit.trace`）：**
  - 你的模型或模块**不含数据依赖的控制流**。
  - 计算图是静态的，无论输入值如何（尽管如果适当追踪，它可以依赖输入*形状*）。
  - 你希望快速捕获简单模块的操作。
- **在以下情况下使用脚本化（`torch.jit.script`）：**
  - 你的模型包含 `if` 语句、`for` 循环或其行为依赖于所处理张量值的其他结构。
  - 你需要一种能够在不同输入下正确运行的表示，这些输入可能触发不同的执行路径。
  - 你愿意确保代码符合 TorchScript 子集。

**混合方法：** 也可以混合使用追踪和脚本化。你可以脚本化一个内部调用追踪子模块的模块，反之亦然。通常，你可能会脚本化包含控制流的主模型，并在其中追踪更简单、静态的组件。

### 序列化与使用

一旦你拥有了一个 `ScriptModule`（无论是通过追踪还是脚本化获得），你就可以方便地将其保存到文件并稍后加载，可能在不同的环境中：

```python
# 保存脚本化模型
torch.jit.save(scripted_model, 'control_flow_model.pt')

# 稍后加载模型（可能在另一个进程或 C++ 中）
loaded_model = torch.jit.load('control_flow_model.pt')
loaded_model.eval()

# 使用加载的模型
output_loaded = loaded_model(input2)
print(f"加载模型的输出: {output_loaded.item()}")
```

这个保存的 `.pt` 文件包含模型的架构、参数 (parameter)以及执行所需的 TorchScript 代码/图，使其成为一个用于部署的自包含工件。

掌握 TorchScript，特别是追踪和脚本化之间的区别，是使你的 PyTorch 模型为在生产环境中进行高效且稳定部署做准备的重要一步。通过选择合适的方法，你可以创建优化的、独立的模型版本，以进行推理 (inference)。

## 参考资料

- [TorchScript Documentation](https://pytorch.org/docs/stable/jit.html) — PyTorch Team (2024)
  Publisher: PyTorch Foundation
  TorchScript 的权威指南，详细介绍了追踪、脚本化、TorchScript 语言子集以及部署。
- [Introducing TorchScript: A Way to Create Serializable and Optimizable Models from PyTorch](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html) — PyTorch Team (2018)
  Journal: PyTorch Blog; Publisher: PyTorch
  PyTorch 官方对 TorchScript 的介绍，阐述了其动机以及将 PyTorch 模型转换为可部署模型的关键概念。
- [Deep Learning with PyTorch](https://www.manning.com/books/deep-learning-with-pytorch) — Eli Stevens, Luca Antiga, and Thomas Viehmann (2020)
  Publisher: Manning Publications; Pages: Chapter 14
  提供了一份使用 PyTorch 的实践性指南，包含一章专门介绍使用 TorchScript 进行模型部署。
- [Introduction to TorchScript Tutorial](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html) — PyTorch Team (2024)
  Publisher: PyTorch Team
  PyTorch 官方网站上的综合教程，提供了使用 TorchScript 的实用示例和解释。
