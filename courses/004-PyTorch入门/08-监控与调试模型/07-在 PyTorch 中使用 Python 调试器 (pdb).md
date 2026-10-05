# 在 PyTorch 中使用 Python 调试器 (pdb)

来源：[原文](https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-8-monitoring-debugging-models/python-debugger-pytorch)

[返回章节目录](README.md) · [返回课程目录](../README.md)

尽管像 TensorBoard 这样的工具可以帮助监视训练趋势，并且可视化网络图可以提供架构理解，但有时你需要在程序执行到某个特定点时，仔细检查其确切状态。形状不匹配可能在模型的 `forward` 传递中发生，梯度可能意外地变成 `NaN`，或者张量值可能无故发散。对于这些情况，使用调试器逐行执行代码通常是找出根本原因最直接的方法。

Python 的内置调试器 `pdb` 是一个强大的文本工具，它能与 PyTorch 代码一同使用。它允许你暂停执行、检查变量（包括张量）、逐行执行代码，并在问题发生时准确了解程序的流程。

### 启用调试器

使用 `pdb` 启动调试会话最常见的方法是，在你希望执行暂停的位置，直接将以下两行代码插入到你的 Python 脚本中：

```python
import pdb
pdb.set_trace()
```

当 Python 解释器遇到 `pdb.set_trace()` 时，它会停止执行，并将你带入终端中的 `pdb` 交互式控制台。`(Pdb)` 提示符表示你当前处于调试器中。

你应该把 `pdb.set_trace()` 放在哪里？

- **在可能出错的代码前：** 如果某行代码（例如，矩阵乘法、损失计算）引发错误，请将 `pdb.set_trace()` 放在紧邻该行之前。这能让你检查该操作的输入。
- **在模型的 `forward` 方法中：** 为了了解数据在经过层时如何变化，请将跟踪点放在 `forward` 方法内。你可以逐层执行应用，并检查张量的形状和值。
- **在训练循环中：** 为了查看特定迭代或 epoch 的状态，请将其放在循环内。你可以将其包装在条件语句中（例如，`if batch_idx == problematic_index: import pdb; pdb.set_trace()`）。
- **在 `loss.backward()` 之后：** 为了在梯度计算完毕但优化器步骤执行之前检查梯度，请将跟踪点放在 `backward()` 调用之后。

此外，你也可以从命令行在 `pdb` 的控制下启动整个脚本，这会在脚本的第一行启动调试器：

```bash
python -m pdb your_pytorch_script.py
```

这对于诊断脚本执行早期发生的问题很有用，例如导入错误或设置问题。

### PDB 基本命令

一旦你在 `(Pdb)` 提示符下，就可以使用各种命令来控制执行和检查状态。下面是一些最常用的命令：

- `n` (next)：执行当前行，并在当前函数的下一行停止。如果当前行是一个函数调用，`n` 会执行整个函数并在它返回*后*停止。
- `s` (step)：与 `n` 类似，但如果当前行是一个函数调用，`s` 会*进入*函数，并在其第一行停止。
- `c` (continue)：恢复正常执行，直到遇到下一个断点（或 `pdb.set_trace()` 调用），或者直到脚本结束或出错。
- `l` (list)：显示当前执行行周围的源代码。使用 `l .` 再次列出以当前行为中心的源代码。
- `p <expression>` (print)：在当前上下文 (context)中评估 `<expression>` 并打印其值。这可以说是调试 PyTorch 最重要的命令。你可以检查张量、变量、模型参数 (parameter)等。
  - `p my_tensor.shape`
  - `p my_tensor.dtype`
  - `p my_tensor.device`
  - `p my_tensor` (打印张量本身；可能很大)
  - `p model.layer1.weight.grad` (在 `backward()` 之后)
  - `p loss.item()`
- `a` (args)：打印当前函数的参数列表。
- `r` (return)：继续执行直到当前函数返回。
- `b <line_number>` (breakpoint)：在当前文件的特定 `<line_number>` 处设置断点。当执行到达该行时会暂停。你也可以在其他文件 (`b path/to/file.py:<line_number>`) 或方法上 (`b self.my_method`) 指定断点。
- `cl` 或 `clear`：清除所有断点。`cl <bp_number>` 清除特定断点。
- `q` (quit)：退出调试器并立即终止脚本。
- `h` (help)：显示可用命令列表。`h <command>` 提供特定命令的帮助。

### 使用 PDB 调试 PyTorch 代码：示例

让我们看看 `pdb` 如何帮助处理常见的 PyTorch 调试情况。

**情况 1：调试模型中的形状不匹配问题**

假设你有一个简单模型，并且在前向传播时遇到了形状不匹配错误。

```python
import torch
import torch.nn as nn
import pdb

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.activation = nn.ReLU()
        # 潜在错误：输入尺寸与 layer1 的输出不匹配
        self.layer2 = nn.Linear(25, 5) # 错误可能在此处 (25 != 20)

    def forward(self, x):
        print(f"Initial shape: {x.shape}")
        x = self.layer1(x)
        print(f"After layer1: {x.shape}")
        x = self.activation(x)
        print(f"After activation: {x.shape}")
        # 在 layer2 之前进行调试
        pdb.set_trace()
        # 这行代码很可能会导致运行时错误
        x = self.layer2(x)
        print(f"After layer2: {x.shape}")
        return x

# 示例用法
net = SimpleNet()
# 创建一个虚拟输入张量
input_tensor = torch.randn(32, 10) # 批次大小 32，特征数 10
output = net(input_tensor)
```

当你运行这段代码时，它会打印形状，然后在 `pdb.set_trace()` 处停止。

```text
Initial shape: torch.Size([32, 10])
After layer1: torch.Size([32, 20])
After activation: torch.Size([32, 20])
-> x = self.layer2(x)
(Pdb)
```

在 `(Pdb)` 提示符下，你可以检查：

- `p x.shape`：这将输出 `torch.Size([32, 20])`。
- `p self.layer2`：这将显示定义 `Linear(in_features=25, out_features=5, bias=True)`。
- `p self.layer2.in_features`：这将输出 `25`。

通过比较输入形状 `[32, 20]`（特别是特征维度 `20`）与 `layer2.in_features`（`25`），不匹配之处变得显而易见。然后你可以退出 (`q`) 并修正 `nn.Linear` 的定义。

**情况 2：检查梯度**

假设你的损失没有下降，并且你怀疑出现了梯度消失或梯度爆炸。你可以在反向传播 (backpropagation)后检查它们。

```python
# 假设 model、data (inputs, targets)、loss_fn、optimizer 都已定义

# 前向传播
outputs = model(inputs)
loss = loss_fn(outputs, targets)

# 反向传播
optimizer.zero_grad()
loss.backward()

# 在此处插入调试器以检查梯度
import pdb
pdb.set_trace()

# 优化器步骤（将在调试后发生）
# optimizer.step()
```

当执行暂停时，你可以检查特定参数 (parameter)的梯度：

- `p model.some_layer.weight.grad`：检查 `some_layer` 权重 (weight)部分的梯度张量。查看是否存在 `NaN` 值、非常大的值（爆炸）或非常小的值（消失）。
- `p model.some_layer.weight.grad.abs().mean()`：计算梯度的平均绝对值，以了解其大小。
- `p loss.item()`：提醒自己当前的损失值。

### 有效使用 PDB 的建议

- **有针对性：** 将 `pdb.set_trace()` 放置在离你怀疑问题所在位置尽可能近的地方。
- **使用 Print (`p`)：** 广泛运用 `p` 命令来检查张量形状、数据类型、设备位置和实际值。
- **谨慎执行：** 使用 `n`（next）逐行移动。只有当你需要检查你编写的函数调用时才使用 `s`（step）（检查 PyTorch 的内部函数可能会很冗长）。
- **移除跟踪：** 在最终确定代码之前，请记得移除或注释掉 `import pdb; pdb.set_trace()` 调用，尤其是在提交到版本控制或部署之前。
- **考虑长时间运行的替代方案：** `pdb` 会停止执行。对于仅在训练数小时后才出现的问题进行调试，交互式调试可能不切实际。在这种情况下，定期记录详细信息（张量形状、损失值、梯度范数）可能更适合，或许可以结合条件断点或断言检查。

有效使用 `pdb` 需要一些练习，但它是理解 PyTorch 代码详细的、逐步执行过程以及解决许多从堆栈跟踪或高级指标中不明显问题的必不可少的工具。

## 参考资料

- [pdb - The Python Debugger](https://docs.python.org/3/library/pdb.html) — Python Software Foundation (2023)
  Publisher: Python Software Foundation
  Python内置调试器的官方参考资料，详细说明了其命令和用法。
- [Common Pitfalls & Debugging (within PyTorch documentation)](https://pytorch.org/docs/stable/notes/faq.html#common-pitfalls-debugging) — PyTorch Authors (2021)
  Publisher: PyTorch Foundation
  提供了PyTorch中常见问题和调试策略的指导，包括`pdb`的使用。
- [Deep Learning with PyTorch](https://www.manning.com/books/deep-learning-with-pytorch) — Eli Stevens, Luca Antiga, and Thomas Viehmann (2020)
  Publisher: Manning Publications
  一本使用PyTorch构建深度学习模型的实用指南，涵盖了识别和解决问题相关的技术。

---

[上一节](06-%E8%AE%AD%E7%BB%83-%E8%AF%84%E4%BC%B0%E6%9C%9F%E9%97%B4%E7%9A%84%E6%8C%87%E6%A0%87%E8%AE%B0%E5%BD%95.md) · [下一节](08-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E8%B0%83%E8%AF%95%E4%B8%8E%E5%8F%AF%E8%A7%86%E5%8C%96.md)
