# 自定义 Autograd 函数：前向与反向

来源：[原文](https://apxml.com/zh/courses/advanced-pytorch/chapter-1-pytorch-internals-autograd/custom-autograd-functions)

[返回章节目录](README.md) · [返回课程目录](../README.md)

虽然 PyTorch 的 `autograd` 引擎能自动处理各种内置操作的微分，但有时你会需要更多控制，或需要为 PyTorch 未知的操作定义梯度。这可能发生在以下情况：

- 实现 PyTorch 中没有的新颖数学操作。
- 集成来自外部库（例如：自定义 C++ 或 CUDA 核，稍后介绍）的代码，以执行部分计算。
- 通过提供比自动推导更高效的梯度计算来优化性能。
- 为数学上不可微分的操作定义“梯度”，或你希望覆盖标准导数的情况（例如：对离散操作使用直通估计器）。

对于这些情况，PyTorch 提供了一种机制，通过继承 `torch.autograd.Function` 来定义自己的可微分操作。这个类允许你精确指定前向计算的执行方式以及在反向传播 (backpropagation)期间如何计算梯度。

区分 `torch.autograd.Function` 和 `torch.nn.Module` 很重要。`nn.Module` 通常表示神经网络 (neural network)中包含参数 (parameter)（`torch.nn.Parameter`）的层，并且可以由其他模块或函数组成；而 `autograd.Function` 定义的是一个单一、特定的计算操作及其梯度。它本身不持有参数。

### 定义前向传播

要创建一个自定义操作，你需要定义一个继承自 `torch.autograd.Function` 的类。前向计算的核心在于实现一个名为 `forward` 的静态方法。

```python
import torch

class MyLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, weight, bias=None):
        # ctx 是一个上下文对象，用于保存反向传播所需的信息
        # input_tensor, weight, bias 是函数的输入

        # 执行操作
        output = input_tensor.mm(weight.t()) # 矩阵乘法
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        # 保存反向传播所需的张量
        # 我们需要 input_tensor 和 weight 来计算梯度
        ctx.save_for_backward(input_tensor, weight, bias)

        return output
```

`forward` 方法的重要方面：

1. **静态方法：** 它必须声明为 `@staticmethod`。它不作用于类的实例，而是定义操作本身。
2. **`ctx` 参数 (parameter)：** 第一个参数始终是 `ctx`，一个上下文 (context)对象。它的主要作用是充当 `forward` 和 `backward` 传播之间的桥梁。你使用 `ctx` 来存储在 `forward` 期间计算的、稍后在 `backward` 中计算梯度所需的任何张量或信息。
3. **输入参数：** 紧随 `ctx` 之后，你列出函数接受的输入参数。这些可以是张量或其他 Python 对象。
4. **计算：** 在 `forward` 内部，你使用标准 PyTorch 张量操作或可能调用外部库来实现操作的逻辑。
5. **`ctx.save_for_backward(*tensors)`：** 这是保存梯度计算所需张量的重要方法。只保存必需的内容以避免不必要的内存消耗。PyTorch 处理好记录，以确保这些张量在 `backward` 传播中可用。你也可以将非张量属性直接保存到 `ctx` 上（例如，`ctx.some_flag = True`），这些属性稍后可在 `backward` 中获取。
6. **返回值：** 该方法应返回操作产生的一个或多个输出张量。

### 定义反向传播 (backpropagation)

`forward` 的对应部分是静态的 `backward` 方法。此方法定义了如何在给出损失函数 (loss function)对 `forward` 方法的 *输出* 的梯度的前提下，计算损失函数对 `forward` 方法的 *输入* 的梯度。

```python
import torch

class MyLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, weight, bias=None):
        output = input_tensor.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        # 保存 input_tensor 和 weight。如果提供了 bias，也会保存。
        saved_tensors = [input_tensor, weight]
        if bias is not None:
            saved_tensors.append(bias)
        ctx.save_for_backward(*saved_tensors)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output 是损失函数相对于 forward 输出的梯度
        # 我们需要计算损失函数相对于 forward 输入的梯度：
        # input_tensor, weight, bias

        # 获取已保存的张量
        saved_tensors = ctx.saved_tensors
        input_tensor = saved_tensors[0]
        weight = saved_tensors[1]
        bias = saved_tensors[2] if len(saved_tensors) > 2 else None

        # 使用链式法则计算梯度
        # dL/d(输入) = dL/d(输出) * d(输出)/d(输入)
        # d(输出)/d(输入) = 权重^T
        grad_input = grad_output.mm(weight)

        # dL/d(权重) = dL/d(输出) * d(输出)/d(权重)
        # d(输出)/d(权重) = 输入^T
        grad_weight = grad_output.t().mm(input_tensor)

        # dL/d(偏置) = dL/d(输出) * d(输出)/d(偏置)
        # d(输出)/d(偏置) = 1
        grad_bias = None
        if bias is not None:
            # 在批处理维度上对梯度求和
            grad_bias = grad_output.sum(0)

        # 按相同顺序返回 forward 的每个输入参数的梯度
        # 对于不需要梯度（如 ctx）或非张量输入，返回 None。
        # 返回值的数量必须与 forward 输入的数量匹配。
        return grad_input, grad_weight, grad_bias
```

`backward` 方法的重要方面：

1. **静态方法：** 与 `forward` 类似，它必须是一个 `@staticmethod`。
2. **`ctx` 参数 (parameter)：** 第一个参数再次是上下文 (context)对象 `ctx`，用于获取已保存的信息。
3. **`grad_output` 参数：** 紧随 `ctx` 之后，它接收表示最终损失函数相对于 `forward` 方法每个输出的梯度的参数。如果 `forward` 返回单个张量，`backward` 接收单个 `grad_output` 张量。如果 `forward` 返回多个张量，`backward` 接收多个梯度张量，每个输出对应一个，按相应顺序排列。这些梯度（$\frac{\partial L}{\partial \text{输出}}$）由 autograd 引擎在反向传播期间提供。
4. **`ctx.saved_tensors`：** 你使用 `ctx` 的 `saved_tensors` 属性来获取在 `forward` 中保存的张量。它们以元组形式返回，顺序与保存时相同。直接保存到 `ctx` 上的任何非张量属性也可以被访问（例如，`ctx.some_flag`）。
5. **梯度计算：** 这是你实现核心梯度逻辑的地方，通常应用链式法则：$\frac{\partial L}{\partial \text{输入}} = \frac{\partial L}{\partial \text{输出}} \times \frac{\partial \text{输出}}{\partial \text{输入}}$。你使用传入的 `grad_output` （$\frac{\partial L}{\partial \text{输出}}$）和从 `ctx` 中获取的张量（或原始输入，如果已保存）来计算 $\frac{\partial \text{输出}}{\partial \text{输入}}$。
6. **返回值：** `backward` 方法*必须*为 `forward` 方法的*每个*输入参数返回一个梯度，顺序必须完全相同。
   - 如果输入张量需要梯度（`requires_grad=True`），返回计算出的梯度张量。
   - 如果输入张量不需要梯度（`requires_grad=False`），你可以返回 `None`。PyTorch 通常通过不保存仅用于计算不需要梯度的输入的张量来进行优化。
   - 如果输入不是张量（例如，布尔标志、整型维度），返回 `None`。
   - `backward` 的返回值数量必须精确匹配 `forward` 接受的参数数量（不包括 `ctx`）。

> 该图说明了流程：输入进入 `forward`，它计算输出并通过 `ctx` 保存必要的张量。之后，相对于输出的梯度（`grad_output`）流入 `backward`，它从 `ctx` 获取已保存的张量并计算相对于原始输入的梯度。

### 使用自定义函数

你不会直接调用 `forward` 或 `backward` 方法。相反，你使用 `apply` 类方法。这个方法接受与你的 `forward` 函数相同的参数 (parameter)（不包括 `ctx`），执行前向传播，并设置必要的记录，以便 `autograd` 在需要时知道调用你的 `backward` 方法。

```python
# 使用示例
input_features = 10
output_features = 5
batch_size = 3

# 创建需要梯度的张量
x = torch.randn(batch_size, input_features, requires_grad=True)
w = torch.randn(output_features, input_features, requires_grad=True) # 注意：用于 mm(weight.t()) 的形状
b = torch.randn(output_features, requires_grad=True)

# 应用自定义函数
# 使用 MyLinearFunction.apply，而不是直接调用 MyLinearFunction.forward
y = MyLinearFunction.apply(x, w, b)

# 示例：计算一个虚拟损失并反向传播
loss = y.mean()
loss.backward()

# 检查梯度（可选）
print("x 的梯度:", x.grad is not None)
print("w 的梯度:", w.grad is not None)
print("b 的梯度:", b.grad is not None)
```

调用 `MyLinearFunction.apply(x, w, b)` 会执行在 `MyLinearFunction.forward` 中定义的前向计算，并在计算图中注册该操作。当稍后调用 `loss.backward()` 时，autograd 引擎会遇到这个自定义操作，并使用适当的 `grad_output` 调用 `MyLinearFunction.backward`。

### 使用 `gradcheck` 验证正确性

正确实现 `backward` 传播非常必要且容易出错。PyTorch 提供了一个实用函数 `torch.autograd.gradcheck` 来帮助验证你的实现。`gradcheck` 通过轻微扰动每个输入（有限差分）来数值计算梯度，并将这些数值梯度与你的 `backward` 函数计算的解析梯度进行比较。

```python
from torch.autograd import gradcheck

# 为 gradcheck 创建输入。通常需要双精度以确保稳定性。
x_check = torch.randn(batch_size, input_features, dtype=torch.double, requires_grad=True)
w_check = torch.randn(output_features, input_features, dtype=torch.double, requires_grad=True)
b_check = torch.randn(output_features, dtype=torch.double, requires_grad=True)

# 定义要测试的函数（使用 apply）
test_func = MyLinearFunction.apply

# 执行检查
# inputs 是一个包含函数参数的元组
inputs = (x_check, w_check, b_check)
is_correct = gradcheck(test_func, inputs, eps=1e-6, atol=1e-4)
print("梯度检查通过:", is_correct)

# bias=None 的示例（可选参数处理）
x_check_no_bias = torch.randn(batch_size, input_features, dtype=torch.double, requires_grad=True)
w_check_no_bias = torch.randn(output_features, input_features, dtype=torch.double, requires_grad=True)

# 如果函数签名根据输入而变化，则需要一个小包装器
def test_func_no_bias(x, w):
    return MyLinearFunction.apply(x, w, None)

inputs_no_bias = (x_check_no_bias, w_check_no_bias)
is_correct_no_bias = gradcheck(test_func_no_bias, inputs_no_bias, eps=1e-6, atol=1e-4)
print("梯度检查（无偏置）通过:", is_correct_no_bias)
```

无论何时你实现自定义 `autograd.Function`，都强烈推荐使用 `gradcheck`。它能发现梯度公式中的许多常见错误。请注意，`gradcheck` 通常要求输入为 `torch.double` 以获得足够的数值精度，并且对于大型输入可能速度较慢。它通常在小型、有代表性的测试用例上执行。

### 重要注意事项

- **使用 `.apply()`：** 始终使用 `YourFunction.apply(...)` 调用你的自定义函数。直接调用 `forward` 将绕过 autograd 机制。
- **内存与重新计算：** `ctx.save_for_backward` 会存储张量，并在反向传播 (backpropagation)完成前一直消耗内存。只保存梯度计算严格必需的张量。如果中间值的重新计算成本较低，你可以在 `backward` 中进行，而不是保存它们。
- **就地操作：** 在自定义函数中对输入或已保存的张量进行就地操作时，要极其小心。它们可能会干扰梯度计算，特别是如果反向传播需要缓冲区。在 `backward` 方法中修改通过 `ctx.save_for_backward` 保存的张量通常是不安全的。通常更安全的方法是使用副本或为结果分配新的张量。
- **高阶梯度：** 如果你需要计算梯度的梯度（例如，用于梯度范数的惩罚项），你的 `backward` 方法中执行的操作本身必须是可微分的。如果你在 `backward` 中使用标准可微分 PyTorch 操作，PyTorch 的 autograd 引擎可以自动处理。创建正确支持高阶梯度的自定义函数需要仔细的实现。

掌握 `torch.autograd.Function` 可以对微分过程进行细粒度控制，从而实现标准库功能之外的复杂模型和优化策略。它是高级 PyTorch 开发和研究的基本工具。

## 参考资料

- [torch.autograd.Function](https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function) — PyTorch Contributors (2024)
  Publisher: PyTorch Foundation
  提供PyTorch中创建自定义可微分操作的官方API参考和示例。
- [Autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html) — PyTorch Contributors (2024)
  Publisher: PyTorch Foundation
  详细介绍了PyTorch自动微分引擎的机制，这对于理解自定义`Function`的实现至关重要。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  第6章详细介绍了反向传播和数值微分技术，这些技术是PyTorch自动微分系统等的基础。
- [Deep Learning with PyTorch](https://www.manning.com/books/deep-learning-with-pytorch) — Eli Stevens, Luca Antiga, and Thomas Viehmann (2020)
  Publisher: Manning Publications
  提供在PyTorch中实现自定义`autograd.Function`的实用示例和深入理解。

---

[上一节](03-Autograd%20%E5%BC%95%E6%93%8E%E6%9C%BA%E5%88%B6.md) · [下一节](05-%E9%AB%98%E9%98%B6%E6%A2%AF%E5%BA%A6%E8%AE%A1%E7%AE%97.md)
