---
course: "advanced-pytorch"
chapter: "pytorch-internals-autograd"
lesson: "practice-custom-autograd"
sourceId: 2679
sourceUrl: "https://apxml.com/zh/courses/advanced-pytorch/chapter-1-pytorch-internals-autograd/practice-custom-autograd"
title: "实践操作：构建自定义自动求导函数"
description: "实现并验证用于特定数学运算的自定义自动求导函数。"
order: 8
plots: []
sourceHash: "4d2942a565619b2fc3422b29b233dd8807d35ae328a86f0c51495a53a3f0bdb3"
sourceCorrections: []
---

虽然PyTorch的自动微分功能可以处理大多数标准操作，但您会遇到需要自定义梯度逻辑的情况。这可能是因为您正在实现一项新颖的操作，优化特定计算，或者处理自动求导无法直接推导出梯度的函数。获得使用`torch.autograd.Function`定义自己可微分操作的实践经验。

### 构建模块：`torch.autograd.Function`

定义具有特定梯度规则的自定义操作的核心机制是继承`torch.autograd.Function`。该类要求您实现两个静态方法：

1. `forward()`: 此方法执行操作的实际计算。它接收输入张量，并可以接受额外参数 (parameter)。重要的是，它还接收一个上下文 (context)对象`ctx`，该对象作为通向`backward`方法的桥梁。您可以使用`ctx.save_for_backward()`来存储后续梯度计算所需的任何张量。它应返回操作的输出张量。
2. `backward()`: 此方法定义梯度计算。它接收上下文对象`ctx`（包含从`forward`保存的张量）以及损失相对于`forward`方法输出（`grad_output`）的梯度。其职责是计算并返回损失相对于`forward`方法**每个**输入的梯度。返回的梯度数量和顺序必须与`forward`的输入数量和顺序匹配。如果一个输入不需要梯度（例如，它不是张量或`requires_grad=False`），您应为其对应的梯度返回`None`。

### 示例：实现一个截断ReLU函数

我们来实作一个自定义激活函数 (activation function)：截断ReLU。该函数行为类似标准ReLU，但将最大输出值限制在特定阈值。

从数学上说，对于截断值$C$：


$$
\text{截断ReLU}(x, C) = \min(\max(0, x), C)
$$


相对于$x$的导数是：


$$
\frac{\partial}{\partial x} \text{截断ReLU}(x, C) =
\begin{cases}
1 & \text{如果 } 0 < x < C \\
0 & \text{否则}
\end{cases}
$$


现在，我们使用`torch.autograd.Function`来实作它。

```python
import torch

class ClippedReLUFunction(torch.autograd.Function):
    """
    实现截断ReLU函数：min(max(0, x), clip_val)。
    """

    @staticmethod
    def forward(ctx, input_tensor, clip_val):
        """
        前向传播：计算截断ReLU。

        参数：
            ctx: 用于保存信息供反向传播的上下文对象。
            input_tensor: 输入张量。
            clip_val: 输出的截断最大值。

        返回：
            应用截断ReLU后的输出张量。
        """
        # 确保clip_val为浮点数以便一致比较
        clip_val = float(clip_val)

        # 保存输入张量和clip_val供反向传播使用
        # 我们只需要输入张量来计算梯度掩码
        ctx.save_for_backward(input_tensor)
        # 将非张量参数直接存储在ctx上
        ctx.clip_val = clip_val

        # 应用截断ReLU操作
        output = input_tensor.clamp(min=0, max=clip_val)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：计算截断ReLU的梯度。

        参数：
            ctx: 带有保存信息的上下文对象。
            grad_output: 损失相对于此函数输出的梯度。

        返回：
            相对于input_tensor的梯度，相对于clip_val的梯度（无）
        """
        # 检索已保存的张量和值
        input_tensor, = ctx.saved_tensors
        clip_val = ctx.clip_val

        # 根据输入值范围创建梯度掩码
        # 当 0 < 输入 < clip_val 时梯度为1，否则为0
        grad_input_mask = (input_tensor > 0) & (input_tensor < clip_val)
        grad_input = grad_output * grad_input_mask.float()

        # 由于clip_val是一个超参数，因此不需要计算其梯度，
        # 它不是我们通常进行微分的输入张量。
        # 对于非张量输入或不需要梯度的输入，
        # 返回None作为其梯度。
        return grad_input, None

# 辅助函数，使其更易于像标准PyTorch函数一样使用
def clipped_relu(input_tensor, clip_val=1.0):
    """逐元素应用截断ReLU函数。"""
    return ClippedReLUFunction.apply(input_tensor, clip_val)

# 使用示例
x = torch.randn(5, requires_grad=True, dtype=torch.float64) # 使用float64以获得gradcheck所需更高精度
clip_value = 2.0
y = clipped_relu(x, clip_value)
z = y.mean() # 下游计算示例

# 计算梯度
z.backward()

print("输入张量 (x):\n", x)
print("截断输出 (y):\n", y)
print("平均输出 (z):\n", z)
print("x的梯度 (x.grad):\n", x.grad)
```

在此代码中：

- `ClippedReLUFunction`继承自`torch.autograd.Function`。
- `forward`计算 $y = \min(\max(0, x), C)$，使用`ctx.save_for_backward(input_tensor)`保存梯度计算所需的输入张量`x`，并将非张量`clip_val`直接保存到`ctx`上。
- `backward`使用`ctx.saved_tensors`检索`input_tensor`。它计算梯度掩码（如果$0 < x < C$则为$1$，否则为$0$），并将其与传入梯度`grad_output`逐元素相乘。它返回`input_tensor`的计算梯度，并为`clip_val`返回`None`，因为`clip_val`不是需要梯度的张量输入。
- `clipped_relu`辅助函数提供了一个用户友好的接口，调用`ClippedReLUFunction.apply(...)`。使用`.apply`对于在自动求导图中正确注册该操作是必需的。

### 图中自定义操作的可视化

当您使用`ClippedReLUFunction.apply`时，PyTorch会将其集成到计算图中，就像任何内置操作一样。您定义的`backward`方法确保梯度正确地流经此自定义节点。

> 包含自定义`ClippedReLUFunction`的计算图表示。虚线表示非张量输入或数据流。点线表示反向传播 (backpropagation)。

### 使用`gradcheck`验证正确性

实现自定义反向函数可能容易出错。您的`forward`逻辑与`backward`梯度计算之间的不匹配会导致不正确的训练行为，这可能难以调试。PyTorch提供了一个有用的工具`torch.autograd.gradcheck`，用于数值验证您的自定义函数计算的梯度。

`gradcheck`通过将您的`backward`方法计算的解析梯度与使用有限差分计算的数值梯度进行比较来工作。

```python
from torch.autograd import gradcheck

# 使用float64以获得gradcheck所需更高精度
input_data = torch.randn(5, requires_grad=True, dtype=torch.float64)
clip_value = 2.0 # 保持为浮点数

# gradcheck接受一个函数（或lambda）和一组输入元组
# 该函数应执行我们想要检查的操作
test_passed = gradcheck(lambda x: clipped_relu(x, clip_value), (input_data,), eps=1e-6, atol=1e-4)
print(f"\n梯度检查通过: {test_passed}")

# 使用不同截断值检查的示例
input_data_2 = torch.randn(3, 4, requires_grad=True, dtype=torch.float64)
clip_value_2 = 0.5
test_passed_2 = gradcheck(lambda x: clipped_relu(x, clip_value_2), (input_data_2,), eps=1e-6, atol=1e-4)
print(f"梯度检查2通过: {test_passed_2}")

# 显示失败的示例（如果反向逻辑有误）
# 让我们模拟一个错误的反向传播：
class IncorrectClippedReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, clip_val):
        ctx.save_for_backward(input_tensor)
        ctx.clip_val = float(clip_val)
        return input_tensor.clamp(min=0, max=ctx.clip_val)

    @staticmethod
    def backward(ctx, grad_output):
        # 错误的梯度计算（例如，忘记了掩码）
        grad_input = grad_output.clone() # 错误！
        return grad_input, None

try:
    input_fail = torch.randn(5, requires_grad=True, dtype=torch.float64)
    clip_fail = 1.5
    gradcheck(lambda x: IncorrectClippedReLU.apply(x, clip_fail), (input_fail,), eps=1e-6, atol=1e-4)
except RuntimeError as e:
    print(f"\n梯度检查如预期般失败：\n{e}")
```

如果`gradcheck`返回`True`，则表示您的解析梯度与数值近似值非常接近，这让您对自己的实作有信心。如果失败，通常指向您的`backward`逻辑中的错误或潜在的数值稳定性问题（尤其是在`float32`等较低精度下）。请务必彻底测试您的自定义函数。强烈建议在`gradcheck`中使用`float64`（双精度）以获得稳定性。

这项实践练习说明了扩展PyTorch自动微分能力的过程。通过熟练掌握`torch.autograd.Function`，您能够实作模型中的几乎任何操作，同时确保正确的梯度传播以进行有效的训练。这是构建高度定制化和高效深度学习 (deep learning)解决方案的重要一步。

## 参考资料

- [Extending Autograd with Custom Functions](https://pytorch.org/docs/stable/notes/extending.html) — PyTorch Documentation (2017)
  Publisher: PyTorch Foundation
  PyTorch自动求导系统中实现自定义可微操作的官方指南。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 6
  提供自动微分和反向传播的理论基础，这是理解自动求导的根本。
- [Deep Learning with PyTorch](https://www.manning.com/books/deep-learning-with-pytorch) — Eli Stevens, Luca Antiga, and Thomas Viehmann (2020)
  Publisher: Manning Publications; Pages: Chapter 8
  一本实用的书籍，包含PyTorch自动求导引擎和自定义函数实现的示例及解释。
- [\`torch.autograd.gradcheck\`](https://pytorch.org/docs/stable/generated/torch.autograd.gradcheck.html) — PyTorch Documentation (2024)
  `gradcheck`的官方文档，一个用于验证自定义`backward`实现正确性的工具。
