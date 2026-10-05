# 张量与梯度计算 (`requires_grad`)

来源：[原文](https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-3-automatic-differentiation-autograd/tensors-gradient-calculation)

[返回章节目录](README.md) · [返回课程目录](../README.md)

神经网络 (neural network)训练的根基在于计算损失函数 (loss function)相对于模型参数 (parameter)的梯度。PyTorch 的 Autograd 引擎自动处理这项复杂的任务。但是 Autograd 怎么知道*哪些*计算需要被追踪以便进行微分呢？答案在于 PyTorch 张量的一个特定属性：`requires_grad`。

### `requires_grad` 属性

每个 PyTorch 张量都有一个布尔属性，名为 `requires_grad`。此属性充当一个标志，告诉 Autograd 是否应记录涉及此张量的操作，以便稍后进行可能的梯度计算。

默认情况下，当你创建一个张量时，它的 `requires_grad` 属性被设置为 `False`。

```python
import torch

# 默认行为：requires_grad 为 False
x = torch.tensor([1.0, 2.0, 3.0])
print(f"Tensor x: {x}")
print(f"x.requires_grad: {x.requires_grad}")

# 显式创建另一个张量并将 requires_grad 设置为 False
y = torch.tensor([4.0, 5.0, 6.0], requires_grad=False)
print(f"\nTensor y: {y}")
print(f"y.requires_grad: {y.requires_grad}")
```

这种默认行为对效率来说是合理的。在典型的工作流程中，许多张量不需要梯度。例如，输入数据或目标标签通常是固定的，不需要计算相对于它们自身的梯度。不必要地追踪操作会消耗额外的内存和计算资源。

### 启用梯度追踪

要指示 PyTorch 追踪某个特定张量的操作并准备进行梯度计算，你需要将其 `requires_grad` 属性设置为 `True`。有两种主要方式可以做到这一点：

1. **在张量创建时：** 将 `requires_grad=True` 作为参数 (parameter)传递给张量创建函数。

   ```python
   # 在创建时启用梯度追踪
   w = torch.tensor([0.5, -1.0], requires_grad=True)
   print(f"Tensor w: {w}")
   print(f"w.requires_grad: {w.requires_grad}")
   ```
2. **在张量创建后（原地修改）：** 对现有张量使用原地方法 `.requires_grad_(True)`。

   ```python
   b = torch.tensor([0.1])
   print(f"Tensor b (before): {b}")
   print(f"b.requires_grad (before): {b.requires_grad}")

   # 在创建后启用梯度追踪
   b.requires_grad_(True)
   print(f"\nTensor b (after): {b}")
   print(f"b.requires_grad (after): {b.requires_grad}")
   ```

**重要提示：** 梯度计算通常只对浮点张量（如 `torch.float32` 或 `torch.float64`）有意义。导数涉及连续变化，这与浮点类型相符。尝试对整数张量设置 `requires_grad=True` 通常会导致错误或出现意料之外的行为，因为梯度并非以相同方式为离散值定义的。如果你尝试计算直接涉及被追踪操作的整数张量的梯度，PyTorch 通常会抛出 `RuntimeError`。

```python
# 尝试对整数张量设置 requires_grad
try:
    int_tensor = torch.tensor([1, 2], dtype=torch.int64, requires_grad=True)
    # 这一行可能不会立即出错，但后续涉及它的 backward() 调用会出错。
    print(f"Integer tensor created with requires_grad=True: {int_tensor.requires_grad}")
    # 让我们尝试一个简单的操作，这可能会在以后导致问题
    result = int_tensor * 2.0 # 乘以浮点数看看是否会引起问题
    print(f"Result requires_grad: {result.requires_grad}")
    # result.backward() # 如果我们尝试反向传播，这很可能会失败
except RuntimeError as e:
    print(f"\n对整数张量设置 requires_grad 时出错: {e}")

# 最佳实践：对需要梯度的参数/计算使用浮点张量
float_tensor = torch.tensor([1.0, 2.0], requires_grad=True)
print(f"\n已创建 requires_grad=True 的浮点张量: {float_tensor.requires_grad}")
```

### `requires_grad` 的传播

重要的一点是，`requires_grad` 状态会在操作中传播。如果参与操作的*任何*输入张量具有 `requires_grad=True`，则该操作产生的输出张量将自动具有 `requires_grad=True`。这确保了涉及参数 (parameter)（通常具有 `requires_grad=True`）的整个计算链都得到追踪。

让我们通过一个例子说明：

```python
# 定义张量：x（输入）、w（权重）、b（偏置）
x = torch.tensor([1.0, 2.0]) # 输入数据，不需要梯度
w = torch.tensor([0.5, -1.0], requires_grad=True) # 权重参数，追踪梯度
b = torch.tensor([0.1], requires_grad=True) # 偏置参数，追踪梯度

print(f"x requires_grad: {x.requires_grad}")
print(f"w requires_grad: {w.requires_grad}")
print(f"b requires_grad: {b.requires_grad}")

# 执行操作：y = w * x + b
# 注意：PyTorch 处理 b 的广播
intermediate = w * x
print(f"\nintermediate (w * x) requires_grad: {intermediate.requires_grad}")

y = intermediate + b
print(f"y requires_grad: {y.requires_grad}")
```

注意，即使 `x` 不需要梯度，但由于 `w` 需要梯度，所以 `w * x` 的结果 (`intermediate`) 也需要梯度。接着，由于 `intermediate` 需要梯度（并且 `b` 也需要），最终输出 `y` 也具有 `requires_grad=True`。

### `.grad_fn` 属性

这种传播与 PyTorch 构建计算图的方式紧密关联。当一个新张量由某个操作创建，并且其 `requires_grad` 为 `True` 时，PyTorch 会将一个 `.grad_fn` 属性附加到这个新张量上。该属性引用了执行此操作的函数（例如 `AddBackward0` 或 `MulBackward0`），并且知道如何在反向传播 (backpropagation)过程中计算相应的梯度。

用户直接创建的张量（如我们上面的 `x`、`w` 和 `b` 示例）在图中被认为是“叶”张量。如果它们具有 `requires_grad=True`，它们的 `.grad_fn` 为 `None`，因为它们不是由图中被追踪的操作创建的。对需要梯度的张量进行操作所产生的张量是“非叶”张量，并将具有 `.grad_fn`。

让我们查看前面示例中的 `.grad_fn`：

```python
print(f"\nx.grad_fn: {x.grad_fn}")
print(f"w.grad_fn: {w.grad_fn}")
print(f"b.grad_fn: {b.grad_fn}")
print(f"intermediate.grad_fn: {intermediate.grad_fn}") # 乘法的结果
print(f"y.grad_fn: {y.grad_fn}") # 加法的结果
```

你可以看到 `x`、`w` 和 `b`（我们的叶张量）的 `grad_fn` 为 `None`。相比之下，`intermediate` 有一个 `MulBackward0` 函数，而 `y` 有一个 `AddBackward0` 函数，这表明了创建它们的那些操作。这条 `grad_fn` 引用链*就是*Autograd 使用的动态计算图。

> `y = w * x + b` 的计算图简化视图。需要梯度的张量用蓝色突出显示。注意操作符（`*`、`+`）如何创建新张量（`intermediate`、`y`），如果通过其输入启用了梯度追踪，这些新张量将通过 `grad_fn` 引用其创建操作。

通过对我们希望优化的张量（通常是模型参数 (parameter)，如权重 (weight) `w` 和偏置 (bias) `b`）设置 `requires_grad=True`，我们让 Autograd 能够构建此图，并将计算从最终输出（通常是损失）追溯到这些参数，为使用 `.backward()` 进行梯度计算的步骤做好准备，我们将在接下来介绍这一点。

## 参考资料

- [Automatic differentiation package - torch.autograd](https://pytorch.org/docs/stable/autograd.html) — PyTorch Developers (2024)
  Publisher: PyTorch Foundation
  全面介绍了 PyTorch 的自动微分引擎，包括 `requires_grad` 和计算图的工作原理。
- [Tensors](https://pytorch.org/docs/stable/tensors.html) — PyTorch Developers (2024)
  Publisher: PyTorch Foundation
  `torch.Tensor` 的官方参考文档，详细说明了其属性、创建方法以及 `requires_grad` 属性。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 6
  解释反向传播和自动微分数学原理的理论书籍。
- [Deep Learning with PyTorch](https://www.manning.com/books/deep-learning-with-pytorch) — Eli Stevens, Luca Antiga, and Thomas Viehmann (2020)
  Publisher: Manning Publications
  一本实用书籍，从实践者的角度解释 PyTorch 的自动微分、张量和 `requires_grad`。

---

[上一节](02-PyTorch%20%E8%AE%A1%E7%AE%97%E5%9B%BE.md) · [下一节](04-%E6%89%A7%E8%A1%8C%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD%20%28%60backward%28%29%60%29.md)
