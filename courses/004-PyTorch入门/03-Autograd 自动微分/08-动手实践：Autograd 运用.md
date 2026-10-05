# 动手实践：Autograd 运用

来源：[原文](https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-3-automatic-differentiation-autograd/practice-autograd-exploration)

[返回章节目录](README.md) · [返回课程目录](../README.md)

实际例子演示 PyTorch Autograd 系统。这些练习会引导您设置梯度要求、执行反向传播 (backpropagation)、查看梯度、观察累积以及禁用梯度跟踪。请确保您已安装 PyTorch 并能导入 `torch` 库。

### 设置

首先，导入 PyTorch：

```python
import torch
```

### 例子 1：基本梯度计算

我们从一个非常简单的计算开始，并跟踪梯度。我们将定义两个张量 `x` 和 `w`，其中 `w` 表示我们想要优化的权重 (weight)。我们将计算一个简单的输出 `y`，然后计算一个标量损失 `L`。

1. **创建张量**：将 `x` 定义为一个包含一些数据的张量，将 `w` 定义为一个需要计算其梯度的张量（使用 `requires_grad=True`）。

   ```python
   # 输入数据
   x = torch.tensor([2.0, 4.0, 6.0])

   # 权重张量 - 需要计算梯度
   w = torch.tensor([0.5], requires_grad=True)

   print(f"x: {x}")
   print(f"w: {w}")
   print(f"x.requires_grad: {x.requires_grad}")
   print(f"w.requires_grad: {w.requires_grad}")
   ```

   请注意，`x` 默认情况下不需要梯度，而我们为 `w` 显式设置了它。
2. **定义计算**：执行一个简单的运算。任何通过涉及 `requires_grad=True` 的张量运算而得到的张量，其 `requires_grad` 也会是 `True`。

   ```python
   # 前向传播：y = w * x
   y = w * x

   # 定义一个简单的标量损失 L（例如，y 的均值）
   L = y.mean()

   print(f"y: {y}")
   print(f"L: {L}")
   print(f"y.requires_grad: {y.requires_grad}")
   print(f"L.requires_grad: {L.requires_grad}")
   ```

   您会看到 `y` 和 `L` 现在都需要梯度，因为它们依赖于 `w`。
3. **计算梯度**：在最终的标量输出 (`L`) 上使用 `.backward()` 方法来计算整个图中的梯度。

   ```python
   # 执行反向传播
   L.backward()
   ```
4. **查看梯度**：查看张量 `w` 的 `.grad` 属性。

   ```python
   # 梯度存储在 w.grad 中
   print(f"Gradient dL/dw: {w.grad}")

   # x 不需要梯度，因此它的梯度为 None
   print(f"Gradient dL/dx: {x.grad}")
   ```

   我们来分析 `w.grad` 的结果。计算过程为：
   $y_i = w * x_i$
   $L = \frac{1}{3} \sum y_i = \frac{1}{3} (w x_1 + w x_2 + w x_3)$
   梯度 $\frac{\partial L}{\partial w}$ 为：

   
   $$
   \frac{\partial L}{\partial w} = \frac{1}{3} (x_1 + x_2 + x_3)
   $$
   

   当 $x = [2.0, 4.0, 6.0]$ 时，梯度为 $\frac{1}{3} (2.0 + 4.0 + 6.0) = \frac{12.0}{3} = 4.0$。这与输出 `tensor([4.])` 相符。因为 `x` 在创建时没有设置 `requires_grad=True`，所以它的梯度未被计算，仍为 `None`。

### 例子 2：梯度与计算图

Autograd 动态构建图。我们来看一个稍微复杂一点的例子。

1. **创建张量**：

   ```python
   a = torch.tensor(2.0, requires_grad=True)
   b = torch.tensor(3.0, requires_grad=True)
   c = torch.tensor(4.0, requires_grad=False) # 不需要梯度

   print(f"a: {a}, requires_grad={a.requires_grad}")
   print(f"b: {b}, requires_grad={b.requires_grad}")
   print(f"c: {c}, requires_grad={c.requires_grad}")
   ```
2. **定义计算**：

   ```python
   d = a * b
   e = d + c
   f = e * 2

   print(f"d: {d}, requires_grad={d.requires_grad}") # True（依赖于 a, b）
   print(f"e: {e}, requires_grad={e.requires_grad}") # True（依赖于 d）
   print(f"f: {f}, requires_grad={f.requires_grad}") # True（依赖于 e）
   ```
3. **计算并查看梯度**：

   ```python
   # 从最终的标量输出 f 进行反向传播
   f.backward()

   # 检查梯度
   print(f"Gradient df/da: {a.grad}")
   print(f"Gradient df/db: {b.grad}")
   print(f"Gradient df/dc: {c.grad}") # 预期结果：None
   ```

   我们手动计算一下：
   $d = a \times b$
   $e = d + c = a \times b + c$
   $f = 2 \times e = 2(a \times b + c)$

   $\frac{\partial f}{\partial a} = 2 \times b = 2 \times 3.0 = 6.0$
   $\frac{\partial f}{\partial b} = 2 \times a = 2 \times 2.0 = 4.0$
   $\frac{\partial f}{\partial c} = 2$

   `a` 和 `b` 的计算梯度是匹配的。由于 `c` 定义时 `requires_grad=False`，Autograd 没有跟踪涉及 `c` 的操作来计算关于 `c` 本身的梯度，因此 `c.grad` 为 `None`。

### 例子 3：梯度累积

默认情况下，每次调用 `.backward()` 时，梯度都会累积到 `.grad` 属性中。这对于计算多个损失的梯度或模拟更大的批次大小等情况很有用，但在标准训练循环中，需要显式地将梯度清零。

1. **设置**：我们再次使用一个简单的设置。

   ```python
   x = torch.tensor(5.0, requires_grad=True)
   y = x * x
   print(f"Initial x.grad: {x.grad}") # 最初应为 None
   ```
2. **第一次反向传播 (backpropagation)**：

   ```python
   # 对 y 执行反向传播。注意：backward() 通常需要一个标量。
   # 如果在非标量张量上调用，需要提供梯度参数。
   # 为了演示，我们计算 y 对 x 的梯度（即 2x）。
   # 我们将使用 y.backward(gradient=torch.tensor(1.0)) 来实现。
   # 通常，您会有一个从 y 导出的标量损失 L。
   # 如果 y 是多元素，则 L = y.mean()；如果 y 是标量，则 L = y。
   y.backward(retain_graph=True) # 对于多次反向传播需要 retain_graph=True
   print(f"x.grad after 1st backward: {x.grad}") # 预期结果：2*x = 10.0
   ```
3. **第二次反向传播（累积）**：再次调用 `backward`，*不*清零梯度。

   ```python
   y.backward(retain_graph=True) # 再次调用 backward
   print(f"x.grad after 2nd backward: {x.grad}") # 预期结果：10.0 + 10.0 = 20.0
   ```

   梯度被累积（相加）到之前的值上。
4. **清零梯度**：手动清零梯度。在典型的训练循环中，这通常通过 `optimizer.zero_grad()` 完成。

   ```python
   if x.grad is not None:
       x.grad.zero_() # 原位清零
   print(f"x.grad after zeroing: {x.grad}") # 预期结果：0.0
   ```
5. **第三次反向传播（清零后）**：

   ```python
   y.backward() # 最后一次反向传播不需要 retain_graph
   print(f"x.grad after 3rd backward: {x.grad}") # 预期结果：10.0
   ```

   梯度清零后会重新计算。在训练循环中忘记清零梯度是常见的错误原因。

### 例子 4：禁用梯度跟踪

有时，您需要执行操作而不跟踪其梯度计算，最常见的情况是在模型评估（推理 (inference)）期间或在优化步骤之外调整参数 (parameter)时。

1. **使用 `torch.no_grad()`**：这个上下文 (context)管理器是禁用代码块梯度跟踪的标准方法。

   ```python
   a = torch.tensor(2.0, requires_grad=True)
   print(f"Outside context: a.requires_grad = {a.requires_grad}")

   with torch.no_grad():
       print(f"Inside context: a.requires_grad = {a.requires_grad}") # 仍然是 True
       b = a * 2
       print(f"Inside context: b = {b}, b.requires_grad = {b.requires_grad}") # False！

   # 在上下文之外，如果输入需要梯度，计算会恢复跟踪
   c = a * 3
   print(f"Outside context: c = {c}, c.requires_grad = {c.requires_grad}") # True
   ```

   在 `torch.no_grad()` 块内部，尽管 `a` 需要梯度，但生成的张量 `b` 却不需要。这使得块内的操作更节省内存且更快，因为反向传播 (backpropagation)的历史不会被保存。
2. **使用 `.detach()`**：这个方法会创建一个*新*张量，它共享相同的数据，但与计算历史分离。它不需要梯度。

   ```python
   a = torch.tensor(5.0, requires_grad=True)
   b = a * a # b 需要梯度，并且是连接到 a 的图的一部分

   # 分离 a，创建一个不需要梯度的新张量 c
   c = a.detach()
   print(f"a.requires_grad: {a.requires_grad}") # True
   print(f"c.requires_grad: {c.requires_grad}") # False

   # 涉及 c 的操作不会跟踪回 a
   d = c * 3 # d 不需要梯度
   print(f"d.requires_grad: {d.requires_grad}") # False

   # 如果您对涉及 'b' 的计算执行反向传播，
   # 它会流回 'a'。如果您使用 'd'，则不会。
   L1 = b.mean() # 依赖于 'a'
   L1.backward()
   print(f"Gradient dL1/da: {a.grad}") # 预期结果：2*a = 10.0

   # 在下一次反向调用前清零梯度
   if a.grad is not None:
       a.grad.zero_()

   # 尝试通过 'd' 进行反向传播 - 它不会影响 'a' 的梯度
   try:
       # L2 = d.mean() # 最终需要一个需要梯度的计算
       # 例子：再次使用 'a' 与分离后的结果
       L2 = (a + d).mean() # L2 = (a + a.detach()*3).mean()
       L2.backward()
       print(f"Gradient dL2/da: {a.grad}") # 只计算来自 'a' 路径的梯度 (1.0)
   except RuntimeError as e:
       print(f"Error demonstrating backward with detached: {e}")
       # 如果最终的标量不依赖于
       # 分离后任何需要梯度的输入，您可能会得到一个错误。
       # 这里，L2 依赖于 'a'，所以梯度是 1.0。
       # 经由 'd' 的路径对 a.grad 没有贡献。

   # 修改 c（分离的张量） - 它会影响 a，因为它们共享数据！
   with torch.no_grad():
     c[0] = 100.0 # 原位修改 c（对标量使用索引）
   print(f"After modifying c, a = {a}") # 'a' 也改变了！
   print(f"After modifying c, c = {c}")
   ```

   `detach()` 在您想在计算中使用张量的值但阻止梯度通过该特定路径回流时很有用，或者当您需要一个没有梯度历史的张量时（例如，用于绘图或日志记录）。请注意，它共享数据存储，因此原位修改会影响原始张量，除非您先 `.clone()` 它（`c = a.detach().clone()`）。

这些练习展示了 Autograd 的核心机制。您已经练习了启用梯度跟踪、执行反向传播、查看计算出的梯度、理解累积以及在需要时禁用跟踪。掌握这些操作对在 PyTorch 中构建和训练神经网络 (neural network)来说非常重要。

## 参考资料

- [Automatic differentiation in machine learning: a survey](http://www.jmlr.org/papers/v18/15-153/15-153.pdf) — Atılım Güneş Baydin, Barak A. Pearlmutter, Alexey Andreyevich Radul, Jeffrey Mark Siskind (2018)
  Journal: Journal of Machine Learning Research; Publisher: Journal of Machine Learning Research; Volume: 18; Pages: 1-43
  一篇全面的学术调查，涵盖了自动微分的原理和技术，它们构成了PyTorch Autograd等系统的基础。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press; Pages: 201-232
  第六章（反向传播和其他微分算法）提供了神经网络中梯度计算的基础说明，与Autograd的操作直接相关。
- [Autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html) — PyTorch Documentation (2017)
  Publisher: PyTorch Foundation
  官方文档，深入说明了PyTorch的Autograd引擎，包括它如何构建计算图、跟踪操作和计算梯度。
- [A Gentle Introduction to \`torch.autograd\`](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) — PyTorch Documentation (2024)
  一个官方教程，通过实际例子介绍了PyTorch Autograd系统的核心功能，涵盖了梯度跟踪和反向传播。

---

[上一节](07-%E6%A2%AF%E5%BA%A6%E7%B4%AF%E7%A7%AF.md) · [下一节](../04-%E4%BD%BF%E7%94%A8%20%60torch.nn%60%20%E6%90%AD%E5%BB%BA%E6%A8%A1%E5%9E%8B/01-%60torch.nn.Module%60%20%E5%9F%BA%E7%B1%BB.md)
