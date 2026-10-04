---
course: "pytorch-for-tensorflow-developers"
chapter: "pytorch-tensorflow-core-concepts"
lesson: "autodiff-gradienttape-autograd"
sourceId: 5740
sourceUrl: "https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-1-pytorch-tensorflow-core-concepts/autodiff-gradienttape-autograd"
title: "自动微分：GradientTape 与 Autograd 对比"
description: "了解 PyTorch 的 `autograd` 系统与 TensorFlow 的 `tf.GradientTape` 在梯度计算方面的差异。"
order: 5
plots: []
sourceHash: "4b5a01ea42e2d4e04d15dcc300c67ba703c3dfea1ac47c49f988c05ae5b427f0"
sourceCorrections: []
---

自动微分是驱动现代神经网络 (neural network)训练的引擎，它使我们能够高效计算损失函数 (loss function)相对于模型参数 (parameter)的梯度。TensorFlow 和 PyTorch 都为此提供了完善的系统，但它们在处理方式上采用略有不同的理念和 API。对 TensorFlow 的 `tf.GradientTape` 与 PyTorch 的 `autograd` 系统进行比较，能够阐明它们之间的区别，并有助于开发者理解。

本质上，自动微分会追踪张量上执行的操作，以构建一个计算图。当需要梯度时，这些系统会从输出（通常是标量损失）反向遍历此图到输入（通常是模型权重 (weight)），在每一步应用链式法则。

## TensorFlow 的方法：`tf.GradientTape`

在 TensorFlow 中，梯度计算主要通过 `tf.GradientTape` 上下文 (context)管理器进行管理。涉及 `tf.Variable` 对象（或被磁带明确“监视”的张量）的操作，会在其作用域内被记录到这个“磁带”上。一旦 `with` 块内的前向传播完成，你可以调用 `tape.gradient(target, sources)` 来计算 `target` 张量（例如，损失）相对于一个或多个 `source` 张量（例如，模型权重 (weight)）的梯度。

我们来看一个简单的例子。假设我们有一个函数 $y = w^2 + b$，我们想计算 $\frac{\partial y}{\partial w}$ 和 $\frac{\partial y}{\partial b}$。

```python
import tensorflow as tf

# 定义可训练变量
w_tf = tf.Variable(2.0, name='weight')
b_tf = tf.Variable(1.0, name='bias')

with tf.GradientTape() as tape:
    # 定义计算。涉及 w_tf 和 b_tf 的操作会被记录。
    y_tf = w_tf * w_tf + b_tf  # y = w^2 + b

# 计算梯度
# dy/dw = 2w = 2*2 = 4
# dy/db = 1
gradients_tf = tape.gradient(y_tf, {'w': w_tf, 'b': b_tf})

print(f"TensorFlow y: {y_tf.numpy()}")
print(f"TensorFlow d(y)/d(w): {gradients_tf['w'].numpy()}")
print(f"TensorFlow d(y)/d(b): {gradients_tf['b'].numpy()}")
# 输出：
# TensorFlow y: 5.0
# TensorFlow d(y)/d(w): 4.0
# TensorFlow d(y)/d(b): 1.0
```

默认情况下，`tf.GradientTape` 在调用一次 `tape.gradient()` 后即被消耗。如果你需要从相同的已记录操作中计算多个梯度（例如，针对不同的损失或高阶导数），则必须在初始化时通过设置 `persistent=True` 来创建一个持久的磁带：`with tf.GradientTape(persistent=True) as tape:`。

## PyTorch 的方法：`autograd`

PyTorch 的自动微分系统 `torch.autograd` 与 `torch.Tensor` 对象本身的集成度更高。张量具有 `requires_grad` 属性（你创建的张量默认为 `False`，但对于 `nn.Linear` 权重 (weight)等模型参数 (parameter)则为 `True`）。如果一个张量的 `requires_grad=True`，PyTorch 会自动追踪所有涉及该张量的操作，构建一个动态计算图。

在你计算最终输出（通常是标量损失）的前向传播完成后，你在此输出张量上调用 `.backward()` 方法。这会触发图中所有 `requires_grad=True` 且是输出张量祖先的张量的梯度计算。计算出的梯度会累积在那些相应叶张量的 `.grad` 属性中。

让我们在 PyTorch 中复现之前的例子：$y = w^2 + b$。

```python
import torch

# 定义需要梯度追踪的张量
w_pt = torch.tensor(2.0, requires_grad=True)
b_pt = torch.tensor(1.0, requires_grad=True)

# 定义计算。PyTorch 追踪 w_pt 和 b_pt 上的操作。
y_pt = w_pt * w_pt + b_pt  # y = w^2 + b

# y_pt 是一个标量，因此我们可以直接调用 backward()
# 这会计算所有 requires_grad=True 的张量的梯度
# 这些张量对 y_pt 有贡献。
y_pt.backward()

# 梯度存储在张量的 .grad 属性中。
# dy/dw = 2w = 2*2 = 4
# dy/db = 1
print(f"PyTorch y: {y_pt.item()}")
print(f"PyTorch d(y)/d(w): {w_pt.grad.item()}")
print(f"PyTorch d(y)/d(b): {b_pt.grad.item()}")
# 输出：
# PyTorch y: 5.0
# PyTorch d(y)/d(w): 4.0
# PyTorch d(y)/d(b): 1.0
```

PyTorch 中一个重要的细节是，梯度会累积。这意味着如果你多次调用 `.backward()`（例如，在训练循环中），新的梯度将被加到 `.grad` 属性中已有的梯度上。因此，在典型的训练迭代中，每次反向传播 (backpropagation)之前，你必须明确地将梯度清零，通常是通过调用 `optimizer.zero_grad()`，或者如果 `param.grad` 不为 `None`，则手动遍历参数并调用 `param.grad.zero_()`。

```python
# 梯度累积和清零的例子
w = torch.tensor(2.0, requires_grad=True)
optimizer = torch.optim.SGD([w], lr=0.1) # 优化器需要参数

# 第一次传播
y1 = w * w
y1.backward()
print(f"After first backward(): w.grad = {w.grad}") # w.grad = 4.0
# optimizer.step() # 将根据此梯度更新 w

# 如果我们不清零梯度：
# optimizer.zero_grad() # <--- 忘记了这一步！
y2 = w * 3
y2.backward() # 这将加到现有梯度上
# y2 的新梯度是 3.0。累积后：4.0 + 3.0 = 7.0
print(f"After second backward() (no zero_grad): w.grad = {w.grad}")

# 正确做法：在下次计算前清零梯度
optimizer.zero_grad() # 或 w.grad.zero_()
y3 = w * 4
y3.backward()
print(f"After third backward() (with zero_grad): w.grad = {w.grad}") # w.grad = 4.0
```

`backward()` 方法通常在一个标量张量（如损失值）上调用。如果 `y_pt` 是一个非标量张量，`y_pt.backward()` 将需要一个与 `y_pt` 形状相同的 `gradient` 参数，表示“最终标量损失相对于 `y_pt` 的梯度”。如果你只想获取每个元素的梯度总和，这通常是 `torch.ones_like(y_pt)`。

## 比较 `GradientTape` 和 `autograd`

以下是差异和相似点的总结：

> 此图示了 TensorFlow 使用 `GradientTape` 和 PyTorch 使用 `autograd` 进行梯度计算的操作流程。TensorFlow 依赖于显式记录上下文 (context)，而 PyTorch 的 `autograd` 根据张量的 `requires_grad` 属性追踪操作。

**主要区别：**

1. **记录机制**：

   - **TensorFlow**：显式。你定义一个 `with tf.GradientTape() as tape:` 块。只有在此块内对被监视张量进行的操作才会被记录。
   - **PyTorch**：隐式，基于张量属性。如果操作中涉及的张量具有 `requires_grad=True`，则该操作会被追踪并添加到与其输入关联的计算图。
2. **梯度计算的启动**：

   - **TensorFlow**：你调用 `tape.gradient(target, sources)`。这会返回梯度。
   - **PyTorch**：你调用 `target.backward()`（其中 `target` 通常是一个标量损失）。梯度不直接返回，而是累积在具有 `requires_grad=True` 的叶张量的 `.grad` 属性中。
3. **梯度累积**：

   - **TensorFlow**：`tape.gradient()` 每次都会计算新的梯度。
   - **PyTorch**：`tensor.grad` 会累积梯度。如果不需要累积，你必须在训练循环中每次新的反向传播 (backpropagation)之前手动调用 `optimizer.zero_grad()` 或 `tensor.grad.zero_()`。这种累积对于多小批次的梯度累积等场景很有用。
4. **计算图的持久性**：

   - **TensorFlow `GradientTape`**：默认情况下，磁带所持有的资源在调用一次 `tape.gradient()` 后即被释放。要多次调用它（例如，用于高阶导数或来自相同计算的多个不同梯度），你需要设置 `persistent=True`。
   - **PyTorch `autograd`**：默认情况下，用于计算梯度的图在调用 `.backward()` 后被释放（这由 `backward()` 中 `retain_graph=False` 的默认值控制）。如果你需要在图的同一部分再次调用 `.backward()`（例如，用于多个损失组件贡献给相同的参数 (parameter)，或用于高阶梯度），你必须指定 `loss.backward(retain_graph=True)`。
5. **高阶梯度**：

   - **TensorFlow**：通过嵌套 `GradientTape` 上下文或使用持久磁带实现。

     ```python
     x_tf = tf.Variable(2.0)
     with tf.GradientTape() as tape2:
         with tf.GradientTape() as tape1:
             y_tf = x_tf * x_tf * x_tf # x的三次方
         dy_dx_tf = tape1.gradient(y_tf, x_tf) # dy/dx = 3x^2 = 12
     d2y_dx2_tf = tape2.gradient(dy_dx_tf, x_tf) # d2y/dx2 = 6x = 12
     print(f"TensorFlow dy/dx: {dy_dx_tf.numpy()}, d2y/dx2: {d2y_dx2_tf.numpy()}")
     ```
   - **PyTorch**：通过对中间梯度调用 `.backward(create_graph=True)` 实现，这允许通过梯度计算本身进行进一步的微分。

     ```python
     x_pt = torch.tensor(2.0, requires_grad=True)
     y_pt = x_pt * x_pt * x_pt # x的三次方

     # 一阶导数
     dy_dx_pt = torch.autograd.grad(y_pt, x_pt, create_graph=True)[0] # dy/dx = 3x^2 = 12
     # 二阶导数
     d2y_dx2_pt = torch.autograd.grad(dy_dx_pt, x_pt)[0] # d2y/dx2 = 6x = 12
     print(f"PyTorch dy/dx: {dy_dx_pt.item()}, d2y/dx2: {d2y_dx2_pt.item()}")
     ```

     请注意这里使用 `torch.autograd.grad`，这是一个用于计算梯度的更通用函数，对高阶导数很有用。当调用 `y_pt.backward(create_graph=True)` 时，参与计算 `y_pt` 的张量的 `.grad` 属性也将附带图，允许进一步的 `backward()` 调用或 `torch.autograd.grad` 调用。
6. **停止梯度流**：

   - **TensorFlow**：如果一个 `tf.Variable` 未被磁带监视，或者通过对张量使用 `tf.stop_gradient()`。
   - **PyTorch**：通过将张量的 `requires_grad` 属性设置为 `False`，使用 `tensor.detach()` 创建一个不要求梯度且与当前图分离的新张量，或者将代码封装在 `with torch.no_grad():` 块中。

```python
# PyTorch：停止梯度流
x = torch.randn(2, 2, requires_grad=True)
y = x * 2
print(f"y.requires_grad: {y.requires_grad}") # True

# 使用 torch.no_grad()
with torch.no_grad():
    z = x * 2
print(f"z.requires_grad: {z.requires_grad}") # False，z 不追踪历史记录

# 使用 .detach()
w = x * 2
w_detached = w.detach() # 创建一个新张量，共享数据，但没有梯度历史
print(f"w.requires_grad: {w.requires_grad}")             # True
print(f"w_detached.requires_grad: {w_detached.requires_grad}") # False
```

对于来自 TensorFlow 的开发者来说，最重要的调整将是：

- 习惯于将 `requires_grad` 作为梯度追踪的主要开关，而不是每次梯度计算都使用显式的 `GradientTape` 范围。
- 记住在训练循环中调用 `optimizer.zero_grad()` 以防止梯度累积。
- 理解 `loss.backward()` 会填充模型参数的 `.grad` 字段，这些字段随后被优化器使用。

这两个系统都强大且灵活，旨在处理深度学习 (deep learning)模型复杂的微分需求。PyTorch 的 `autograd` 通常感觉与张量系统集成度更高，体现了其即时执行的特性，即图随着操作的发生而构建。TensorFlow 的 `GradientTape` 提供了一种更显式的方式来控制何时以及哪些操作被记录用于微分。随着你更多地使用 PyTorch，其 `autograd` 系统将变得得心应手，尤其是在构建自定义训练循环时。

## 参考资料

- [Introduction to gradients and automatic differentiation](https://www.tensorflow.org/guide/autodiff) — TensorFlow Developers (2023)
  TensorFlow 自动微分的官方指南，详细介绍了 `tf.GradientTape` 及其用法。
- [Automatic differentiation with torch.autograd](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) — PyTorch Developers (2025)
  Publisher: PyTorch Foundation
  PyTorch `autograd` 引擎的官方教程，涵盖了其核心概念和 API。
- [Automatic Differentiation in Machine Learning: A Survey](https://www.jmlr.org/papers/volume18/17-468/17-468.pdf) — Atılım Güneş Baydin, Barak A. Pearlmutter, and Jeffrey Mark Siskind (2018)
  Journal: Journal of Machine Learning Research; Publisher: Microtome Publishing; Volume: 18; Pages: 1-65; DOI: [10.5555/3342930.3342931](https://doi.org/10.5555/3342930.3342931)
  一份全面介绍自动微分技术及其在机器学习中应用的数学和算法概述性调查报告。
- [Deep Learning with PyTorch](https://www.manning.com/books/deep-learning-with-pytorch) — Eli Stevens, Luca Antiga, and Thomas Viehmann (2020)
  Publisher: Manning Publications
  一本使用 PyTorch 进行深度学习的实用指南，详细解释了其自动微分系统及其在实际模型中的作用。
