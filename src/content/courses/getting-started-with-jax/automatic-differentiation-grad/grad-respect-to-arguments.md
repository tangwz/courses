---
course: "getting-started-with-jax"
chapter: "automatic-differentiation-grad"
lesson: "grad-respect-to-arguments"
sourceId: 1726
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-jax/chapter-3-automatic-differentiation-grad/grad-respect-to-arguments"
title: "关于参数求导"
description: "指定对哪些函数参数进行求导。"
order: 4
plots: []
sourceHash: "c9f38982cee79d310f3f995c8857cabcf60cc2327fcf68c656a3d5ea56559f91"
sourceCorrections: []
---

默认情况下，`jax.grad` 计算函数对其第一个参数 (parameter)的梯度。然而，许多函数，特别是在机器学习 (machine learning)场景中（例如损失函数 (loss function)），会接受多个输入（例如，参数和数据）。你通常需要只针对这些输入中的一个或一个特定子集来计算梯度。JAX 在 `jax.grad` 中提供了 `argnums` 参数，以便精确地控制此行为。

### 指定单个参数 (parameter)进行求导

我们来看一个接受两个参数的简单函数：$f(x, y) = x^2 \cdot y$。

```python
import jax
import jax.numpy as jnp

def power_product(x, y):
  """计算 x^2 * y"""
  return (x**2) * y

# 定义一些输入值
x_val = 3.0
y_val = 4.0

print(f"函数输出: {power_product(x_val, y_val)}")
```

如果我们不带任何额外参数应用 `jax.grad`，它将对第一个参数 $x$ 求导。偏导数 $\frac{\partial f}{\partial x}$ 为 $2xy$。

```python
# 对第一个参数 (x) 求导 - 默认行为
grad_f_wrt_x = jax.grad(power_product)
gradient_x = grad_f_wrt_x(x_val, y_val)

print(f"关于 x 的梯度 (默认): {gradient_x}") # 预期: 2 * 3.0 * 4.0 = 24.0
```

现在，假设我们需要对第二个参数 $y$ 求梯度。偏导数 $\frac{\partial f}{\partial y}$ 为 $x^2$。我们可以通过使用 `argnums=1` 来实现这一点（记住参数索引从0开始）：

```python
# 使用 argnums=1 对第二个参数 (y) 求梯度
grad_f_wrt_y = jax.grad(power_product, argnums=1)
gradient_y = grad_f_wrt_y(x_val, y_val)

print(f"关于 y 的梯度 (argnums=1): {gradient_y}") # 预期: 3.0**2 = 9.0
```

在这里，`argnums=1` 指示 `jax.grad` 在求导时，将第二个参数（本例中的 `y`）视为变量，并把其他参数（`x`）视为常数来计算导数。

### 指定多个参数 (parameter)进行求导

你可能需要同时对多个参数求梯度。比如，考虑一个函数 $g(w, b, x) = wx + b$。我们可能需要同时求关于 $w$ 和 $b$ 的梯度。

你可以通过向 `argnums` 传递一个整数元组来实现这一点。

```python
def affine(w, b, x):
  """计算 w*x + b"""
  return w * x + b

# 定义输入值
w_val = 2.0
b_val = 1.0
x_data = 5.0

print(f"仿射函数输出: {affine(w_val, b_val, x_data)}")

# 对第一个 (w) 和第二个 (b) 参数求梯度
grad_g_wrt_wb = jax.grad(affine, argnums=(0, 1))

# 注意: 在求导过程中，输入 x_data 被视为常数
gradients_wb = grad_g_wrt_wb(w_val, b_val, x_data)

print(f"关于 (w, b) 的梯度 (使用 argnums=(0, 1)): {gradients_wb}")
# 预期: (关于 w 的梯度, 关于 b 的梯度) = (x, 1) = (5.0, 1.0)
```

当 `argnums` 是一个元组时，`jax.grad` 返回的函数也会返回一个元组。输出元组的元素直接对应 `argnums` 中指定的参数，并按相同顺序排列。在上面的例子中：

- `gradients_wb` 的第一个元素（即 `5.0`）是 $\frac{\partial g}{\partial w} = x$。
- 第二个元素（即 `1.0`）是 $\frac{\partial g}{\partial b} = 1$。

### 与机器学习 (machine learning)的联系

此功能在训练机器学习模型中非常基本。一个典型的损失函数 (loss function)可能形如 `loss(params, data_batch)`。要使用梯度下降 (gradient descent)更新模型参数 (parameter)，你需要损失函数关于 `params` 的梯度，同时将 `data_batch` 视为固定输入。这可以很自然地表示为：

```python
# 示例结构
# def loss_function(params, data_batch):
#   predictions = model_apply(params, data_batch['inputs'])
#   error = predictions - data_batch['targets']
#   return jnp.mean(error**2) # 示例: 均方误差

# grad_loss_wrt_params = jax.grad(loss_function, argnums=0)
# gradients = grad_loss_wrt_params(current_params, batch)
# updated_params = current_params - learning_rate * gradients
```

使用 `argnums=0` 确保 `jax.grad` 计算 $\nabla_{\text{params}} \text{loss}(\text{params}, \text{data\_batch})$，这正是优化所需要的。

掌握 `argnums` 参数，你就能对 JAX 中的自动求导过程进行精细控制，从而能够针对特定输入计算梯度，这对于更复杂的函数和标准的机器学习工作流程来说非常重要。

## 参考资料

- [\`jax.grad\`](https://jax.readthedocs.io/en/latest/_autosummary/jax.grad.html) — JAX core developers (2023)
  详细说明 `jax.grad` 函数的官方文档，包括 `argnums` 参数用于定向微分的用法。
- [Evaluating Derivatives: Principles and Techniques for Algorithmic Differentiation](https://www.siam.org/store/book/ot105) — Andreas Griewank, Andrea Walther (2008)
  Publisher: Society for Industrial and Applied Mathematics (SIAM); DOI: [10.1137/1.9780898717761](https://doi.org/10.1137/1.9780898717761)
  一本综合性教材，解释了自动微分的基本原理和各种技术。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  提供机器学习中基于梯度的优化方法和反向传播的基础知识，其中定向微分至关重要。
