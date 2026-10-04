---
course: "getting-started-with-jax"
chapter: "automatic-differentiation-grad"
lesson: "hands-on-computing-gradients"
sourceId: 1738
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-jax/chapter-3-automatic-differentiation-grad/hands-on-computing-gradients"
title: "动手实践：计算梯度"
description: "练习使用 `jax.grad` 计算各种函数的梯度。"
order: 9
plots: []
sourceHash: "8703262be915932ffa0682381e9eeb044b8b2b464c9b11f310f3085c1b179240"
sourceCorrections: []
---

本练习通过实际操作，强化对 `jax.grad` 的理解。练习从简单的函数入手，展示如何对特定参数 (parameter)求导和计算高阶导数。

请确保您已安装并导入 JAX：

```python
import jax
import jax.numpy as jnp
import numpy as np # 常用于比较或作为初始数据

# 如果需要，启用 64 位精度以便更好地与解析结果进行比较
from jax.config import config
config.update("jax_enable_x64", True)

print(f"JAX 版本：{jax.__version__}")
print(f"默认后端：{jax.default_backend()}")
```

### 练习 1：简单多项式的梯度

考虑多项式函数 $f(x) = 3x^2 + 2x + 5$。从解析角度看，其导数为 $f'(x) = 6x + 2$。让我们使用 `jax.grad` 验证这一点。

1. **定义函数：**

   ```python
   def poly_func(x):
     """一个简单的多项式函数。"""
     return 3 * x**2 + 2 * x + 5
   ```
2. **创建梯度函数：** 使用 `jax.grad` 获得一个计算 `poly_func` 梯度的新函数。

   ```python
   grad_poly_func = jax.grad(poly_func)
   ```
3. **在特定点计算梯度：** 让我们评估在 $x = 4$ 处的梯度。

   ```python
   x_value = 4.0 # 使用浮点数进行求导
   gradient_at_x = grad_poly_func(x_value)
   print(f"f(x) 在 x = {x_value} 处的梯度：{gradient_at_x}")

   # 解析验证：f'(4) = 6*4 + 2 = 24 + 2 = 26
   analytical_gradient = 6 * x_value + 2
   print(f"在 x = {x_value} 处的解析梯度：{analytical_gradient}")
   ```

您应该看到 `jax.grad` 的输出与解析结果 (26.0) 一致。

### 练习 2：对特定参数 (parameter)求导

现在，让我们处理一个二元函数：$g(x, y) = x^3 y + 2x^2$。我们想要计算偏导数 $\frac{\partial g}{\partial x}$ 和 $\frac{\partial g}{\partial y}$。

解析结果：
$\frac{\partial g}{\partial x} = 3x^2 y + 4x$
$\frac{\partial g}{\partial y} = x^3$

1. **定义函数：**

   ```python
   def multi_var_func(x, y):
     """一个二元函数。"""
     return x**3 * y + 2 * x**2
   ```
2. **计算对第一个参数 (x) 的梯度：** 默认情况下，`jax.grad` 对第一个参数 (`argnums=0`) 求导。

   ```python
   grad_g_wrt_x = jax.grad(multi_var_func, argnums=0)

   x_val = 2.0
   y_val = 3.0

   gradient_x = grad_g_wrt_x(x_val, y_val)
   print(f"在 ({x_val}, {y_val}) 处对 x 的梯度：{gradient_x}")

   # 解析验证：3*(2^2)*3 + 4*2 = 3*4*3 + 8 = 36 + 8 = 44
   analytical_grad_x = 3 * x_val**2 * y_val + 4 * x_val
   print(f"对 x 的解析梯度：{analytical_grad_x}")
   ```
3. **计算对第二个参数 (y) 的梯度：** 使用 `argnums=1`。

   ```python
   grad_g_wrt_y = jax.grad(multi_var_func, argnums=1)

   gradient_y = grad_g_wrt_y(x_val, y_val)
   print(f"\n在 ({x_val}, {y_val}) 处对 y 的梯度：{gradient_y}")

   # 解析验证：2^3 = 8
   analytical_grad_y = x_val**3
   print(f"对 y 的解析梯度：{analytical_grad_y}")
   ```
4. **计算对两个参数的梯度：** 使用 `argnums=(0, 1)`。这会返回一个包含梯度的元组。

   ```python
   grad_g_wrt_xy = jax.grad(multi_var_func, argnums=(0, 1))

   gradient_xy = grad_g_wrt_xy(x_val, y_val)
   print(f"\n在 ({x_val}, {y_val}) 处对 (x, y) 的梯度：{gradient_xy}")
   print(f"对 (x, y) 的解析梯度：({analytical_grad_x}, {analytical_grad_y})")
   ```

结果应与解析偏导数一致。

### 练习 3：高阶导数

让我们找到原始多项式 $f(x) = 3x^2 + 2x + 5$ 的二阶导数。一阶导数为 $f'(x) = 6x + 2$，二阶导数为 $f''(x) = 6$。

1. **计算二阶导数：** 两次应用 `jax.grad`。

   ```python
   # 我们已经有了 grad_poly_func = jax.grad(poly_func)
   grad_grad_poly_func = jax.grad(grad_poly_func) # 再次应用 grad

   x_value = 4.0 # 对于 f''(x) = 6，此点不重要
   second_derivative = grad_grad_poly_func(x_value)
   print(f"\nf(x) 在 x = {x_value} 处的二阶导数：{second_derivative}")
   print(f"解析二阶导数：6.0")
   ```

结果应为 6.0，与输入 `x_value` 无关。

### 练习 4：使用 `jax.value_and_grad`

在优化中，您通常需要函数值（例如，损失）及其梯度。`jax.value_and_grad` 可以同时计算两者，这比分别调用函数及其梯度函数更高效。

让我们使用函数 $h(w) = (w - 5)^2$，这是一个简单的二次函数，常用于表示一个基础损失函数 (loss function)，我们想找到使 $h(w)$ 最小化的 $w$（最小值为 $w=5$ 处）。梯度为 $h'(w) = 2(w - 5)$。

1. **定义函数：**

   ```python
   def simple_loss(w):
     """一个简单的二次损失函数。"""
     return (w - 5.0)**2
   ```
2. **创建值-和-梯度函数：**

   ```python
   value_and_grad_loss = jax.value_and_grad(simple_loss)
   ```
3. **在特定点进行评估：** 让我们尝试 $w = 2.0$。

   ```python
   w_value = 2.0
   value, gradient = value_and_grad_loss(w_value)

   print(f"\n在 w = {w_value} 处使用 jax.value_and_grad：")
   print(f"  函数值 h(w)：{value}")
   print(f"  梯度 h'(w)：{gradient}")

   # 解析验证：
   # h(2) = (2 - 5)^2 = (-3)^2 = 9
   # h'(2) = 2 * (2 - 5) = 2 * (-3) = -6
   analytical_value = (w_value - 5.0)**2
   analytical_gradient_h = 2 * (w_value - 5.0)
   print(f"解析值：{analytical_value}")
   print(f"解析梯度：{analytical_gradient_h}")
   ```

您将在一次调用中同时获得函数值 (9.0) 和梯度 (-6.0)。

### 练习 5：使用 `jax.numpy` 的函数梯度

`jax.grad` 适用于使用 `jax.numpy` 构建的函数。让我们计算一个包含 `jnp.sum` 和 `jnp.sin` 的函数的梯度。

考虑 $k(\mathbf{v}) = \sum_{i} \sin(v_i)$，假设 $\mathbf{v}$ 是一个向量 (vector)。梯度 $\nabla k(\mathbf{v})$ 是一个向量，其第 $j$ 个元素为 $\frac{\partial k}{\partial v_j} = \cos(v_j)$。

1. **使用 `jnp` 定义函数：**

   ```python
   def sum_of_sines(v):
     """计算向量元素正弦值的和。"""
     return jnp.sum(jnp.sin(v))
   ```
2. **创建梯度函数：**

   ```python
   grad_sum_of_sines = jax.grad(sum_of_sines)
   ```
3. **使用示例向量进行评估：**

   ```python
   v_vector = jnp.array([0.0, jnp.pi/2, jnp.pi])

   gradient_vector = grad_sum_of_sines(v_vector)
   print(f"\n当 v = {v_vector} 时 sum_of_sines 的梯度：")
   print(f"  梯度：{gradient_vector}")

   # 解析验证：梯度应为 [cos(0), cos(pi/2), cos(pi)]
   analytical_gradient_k = jnp.cos(v_vector)
   print(f"解析梯度：{analytical_gradient_k}")
   ```

输出梯度向量应为 `[1. 0. -1.]`，与 `jnp.cos(v_vector)` 一致。

这些练习体现了 `jax.grad` 在各种情况下的实际应用，从简单多项式到包含 `jax.numpy` 操作和多个参数 (parameter)的函数。掌握这些模式对于在优化和机器学习 (machine learning)任务中使用 JAX 来说非常重要。

## 参考资料

- [JAX documentation: Automatic Differentiation Cookbook](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html) — The JAX team (2024)
  提供使用 `jax.grad`、`jax.value_and_grad` 和 JAX 中高阶导数的实践示例和解释，与练习内容直接相关。
- [Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation](https://epubs.siam.org/doi/book/10.1137/1.9780898717765) — Andreas Griewank, Andrea Walther (2008)
  Publisher: Society for Industrial and Applied Mathematics (SIAM); DOI: [10.1137/1.9780898717765](https://doi.org/10.1137/1.9780898717765)
  一本关于自动微分理论和算法的综合性学术著作，构成了JAX等库的基础。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 6
  涵盖了自动微分（尤其是反向传播）在神经网络背景下的数学和计算原理。
