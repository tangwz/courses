# 值和梯度 (`jax.value_and_grad`)

来源：[原文](https://apxml.com/zh/courses/getting-started-with-jax/chapter-3-automatic-differentiation-grad/jax-value-and-grad)

[返回章节目录](README.md) · [返回课程目录](../README.md)

在进行优化或分析时，经常不仅需要函数的梯度，还需要函数在该点对应的原始输出值。例如，在训练机器学习 (machine learning)模型时，通常需要记录损失值，同时计算其梯度以更新模型参数 (parameter)。

优化或分析任务常常不仅需要函数的梯度，还需要函数在该点的原始输出值。例如，机器学习模型训练通常涉及记录损失值，并同时计算其梯度以更新模型参数。

```python
import jax
import jax.numpy as jnp

# 示例函数
def polynomial(x):
  return x**3 + 2*x**2 - 3*x + 1

# 计算值
x_val = 2.0
value = polynomial(x_val)

# 单独计算梯度
grad_fn = jax.grad(polynomial)
gradient = grad_fn(x_val)

print(f"函数在 x={x_val} 处的值：{value}")
print(f"函数在 x={x_val} 处的梯度：{gradient}")
# 预期梯度：3*x**2 + 4*x - 3 = 3*(2**2) + 4*2 - 3 = 12 + 8 - 3 = 17
```

这种做法虽然可行，但效率不高。回想一下关于反向模式自动微分如何工作的讨论：正向传播通常会计算中间值，这些值与函数的最终输出密切相关，甚至完全相同。计算梯度既需要正向传播（类似于计算原始值），也需要反向传播 (backpropagation)。分别调用函数及其梯度函数，本质上是重复进行了两次正向传播的工作。

JAX 为这种常见模式提供了一种更高效的转换方式：`jax.value_and_grad`。这个函数接受你的原始函数，并返回一个 *新* 函数，该新函数在被调用时，会通过一次优化的计算过程，同时计算出原始函数的值和其梯度。

以下是它的用法：

```python
import jax
import jax.numpy as jnp

# 示例函数（与之前相同）
def polynomial(x):
  return x**3 + 2*x**2 - 3*x + 1

# 创建一个同时返回值和梯度的函数
value_and_grad_fn = jax.value_and_grad(polynomial)

# 调用新函数
x_val = 2.0
value, gradient = value_and_grad_fn(x_val)

print(f"使用 jax.value_and_grad：")
print(f"  函数值：{value}")
print(f"  梯度：{gradient}")
```

`value_and_grad_fn` 返回一个元组，其中第一个元素是 `polynomial(x_val)` 的结果，第二个元素是 `jax.grad(polynomial)(x_val)` 的结果。这通过共享正向传播的工作，避免了重复计算。

### 对多个参数 (parameter)求微分

和 `jax.grad` 一样，`jax.value_and_grad` 接受 `argnums` 参数，以指定对哪些位置参数求微分。

如果 `argnums` 是一个整数，返回的梯度将对应于该单个参数。

```python
def multi_arg_func(a, b):
  return a**2 * jnp.sin(b)

# 对第一个参数（索引 0）求微分
value_and_grad_a_fn = jax.value_and_grad(multi_arg_func, argnums=0)

a_val, b_val = 3.0, jnp.pi / 2
value, grad_a = value_and_grad_a_fn(a_val, b_val)

# 预期 grad_a: 2*a*sin(b) = 2*3*sin(pi/2) = 6 * 1 = 6
print(f"值：{value}")
print(f"相对于 'a' 的梯度：{grad_a}")
```

如果 `argnums` 是一个整数元组，返回的梯度将是一个元组，其中每个元素对应于指定索引处参数的梯度。

```python
# 对两个参数（索引 0 和 1）都求微分
value_and_grad_ab_fn = jax.value_and_grad(multi_arg_func, argnums=(0, 1))

a_val, b_val = 3.0, jnp.pi / 2
value, (grad_a, grad_b) = value_and_grad_ab_fn(a_val, b_val)

# 预期 grad_a: 2*a*sin(b) = 6
# 预期 grad_b: a**2*cos(b) = 3**2*cos(pi/2) = 9 * 0 = 0
print(f"\n值：{value}")
print(f"相对于 'a' 的梯度：{grad_a}")
print(f"相对于 'b' 的梯度：{grad_b}")
```

在实现优化算法（如梯度下降 (gradient descent)）时，`jax.value_and_grad` 是一种标准做法，因为在每一步中都需要损失值及其梯度。它能与其他 JAX 转换（例如 `jit`）很好地配合，使你能够编写高效、可微分且可编译的代码。

## 参考资料

- [\`jax.value_and_grad\`](https://jax.readthedocs.io/en/latest/_autosummary/jax.value_and_grad.html) — JAX Core Developers (2024)
  `jax.value_and_grad` 函数的官方文档，详细说明其用法和参数。
- [Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation](https://epubs.siam.org/doi/book/10.1137/1.9780898717765) — Andreas Griewank, Andrea Walther (2008)
  Publisher: SIAM; DOI: [10.1137/1.9780898717765](https://doi.org/10.1137/1.9780898717765)
  一本关于自动微分的经典著作，涵盖了 `jax.value_and_grad` 用于高效组合值和梯度计算的逆向模式 AD 等技术的理论基础。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  第六章“深度前馈网络”和第八章“深度模型训练优化”解释了反向传播和基于梯度的优化，这是 `jax.value_and_grad` 等函数用于获取损失和梯度的主要应用。

---

[上一节](05-%E9%AB%98%E9%98%B6%E5%AF%BC%E6%95%B0%EF%BC%88%60grad%60%E7%9A%84%60grad%60%EF%BC%89.md) · [下一节](07-%E6%B1%82%E5%AF%BC%E4%B8%8E%E6%8E%A7%E5%88%B6%E6%B5%81.md)
