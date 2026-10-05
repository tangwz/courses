# 核心转换回顾：jit, grad, vmap

来源：[原文](https://apxml.com/zh/courses/advanced-jax/chapter-1-advanced-jax-transformations-control-flow/review-core-transformations)

[返回章节目录](README.md) · [返回课程目录](../README.md)

JAX 编程的基础转换包括 `jax.jit`、`jax.grad` 和 `jax.vmap`。这些转换是 JAX 高级控制流机制（例如 `lax.scan` 和 `lax.cond`）的基础。对于具有扎实 JAX 背景的开发人员，这里巩固了这些转换的主要概念和操作特点，这对于理解它们与更复杂结构之间的配合非常重要。这些转换作用于函数，将 Python 函数作为输入，并返回新的、经过转换的 Python 函数。

### jax.jit: 使用 XLA 加速代码

`jax.jit` 转换通过使用谷歌的加速线性代数 (XLA) 编译器对 Python 函数进行编译，从而加速你的 Python 函数，特别是那些涉及机器学习 (machine learning)中常见数值计算的函数。

当 `jit` 编译的函数首次以特定的输入形状和类型调用时，JAX 会执行*追踪*。在追踪过程中，JAX 不使用实际数值执行 Python 代码，而是使用抽象的*追踪器*对象。这些追踪器会记录所执行的原始操作序列。这个被记录的序列，称为 *jaxpr*（JAX 程序表示），得到计算图。然后 XLA 获取这个 jaxpr，并将其编译成针对目标硬件（CPU、GPU 或 TPU）定制的高度优化的机器代码。后续使用匹配的输入形状和类型进行的调用会直接执行这段预编译代码，从而绕过 Python 解释器的开销，并受益于 XLA 优化，例如运算符融合。

> `jax.jit` 编译过程的简化视图。

此编译过程带来一项重要限制：由 `jit` 转换的函数必须在追踪的操作方面保持*功能纯粹*。这意味着它们不应该有副作用（如打印或修改外部状态）在执行过程中依赖追踪器值，因为这些副作用在追踪期间只发生一次。函数的输出必须仅依赖于其显式输入。

```python
import jax
import jax.numpy as jnp

# 一个简单的函数
def slow_f(x):
  # 模拟一些计算
  return jnp.sin(x) * x + jnp.log(x + 1)

# 应用 jit
fast_f = jax.jit(slow_f)

# 首次调用：进行追踪和编译
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (1000, 1000))
result1 = fast_f(x)
result1.block_until_ready() # 确保计算完成以便计时

# 第二次调用：使用已编译的内核（快得多）
result2 = fast_f(x)
result2.block_until_ready()

print(f"结果接近: {jnp.allclose(result1, result2)}")
# 输出：结果接近：True
```

虽然功能极强，请注意输入数组形状或数据类型的变化，或者函数闭包中 Python 常量值的变化，都可能触发重新编译。我们将在第 2 章中研究减少重新编译的策略。

### jax.grad: 自动微分

自动微分对现代机器学习 (machine learning)中优化模型参数 (parameter)很重要。`jax.grad` 提供 JAX 主要的逆向模式自动微分功能。给定一个计算标量输出的 Python 函数，`jax.grad` 返回一个*新函数*，该函数计算原始函数针对其一个参数（默认情况下是第一个）的梯度。

```python
def predict(params, x):
  w, b = params
  return jnp.dot(w, x) + b

def loss_fn(params, x, y_true):
  y_pred = predict(params, x)
  # 简单的平方误差
  return jnp.mean((y_pred - y_true)**2)

# 获取计算关于 'params'（参数 0）梯度的函数
grad_loss_fn = jax.grad(loss_fn, argnums=0)

# 示例数据
w_init = jnp.array([1.5, -0.5])
b_init = jnp.array(0.3)
params_init = (w_init, b_init)
x_data = jnp.array([0.2, 0.8])
y_target = jnp.array(2.5)

# 计算梯度
gradients = grad_loss_fn(params_init, x_data, y_target)

print(f"关于 w 的梯度: {gradients[0]}")
print(f"关于 b 的梯度: {gradients[1]}")
# 示例输出（具体值取决于计算）：
# 关于 w 的梯度: [-0.392 -1.568]
# 关于 b 的梯度: -1.96
```

在内部实现上，`jax.grad` 构建于向量 (vector)-雅可比积 (VJPs) 之上，我们将在第 4 章中详细查看它。它的组合特性使得通过多次应用 `grad` 来计算高阶导数变得简单。请记住，`grad` 要求被微分的函数返回一个标量值。对于返回多个值或非标量数组的函数，需要不同的技术或与 `vmap` 结合，以计算完整的雅可比矩阵或海森矩阵。

### jax.vmap: 自动向量 (vector)化

`jax.vmap` 是一个向量化 (quantization)映射。它将一个作用于单个数据点的函数转换为一个可以高效地作用于数据批次或轴的函数，无需在 Python 中手动循环。这是通过添加批次维度（在该维度上“映射”函数）完成的。

主要想法是指定哪些输入参数 (parameter)具有批次维度以及该维度应如何映射。`jax.vmap(fun, in_axes, out_axes)` 接受函数 `fun` 和指定映射轴的参数：

- `in_axes`：一个元组/列表/pytree，指示每个输入参数的哪个轴应该被映射。`None` 表示该参数被广播。`0` 通常表示在第一个轴上映射。
- `out_axes`：指定映射轴应出现在输出的什么位置。

```python
# 函数作用于单个向量
def simple_affine(w, b, x):
  # w: 矩阵 [输出维度, 输入维度]
  # b: 向量 [输出维度]
  # x: 向量 [输入维度]
  return jnp.dot(w, x) + b

# 示例参数
w = jnp.array([[1., 2.], [3., 4.], [5., 6.]]) # Shape (3, 2)
b = jnp.array([0.1, 0.2, 0.3])                 # Shape (3,)

# 输入批次（4个大小为2的向量）
x_batch = jnp.arange(8.).reshape(4, 2)       # Shape (4, 2)

# 对 `x_batch` 的批次维度（轴 0）进行 `simple_affine` 的向量化。
# `w` 和 `b` 不被映射（广播）。
batched_affine = jax.vmap(simple_affine, in_axes=(None, None, 0))

# 应用向量化函数
y_batch = batched_affine(w, b, x_batch)

print(f"输入批次形状: {x_batch.shape}")
print(f"输出批次形状: {y_batch.shape}")
# 输出：
# 输入批次形状: (4, 2)
# 输出批次形状: (4, 3)
```

`vmap` 对于在机器学习 (machine learning)中处理数据批次非常重要，避免了缓慢的 Python `for` 循环，并使用并行硬件的功能。它可以任意嵌套并与 `jit` 和 `grad` 组合，使得复杂的批处理计算和梯度计算成为可能。

### 组合即能力

当这些转换组合在一起时，JAX 的真正价值就会显现出来。你可以对梯度函数进行 `jit` 编译（`jit(grad(f))`），对已编译函数进行向量 (vector)化（`vmap(jit(f))`），或者计算批处理梯度（`jit(vmap(grad(f)))`）等等。组合的顺序很重要，并会影响最终的计算结果。

```python
# 示例：JIT 编译的批处理梯度计算
batched_grad_loss_fn = jax.jit(jax.vmap(grad_loss_fn, in_axes=(None, 0, 0)))

# 生成批次数据
key, subkey = jax.random.split(key)
x_batch_data = jax.random.normal(subkey, (16, 2)) # 16 个批次
y_batch_target = jax.random.normal(key, (16,))   # 16 个批次

# 高效计算整个批次的梯度
batch_gradients = batched_grad_loss_fn(params_init, x_batch_data, y_batch_target)

print(f"关于 w 的批次梯度形状: {batch_gradients[0].shape}")
print(f"关于 b 的批次梯度形状: {batch_gradients[1].shape}")
# 输出：
# 关于 w 的批次梯度形状: (16, 2)
# 关于 b 的批次梯度形状: (16,)
```

了解 `jit` 如何追踪代码、`grad` 如何传播导数以及 `vmap` 如何操作批次维度是很基本的。当引入 `lax.scan`、`lax.cond` 和 `lax.while_loop` 等控制流原语时，这种了解变得更加重要，因为这些原语以特定的方式与追踪和转换机制配合，我们将在本章的后续部分进行查看。

## 参考资料

- [JAX: Core Transformations Guide](https://jax.readthedocs.io/en/latest/autodidax.html) — The JAX authors (2023)
  Publisher: The JAX Authors
  JAX核心转换的指南，通过示例解释了`jit`、`grad`和`vmap`。
- [Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation](https://press.siam.org/books/OT105) — Andreas Griewank, Andrea Walther (2008)
  Publisher: Society for Industrial and Applied Mathematics (SIAM)
  关于自动微分的学术书籍，为`jax.grad`提供了理论基础。
- [XLA: Accelerated Linear Algebra](https://www.tensorflow.org/xla) — Google (2024)
  XLA的官方文档，JAX使用它通过`jax.jit`加速数值计算。

---

[下一节](02-%E7%B2%BE%E9%80%9A%20lax.scan%20%E5%A4%84%E7%90%86%E5%BA%8F%E5%88%97%E6%93%8D%E4%BD%9C.md)
