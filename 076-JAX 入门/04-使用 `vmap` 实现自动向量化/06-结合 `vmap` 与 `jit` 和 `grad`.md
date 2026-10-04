# 结合 `vmap` 与 `jit` 和 `grad`

来源：[原文](https://apxml.com/zh/courses/getting-started-with-jax/chapter-4-automatic-vectorization-vmap/combining-vmap-jit-grad)

[返回章节目录](README.md) · [返回课程目录](../README.md)

JAX 设计的一个主要优点是其函数变换的可组合性。JAX 提供了函数变换，例如用于编译的 `jit`、用于微分的 `grad` 以及用于向量 (vector)化的 `vmap`。这些变换可以共同运作，从而实现机器学习 (machine learning)和科学计算中常见的强大高效的计算模式。将它们结合使用，你可以例如对数据批次计算梯度，并编译整个操作以在加速器上获得最佳性能。

### 结合 `vmap` 和 `jit`

你通常希望既使用 `vmap` 对函数进行向量 (vector)化，又使用 `jit` 对其进行编译。这很简单直接：你可以直接一个接一个地应用变换。典型的模式是先应用 `vmap` 来创建函数的向量化 (quantization)版本，然后再应用 `jit` 来编译该向量化函数。

```python
import jax
import jax.numpy as jnp
import time

# 定义一个处理单个数据点的函数
def predict(params, x):
  # 一个简单的线性模型：w*x + b
  w, b = params
  return w * x + b

# 创建一些示例参数和一批数据
key = jax.random.PRNGKey(0)
params = (jnp.array(2.0), jnp.array(1.0)) # (w, b)
batch_x = jnp.arange(10000.0)

# 1. 使用 vmap 对 predict 函数进行向量化
#    在 batch_x 上映射（轴 0），但保持 params 不变 (None)
batched_predict_vmap = jax.vmap(predict, in_axes=(None, 0))

# 2. 使用 jit 编译向量化函数
jitted_batched_predict = jax.jit(batched_predict_vmap)

# --- 计时比较 ---

# 运行 vmap 版本（不带 JIT）
start_time = time.time()
result_vmap = batched_predict_vmap(params, batch_x).block_until_ready()
duration_vmap = time.time() - start_time
print(f"仅 vmap 的执行时间: {duration_vmap:.6f} 秒")

# 运行 JIT 编译的 vmap 版本（首次运行包含编译时间）
start_time = time.time()
result_jit_vmap = jitted_batched_predict(params, batch_x).block_until_ready()
duration_jit_vmap_first = time.time() - start_time
print(f"jit(vmap(...)) 执行时间（首次运行）: {duration_jit_vmap_first:.6f} 秒")

# 再次运行 JIT 编译的 vmap 版本（应该更快）
start_time = time.time()
result_jit_vmap_again = jitted_batched_predict(params, batch_x).block_until_ready()
duration_jit_vmap_second = time.time() - start_time
print(f"jit(vmap(...)) 执行时间（第二次运行）: {duration_jit_vmap_second:.6f} 秒")

# 检查结果是否相同
print(f"结果匹配: {jnp.allclose(result_vmap, result_jit_vmap)}")
```

**为什么选择 `jit(vmap(f))`？**

将 `vmap` 包裹在 `jit` 中（`jit(vmap(f))`）通常是优选的顺序。

1. `vmap(f)` 首先创建一个新的 Python 函数，该函数在输入的映射轴上应用 `f`。在内部，`vmap` 将 `f` 中的 JAX 原语转换为在批次上操作。
2. `jit(...)` 接着接收这个*向量化*的函数，并使用 XLA 将其编译成针对目标硬件（CPU/GPU/TPU）的优化过的内核。这使得编译器能够查看整个批处理操作并作为一个整体对其进行优化。

尽管 `vmap(jit(f))` 也是可行的，但它会首先编译*内部*函数 `f`，然后向量化对已编译函数的调用。这可能效率较低，因为向量化逻辑在编译过的内核“外部”操作，与编译已向量化代码相比，可能导致硬件利用率不如理想。对于大多数常见用例，`jit(vmap(f))` 提供更好的性能。

### 结合 `vmap` 和 `grad`

另一个常见需求，尤其是在机器学习 (machine learning)中，是不只为单个数据点，而是为整个批次计算梯度。`vmap` 使得表达这一点变得简单。你可以使用 `grad` 计算函数的梯度，然后使用 `vmap` 对生成的梯度函数进行向量 (vector)化。

假设你有一个函数，它根据参数 (parameter)和单个数据点计算损失。你通常希望计算此损失关于参数的梯度，并且是*针对批次中的每个项目*计算。

```python
import jax
import jax.numpy as jnp

# 示例：单个数据点的平方误差损失
def squared_error(params, x, y_true):
  w, b = params
  y_pred = w * x + b
  loss = (y_pred - y_true)**2
  return loss

# 计算单个数据点关于参数（第0个参数）的梯度函数
grad_loss_single = jax.grad(squared_error, argnums=0)

# 示例参数和一批数据 (x, y_true)
key = jax.random.PRNGKey(1)
params = (jnp.array(2.0), jnp.array(1.0)) # (w, b)
batch_x = jnp.linspace(0, 1, 5)
batch_y_true = 2.5 * batch_x + 0.5 # 真实关系：w=2.5, b=0.5

# 对梯度函数进行向量化：
# 在 batch_x（轴 0）和 batch_y_true（轴 0）上映射
# 保持 params 在所有计算中不变 (None)
grad_loss_batch = jax.vmap(grad_loss_single, in_axes=(None, 0, 0))

# 计算整个批次的梯度
batch_gradients = grad_loss_batch(params, batch_x, batch_y_true)

# batch_gradients 将是一个元组 (dw, db)。dw 和 db 均为数组，它们包含为批次中每个项目计算的梯度。
print("参数 (w, b):", params)
print("批次 x:", batch_x)
print("批次 y_true:", batch_y_true)
print("\n每个示例的梯度 (dw):", batch_gradients[0])
print("每个示例的梯度 (db):", batch_gradients[1])
print("\ndw 的形状:", batch_gradients[0].shape) # 应该为 (5,)
print("db 的形状:", batch_gradients[1].shape) # 应该为 (5,)
```

此处，`grad_loss_single` 为单个 $(x, y_{true})$ 对计算梯度 $(\frac{\partial L}{\partial w}, \frac{\partial L}{\partial b})$。将 `vmap` 应用于 `grad_loss_single` 有效地在 `batch_x` 和 `batch_y_true` 数组上循环此梯度计算，返回每个示例的梯度。生成梯度的形状反映了 `vmap` 添加的批次维度。

此模式对于需要每个示例梯度的算法很有用。在深度学习 (deep learning)中更常见的是，你可能需要批次中的*平均*梯度。你可以像上面那样计算每个示例的梯度然后取它们的平均值，或者你可以在求梯度*之前*定义损失函数 (loss function)来计算平均损失：

```python
import jax
import jax.numpy as jnp

# 示例：批次上的均方误差损失
def mean_squared_error(params, batch_x, batch_y_true):
  # 在损失函数*内部*使用 vmap 进行预测
  # 注意：这通常不如 vmap(grad(...)) 明确
  batched_predict = jax.vmap(predict, in_axes=(None, 0))
  batch_y_pred = batched_predict(params, batch_x)
  loss = jnp.mean((batch_y_pred - batch_y_true)**2)
  return loss

# 计算*平均*损失关于参数的梯度
grad_mean_loss = jax.grad(mean_squared_error, argnums=0)

# 计算表示批次平均梯度的单个梯度向量
mean_batch_gradient = grad_mean_loss(params, batch_x, batch_y_true)

print("\n--- 平均梯度 ---")
print("平均批次梯度 (dw, db):", mean_batch_gradient)
print("平均 dw 的形状:", mean_batch_gradient[0].shape) # 应该为 () - 标量
print("平均 db 的形状:", mean_batch_gradient[1].shape) # 应该为 () - 标量

# 你可以验证这与每个示例梯度的平均值匹配
print("每个示例 dw 的平均值:", jnp.mean(batch_gradients[0]))
print("每个示例 db 的平均值:", jnp.mean(batch_gradients[1]))
print(f"平均梯度匹配: {jnp.allclose(mean_batch_gradient[0], jnp.mean(batch_gradients[0]))}")
```

当你明确需要每个示例的结果时，`vmap(grad(f, ...), ...)` 模式通常更清晰，而 `grad(mean_loss_fn, ...)` 则是典型梯度下降 (gradient descent)优化的标准做法，因为这种情况下只需要批次中的平均梯度。

### 结合 `vmap`、`grad` 和 `jit`

现在，让我们将这三者结合起来。这是在 JAX 中高效训练机器学习 (machine learning)模型的基本模式。你通常会希望：

1. 定义一个处理*单个*示例（或有时直接处理批次）的损失函数 (loss function)。
2. 获取一个使用 `grad` 计算损失梯度的函数。
3. 使用 `vmap` 对数据批次上的梯度计算进行向量 (vector)化。
4. 使用 `jit` 编译得到的批处理梯度计算，以在加速器上获得高性能。

对于计算批次*平均*梯度，最常见且通常性能最佳的顺序是 `jit(grad(mean_loss_fn))`。如果你需要高效地获取*每个示例*的梯度，模式则是 `jit(vmap(grad(single_loss_fn)))`。

让我们演示 `jit(vmap(grad(f)))` 模式，用于高效地获取每个示例的梯度：

```python
import jax
import jax.numpy as jnp
import time

# 使用之前定义的单个示例的 squared_error
def squared_error(params, x, y_true):
  w, b = params
  y_pred = w * x + b
  loss = (y_pred - y_true)**2
  return loss

# 1. 获取单个示例关于参数的梯度函数
grad_loss_single = jax.grad(squared_error, argnums=0)

# 2. 对单个示例的梯度函数进行向量化
#    在 batch_x（轴 0）和 batch_y_true（轴 0）上映射
#    保持 params 固定 (None)
vmap_grad_loss = jax.vmap(grad_loss_single, in_axes=(None, 0, 0))

# 3. 编译向量化梯度函数
jit_vmap_grad_loss = jax.jit(vmap_grad_loss)

# 准备更大的批次数据用于计时
key = jax.random.PRNGKey(42)
large_batch_size = 100000
params = (jnp.array(2.0), jnp.array(1.0)) # (w, b)
large_batch_x = jax.random.uniform(key, (large_batch_size,))
large_batch_y_true = 2.5 * large_batch_x + 0.5 + 0.1 * jax.random.normal(key, (large_batch_size,))

# --- 计时 ---

# 运行编译过的函数（包含编译时间）
start_time = time.time()
per_example_grads = jit_vmap_grad_loss(params, large_batch_x, large_batch_y_true)
# 等待计算完成再停止计时
per_example_grads[0].block_until_ready()
per_example_grads[1].block_until_ready()
duration_first = time.time() - start_time
print(f"jit(vmap(grad(...))) 执行时间（首次运行）: {duration_first:.6f} 秒")

# 再次运行（应该快很多）
start_time = time.time()
per_example_grads_again = jit_vmap_grad_loss(params, large_batch_x, large_batch_y_true)
per_example_grads_again[0].block_until_ready()
per_example_grads_again[1].block_until_ready()
duration_second = time.time() - start_time
print(f"jit(vmap(grad(...))) 执行时间（第二次运行）: {duration_second:.6f} 秒")

print(f"\n每个示例 dw 的形状: {per_example_grads[0].shape}")
print(f"每个示例 db 的形状: {per_example_grads[1].shape}")
```

通过组合 `jit`、`vmap` 和 `grad`，你可以创建高度优化过的函数，这些函数在现代硬件上高效地计算批处理梯度，构成许多基于 JAX 的机器学习工作流程的核心。

### 注意事项

- **调试：** 堆叠变换有时会使调试更具挑战性，因为回溯可能源于 JAX 变换机制的深层内部。使用 `jax.debug.print` 或暂时禁用 `jit` 等方法可以帮助隔离问题。
- **内存：** 尽管 `vmap` 效率高，但对非常大的批次进行向量 (vector)化可能会消耗大量内存，尤其是在具有固定内存限制的 GPU/TPU 上。请留意你的批次大小相对于可用的设备内存。
- **顺序：** 如前所述，应用变换的顺序对正确性和性能都很重要。`jit(vmap(grad(f)))` 和 `jit(grad(mean_loss))` 是常见且有效的模式，但理解它们为何有效有助于你将它们应用于新情况。

掌握 `vmap`、`grad` 和 `jit` 的组合使用对于编写简洁、高性能的 JAX 代码非常重要，尤其是在处理深度学习 (deep learning)和其他数据并行计算中固有的批处理时。

## 参考资料

- [JAX Quickstart](https://jax.readthedocs.io/en/latest/quickstart.html) — JAX Authors (2024)
  介绍了JAX的核心函数变换，包括`jit`、`grad`和`vmap`，并展示了它们的基本用法和组合性。
- [JAX: Composable transformations of Python+NumPy programs](http://github.com/google/jax) — James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary, Dougal Maclaurin, George Necula, Adam Paszke, Jake VanderPlas, Skye Wanderman-Milne, and Qiao Zhang (2018)
  Publisher: Google
  介绍了JAX的设计原则，特别是其核心变换（`jit`、`grad`、`vmap`）的组合性，这对于本节内容至关重要。
- [Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation](https://books.google.com/books?id=QJbWAAAAMAAJ) — Andreas Griewank, Andrea Walther (2008)
  Publisher: Society for Industrial and Applied Mathematics (SIAM)
  一本关于算法微分（自动微分）的经典著作，为`grad`变换提供了理论基础。
- [TensorFlow's XLA: Optimizing Compiler for Machine Learning](https://www.tensorflow.org/xla) — Google Brain Team (2024)
  Publisher: Google Developers
  描述了XLA，这是JAX `jit`变换使用的优化编译器后端，用于在各种硬件加速器上实现高计算性能。

---

[上一节](05-%E5%B5%8C%E5%A5%97%20%60vmap%60.md) · [下一节](07-%60vmap%60%E7%9A%84%E6%80%A7%E8%83%BD%E8%80%83%E9%87%8F.md)
