# JIT 工作原理：追踪与编译

来源：[原文](https://apxml.com/zh/courses/getting-started-with-jax/chapter-2-accelerating-functions-jit/how-jit-works)

[返回章节目录](README.md) · [返回课程目录](../README.md)

使用 `jax.jit` 包装函数可以使其运行得更快，有时甚至是显著的提速。但其内部机制究竟是怎样的呢？这并非魔法；它是一个两阶段过程：**追踪**，随后是**编译**。理解此过程对于恰当使用 `jit` 并排查潜在问题非常重要。

### 追踪机制

首次调用经过 `jit` 编译的函数时，JAX 不会立即以常规方式运行您的 Python 代码。相反，JAX 会执行**追踪**。它会运行您的函数一次，但使用的是特殊的*追踪器*对象而非实际的数值。这些追踪器充当占位符，按顺序记录在它们之上执行的所有 JAX 操作。

可以这样来理解：想象一下您给某人一个食谱（您的 Python 函数），但要求他们写下*每一个步骤*（例如“加面粉”、“混合配料”），而暂时不实际烘焙蛋糕。他们是在追踪这个*过程*。

在追踪过程中，JAX 操作（如 `jnp.dot`、`jnp.add` 等）不计算数值结果。它们在这些追踪器对象上进行操作并返回新的追踪器，以此构建出一个计算图。涉及这些追踪器*值*的常规 Python 操作或控制流可能会导致问题，我们将在稍后讨论。

此追踪过程的结果是一个中间表示，称为 **Jaxpr** (JAX Program Representation)。Jaxpr 是一种简单、函数式且显式类型化的中间语言，它描述了函数执行的基本操作序列。

让我们来看一个简单的例子：

```python
import jax
import jax.numpy as jnp

def my_simple_func(x, y):
  a = jnp.sin(x)
  b = jnp.cos(y)
  return a + b

# 创建示例输入（追踪器将具有这些形状/数据类型）
x_example = jnp.ones(3)
y_example = jnp.zeros(3)

# 使用 jax.make_jaxpr 查看追踪结果
jaxpr_representation = jax.make_jaxpr(my_simple_func)(x_example, y_example)
print(jaxpr_representation)
```

运行这段代码会输出类似以下内容：

```
{ lambda ; a:f32[3] b:f32[3]. let
    c:f32[3] = sin a
    d:f32[3] = cos b
    e:f32[3] = add c d
  in (e,) }
```

此 Jaxpr 清晰地展现了所涉及的操作（`sin`、`cos`、`add`）和类型（`f32[3]`）。它是您计算的蓝图，与 `x` 和 `y` 的具体*值*无关，但取决于它们的*形状*和*数据类型*（dtypes）。

### 通过 XLA 编译

一旦 JAX 获得了 Jaxpr，它会将其交给 **XLA (Accelerated Linear Algebra)** 编译器。XLA 是谷歌开发的一种特定领域的编译器，专为线性代数计算优化。它接收 Jaxpr 并将其编译成针对您的目标硬件（无论是 CPU、GPU 还是 TPU）高度优化的机器码。

XLA 执行多种优化，例如：

- **融合 (Fusion)**：将多个操作合并为单个、更高效的内核，以减少内存带宽需求和启动开销。例如，我们示例中的 `sin`、`cos` 和 `add` 可能会被融合成一次计算。
- **内核生成 (Kernel Generation)**：生成针对特定设备架构优化的代码。
- **内存布局优化 (Memory Layout Optimization)**：安排内存中的数据以实现更快访问。

此编译步骤通常是 `jit` 编译函数*首次*调用时耗时最长的部分。

### 缓存编译代码

`jit` 真正的好处来自于缓存。函数针对输入*形状*和*数据类型*的特定组合（以及静态参数 (parameter)值，稍后讨论）进行追踪和编译后，生成的优化机器码会被缓存。

后续使用与缓存签名匹配的输入（相同的形状、数据类型、静态值）对同一 `jit` 修饰函数进行的调用，将直接执行高度优化的缓存机器码。它们完全绕过 Python 解释器、追踪和 XLA 编译步骤。这是性能大幅提升的来源。

我们可以将基本流程可视化：

> JIT编译流程：首次调用会启动追踪和编译，后续使用匹配输入签名的调用会使用缓存的优化代码。

请看这个时间比较：

```python
import jax
import jax.numpy as jnp
import time

@jax.jit
def slow_function(x):
  # 模拟一些矩阵运算工作
  for _ in range(5):
      # 确保矩阵乘法有效（方形或兼容形状）
      if x.shape[0] == x.shape[1]:
          x = jnp.dot(x, x.T) + 0.5 * x
      else:
          # 适当处理非方形情况，例如，元素级操作
          x = x * x + 0.5 * x
  return x

key = jax.random.PRNGKey(0)
data = jax.random.normal(key, (100, 100))

# --- 首次调用：追踪和编译 ---
start_time = time.time()
result1 = slow_function(data)
# 用来使计算在计时器停止前完成，尤其是在GPU/TPU上
result1.block_until_ready()
end_time = time.time()
print(f"首次调用（编译）耗时：{end_time - start_time:.4f} 秒")

# --- 第二次调用：使用缓存代码 ---
start_time = time.time()
result2 = slow_function(data)
result2.block_until_ready()
end_time = time.time()
print(f"第二次调用（缓存）耗时：     {end_time - start_time:.4f} 秒")

# --- 使用不同形状调用：重新编译 ---
# 确保新形状也是方形的，以便进行点积逻辑
data_different_shape = jax.random.normal(key, (150, 150))
start_time = time.time()
result3 = slow_function(data_different_shape)
result3.block_until_ready()
end_time = time.time()
print(f"使用新形状调用耗时：      {end_time - start_time:.4f} 秒")

# --- 再次使用原始形状调用：使用缓存 ---
start_time = time.time()
result4 = slow_function(data)
result4.block_until_ready()
end_time = time.time()
print(f"再次使用原始形状调用： {end_time - start_time:.4f} 秒")
```

您会发现，首次调用和使用不同形状的调用明显更长，因为它们涉及追踪和编译。后续使用相同输入形状的调用则快得多，因为它们命中缓存。`block_until_ready()` 方法在此处用来使异步操作（在加速器上常见）在计时器停止前完成，从而提供准确的计时。

### 何时会重新追踪？

如果初始追踪过程中所做的假定不再适用，JAX 需要重新追踪并可能重新编译您的函数。这通常发生在以下情况：

1. **输入形状改变**：调用函数时，数组的形状与之前所见的形状不同（例如，`(100, 100)` 对比 `(150, 150)`）。
2. **输入数据类型改变**：调用时使用不同的数据类型（例如，`float32` 对比 `float64`）。
3. **输入 PyTree 结构改变**：如果您的输入是字典或列表（PyTrees）等嵌套结构，改变其结构（例如，向字典中添加元素）会启动重新追踪。
4. **静态参数 (parameter)值改变**：如果您使用 `static_argnums` 或 `static_argnames`（将在“静态值与追踪值”部分进行介绍），并且调用函数时这些静态参数的值不同。

每种不同的输入形状、数据类型、PyTree 结构和静态参数值组合，都将导致一次独立的追踪和编译，并填充缓存以供后续使用。

理解此追踪-编译-缓存循环对于恰当使用 `jax.jit` 非常重要。它阐明了初始编译成本和后续的加速，并有助于预判何时可能发生重新编译。接下来，我们将考察Python的动态特性，特别是控制流，如何影响此追踪过程。

## 参考资料

- [How JAX JIT compilation works](https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html#how-jax-jit-compilation-works) — The JAX Authors (2024)
  JAX官方文档说明了JIT的追踪、Jaxpr的创建和编译过程。
- [XLA: Accelerated Linear Algebra](https://www.tensorflow.org/xla) — The TensorFlow Authors (2024)
  XLA官方文档描述了XLA的架构、目的和优化能力。
- [Accelerate TensorFlow models with XLA](https://ai.googleblog.com/2018/11/accelerate-tensorflow-models-with-xla.html) — The Google AI Blog Team (2018)
  Journal: Google AI Blog; Publisher: Google
  谷歌AI博客文章，说明XLA的功能、益处以及如何加速机器学习模型。

---

[上一节](02-%E4%BB%8B%E7%BB%8D%20%60jax.jit%60.md) · [下一节](04-Python%20%E6%8E%A7%E5%88%B6%E6%B5%81%E4%B8%8E%20%60jit%60.md)
