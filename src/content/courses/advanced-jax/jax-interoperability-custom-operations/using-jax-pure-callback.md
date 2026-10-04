---
course: "advanced-jax"
chapter: "jax-interoperability-custom-operations"
lesson: "using-jax-pure-callback"
sourceId: 2907
sourceUrl: "https://apxml.com/zh/courses/advanced-jax/chapter-5-jax-interoperability-custom-operations/using-jax-pure-callback"
title: "使用 jax.pure_callback 进行无副作用调用"
description: "在 JIT 编译的代码中安全调用保证纯粹的外部 Python 函数。"
order: 4
plots: []
sourceHash: "5a323a85929ecad0a577f7f79377f16525e25baf5caabeec1ec51d7cbc402621"
sourceCorrections: []
---

虽然 `jax.experimental.host_callback` 提供了一种在 JAX 计算期间在主机上执行任意 Python 代码的方式，但其性质会打断 JAX 转换的流程，主要用于调试或日志记录等副作用。要将 *计算纯粹* 的外部 Python 函数整合到需要转换（例如，使用 `jit`、`vmap` 或 `grad`）的 JAX 图中，JAX 提供了 `jax.pure_callback`。

`jax.pure_callback` 允许你在转换后的 JAX 代码中调用 Python 函数，但它附带一项重要的约定：你调用的函数*必须*是函数式纯粹的。

### 纯粹性约定

函数式纯粹性对于 JAX 的追踪和转换机制的正常运行非常重要。纯函数具有两个主要特性：

1. **确定性：** 对于相同的输入，函数*总是*返回相同的输出。
2. **无副作用：** 函数不得修改任何外部状态、执行输入/输出操作（如打印到控制台、读取/写入文件、与网络交互），或其行为不得基于显式输入之外的因素而改变。

JAX 依赖于这种纯粹性。在追踪阶段（例如，当 `jit` 编译一个函数时），JAX 根据抽象的形状和类型分析操作。它需要相信回调函数的行为只取决于其输入，并且其抽象表示（输出形状/数据类型）准确反映了其在任何有效输入下的运行时行为。如果用 `pure_callback` 封装的函数违反这项约定（例如，对相同输入返回不同的值，或修改全局变量），JAX 转换的结果可能会变得不正确或不可预测，通常不会引发明确的错误。

### `jax.pure_callback` 的工作原理

当你使用 `jax.pure_callback` 时，你需要提供三项主要内容：

1. 要调用的 Python 函数（回调函数）。
2. 用于确定输入抽象类型的示例参数 (parameter)。
3. Python 函数生成的输出的预期形状和数据类型结构。

在 JAX 的追踪阶段，它并不会实际执行 Python 回调函数。相反，它使用提供的输出形状和数据类型信息（`result_shape_dtypes`）在计算图（jaxpr）中创建一个占位符。这使得追踪和 `jit`、`vmap` 或 `grad` 等转换能够进行，将回调视为具有已知输入/输出规范的黑盒 (black box)。

在运行时（当编译后的 JAX 函数执行时），实际的 Python 回调函数会使用具体的输入值被调用。JAX 相信产生的输出会与之前指定的形状和数据类型匹配。

### 语法和用法

让我们通过一个例子来说明。假设我们有一个纯 Python 函数，它执行一个特定的计算，可能使用 JAX 中不直接可用的库，但已知其具有确定性且无副作用。

```python
import jax
import jax.numpy as jnp
import numpy as np # 使用 NumPy 作为“外部”函数

# 假设此函数代表一些复杂的纯计算
# 可能来自外部库或自定义 Python 代码。
def external_pure_python_computation(x: np.ndarray, y: float) -> np.ndarray:
  """一个纯 Python 函数的占位符。"""
  # 确保此函数的输入是 NumPy 数组
  if isinstance(x, jax.Array):
      x = np.array(x)
  # 纯计算示例：
  return np.sin(x) * y + 1.0

def jax_function_using_callback(a, b):
  """一个包含纯回调的 JAX 函数。"""

  # 定义预期的输出结构。
  # 这里，我们期望一个与 'a' 具有相同形状和数据类型的数组。
  output_shape_dtype = jax.ShapeDtypeStruct(a.shape, a.dtype)

  # 创建回调
  result = jax.pure_callback(
      external_pure_python_computation, # 纯 Python 函数
      output_shape_dtype,           # 预期的输出形状/数据类型
      a,                            # 示例/实际参数 'a'
      b                             # 示例/实际参数 'b'
      # 如果回调需要关键字参数，则传递它们
  )
  return result + a # 继续进行标准 JAX 操作

# JIT 内的用法示例
key = jax.random.PRNGKey(0)
input_array = jax.random.normal(key, (3, 3))
input_scalar = 2.0

jitted_function = jax.jit(jax_function_using_callback)

# 执行 JIT 编译的函数
output = jitted_function(input_array, input_scalar)
print("输入数组:\n", input_array)
print("\n输出数组:\n", output)

# 手动验证计算（近似回调）
expected_output = np.sin(np.array(input_array)) * input_scalar + 1.0 + np.array(input_array)
print("\n预期输出（近似）：\n", expected_output)

# 检查结果是否接近
assert np.allclose(output, expected_output)

# 也可以通过 pure_callback 进行 vmap 或 grad
vmapped_function = jax.vmap(jax_function_using_callback, in_axes=(0, None))
batched_input_array = jax.random.normal(key, (10, 3, 3))
batched_output = vmapped_function(batched_input_array, input_scalar)
print("\n批量输出的形状：", batched_output.shape)

grad_function = jax.grad(lambda x: jnp.sum(jax_function_using_callback(x, input_scalar)))
# 注意：Grad 要求回调函数对其输入是可微的，
# 而 pure_callback 本身不保证这一点。你通常会使用
# 自定义 VJP 来通过外部代码进行微分。
# 这个例子之所以有效，是因为 sin(x)*y 是可微的。
gradients = grad_function(input_array)
print("\n关于 input_array 的梯度：\n", gradients)
```

在此示例中：

1. `external_pure_python_computation` 是外部纯 Python 函数的替代表达。它接受一个 NumPy 数组和一个浮点数。
2. 在 `jax_function_using_callback` 中，我们使用 `jax.ShapeDtypeStruct` 定义 `output_shape_dtype`，以告知 JAX 预期从回调中获得的输出类型（一个与输入 `a` 具有相同形状和数据类型的数组）。
3. 调用 `jax.pure_callback`，传递 Python 函数、预期的输出结构和实际输入（`a`、`b`）。
4. 回调的结果随后在后续 JAX 操作中使用（`result + a`）。
5. 我们表明，生成的 JAX 函数可以成功地进行 `jit` 编译、`vmap` 映射，甚至使用 `grad` 进行微分（尽管微分依赖于底层数学运算的可微性，并且对于复杂情况可能需要自定义规则）。

### 指定 `result_shape_dtypes`

提供正确的 `result_shape_dtypes` 非常重要。此参数 (parameter)告知 JAX 的追踪器回调输出的抽象值（形状和数据类型），*而无需运行 Python 代码*。

- **单个输出：** 如果回调返回单个 JAX 兼容的值（如 `jax.Array`、NumPy 数组或标量），请提供 `jax.ShapeDtypeStruct(shape, dtype)`。
- **多个输出：** 如果回调返回多个值（例如，一个元组），请提供一个 `jax.ShapeDtypeStruct` 实例的元组，每个实例对应一个输出元素。
- **PyTrees：** 你可以返回嵌套结构（PyTrees），例如包含数组的元组或字典。`result_shape_dtypes` 必须镜像此结构，在叶子节点处包含 `jax.ShapeDtypeStruct` 对象。

```python
# 多个输出示例
def multi_output_pure_function(x):
    return np.sum(x), np.mean(x)

def jax_multi_output_callback(arr):
    # 指定输出结构：一个 float32 标量和另一个 float32 标量
    output_structure = (
        jax.ShapeDtypeStruct((), arr.dtype), # 标量的形状为 ()
        jax.ShapeDtypeStruct((), arr.dtype)
    )
    sum_val, mean_val = jax.pure_callback(
        multi_output_pure_function,
        output_structure,
        arr
    )
    return sum_val * 2, mean_val * 3

input_arr = jnp.arange(5.0)
res1, res2 = jax.jit(jax_multi_output_callback)(input_arr)
print(f"\n多个输出：{res1=}, {res2=}") # 输出：res1=20.0, res2=6.0
```

### `jax.pure_callback` 的使用场景

- **与外部库集成：** 调用 SciPy、专用数值包或其他提供纯计算的 Python 模块中的函数，这些计算在 JAX/XLA 中没有原生实现。
- **自定义纯 Python 逻辑：** 实现用纯 Python 更易表达的复杂算法或逻辑，同时仍能受益于 JAX 对周围代码的转换。
- **解决 JAX 运算缺失问题：** 在特定低级操作尚无直接 JAX 等效项时，提供临时解决方案，前提是存在纯 Python 实现。

### 重要注意事项

- **纯粹性是你的责任：** 这是最重要的一点。JAX *信任*你对纯粹性的声明。如果回调执行 I/O、修改全局状态或不确定，你的 JAX 程序可能会静默地产生不正确的结果或行为异常，特别是在 `jit` 或 `pmap` 等转换下。
- **性能开销：** 从编译后的 JAX 计算中调用 Python 会产生开销。这涉及加速器（GPU/TPU）与运行 Python 代码的主机 CPU 之间的上下文 (context)切换和数据传输。慎重使用 `pure_callback`，尤其是在性能关键的循环中。XLA 编译的原生 JAX 操作几乎总是快得多。
- **转换兼容性：** 虽然 `pure_callback` 与 `jit`、`vmap` 和 `pmap` 兼容，但自动微分（`grad`、`vjp`、`jvp`）要求被封装的 Python 函数*本身*以 JAX 可理解的方式是可微的，或者你需要为回调操作定义自定义微分规则（使用 `jax.custom_vjp` 或 `jax.custom_jvp`）。`pure_callback` 本身并不能使不可微函数变得可微。
- **序列化：** Python 函数，特别是闭包或依赖外部模块的函数，有时可能对序列化构成挑战，这可能与分布式计算或保存编译函数有关。

### 与 `host_callback` 的比较

| 特性 | `jax.pure_callback` | `jax.experimental.host_callback` |
| --- | --- | --- |
| **纯粹性** | **必需**（用户保证） | 非必需（允许副作用） |
| **用途** | 纯计算，库集成 | 调试，日志记录，I/O，副作用 |
| **`jit`** | 兼容 | 兼容（但在主机上执行） |
| **`vmap` / `pmap`** | 兼容 | 有限（顺序执行回调） |
| **`grad`** | 兼容（如果函数可微或提供了自定义规则） | 不直接可微 |
| **执行** | 在运行时执行 Python 函数 | 在主机上执行 Python 函数 |
| **返回值** | 将计算结果返回给 JAX | 通常不返回任何内容（`None`） |

当你需要将外部的、*纯计算型* Python 代码整合到将进行 `jit`、`vmap` 或 `grad` 等转换的 JAX 函数中时，请选择 `jax.pure_callback`。务必确保该函数严格遵守纯粹性约定。如果你需要执行打印或日志记录等副作用，`host_callback` 是合适的工具，尽管功能更受限。

## 参考资料

- [jax.pure_callback](https://jax.readthedocs.io/en/latest/_autosummary/jax.pure_callback.html) — JAX core contributors (2024)
  Publisher: Google
  提供`jax.pure_callback`的官方API规范、用法示例和契约细节。
- [jax.experimental.host_callback](https://jax.readthedocs.io/en/latest/notebooks/external_callbacks.html) — JAX core contributors (2024)
  Publisher: Google
  记录了`host_callback` API，它提供了一种在JAX计算期间执行带副作用的Python代码的对比方法。
- [JAX: Accelerated Array Programming](https://github.com/google/jax) — James Bradbury, Roy Frostig, Peter Hawkins, Matthew J. Johnson, Chris Leary, Skye Wanderman-Milne, and Adam Paszke (2018)
  Publisher: Google Research
  介绍了JAX框架，详细阐述了其函数式编程模型、自动微分和XLA编译，这些是`pure_callback`设计的基础。
- [Structure and Interpretation of Computer Programs, 2nd Edition](https://mitpress.mit.edu/sites/default/files/sicp/full-text/book/book.html) — Harold Abelson and Gerald Jay Sussman with Julie Sussman (1984)
  Publisher: MIT Press; Pages: Chapters 1 and 3
  为函数式编程范式提供基础介绍，包括对函数纯度、确定性和副作用的全面解释，这对于理解`jax.pure_callback`的契约至关重要。
