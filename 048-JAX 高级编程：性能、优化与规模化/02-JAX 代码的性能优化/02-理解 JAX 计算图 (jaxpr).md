# 理解 JAX 计算图 (jaxpr)

来源：[原文](https://apxml.com/zh/courses/advanced-jax/chapter-2-optimizing-jax-code-performance/understanding-jaxpr)

[返回章节目录](README.md) · [返回课程目录](../README.md)

虽然 `jax.jit` 能为加速器编译您的 Python 函数，但有时需要仔细研究其内部运作，以理解 JAX 究竟在编译什么。调试性能问题或异常行为时，这会特别有用。JAX 使用一种名为 **jaxpr**（JAX 程序表示）的中间表示来表示从您的 Python 代码派生出的计算图，然后将其传递给 XLA 编译器。理解 jaxpr 有助于识别 JAX 如何追踪函数，并能定位效率低下或重新编译的起因。

### 什么是 Jaxpr？

您可以将 jaxpr 视为您计算的简化、函数式和明确的表示。当 JAX 追踪您的 Python 函数时（这发生在您首次使用特定输入类型和形状调用 `jit` 编译的函数时），它不会直接执行 Python 代码。相反，它会用特殊的追踪器对象替换您的输入，并记录在这些追踪器上执行的每个 JAX 原语操作。此追踪过程的结果就是 jaxpr。

jaxpr 的特点包括：

1. **函数式：** Jaxpr 是纯函数。它们仅根据输入定义输出，没有副作用。
2. **静态类型：** jaxpr 中的所有变量都具有明确定义的形状和数据类型 (dtypes)。
3. **明确操作：** Jaxpr 由一系列等式组成，每个等式都将特定的 JAX 原语（如 `add`、`sin`、`dot_general`、`reduce_sum`）应用于输入变量或常量，以生成输出变量。Python 控制流，例如 `if` 语句或 `for` 循环（除非使用 JAX 控制流原语，如 `lax.cond` 或 `lax.scan`），在追踪过程中会被展开。
4. **中间表示 (IR)：** 它位于您的高级 Python 代码和用于编译器优化的低级 XLA HLO（高级操作）表示之间。

### 生成和检查 Jaxpr

在典型的模型开发过程中，您通常不会直接与 jaxpr 交互，但 JAX 提供了在需要时检查它们的工具。主要用于此的函数是 `jax.make_jaxpr`。它接收一个函数和示例输入（如 `jit`），并返回一个表示 jaxpr 的对象，以及其他信息。

我们来看一个简单例子：

```python
import jax
import jax.numpy as jnp

def example_function(x, y):
  a = jnp.sin(x)
  b = jnp.cos(y)
  return a + b * 2.0

# Define example inputs
key = jax.random.PRNGKey(0)
x_example = jax.random.normal(key, (10,))
y_example = jax.random.normal(key, (10,))

# Generate the jaxpr
jaxpr_obj = jax.make_jaxpr(example_function)(x_example, y_example)
print(jaxpr_obj.jaxpr)
```

运行此代码将打印 jaxpr 对象。其结构可能类似于这样（细节可能因 JAX 版本而异）：

```
{ lambda ; a:f32[10] b:f32[10]. let
    c:f32[10] = sin a
    d:f32[10] = cos b
    e:f32[10] = mul d 2.0
    f:f32[10] = add c e
  in (f,) }
```

我们来分解一下这个打印出的表示：

- `{ lambda ; ... }`：将 jaxpr 定义为一个 lambda 函数。
- `a:f32[10] b:f32[10]`：这些是输入变量 (`invars`) 及其类型 (float32) 和形状 (`[10]`)。它们对应于 `example_function` 的 `x` 和 `y` 参数 (parameter)。
- `let ... in ...`：这表示 jaxpr 的主体部分。
- `c:f32[10] = sin a`：这是一个等式 (`eqn`)。它将 `sin` 原语应用于输入变量 `a`，并将结果绑定到一个新的中间变量 `c`，其类型也是 `f32[10]`。
- `d:f32[10] = cos b`：另一个等式，将 `cos` 原语应用于 `b`，得到 `d`。
- `e:f32[10] = mul d 2.0`：将乘法原语 (`mul`) 应用于变量 `d` 和常量 `2.0`。从函数环境 (`constvars`) 捕获的常量也在此处出现。
- `f:f32[10] = add c e`：将加法原语 (`add`) 应用于中间变量 `c` 和 `e`，生成 `f`。
- `in (f,)`：指定 jaxpr 的输出变量 (`outvars`)，在此例中，仅为 `f`。

请注意，Python 代码的结构是如何被转换为作用于类型化变量的线性原语操作序列的。这种明确、简化的形式比原始 Python 代码更容易让编译器进行分析和优化。

### 为何 Jaxpr 对优化很重要

理解 jaxpr 在优化 JAX 代码时有多种帮助：

1. **追踪可见性：** Jaxpr 向您展现 JAX 从您的 Python 代码中导出的操作序列。这可以表明 JAX 是否追踪了超出您预期的计算量，或者是否以不同于您预期的方式使用了原语。例如，NumPy 与 `jax.numpy` 使用上的细微差别可能会导致不同的原语。
2. **理解转换：** 您可以在应用 `jax.grad` 或 `jax.vmap` 等转换后检查 jaxpr，以查看它们如何修改计算图。这对于调试求导或向量 (vector)化的行为很有用。
3. **诊断重新编译：** 如果 jaxpr 结构在函数调用之间发生变化，`jax.jit` 会重新编译您的函数。这通常发生在操作序列依赖于参数 (parameter)的*值*，而不仅仅是它们的形状和类型时（这是基于张量值的 Python 级控制流的常见问题）。比较不同调用生成的 jaxpr 可以突出显示重新编译的原因。如果 `eqns` 列表差异明显，则表明存在 JAX 必须重新追踪的动态行为。
4. **连接 XLA：** Jaxpr 是 JAX 提供给 XLA 编译器的表示。虽然 XLA 执行其自身的复杂优化（如操作符融合，稍后讨论），但输入 jaxpr 的结构会影响 XLA *能够*优化的内容。一个清晰、可预测的 jaxpr 通常比由复杂、展开的 Python 逻辑产生的 jaxpr 更容易被 XLA 有效优化。

### 带控制流的例子

考虑一个使用 `jax.lax.cond` 的函数，这是 JAX 用于 `jit` 编译代码中条件执行的原语：

```python
import jax
import jax.numpy as jnp
from jax import lax

def conditional_function(use_sin, x):
  # 注意：lax.cond 的 `pred` 必须是标量布尔值
  pred = use_sin > 0.5

  # 定义真分支和假分支的函数
  def true_fun(operand):
      return jnp.sin(operand)

  def false_fun(operand):
      return jnp.cos(operand)

  # 使用 lax.cond
  return lax.cond(pred, true_fun, false_fun, x)

# 示例输入
x_example = jnp.ones((3,))
pred_true_example = jnp.array(0.7) # 标量值 > 0.5
pred_false_example = jnp.array(0.3) # 标量值 <= 0.5

# 当 pred 为 True 时的 Jaxpr
jaxpr_true = jax.make_jaxpr(conditional_function)(pred_true_example, x_example)
print("Jaxpr (可能执行真分支):")
print(jaxpr_true.jaxpr)

# 当 pred 为 False 时的 Jaxpr
jaxpr_false = jax.make_jaxpr(conditional_function)(pred_false_example, x_example)
print("\nJaxpr (可能执行假分支):")
print(jaxpr_false.jaxpr)
```

您会注意到生成的 jaxpr 包含一个 `cond` 原语。重要的是，即使我们提供了具体的布尔值（`0.7` 对应 `True`，`0.3` 对应 `False`），jaxpr 本身也*不*仅仅包含一个分支的操作。相反，它包含了 `cond` 原语，该原语封装了*两个*分支的逻辑。无论追踪过程中谓词的具体值如何，只要两个分支的输入和输出的*类型*和*形状*一致，jaxpr 看起来都会相似。这对于 `jit` 编译非常重要，因为编译后的代码需要在运行时处理任一分支。

```
# Jaxpr 的简化表示（结构可能不同）
{ lambda ; a:f32[] b:f32[3]. let
    c:bool[] = gt a 0.5
    # 真分支 jaxpr 的定义（例如，{ lambda ; x:f32[3]. let y = sin x in (y,) }）
    # 假分支 jaxpr 的定义（例如，{ lambda ; x:f32[3]. let y = cos x in (y,) }）
    d:f32[3] = cond c true_branch false_branch b # 传递给 cond 的操作数
  in (d,) }
```

检查 jaxpr 会显示 `cond` 原语和传递给 XLA 的结构，这证实了 JAX 使用其特定原语正确追踪了条件逻辑。

### 局限性

尽管 jaxpr 提供了很多信息，但它仍是一个中间步骤。它不显示 XLA 执行的最终优化，例如操作符融合（将多个原语合并为一个更高效的内核）或为目标加速器生成的精确低级代码。然而，它提供了对编译器输入的重要视图，使其成为调试性能和了解 JAX 内部机制的必备工具，尤其是在简单应用 `@jit` 不足以解决问题时。

通过熟悉 jaxpr，您能更清楚地理解 JAX 如何将您的 Python 代码转换为适合高性能编译的形式，使您能够编写更高效、更可预测的 JAX 程序。

## 参考资料

- [XLA: Optimizing Compiler for Machine Learning](https://www.tensorflow.org/xla) — Google Developers (2024)
  介绍XLA，它是JAX的后端编译器，接收jaxpr并进行硬件特定的优化。
- [Control Flow in JAX (from JAX documentation)](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#control-flow) — JAX developers (2024)
  解释JAX在追踪过程中如何处理Python控制流，并介绍JAX特有的控制流原语如`lax.cond`，展示它们在jaxpr中的表示。

---

[上一节](01-%E5%9C%A8%20CPU%E3%80%81GPU%20%E5%92%8C%20TPU%20%E4%B8%8A%E5%88%86%E6%9E%90%20JAX%20%E4%BB%A3%E7%A0%81.md) · [下一节](03-XLA%20%E7%BC%96%E8%AF%91%E7%9A%84%E4%BD%9C%E7%94%A8.md)
