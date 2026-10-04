# 使用 lax.cond 进行条件执行

来源：[原文](https://apxml.com/zh/courses/advanced-jax/chapter-1-advanced-jax-transformations-control-flow/conditional-execution-lax-cond)

[返回章节目录](README.md) · [返回课程目录](../README.md)

标准 Python 的 `if/else` 语句给 JAX 的编译过程带来困难。当 JAX 使用 `jax.jit` 追踪一个函数时，它会将 Python 代码转换为中间表示 (jaxpr)。标准的 Python `if` 语句在*追踪阶段*只会执行一个分支，这是基于当时可用的*Python 层面*的值。如果条件依赖于一个追踪变量（JAX 数组）的*值*，这种方法就会失效，因为实际值在运行时才知道。JAX 需要一种方法来在计算图*内部*表示条件逻辑。

这就是 `jax.lax.cond` 的作用所在。它提供了函数式、可追踪的条件执行，让你可以在 `jit` 编译的代码中根据运行时值来选择计算路径。

### `lax.cond` 原语

`lax.cond` 函数具有以下签名：

```python
jax.lax.cond(pred, true_fun, false_fun, *operands)
```

或者，当传递单个操作数（可能是 pytree）时，更常用的是：

```python
jax.lax.cond(pred, true_fun, false_fun, operand)
```

我们来分析其组成部分：

1. **`pred`**: 这是一个标量布尔类型的 JAX 数组（即形状为 `()` 且数据类型为 `bool` 的数组）。`pred` 在运行时决定执行哪个分支。如果其值依赖于追踪输入，它*必须*是一个 JAX 数组值，而不能是普通的 Python 布尔值。
2. **`true_fun`**: 一个 Python 可调用对象（如函数或 lambda 表达式），当 `pred` 为 `True` 时执行。它以 `operand`（或 `*operands`）作为输入。
3. **`false_fun`**: 一个 Python 可调用对象，当 `pred` 为 `False` 时执行。它也以 `operand`（或 `*operands`）作为输入。
4. **`operand` (或 `*operands`)**: 传递给 `true_fun` 或 `false_fun` 的输入值。这可以是一个 JAX 数组、多个数组，或者是一个包含 JAX 数组的 pytree（如元组或字典）。

`lax.cond` 的一个重要要求是 `true_fun` 和 `false_fun` 必须作用于相同类型、形状和数据类型的操作数，并且它们返回的输出也必须具有*完全相同的结构*（类型、形状和数据类型）。这种结构上的一致性是必需的，因为 JAX 在追踪阶段（编译时）就决定了输出的形状和数据类型，而此时 `pred` 的实际值尚未可知。

### `lax.cond` 的底层工作原理

与 Python 的 `if` 不同，`lax.cond` 在追踪阶段不会只执行一个分支。相反，JAX 会追踪*两个* `true_fun` 和 `false_fun`，以确保它们是有效的计算，并确定输出结构。然而，生成的编译代码会包含逻辑（通常通过加速器上的专用指令实现），以便在运行时评估 `pred` 并只执行所选的分支。

> `lax.cond` 在运行时评估条件 `pred`。根据结果，它将 `操作数` 路由到 `真分支函数` 或 `假分支函数` 并执行该函数，生成一个在两个分支中结构一致的 `结果`。

### 示例：条件处理

我们来看一个函数，它根据数组 `x` 的元素和是否为正来决定对其进行平方或立方运算。

首先，尝试在 `jit` 编译的函数中使用标准 Python `if` 语句：

```python
import jax
import jax.numpy as jnp

def python_conditional_process(x):
  if jnp.sum(x) > 0: # 条件依赖于 x 的值
    print("执行 Python 'if' 分支 (追踪中)")
    return x * x
  else:
    print("执行 Python 'else' 分支 (追踪中)")
    return x * x * x

# 尝试 JIT 编译
jitted_python_conditional = jax.jit(python_conditional_process)

data_pos = jnp.array([1., 2., 3.])
data_neg = jnp.array([-1., -2., -3.])

# 这很可能会引发 ConcretizationTypeError 错误，或者只追踪一个分支
try:
  print("使用正数数据运行：")
  result_pos = jitted_python_conditional(data_pos)
  print("结果：", result_pos)

  # 这可能会触发重新编译，或使用第一次调用时的追踪结果
  print("\n使用负数数据运行：")
  result_neg = jitted_python_conditional(data_neg)
  print("结果：", result_neg)
except Exception as e:
  print("\n错误：", e)
```

你很可能会遇到 `ConcretizationTypeError`，因为 `jnp.sum(x) > 0` 的布尔结果在*追踪阶段*就需要来决定 Python `if` 的控制流，但 JAX 在此阶段将 `x` 视为抽象的追踪器对象。JAX 无法基于抽象值来确定具体的执行分支。

现在，我们使用 `lax.cond` 正确实现它：

```python
import jax
import jax.numpy as jnp
import jax.lax as lax

def lax_conditional_process(x):
  # 定义两个分支的函数
  # 它们必须接受相同的输入结构 (x)
  # 并返回相同的输出结构（一个与 x 具有相同形状/数据类型的数组）
  def true_branch(operand):
    print("追踪 true_branch (平方)")
    return operand * operand

  def false_branch(operand):
    print("追踪 false_branch (立方)")
    return operand * operand * operand

  # 条件必须是标量布尔类型的 JAX 数组
  pred = jnp.sum(x) > 0

  # 应用 lax.cond
  return lax.cond(pred, true_branch, false_branch, x)

# JIT 编译函数
jitted_lax_conditional = jax.jit(lax_conditional_process)

data_pos = jnp.array([1., 2., 3.])
data_neg = jnp.array([-1., -2., -3.])

# 第一次运行：追踪两个分支，编译，然后执行真分支
print("使用正数数据运行（第一次调用）：")
result_pos = jitted_lax_conditional(data_pos)
# 阻塞直到完成，以查看执行中的打印语句（通常会被优化掉）
result_pos.block_until_ready()
print("结果：", result_pos)
# 预期结果：两个分支的追踪信息，然后是结果 [1., 4., 9.]

print("\n使用负数数据运行（缓存调用）：")
# 第二次运行：使用缓存的编译结果，执行假分支
result_neg = jitted_lax_conditional(data_neg)
result_neg.block_until_ready()
print("结果：", result_neg)
# 预期结果：没有新的追踪信息，然后是结果 [-1., -8., -27.]
```

注意，在第一次执行（这会触发编译）期间，`true_branch` 和 `false_branch` 内的打印语句都会执行。这证实了 JAX 会追踪两条路径来构建完整的计算图。随后的调用即使使用不同的数据（但输入形状/数据类型相同），也会重用已编译的代码，在运行时高效地执行必要的那个分支，没有 Python 开销或重新追踪。

### 与其他转换的配合

`lax.cond` 设计用于与其他 JAX 转换顺畅配合：

- **`jax.vmap`**: 你可以对包含 `lax.cond` 的函数进行矢量化 (quantization)。如果你对操作数数组和相应的条件数组进行映射，`vmap` 将在批处理维度上有效地对每个元素应用 `lax.cond`。批处理中的每个项目将根据其对应的条件值独立选择适当的分支（`true_fun` 或 `false_fun`）。
- **`jax.grad`**: 自动微分可以通过 `lax.cond` 工作。梯度计算将对应于正向传播时在运行时实际执行的分支。如果 `pred` 本身依赖于可微分变量，梯度也会流经该计算。请注意，如果两个分支具有非常不同的数学性质，这可能会影响优化稳定性，但微分机制本身会处理条件结构。

### 与 `jnp.where` 的比较

区分 `lax.cond` 和 `jnp.where` 是很重要的。

- `jnp.where(condition, x, y)`: 此函数按元素操作。它要求 `condition`、`x` 和 `y` 可以广播到相同的形状。它会完整地评估 `x` 和 `y`，然后从 `x` 中选择 `condition` 为真的元素，从 `y` 中选择 `condition` 为假的元素来构建输出数组。
- `lax.cond(pred, true_fun, false_fun, operand)`: 此函数根据单个标量布尔值 `pred` 来选择执行哪个*计算*（`true_fun` 或 `false_fun`）。它在运行时只执行其中一个函数。

当你需要根据一个标量条件在根本不同的计算路径之间进行选择时，使用 `lax.cond`。当你需要根据布尔掩码按元素选择值时，使用 `jnp.where`。

### 性能与限制

- **编译开销**: 由于两个分支都会被追踪和编译，因此会产生相应的编译时成本。然而，运行时成本只涉及执行所选的分支。
- **标量条件**: `lax.cond` 需要一个标量条件。如果你需要基于多个布尔值（例如，按元素条件选择不同操作）的条件逻辑，你可能需要将 `lax.cond` 与 `vmap` 结合使用，或使用 `jnp.where`，或考虑调整你的逻辑结构。
- **结构一致性**: `lax.cond` 对分支之间输入/输出结构匹配的严格要求是使用时最常见的错误源。需要仔细进行函数设计。

`lax.cond` 是一个必不可少的工具，用于在 JAX 的高性能、编译环境中实现具有数据依赖控制流的算法，例如某些优化程序、强化学习 (reinforcement learning)策略或具有条件计算层的模型。

## 参考资料

- [\`jax.lax.cond\` JAX documentation](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.cond.html) — JAX developers (2024)
  Publisher: JAX Project
  提供`jax.lax.cond`的官方API规范和使用细节。
- [Thinking in JAX: Python Control Flow vs. JAX Control Flow](https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html#python-control-flow-vs.-jax-control-flow) — JAX developers (2024)
  Publisher: JAX Project
  讨论Python原生控制流与JAX函数式控制流的关键区别，直接解决`lax.cond`所处理的问题。

---

[上一节](02-%E7%B2%BE%E9%80%9A%20lax.scan%20%E5%A4%84%E7%90%86%E5%BA%8F%E5%88%97%E6%93%8D%E4%BD%9C.md) · [下一节](04-%E4%BD%BF%E7%94%A8%20lax.while_loop%20%E8%BF%9B%E8%A1%8C%E5%BE%AA%E7%8E%AF.md)
