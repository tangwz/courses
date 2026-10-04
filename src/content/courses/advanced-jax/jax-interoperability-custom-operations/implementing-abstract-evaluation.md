---
course: "advanced-jax"
chapter: "jax-interoperability-custom-operations"
lesson: "implementing-abstract-evaluation"
sourceId: 2914
sourceUrl: "https://apxml.com/zh/courses/advanced-jax/chapter-5-jax-interoperability-custom-operations/implementing-abstract-evaluation"
title: "实现抽象求值规则"
description: "定义自定义原语在JAX追踪中如何根据形状和dtype来行为。"
order: 7
plots: []
sourceHash: "d69b02104236508efa28b13c6d845169b9e238f694bc2a90fb0886a000051fbf"
sourceCorrections: []
---

JAX原语是JAX系统所知的基础构造块。在定义您自己的自定义操作时，主要的一步是让JAX知晓其*形状和类型特征*。这通过实现**抽象求值规则**来完成。

抽象求值是JAX追踪机制的一个重要组成部分。在JAX能够对包含您自定义原语的函数进行JIT编译、自动微分或向量 (vector)化之前，它需要确定该原语产生的输出的属性（如形状和数据类型，即`dtype`），*而无需实际在真实数据上运行计算*。它通过操作抽象值来执行此分析，主要是`jax.core.ShapedArray`实例，这些实例封装了形状和`dtype`信息，但不持有实际的数值。

### 抽象求值的目的

可以将抽象求值理解为在类型和形状层面定义您的原语的函数签名。当JAX在追踪期间遇到您的原语时（例如，当`jax.jit`应用于使用它的函数时），它会调用该原语的抽象求值规则。此规则接收与原语输入对应的抽象值以及任何必要的元数据（原语行为的特定参数 (parameter)），并计算原语输出的抽象值（形状和`dtype`）。

此信息之所以重要，有以下几个原因：

1. **静态分析：** 这使得JAX能够在追踪阶段，在尝试编译或执行之前，及早检查形状不匹配或类型错误。
2. **变换兼容性：** `vmap`等变换需要知晓形状如何变化才能正确实现批处理规则。`grad`需要知晓形状和类型才能正确设置反向传播 (backpropagation)。
3. **编译：** XLA编译器需要计算图中所有操作的精确形状和类型信息，以便为目标硬件（CPU、GPU、TPU）生成高效、专门的代码。

### 实现`abstract_eval`方法

为了定义自定义原语的抽象求值规则，您通常需要继承`jax.core.Primitive`并实现其`abstract_eval`方法。

```python
import jax
import jax.numpy as jnp
from jax.core import Primitive, ShapedArray
from jax.interpreters import xla

# 示例：一个将输入加1.0的原语
custom_add_one_p = Primitive('custom_add_one')

@custom_add_one_p.def_abstract_eval
def custom_add_one_abstract_eval(*avals, **params):
  # avals 包含输入的抽象值（例如 ShapedArray）
  # params 包含在原语绑定期间传递的任何参数

  # 基本验证（可选但推荐）
  if len(avals) != 1:
    raise ValueError("custom_add_one 需要恰好一个输入操作数。")

  input_aval = avals[0]

  # 检查输入是否为 ShapedArray（最常见的情况）
  if not isinstance(input_aval, ShapedArray):
      raise TypeError(f"输入必须是 ShapedArray，得到 {type(input_aval)}")

  # 确保此特定操作的 dtype 是浮点类型
  if not jnp.issubdtype(input_aval.dtype, jnp.floating):
      # 或者您可能定义特定的类型提升规则
      raise TypeError(f"输入 dtype 必须是浮点类型，得到 {input_aval.dtype}")

  # 核心逻辑：确定输出形状和 dtype
  # 对于 'custom_add_one'，形状和 dtype 与输入相同。
  output_shape = input_aval.shape
  output_dtype = input_aval.dtype

  # 返回输出的抽象值
  return ShapedArray(output_shape, output_dtype)

# 示例用法（说明性，如果没有实现规则则无法运行）
# def my_func(x):
#   # 将原语与输入 x 绑定
#   return custom_add_one_p.bind(x) 
```

在此示例中：

1. 我们定义了一个原语`custom_add_one_p`。
2. 我们使用`@custom_add_one_p.def_abstract_eval`装饰器修饰函数`custom_add_one_abstract_eval`。这将该函数注册为我们原语的抽象求值规则。
3. 该函数接收传递给原语`bind`方法的每个位置参数 (parameter)的抽象值（`avals`），以及传递给`bind`的任何关键字参数（`params`）。
4. 我们对输入数量和类型执行基本验证。对于数组操作，检查`isinstance(aval, ShapedArray)`很常见。我们还检查了`dtype`。
5. 主要任务是计算`output_shape`和`output_dtype`。对于我们简单的`custom_add_one`，它们与输入的形状和`dtype`保持一致。更复杂的原语在此处会有逻辑，以体现它们如何变换形状（例如矩阵乘法、卷积）。
6. 最后，它返回一个`ShapedArray`，表示原语输出的抽象属性。如果一个原语返回多个输出，该函数将返回一个`ShapedArray`实例的元组。

### 处理参数 (parameter)（`params`）

有时，原语的行为以及其输出形状或类型，取决于并非输入数组本身的参数。例如，卷积原语需要步长和填充信息，或者归约原语需要归约的轴。这些参数通常作为关键字参数传递给原语的`bind`方法，并在`abstract_eval`函数内部的`params`字典中接收。

```python
# 归约原语
# custom_reduce_sum_p = Primitive('custom_reduce_sum')

# @custom_reduce_sum_p.def_abstract_eval
# def custom_reduce_sum_abstract_eval(aval, *, axis): # axis 作为关键字参数传递给 bind
#   if not isinstance(aval, ShapedArray):
#      raise TypeError("输入必须是数组")
#
#   # 根据输入形状和轴参数计算输出形状
#   output_shape = tuple(d for i, d in enumerate(aval.shape) if i not in axis)
#   output_dtype = aval.dtype # 假设 dtype 不变
#
#   return ShapedArray(output_shape, output_dtype)

# 用法：
# result = custom_reduce_sum_p.bind(my_array, axis=(0,))
```

在这里，`axis`参数影响`output_shape`的计算。`abstract_eval`规则直接从`params`字典（或者如上所示的关键字参数，JAX对此处理得很好）中使用此参数来确定正确的输出形状。

### 在原语定义工作流中的位置

抽象求值是您需要为新原语定义的第一个规则。它为JAX理解原语的签名奠定了基础。只有在定义了`abstract_eval`之后，您才能继续实现实际的计算逻辑（针对CPU/GPU/TPU等特定后端的“降级”规则）及其微分规则（JVP/VJP），这些都依赖于抽象求值提供的形状和类型信息。

> 抽象求值规则（`abstract_eval`）在JAX追踪期间被调用，以确定原语输出的形状和类型，从而能够创建`jaxpr`（JAX的中间表示），`jaxpr`随后用于编译、通过降级规则执行以及定义微分行为。

通过仔细实现`abstract_eval`方法，您可以确保您的自定义原语与JAX的追踪和变换机制结合，在JIT编译、向量 (vector)化和自动微分等不同环境中，在形状和数据类型方面表现出可预测性。这一步对于在JAX中创建自定义操作来说是基础。

## 参考资料

- [How JAX primitives work](https://jax.readthedocs.io/en/latest/notebooks/How_JAX_primitives_work.html) — JAX core developers (2024)
  Publisher: Google LLC
  这份官方指南解释了JAX原语的核心概念，包括抽象求值、ShapedArray以及定义自定义操作的过程。
- [Accelerated Linear Algebra (XLA) Overview](https://www.tensorflow.org/xla) — Google Developers (2024)
  Publisher: Google Developers
  提供了XLA（JAX使用的线性代数编译器）的概述，解释了其在优化计算中的作用以及静态形状和类型信息的重要性。
