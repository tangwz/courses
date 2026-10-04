# 为后端（CPU/GPU/TPU）实现转换规则

来源：[原文](https://apxml.com/zh/courses/advanced-jax/chapter-5-jax-interoperability-custom-operations/implementing-lowering-rules)

[返回章节目录](README.md) · [返回课程目录](../README.md)

为了在 CPU、GPU 或 TPU 等特定硬件上实际执行自定义 JAX 原语，需要一个重要的翻译过程。这个过程被称为“转换”，其中高级原语被转换为硬件后端能够理解的低级指令。尽管自定义原语在形状和数据类型方面已抽象定义，但使它们能够在不同硬件后端上实际执行是一个必要的步骤，通常通过 XLA (加速线性代数) 编译器完成。

### 为什么需要转换规则

JAX 通过使用 XLA 将 Python 函数即时编译为优化的可执行代码来提高性能。XLA 在其自身的中间表示形式 HLO (高级优化器 IR) 上运行。当 JAX 在编译过程中遇到一个原语（例如 `jax.lax.add` 或您的自定义原语）时，它需要一种方法将该原语转换为相应的 HLO 指令序列。如果没有转换规则，XLA 将不知道为您的自定义操作生成什么机器代码。

每个后端（CPU、GPU、TPU）可能有不同的最佳方式来实现相同的操作。因此，您需要为打算支持的每个后端提供专门的转换规则。

### 目标：XLA HLO

JAX 中的转换过程几乎总是以 XLA HLO 为目标。您的转换规则的任务是接收原语的输入（表示为 XLA 值），并使用 XLA 构建器对象构建执行预期操作的 HLO 计算图。XLA 随后会将此 HLO 图进一步编译为目标加速器的高度优化机器代码。

### 注册转换函数

您可以使用 JAX 的内部 API 为特定原语和后端注册转换规则。一种常见的方法涉及 `xla_client.register_translation` 等函数。您将您的原语对象与执行转换的 Python 函数进行关联。

```python
# 示例：为 'my_custom_op_p' 原语注册转换规则
from jax.lib import xla_client
from jax.interpreters import xla

# 假设 'my_custom_op_p' 是您的 Primitive 对象

def my_custom_op_lowering_cpu(builder, *args, **params):
  """my_custom_op 在 CPU 上的转换规则。"""
  # 'builder' 是一个 XlaBuilder 实例
  # 'args' 是输入的 XLA 计算值
  # 'params' 是原语的静态参数

  # 使用构建器来构建 HLO 操作...
  # 示例：如果 my_custom_op 添加 1.0
  # operand = args[0]
  # one = builder.ConstantF32Scalar(1.0)
  # result = builder.Add(operand, one)
  # return [result] # 返回输出 XLA 值的列表

  # 替换为您的原语的实际 HLO 构建
  pass 

# 为 CPU 后端注册
xla_client.register_translation(my_custom_op_p, 
                                my_custom_op_lowering_cpu, 
                                platform='cpu')

# 类似地，为 'gpu' 和 'tpu' 平台定义和注册
# def my_custom_op_lowering_gpu(builder, *args, **params): ...
# xla_client.register_translation(my_custom_op_p, my_custom_op_lowering_gpu, platform='gpu') 

# def my_custom_op_lowering_tpu(builder, *args, **params): ...
# xla_client.register_translation(my_custom_op_p, my_custom_op_lowering_tpu, platform='tpu') 
```

*(注意：具体的 API 细节可能会有所变化；请查阅当前的 JAX 文档以获取最新的注册方法。)*

### 使用 XLA 构建器

传递给转换函数的 `builder` 对象是构建 HLO 的主要工具。它提供了以下方法：

- 创建常量（例如，`builder.ConstantF32Scalar(1.0)`）。
- 执行逐元素算术运算（例如，`builder.Add`、`builder.Mul`、`builder.Log1p`）。
- 矩阵操作（例如，`builder.DotGeneral`）。
- 数据操作（例如，`builder.Reshape`、`builder.Transpose`、`builder.Slice`）。
- 控制流（尽管在 JAX 中通常在更高层次处理）。
- 以及许多更专业的操作。

您使用这些方法，将它们链接在一起，接收输入 `args`（它们是 XLA 值的句柄）并生成表示您的原语结果的输出 XLA 值。您之前定义的抽象评估规则提供了构建器通常需要正确构建这些操作的形状和数据类型信息。

### 后端特定考量

尽管某些原语在不同后端上可能具有相同的转换规则，但性能要求高的操作通常受益于后端专门化：

- **CPU：** 转换可能会生成标准 HLO 操作，XLA 可以有效地为多核 CPU 进行编译，并可能使用向量 (vector)指令（如 AVX）。
- **GPU：** 转换规则可能会生成与 CUDA 内核很好匹配的 HLO。这可能涉及使用已知在 GPU 上高效的特定 HLO 操作，或者构造计算以最大化并行度并最小化内存传输。对于 HLO 中不直接可用的操作，您可能需要在转换规则中使用 `gpu_kernel` 调用，嵌入 (embedding) CUDA 代码（这会大大增加复杂性）。
- **TPU：** 转换需要考虑 TPU 的架构，特别是其强大的矩阵乘法单元 (MXU) 和向量处理单元。高效的 TPU 代码通常涉及将计算构造为大型矩阵乘法或为 TPU 硬件优化的特定 HLO 操作。

### 示例：简单 `relu` 的转换

假设我们正在定义一个自定义 ReLU 原语 `my_relu_p`。ReLU 操作是 $max(0, x)$。

```python
# my_relu_p 的转换
# 假设 my_relu_p 是 Primitive 对象
# 假设抽象评估规则已定义

def my_relu_lowering(builder, x_operand, **params):
  """将 my_relu(x) = max(0, x) 转换为 HLO。"""

  # 从操作数获取形状和数据类型
  # 抽象评估确保 x_operand 具有正确的抽象值
  shape = builder.GetShape(x_operand) 
  dtype = shape.element_type() # 例如，F32

  # 创建一个与操作数相同类型和形状的常数零
  zero = builder.Broadcast(builder.ConstantElement(0, dtype), shape.dimensions())

  # 计算逐元素最大值
  result = builder.Max(zero, x_operand)

  return [result] # 返回包含单个输出的列表

# 为相关平台注册此规则
# xla_client.register_translation(my_relu_p, my_relu_lowering, platform='cpu')
# xla_client.register_translation(my_relu_p, my_relu_lowering, platform='gpu')
# xla_client.register_translation(my_relu_p, my_relu_lowering, platform='tpu') 
```

在此示例中，转换很简单：创建一个与输入形状和类型相同的零张量，并使用 `Max` HLO 操作。这条规则可能对所有后端都足够，因为 XLA 知道如何为每个后端高效地编译 `Max`。

### 复杂性和高级用法

编写转换规则可以从简单（如 ReLU 示例）到高度复杂。如果您的操作不能很好地映射到现有的 HLO 指令，您可能需要：

1. **组合多个 HLO 操作：** 从现有的 HLO 原语构建您的功能。
2. **实现后端特定内核：** 对于 GPU，编写自定义 CUDA 代码并通过 HLO 的 `CustomCall` 调用它。这需要 HLO 和目标硬件编程模型（例如 CUDA）方面的专业知识。
3. **理解 XLA 优化：** 注意 XLA 如何融合或优化您生成的 HLO。

调试转换规则也可能具有挑战性，通常需要使用 JAX 或 XLA 提供的工具检查生成的 HLO 本身。

通过实现转换规则，您提供了 JAX 在所需硬件上高效编译和运行您的自定义原语所需的最后一部分，使其成为 JAX 生态系统中的一等公民，可以用于 `jit`、`vmap`、`pmap`，甚至进行微分（一旦您定义了其微分规则）。

## 参考资料

- [How to write JAX XLA Lowering](https://jax.readthedocs.io/en/latest/aot.html) — JAX developers (2024)
  Publisher: JAX documentation
  解释如何为自定义 JAX 原语定义降低规则，涵盖 XLA HLO 和 `xla_client.register_translation` 等 API 的使用。
- [XLA: Accelerated Linear Algebra](https://www.tensorflow.org/xla) — Google (2024)
  提供了 XLA 编译器的概述、其功能，并链接到有关其中间表示 (HLO) 和硬件目标的信息。
- [A Domain-Specific Compiler for Tensor Processing Units](https://dl.acm.org/doi/10.1145/3445814.3446700) — Heejin Jo, Andrew Siena, Mark Weiser, Nicholas P. Johnson, Robert S. French, Paul M. Smith, Kevin S. Lee, Cliff L. Biffle, Eric S. Chung, and Norman P. Jouppi (2021)
  Journal: Proceedings of the 26th ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS '21); Publisher: ACM; Pages: 385-399; DOI: [10.1145/3445814.3446700](https://doi.org/10.1145/3445814.3446700)
  研究了 XLA 编译器如何针对 Google 的张量处理单元 (TPU) 进行优化，提供了针对专用硬件的后端特定编译器策略的示例。

---

[上一节](07-%E5%AE%9E%E7%8E%B0%E6%8A%BD%E8%B1%A1%E6%B1%82%E5%80%BC%E8%A7%84%E5%88%99.md) · [下一节](09-%E5%88%B6%E5%AE%9A%E8%87%AA%E5%AE%9A%E4%B9%89%E5%8E%9F%E8%AF%AD%E7%9A%84%E6%B1%82%E5%AF%BC%E8%A7%84%E5%88%99.md)
