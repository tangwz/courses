---
course: "advanced-tensorflow"
chapter: "tensorflow-execution-graphs"
lesson: "tf-function-autograph"
sourceId: 2794
sourceUrl: "https://apxml.com/zh/courses/advanced-tensorflow/chapter-1-tensorflow-execution-graphs/tf-function-autograph"
title: "理解 tf.function 和 AutoGraph"
description: "学习 tf.function 如何使用 AutoGraph 将 Python 代码转换为 TensorFlow 图，以获得性能提升。"
order: 2
plots: []
sourceHash: "d792747ff5f489132f154b1338e2bb496aed47973b8030b72a3d9d40a05196d8"
sourceCorrections: []
---

虽然即时执行（eager execution）提供了与标准Python相似的灵活性和调试便利性，但通常会带来性能上的代价。对于像训练深度神经网络 (neural network)这样计算量大的任务，通过Python解释器逐个执行操作的开销可能会成为一个重要的性能瓶颈。TensorFlow 提供了一种有效方法，连接Python的易用性与静态计算图的性能优势：`tf.function`。

`tf.function` 通常作为装饰器（`@tf.function`）应用于Python函数。它的主要作用是将此Python函数转换为可调用的TensorFlow图。此过程使TensorFlow能够执行图级优化，并更高效地执行计算，特别是在GPU和TPU等硬件加速器上。

### 追踪过程

当您首次调用用 `@tf.function` 装饰的函数时（或者使用新的输入类型或形状时），TensorFlow 会执行一个称为“追踪”（tracing）的过程。在追踪期间：

1. **执行与记录：** TensorFlow 会*一次性*执行函数内的Python代码。当Python代码运行时，TensorFlow 会记录所执行的TensorFlow操作序列（例如，`tf.matmul`、`tf.add`、`tf.nn.relu`）。
2. **图构建：** 它使用这个记录的操作序列来构建一个静态计算图（`tf.Graph`），此图代表函数的逻辑。这个图捕获了操作之间的数据流和依赖关系。
3. **缓存：** 生成的图（具体而言，是一个 `tf.ConcreteFunction`）会被缓存起来，以输入参数 (parameter)的特征（数据类型和形状，即输入签名）作为键。

随后对已装饰函数使用匹配先前缓存签名的参数进行调用时，将完全跳过Python执行。相反，TensorFlow 直接执行相应的预编译图，从而通过减少Python开销和实现图优化，带来显著的性能提升。

```python
import tensorflow as tf

# 定义一个简单的Python函数
def simple_computation(x, y):
  print(f"Tracing with inputs: {x}, {y}") # 这只会在追踪时打印
  a = tf.add(x, y)
  b = tf.multiply(a, 2)
  return b

# 使用 tf.function 装饰函数
@tf.function
def optimized_computation(x, y):
  print(f"Tracing optimized function with inputs: {x}, {y}") # 这也只会在追踪时打印
  a = tf.add(x, y)
  b = tf.multiply(a, 2)
  return b

# 即时执行（每次调用都运行Python）
print("Eager Execution:")
result1_eager = simple_computation(tf.constant(1), tf.constant(2))
print(result1_eager)
result2_eager = simple_computation(tf.constant(3), tf.constant(4))
print(result2_eager)

print("\n图执行 (tf.function):")
# 首次调用：追踪函数并构建图
result1_graph = optimized_computation(tf.constant(1), tf.constant(2))
print(result1_graph)

# 第二次调用（相同的输入类型/形状）：重用缓存的图
result2_graph = optimized_computation(tf.constant(3), tf.constant(4))
print(result2_graph)

# 第三次调用（不同的输入类型 - float32）：触发重新追踪
result3_graph = optimized_computation(tf.constant(1.0), tf.constant(2.0))
print(result3_graph)

# 第四次调用（与第三次相同）：重用 float32 图
result4_graph = optimized_computation(tf.constant(3.0), tf.constant(4.0))
print(result4_graph)
```

请注意，在装饰过的函数 `optimized_computation` 内部的 `print` 语句只在追踪发生时执行（`int32` 张量的首次调用和 `float32` 张量的首次调用），而在普通的Python函数 `simple_computation` 中，它们在每次调用时都会执行。

### AutoGraph：转换Python控制流

`tf.function` 如何处理图中 `if`、`for` 和 `while` 循环等Python控制流结构？这就是 **AutoGraph** (`tf.autograph`) 的作用。AutoGraph 是 `tf.function` 内部使用的一个子模块，用于自动将包含这些结构的Python代码重写为等效的TensorFlow图操作。

例如：

- 依赖于 `Tensor` 值的Python `if`/`else` 语句会被转换为 `tf.cond`。
- 依赖于 `Tensor` 条件的Python `while` 循环会变为 `tf.while_loop`。
- 迭代 `Tensor` 的Python `for` 循环可以转换为 `tf.while_loop`，或者在迭代Python列表/元组时可能被展开。

考虑这个函数：

```python
@tf.function
def conditional_function(x):
  if tf.reduce_sum(x) > 0:
    # This branch uses tf.abs
    return tf.abs(x)
  else:
    # This branch uses tf.square
    return tf.square(x)

# 正和调用
print(conditional_function(tf.constant([1, 2, -1])))
# 负和调用
print(conditional_function(tf.constant([-1, -2, 1])))
```

AutoGraph 分析 `if tf.reduce_sum(x) > 0:` 条件。因为该条件依赖于 `Tensor` 的值（这只在图执行时才已知），AutoGraph 将 `if`/`else` 块转换为 `tf.cond` 操作。此操作确保在运行时根据输入 `x` 在图中执行正确的条件分支（`tf.abs` 或 `tf.square`）。

> `@tf.function` 如何在首次调用（追踪）时，使用 AutoGraph 将带有控制流的 Python 代码转换为优化的 TensorFlow 图，并在后续调用中重用该图的流程示意图。

**AutoGraph 的重要考量：**

- **支持的Python子集：** AutoGraph 可以转换Python的很大一部分子集，但并非全部。非常动态的特性、修改外部状态的复杂列表推导式，或者生成非Tensor对象的生成器可能无法正确转换。
- **副作用：** Python 副作用（例如打印到控制台、向列表添加元素、修改外部对象）在 `@tf.function` 内部只在*追踪期间*发生。它们不是图本身的一部分，并且在重用图的后续调用中不会执行。在图执行内部打印 Tensor 值时请使用 `tf.print`，并在适当情况下使用 `tf.Variable` 管理状态。
- **调试：** 调试 AutoGraph 转换有时会比较棘手。`tf.autograph.to_code` 等工具可以显示生成的代码，而使用 `tf.config.run_functions_eagerly(True)` 可以暂时禁用 `tf.function` 行为，以便于单步调试。

### 管理追踪和多态

因为追踪取决于输入签名（Tensor 参数 (parameter)的数据类型和形状，Python 参数的类型），`tf.function` 可以为同一个Python函数创建多个图。这称为**多态**。尽管这很灵活，但过度的重新追踪会抵消性能提升。

以下情况会触发重新追踪：

1. `Tensor` 参数数据类型（`dtypes`）的组合不同。
2. `Tensor` 参数的秩（维度数量）不同。
3. 如果图是基于特定形状构建的，而 `Tensor` 参数的形状不兼容。
4. 非Tensor（Python）参数不同，并且它们影响图结构。

频繁的重新追踪，通常是由于使用Python标量或形状不断变化的张量进行调用引起的，可能会带来负面影响。为避免这种情况，您可以为 `@tf.function` 提供一个 `input_signature`。这会指定输入张量的预期形状和数据类型，只创建一个*特定*的图，并在使用不兼容签名调用时引发错误。

```python
@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
def specific_function(x):
  print(f"Tracing specific function with input shape: {x.shape}")
  return x * 2.0

# 首次调用：为 shape=[None], dtype=float32 追踪并创建图
result1 = specific_function(tf.constant([1.0, 2.0, 3.0]))
print(result1)

# 第二次调用：重用图（兼容的形状）
result2 = specific_function(tf.constant([4.0, 5.0]))
print(result2)

# 第三次调用：错误！不兼容的数据类型 (int32 与 float32)
try:
    specific_function(tf.constant([1, 2]))
except TypeError as e:
    print(f"\n错误: {e}")

# 第四次调用：错误！不兼容的形状（标量与向量 [None]）
try:
    specific_function(tf.constant(1.0))
except ValueError as e: # 可能是 ValueError 或 TypeError，取决于 TF 版本/细节
    print(f"\n错误: {e}")
```

在保存模型（SavedModel 格式）或部署函数时，使用 `input_signature` 尤其重要，因为它定义了预期的接口。

理解 `tf.function` 和 AutoGraph 对于编写高性能 TensorFlow 代码非常重要。它让您可以在发挥Python可读性的同时，从TensorFlow图执行引擎的优化中获益。这有助于实现高性能（第2章中涵盖）、启用分布式训练策略（第3章）以及构建高效的自定义组件（第4章）。掌握追踪行为并知道何时使用 `input_signature` 限制多态性，是任何进阶TensorFlow开发者的实用技能。

## 参考资料

- [Better performance with tf.function](https://www.tensorflow.org/guide/function) — TensorFlow Authors (2024)
  官方指南，详细介绍了 tf.function 的使用、追踪过程、AutoGraph 功能以及性能方面的考量。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781098122461/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本被广泛使用的书籍，提供了关于 TensorFlow 2.x 中 tf.function、图执行和性能优化的实用内容。
