---
course: "advanced-tensorflow"
chapter: "tensorflow-execution-graphs"
lesson: "tracing-graph-representation"
sourceId: 2797
sourceUrl: "https://apxml.com/zh/courses/advanced-tensorflow/chapter-1-tensorflow-execution-graphs/tracing-graph-representation"
title: "追踪机制与图表示"
description: "审视 TensorFlow 如何追踪 Python 函数并将其在内部表示为计算图。"
order: 3
plots: []
sourceHash: "e7aea3367d74d16313d2014320834ebd3d72e14ec90453ec32186eaa88fe2fd8"
sourceCorrections: []
---

当您使用 `tf.function` 装饰 Python 函数时，您是在指示 TensorFlow 可能会将其转换为可调用的 TensorFlow 图。这种转换过程，称为“追踪”，是理解 `tf.function` 如何获得性能提升和可移植性的根本。追踪是指执行一次 Python 函数代码（或在特定情况下多次），以将 TensorFlow 操作序列捕获为静态图。

### 追踪过程

第一次使用特定输入参数 (parameter)调用被 `tf.function` 装饰的函数时，TensorFlow 会执行几个步骤：

1. **执行与图构建：** TensorFlow 运行您函数的 Python 代码。然而，它不像标准 Python 那样立即返回结果，而是按执行顺序记录遇到的所有 TensorFlow 操作（例如，`tf.add`、`tf.matmul`、`tf.reduce_sum`）。这些被记录的操作构成 `tf.Graph` 的节点。在这些操作之间流动的数据张量则成为图的边。
2. **AutoGraph 转换：** 如果您的 Python 函数包含 `if`、`for`、`while` 或断言等 Python 控制流结构，`tf.function` 会采用一种称为 AutoGraph 的机制。AutoGraph 将这些 Python 代码重写为等效的 TensorFlow 图操作，例如用于条件判断的 `tf.cond` 和用于循环的 `tf.while_loop`。这种转换确保逻辑可以直接嵌入 (embedding)到静态计算图中。
3. **图定型：** 函数执行完成后，生成的 `tf.Graph` 会被定型。这个图表示您的 Python 函数针对其追踪时所用的*特定输入*定义的计算。TensorFlow 可能会对此图进行优化，例如修剪未使用的操作或融合操作。
4. **缓存：** 生成的图（特别是 `ConcreteFunction`）会被缓存。缓存键是从追踪期间使用的参数的*输入签名*中派生出来的。

### 输入签名与重新追踪

**输入签名**的理念非常重要。签名包含参数 (parameter)的数量、它们的数据类型（`dtype`），以及对于张量而言，重要的是它们的形状。

考虑以下函数：

```python
import tensorflow as tf
import time

@tf.function
def dynamic_resize(x, new_height):
  print(f"Tracing dynamic_resize with x shape: {x.shape}, new_height: {new_height}")
  # 模拟一些工作
  tf.print("Executing graph for shape:", tf.shape(x), "and height:", new_height)
  resized = tf.image.resize(x, [new_height, tf.shape(x)[1]])
  return tf.reduce_sum(resized)

# 第一次调用：针对形状 (1, 100, 100, 3) 和 int 类型的 new_height 进行追踪
img1 = tf.random.normal((1, 100, 100, 3))
start = time.time()
result1 = dynamic_resize(img1, 50)
print(f"First call time: {time.time() - start:.4f}s")

# 第二次调用：使用相同签名的缓存图
img2 = tf.random.normal((1, 100, 100, 3))
start = time.time()
result2 = dynamic_resize(img2, 50)
print(f"Second call time (cached): {time.time() - start:.4f}s")

# 第三次调用：形状不同，触发重新追踪
img3 = tf.random.normal((1, 120, 120, 3))
start = time.time()
result3 = dynamic_resize(img3, 50)
print(f"Third call time (re-trace): {time.time() - start:.4f}s")

# 第四次调用：new_height 的 Python 类型不同，触发重新追踪
start = time.time()
result4 = dynamic_resize(img1, tf.constant(60)) # new_height is now a Tensor
print(f"Fourth call time (re-trace): {time.time() - start:.4f}s")
```

**输出：**

```text
Tracing dynamic_resize with x shape: (1, 100, 100, 3), new_height: 50
Executing graph for shape: [  1 100 100   3] and height: 50
First call time: 0.1523s  # 包含追踪时间
Executing graph for shape: [  1 100 100   3] and height: 50
Second call time (cached): 0.0015s # 快得多，使用缓存图
Tracing dynamic_resize with x shape: (1, 120, 120, 3), new_height: 50
Executing graph for shape: [  1 120 120   3] and height: 50
Third call time (re-trace): 0.0876s # 再次包含追踪时间
Tracing dynamic_resize with x shape: (1, 100, 100, 3), new_height: Tensor("Const:0", shape=(), dtype=int32)
Executing graph for shape: [  1 100 100   3] and height: 60
Fourth call time (re-trace): 0.0751s # 包含追踪时间
```

请注意，第二次调用明显更快，因为它重用了第一次调用时追踪的图。第三次和第四次调用触发了重新追踪，因为张量形状 (`img3`) 或参数的 Python 类型 (`tf.constant(60)` 与 Python `int` 50) 发生了变化，导致输入签名不同。

过度的重新追踪会抵消 `tf.function` 的性能优势。如果一个函数经常以不同的张量形状或参数类型被调用，它可能会被反复追踪。您可以通过向 `tf.function` 提供 `input_signature` 来限制追踪行为：

```python
@tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None, 3], dtype=tf.float32),
                               tf.TensorSpec(shape=[], dtype=tf.int32)])
def stable_resize(x, new_height):
  print(f"Tracing stable_resize with spec: {x.shape}, {new_height.shape}")
  tf.print("Executing stable graph for shape:", tf.shape(x), "and height:", new_height)
  # 对动态维度使用 tf.shape
  new_width = tf.shape(x)[2]
  resized = tf.image.resize(x, [new_height, new_width])
  return tf.reduce_sum(resized)

# 调用不同形状但符合规范的函数
img1 = tf.random.normal((1, 100, 120, 3))
img2 = tf.random.normal((2, 80, 90, 3))

# 第一次调用进行追踪
print("First call:")
res1 = stable_resize(img1, tf.constant(50))

# 第二次调用重用图，即使形状不同，因为它符合规范
print("\nSecond call:")
res2 = stable_resize(img2, tf.constant(40))

# 这会因为数据类型错误而失败
# try:
#   stable_resize(img1, tf.constant(50.0)) # float32 类型的高度
# except TypeError as e:
#   print(f"\n第三次调用出错： {e}")
```

**输出：**

```text
First call:
Tracing stable_resize with spec: (None, None, None, 3), ()
Executing stable graph for shape: [  1 100 120   3] and height: 50

Second call:
Executing stable graph for shape: [ 2 80 90  3] and height: 40
```

对未知维度使用 `tf.TensorSpec` 中的 `None`，可以让函数在不重新追踪的情况下处理这些维度中不同形状，前提是秩和 `dtype` 匹配。这让您可以更好地控制追踪的发生时机。

### 图表示：`tf.Graph`、操作和张量

在内部，TensorFlow 将追踪到的计算表示为 `tf.Graph` 对象。此图包含两种主要类型的对象：

1. **`tf.Operation`**：它们是图的节点，表示计算单元（例如，`MatMul`、`AddV2`、`Conv2D`、`Relu`）。操作消耗零个或多个张量，并产生零个或多个张量。
2. **`tf.Tensor`**：它们是图的边，表示在操作之间流动的数据。

您可以访问已追踪函数（即 `ConcreteFunction`）的底层图，以检查其结构。

```python
@tf.function
def simple_computation(a, b):
  c = tf.matmul(a, b)
  d = tf.add(c, 1.0)
  return tf.nn.relu(d)

# 追踪函数
input_spec = (tf.TensorSpec(shape=[2, 2], dtype=tf.float32),
              tf.TensorSpec(shape=[2, 2], dtype=tf.float32))
concrete_func = simple_computation.get_concrete_function(*input_spec)

# 获取图
graph = concrete_func.graph

print(f"Function captures: {graph.captures}") # 从外部范围捕获的张量（此处通常为空）
print(f"Function variables: {graph.variables}") # 使用的 tf.Variable（此处为空）
print("\n图中的操作：")
for op in graph.get_operations():
  print(f"- {op.name} (type: {op.type})")

print("\n图输入（占位符）：")
print(graph.inputs)
print("\n图输出：")
print(graph.outputs)
```

**输出：**

```text
Function captures: []
Function variables: []

图中的操作：
- args_0 (type: Placeholder)
- args_1 (type: Placeholder)
- MatMul (type: MatMul)
- AddV2/y (type: Const)
- AddV2 (type: AddV2)
- Relu (type: Relu)
- Identity (type: Identity)

图输入（占位符）：
[<tf.Tensor 'args_0:0' shape=(2, 2) dtype=float32>, <tf.Tensor 'args_1:0' shape=(2, 2) dtype=float32>]

图输出：
[<tf.Tensor 'Identity:0' shape=(2, 2) dtype=float32>]
```

输出显示了为输入（`args_0`、`args_1`）创建的占位符操作、核心计算操作（`MatMul`、`AddV2`、`Relu`）、为加法创建的常量（`AddV2/y`），以及常用于最终返回值的 `Identity` 操作。

我们可以将这个简单的图可视化：

> `simple_computation` 函数追踪生成的 `tf.Graph` 的简化可视化图。占位符表示输入，其他节点表示 TensorFlow 操作，边表示张量流。

### 状态管理：Python 变量与 `tf.Variable`

追踪期间如何处理状态是另一个重要的细节：

- **Python 变量：** 在 `tf.function` 中访问的标准 Python 变量通常在*追踪时*按值捕获。它们的值成为嵌入 (embedding)图中的常量。追踪后在函数内部修改 Python 变量不会影响图的执行，外部对变量的更改也不会反映在后续的图调用中（除非触发重新追踪）。
- **`tf.Variable`**：这些对象旨在表示 TensorFlow 图中可变、有状态的张量。当 `tf.function` 访问 `tf.Variable`（在函数外部创建）时，它会在图中为其创建一个符号占位符。`assign`、`assign_add` 等操作会修改 `tf.Variable` 的底层状态，这些更改会在对已追踪函数的调用之间持续存在。

```python
external_python_var = 10
external_tf_var = tf.Variable(10, dtype=tf.int32)

@tf.function
def state_example():
  # Python 变量在追踪时捕获
  result_py = external_python_var * 2
  tf.print("Python var based result (trace time value):", result_py)

  # tf.Variable 以有状态方式访问
  external_tf_var.assign_add(1) # 修改变量状态
  tf.print("TF Variable current value:", external_tf_var)

print("--- 首次调用（追踪中）---")
state_example()

external_python_var = 100 # 在外部更改 Python 变量
print("\n--- 第二次调用 ---")
state_example() # 使用缓存图

print("\n--- 第三次调用 ---")
state_example() # 使用缓存图

print(f"\n最终 Python 变量值： {external_python_var}")
print(f"最终 TF 变量值： {external_tf_var.numpy()}")
```

**输出：**

```text
--- Initial Call (Tracing) ---
Python var based result (trace time value): 20 # 捕获 external_python_var=10
TF Variable current value: 11

--- Second Call ---
Python var based result (trace time value): 20 # 仍使用追踪到的值 10*2
TF Variable current value: 12 # 变量状态已更新

--- Third Call ---
Python var based result (trace time value): 20 # 仍使用追踪到的值 10*2
TF Variable current value: 13 # 变量状态再次更新

Final Python var value: 100
Final TF var value: 13
```

如输出所示，基于 `external_python_var` 的计算始终使用在首次追踪期间捕获的值 `10`，即使外部变量已更改为 `100`。相反，`external_tf_var` 是有状态的；在图执行的每次调用中，其值都会正确更新。这种区别对于实现带有可训练权重 (weight)（它们是 `tf.Variable` 对象）的模型至关重要。

了解追踪机制、输入签名、图表示和状态处理，使您能够编写正确且高性能的 `tf.function` 装饰代码，避免因不必要的重新追踪而导致的意外行为或性能下降。

## 参考资料

- [Better performance with tf.function and ConcreteFunctions](https://www.tensorflow.org/guide/function) — TensorFlow Authors (2024)
  Publisher: TensorFlow (Google)
  该指南是理解 `tf.function`、追踪、AutoGraph 和 `tf.Variable` 交互的官方最新资源。它包含与本节内容直接相关的解释和示例。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  本书提供了实用的示例和对 `tf.function` 及其在 TensorFlow 2.x 中的作用以及如何管理图和状态的全面说明，是对官方文档的补充。
- [TensorFlow: A System for Large-Scale Machine Learning](https://www.usenix.org/conference/osdi16/technical-sessions/presentation/abadi) — Martín Abadi, Paul Barham, Jianmin Chen, Zhifeng Chen, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Geoffrey Irving, Michael Isard, Manjunath Kudlur, Josh Levenberg, Rajat Monga, Sherry Moore, Derek G. Murray, Benoit Steiner, Paul Tucker, Vijay Vasudevan, Pete Warden, Martin Wicke, Yuan Yu, Xiaoqiang Zheng (2016)
  Journal: 12th USENIX Symposium on Operating Systems Design and Implementation (OSDI 16); Publisher: USENIX Association; Pages: 265-283
  这篇基础论文介绍了最初的 TensorFlow 系统，包括其图执行模型，为理解 `tf.function` 如何在 TensorFlow 2.x 中利用和优化这些图提供了背景。
