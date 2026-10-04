---
course: "getting-started-with-tensorflow"
chapter: "core-tensorflow-concepts"
lesson: "introduction-tf-function"
sourceId: 1701
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-tensorflow/chapter-2-core-tensorflow-concepts/introduction-tf-function"
title: "tf.function 介绍"
description: "学习如何使用 tf.function 装饰器将 Python 函数编译成 TensorFlow 图以提高性能。"
order: 5
plots: []
sourceHash: "88f68a71a33f80ff3e70bef8bcf760b15fede2fefafb0c6d069d91fbf92feb53"
sourceCorrections: []
---

TensorFlow 2.x 默认以“即时执行”模式运行。这意味着 TensorFlow 操作会立即评估，很像标准 Python 代码。虽然这提供了灵活性并使调试更轻松（您可以使用 `print()` 和调试器等标准 Python 工具），但有时可能会错失静态计算图所能提供的性能优化，而静态计算图是 TensorFlow 1.x 中的标准做法。图使 TensorFlow 能够分析计算，对其进行优化（例如，合并操作，消除冗余计算），并可能更高效地执行，尤其是在 GPU 或 TPU 等多个设备上。

那么，我们如何才能两全其美：即时执行的便捷性与图执行的性能？这时 `tf.function` 就派上用场了。

`tf.function` 是一个装饰器，它将包含 TensorFlow 操作的 Python 函数转换为可调用的 TensorFlow 图。当您使用 `@tf.function` 装饰一个 Python 函数时，TensorFlow 会执行一个称为“追踪”的过程。在第一次使用特定输入类型和形状（称为输入签名）进行调用期间，TensorFlow 会在 Python 中执行函数，追踪 TensorFlow 操作以构建静态计算图。对于具有相同输入签名的后续调用，TensorFlow 可以直接执行优化后的图，跳过 Python 执行步骤，通常会带来显著的速度提升。

让我们看一个简单示例。考虑一个使用 TensorFlow 操作的普通 Python 函数：

```python
import tensorflow as tf

# 使用 TF 操作的普通 Python 函数
def simple_math(x, y):
  print(f"Running Python function with x={x}, y={y}") # Python 副作用
  a = tf.matmul(x, y)
  b = tf.add(a, y)
  return b

# 创建一些张量
tensor_a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
tensor_b = tf.constant([[1.0, 1.0], [0.0, 1.0]])

# 即时调用函数
result1 = simple_math(tensor_a, tensor_b)
print("第一次即时调用结果:\n", result1.numpy())

result2 = simple_math(tensor_b, tensor_a) # 不同输入，仍运行 Python
print("\n第二次即时调用结果:\n", result2.numpy())
```

每次调用 `simple_math` 时，Python 代码都会执行，包括 `print` 语句。

现在，让我们应用 `@tf.function` 装饰器：

```python
import tensorflow as tf
import time

# A regular Python function using TF ops
def simple_math(x, y):
  print(f"Running Python function with x={x}, y={y}") # Python side-effect
  a = tf.matmul(x, y)
  b = tf.add(a, y)
  return b

# 装饰后的函数
@tf.function
def graph_math(x, y):
  print(f"Tracing function with x={x}, y={y}") # 这个 print 只会在追踪期间执行！
  a = tf.matmul(x, y)
  b = tf.add(a, y)
  return b

# 创建张量
tensor_a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
tensor_b = tf.constant([[1.0, 1.0], [0.0, 1.0]])

# Call the function eagerly
result1 = simple_math(tensor_a, tensor_b)
print("First eager call result:\n", result1.numpy())

result2 = simple_math(tensor_b, tensor_a) # Different inputs, still runs Python
print("\nSecond eager call result:\n", result2.numpy())

print("第一次调用（触发追踪）：")
result_graph1 = graph_math(tensor_a, tensor_b)
print("第一次图调用结果:\n", result_graph1.numpy())

print("\n第二次调用（重用追踪的图）：")
result_graph2 = graph_math(tensor_a, tensor_b) # 相同的输入签名，使用缓存的图
print("第二次图调用结果:\n", result_graph2.numpy())

print("\n第三次调用（形状不同，触发重新追踪）：")
# 修正 tensor_c 和 tensor_d 以实现兼容的形状
tensor_c = tf.constant([[1.0, 2.0], [3.0, 4.0]])
tensor_d = tf.constant([[1.0], [1.0]])
result_graph3 = graph_math(tensor_c, tensor_d) # 由于形状不同，触发新的追踪
print("第三次图调用结果:\n", result_graph3.numpy())

# 演示性能差异（简单计时）
# 注意：实际性能提升在复杂操作和 GPU 上更为显著
n_runs = 1000

start_time = time.time()
for _ in range(n_runs):
    simple_math(tensor_a, tensor_b)
eager_time = time.time() - start_time

start_time = time.time()
# 调用一次以确保追踪在循环外完成
graph_math(tensor_a, tensor_b)
for _ in range(n_runs):
    graph_math(tensor_a, tensor_b) # 重用图
graph_time = time.time() - start_time

print(f"\n运行 {n_runs} 次所需时间（即时执行）：{eager_time:.4f} 秒")
print(f"运行 {n_runs} 次所需时间（@tf.function）：{graph_time:.4f} 秒")
```

请注意以下几点重要事项：

1. **追踪：** `graph_math` 内部的 `print` 语句只在*第一次*调用时执行（对于给定输入签名）。这是图的构建阶段。后续具有相同签名的调用会直接执行图。具有不同张量形状的调用（如 `tensor_c` 和 `tensor_d`）会触发针对该特定签名的一次*新的*追踪。
2. **性能：** 即使在这个简单示例中，`@tf.function` 版本在循环中也通常运行得更快，因为它避免了初始追踪后每次调用时 Python 解释的开销。在复杂计算、自定义训练循环以及在 GPU 等硬件加速器上运行时，益处会更加明显。
3. **Python 副作用：** 请注意，装饰函数内部的 Python 操作（如打印、向 Python 列表追加或进行外部库调用）通常只在追踪期间发生。如果您需要操作在每次图执行时都运行，请改用 TensorFlow 操作（例如，`tf.print` 代替 Python 的 `print`）。

### AutoGraph：处理 Python 控制流

Python 控制流如 `if`、`for` 和 `while` 语句怎么办？`tf.function` 使用一个名为 AutoGraph 的库，自动将此类 Python 结构转换为其 TensorFlow 图等效项（如 `tf.cond` 和 `tf.while_loop`）。这让您能够编写自然的 Python 代码，并且 `tf.function` 会处理转换为高性能图结构。

```python
import tensorflow as tf

@tf.function
def dynamic_choice(x, threshold):
  if tf.reduce_sum(x) > threshold:
    # 此分支使用 tf.square
    return tf.square(x)
  else:
    # 此分支使用 tf.sqrt（确保 sqrt 的输入为非负数）
    return tf.sqrt(tf.abs(x))

tensor_low = tf.constant([1.0, 2.0, 3.0]) # 和 = 6.0
tensor_high = tf.constant([5.0, 6.0, 7.0]) # 和 = 18.0
threshold_val = tf.constant(10.0)

print("低张量的结果:", dynamic_choice(tensor_low, threshold_val).numpy())
print("高张量的结果:", dynamic_choice(tensor_high, threshold_val).numpy())
```

AutoGraph 会分析 `if` 语句，并将其转换为图操作，这些操作可以根据运行时张量的值选择正确的计算路径。

### 何时使用 `tf.function`

虽然您*可以*装饰几乎任何执行 TensorFlow 操作的 Python 函数，但它最适用于：

- **训练循环：** 封装您的主要训练步骤函数可以显著加快训练速度。
- **模型推理 (inference)：** 将其应用于模型的 `call` 方法或专门的预测函数可提高推理速度。
- **计算密集型预处理：** `tf.data` 管道中的复杂数据转换会从中受益。

不要在不重要的函数上过度使用它，因为追踪的开销可能大于其益处。从装饰更大的计算块开始。

`tf.function` 是编写高性能 TensorFlow 2.x 代码的基本工具。它让您能够编写直观的 Python 风格代码，同时获得以前只与静态图相关的优化益处。理解它是如何追踪函数并转换控制流的，这对于有效地加速您的模型和数据管道很重要。

## 参考资料

- [Better performance with tf.function](https://www.tensorflow.org/guide/function) — TensorFlow Authors (2024)
  提供关于`tf.function`的官方说明，包括其用法、与Eager Execution的交互，以及如何通过将Python函数转换为可调用图来优化TensorFlow程序。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEhbovopJi45vudtHp5YEY_yS10p6UowQONYQlQTgD7RNkxwHoAmFnGU9vLkh55dI9eFeCJYqoVy8iq_HbaiQ-ozuPaBXBI9mGm6sZsKBH_OESfLlEJLHc3hNHWcVUK58gETd64aRsrNG9Fhcuw_myxz5_w6K2nluQZ9pxDQSFANtbxfLjWXNutfAo7Hl0_01OkDj7utrw=) — Aurélien Géron (2022)
  Publisher: O'Reilly Media; Pages: 864
  一本广泛使用的实践指南，包含TensorFlow 2.x Eager Execution和`tf.function`的章节，说明它们在构建和优化机器学习模型中的作用。
- [TensorFlow 2.0: An Introduction to Eager Execution and tf.function](https://blog.tensorflow.org/2019/01/tensorflow-2.0-alpha-eager-execution-tf_function.html) — Josh Gordon, Robert Crowe (2019)
  Publisher: TensorFlow Blog
  TensorFlow官方博客的一篇早期文章，介绍TensorFlow 2.0过渡期间Eager Execution的动机和`tf.function`的功能。
