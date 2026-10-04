---
course: "advanced-tensorflow"
chapter: "tensorflow-execution-graphs"
lesson: "debugging-tensorflow-programs"
sourceId: 2812
sourceUrl: "https://apxml.com/zh/courses/advanced-tensorflow/chapter-1-tensorflow-execution-graphs/debugging-tensorflow-programs"
title: "调试 TensorFlow 程序"
description: "查看调试复杂 TensorFlow 模型及执行图的工具与技巧。"
order: 7
plots: []
sourceHash: "3aaedd16f2aeae379f1a154d373275277db66ffec165c4e91eb31ab5273881d4"
sourceCorrections: []
---

即使对 TensorFlow 的执行模型有扎实的了解，在复杂模型或 `tf.function` 修饰的代码中查找错误也可能很困难。处理编译图时，标准 Python 调试技术有时会不够用。在此介绍调试 TensorFlow 程序的特定工具和方法，特别关注图执行、AutoGraph 转换和梯度计算中出现的问题。

### 调试方法：即时执行与图执行

您的调试方式通常取决于有问题代码是即时运行还是在 `tf.function` 图中运行。

- **即时执行：** 当即时运行（TensorFlow 2 中的默认设置）时，您的代码会逐行执行，很像标准 Python。这意味着您通常可以使用熟悉的 Python 调试工具：

  - **`print()` 函数：** 使用标准 Python `print()` 语句检查 Python 变量的值或 TensorFlow 操作的结果（它们将是即时张量）。
  - **Python 调试器（`pdb` 或 IDE 调试器）：** 设置断点，逐步执行代码，并交互式地检查变量。这对于在图追踪发生之前理解逻辑流程和变量状态非常有效。
- **图执行（`tf.function`）：** 一旦代码被 `tf.function` 封装，TensorFlow 就会对其进行追踪以创建静态计算图。放置在修饰函数 *内部* 的标准 Python `print()` 或 `pdb` 断点通常只会在初始追踪阶段执行，而不会在后续图执行期间执行。这种行为可能会产生误导。图模式调试需要新的方法。

### `tf.function` 内部调试

当您需要 *在* TensorFlow 图执行期间检查值或控制流时，请使用以下技巧：

#### 使用 `tf.print`

`tf.print` 函数是 Python `print` 的图感知等效项。它将打印操作直接插入到 TensorFlow 图中。这确保了每当图执行时（而不仅仅是追踪期间）都会打印张量的值。

```python
import tensorflow as tf

@tf.function
def problematic_function(x):
  # 使用 tf.print 检查图中的张量值
  tf.print("Inside tf.function, x =", x)
  y = x * 2
  tf.print("Intermediate value y =", y)
  # 潜在问题：整数除法可能意外截断
  z = y // 3
  tf.print("Final value z =", z)
  return z

# 调用函数
input_tensor = tf.constant([1, 5, 10], dtype=tf.int32)
result = problematic_function(input_tensor)
print("Result outside tf.function:", result)

# 示例输出：
# Inside tf.function, x = [1 5 10]
# Intermediate value y = [2 10 20]
# Final value z = [0 3 6]
# Result outside tf.function: tf.Tensor([0 3 6], shape=(3,), dtype=int32)
```

`tf.print` 对于观察图执行路径中不同阶段的张量值非常有用。请记住，`tf.print` 操作在计算所在的设备（CPU/GPU/TPU）上执行，并且输出可能会根据执行环境，特别是在分布式设置中，出现在不同的位置（例如日志）。

#### 暂时禁用 `tf.function`

对于复杂的调试情况，强制函数即时运行可能很有帮助，这使您可以使用标准 Python 调试器。您可以在全局实现这一点：

```python
import tensorflow as tf

# 全局禁用 tf.function
tf.config.run_functions_eagerly(True)

@tf.function
def my_complex_logic(a, b):
  # 现在您可以在此处有效地使用 pdb 或 print
  # import pdb; pdb.set_trace()
  print("Running eagerly, a:", a)
  c = tf.matmul(a, b)
  print("Running eagerly, c:", c)
  return c

# 调用现在将即时执行
matrix_a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
matrix_b = tf.constant([[5.0], [6.0]])
result = my_complex_logic(matrix_a, matrix_b)

# 调试完成后记得将其关闭
tf.config.run_functions_eagerly(False)
```

即时运行简化了调试，但有其代价：

1. **性能：** 即时执行通常比图执行慢。
2. **行为差异：** 即时执行和图执行之间的一些行为可能略有不同，特别是 AutoGraph 处理控制流或细微类型强制转换方面。错误可能只在图模式下出现。

请谨慎使用此技术来隔离问题，但务必在启用 `tf.function` 后验证修复是否正常工作。

#### 使用 TensorBoard 可视化图

了解 `tf.function` 和 AutoGraph 如何将您的 Python 代码转换为计算图，可以显示出意想不到的结构或操作。TensorBoard 提供了一个图可视化工具。

要使用它，请在您的 `tf.function` 上下文 (context)中创建一个摘要文件写入器并追踪该函数：

```python
import tensorflow as tf
import datetime

@tf.function
def simple_graph(x, y):
  return tf.add(x, y)

# 设置日志记录
stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = f'./logs/func/{stamp}'
writer = tf.summary.create_file_writer(logdir)

# 追踪函数
input_tensor = tf.constant(1.0)
tf.summary.trace_on(graph=True, profiler=False) # 开始追踪图
result = simple_graph(input_tensor, input_tensor) # 调用函数以追踪它
with writer.as_default():
  tf.summary.trace_export(name="simple_graph_trace", step=0) # 导出图
tf.summary.trace_off() # 停止追踪

print(f"Graph trace written to {logdir}")
# 现在运行：tensorboard --logdir ./logs/func
```

启动 TensorBoard (`tensorboard --logdir ./logs/func`) 并导航到“图”选项卡将显示 TensorFlow 创建的结构。这有助于确定控制流（`tf.cond`、`tf.while_loop`）是否按预期生成，或者是否存在特定操作。

> `simple_graph` 函数计算图的简化表示，显示输入流向加法操作。

#### 使用 `tf.debugging` 模块

TensorFlow 提供了一个专门的调试模块 `tf.debugging`，其中包含在图内部运行的断言。这些断言对于在图执行期间检查条件以及在检查失败时引发错误很有用。

- **`tf.debugging.check_numerics(tensor, message)`：** 检查张量是否包含任何 `NaN`（非数字）或 `Inf`（无穷大）值。这对于在训练期间检测数值不稳定性（例如梯度爆炸）非常有帮助。
- **`tf.debugging.assert_equal(x, y)`：** 断言两个张量 `x` 和 `y` 逐元素值相同。
- **`tf.debugging.assert_shapes(shapes, data=None)`：** 断言张量的形状与指定的形状列表匹配。这有助于及早捕获形状不匹配错误。示例：`tf.debugging.assert_shapes([(tensor1, (None, 10)), (tensor2, (5, None, 3))])` 检查 `tensor1` 的形状是否为 `(batch, 10)`，`tensor2` 的形状是否为 `(5, time, 3)`。
- **`tf.debugging.Assert(condition, data)`：** 一个通用断言，如果布尔型 `condition` 张量评估为 `False`，则会引发错误。

这些断言向图中添加操作，这些操作在执行期间执行检查。

```python
import tensorflow as tf

@tf.function
def safe_divide(numerator, denominator):
  tf.debugging.assert_greater(tf.abs(denominator), 1e-6,
                              message="分母接近零！")
  result = numerator / denominator
  # 检查可能由接近零的除法引起的 NaN/Inf
  tf.debugging.check_numerics(result, "结果中的数值问题")
  return result

# 这将正常执行
print(safe_divide(tf.constant(10.0), tf.constant(2.0)))

# 这将因为 assert_greater 检查而引发 InvalidArgumentError
# try:
#   print(safe_divide(tf.constant(1.0), tf.constant(1e-8)))
# except tf.errors.InvalidArgumentError as e:
#   print(f"捕获到预期错误：{e}")
```

### 调试梯度和 `tf.GradientTape`

与自动微分相关的问题很常见，特别是在构建自定义训练循环或复杂模型时。

#### `None` 梯度

一个常见问题是，从 `tf.GradientTape` 请求梯度时收到 `None`。这通常是由于以下原因之一：

1. **目标未被追踪：** 默认情况下，`tf.GradientTape` 只追踪可训练的 `tf.Variable` 对象。如果需要对 `tf.Tensor` 求梯度，必须使用 `tape.watch(tensor)` 显式告诉 Tape 追踪它。
2. **没有路径：** 图中可能没有连接源张量（损失所依赖的值）和目标张量（您想要梯度的变量）的可微分路径。如果操作断开连接或切片/索引无意中破坏了连接，就可能发生这种情况。
3. **不可微分操作：** 路径可能涉及一个相对于目标本身不可微分的操作（例如，`tf.cast` 到整数类型、`tf.round`、布尔操作、使用非常量张量进行索引）。
4. **整数类型：** 梯度通常不为整数值张量定义。确保相关计算使用浮点类型（`float32`、`float64`）。
5. **在 Tape 上下文 (context)内创建变量：** 通常不鼓励在梯度 Tape 的上下文 *内部* 创建 `tf.Variable`，这可能导致意外行为或 `None` 梯度。在 Tape 范围外初始化变量。

#### 检查梯度值

您可以直接检查计算出的梯度，以查看它们是否合理（例如，不全为零，不太大）。

```python
import tensorflow as tf

x = tf.Variable(3.0)
with tf.GradientTape() as tape:
  y = x * x

# 计算梯度 dy/dx
grad = tape.gradient(y, x)

# 检查梯度是否为 None 或检查其值
if grad is not None:
  print(f"Gradient dy/dx at x={x.numpy()}: {grad.numpy()}") # 应该为 6.0
  tf.debugging.check_numerics(grad, "检查梯度是否存在 NaN/Inf")
else:
  print("梯度为 None。检查 Tape 追踪和可微分性。")
```

在梯度本身上使用 `tf.debugging.check_numerics` 是一种很好的做法，可以在训练过程的早期捕获梯度爆炸。

### 一般调试技巧

- **简化：** 隔离有问题的代码。尝试单独运行特定层、函数或计算，初始时可以尝试即时模式。
- **检查形状：** 形状不匹配是常见的错误来源。请随意使用 `tf.print(tf.shape(tensor))` 或 `tf.debugging.assert_shapes`。请记住，在图构建期间（`None` 维度），形状的定义可能不如执行期间明确。
- **检查数据类型（`dtype`）：** 确保操作中组合的张量具有兼容和适当的数据类型（通常是 `float32`）。如果需要，请显式使用 `tf.cast`，但请注意，如果转换为非浮点类型，它可能会破坏梯度流。
- **从即时模式开始：** 首先在即时模式下开发和测试组件，以使用标准 Python 工具捕获基本逻辑错误。然后，使用 `tf.function` 封装并调试任何图特有的问题。
- **增量转换：** 将复杂 Python 代码转换为 `tf.function` 时，先转换较小的部分并验证它们正常工作，然后再将它们组合起来。这有助于找出 AutoGraph 可能存在困难的地方。
- **查阅 AutoGraph 日志：** 对于复杂的 AutoGraph 问题，设置 `tf.autograph.set_verbosity` 级别可以提供有关转换过程的详细日志，可能突显有问题的 Python 结构。

调试 TensorFlow 代码，特别是在 `tf.function` 图中，需要调整您的方法。借助 `tf.print`、`tf.debugging`、TensorBoard 可视化以及切换即时执行的能力，为识别和解决高级 TensorFlow 应用程序中的问题提供了一套强大的工具。这些技术为构建和维护后续章节中讨论的复杂、高性能模型奠定了重要基础。

## 参考资料

- [Debugging TensorFlow programs](https://www.tensorflow.org/guide/debugging) — TensorFlow Authors (2024)
  官方指南，详细介绍了 tf.print、tf.debugging 模块、急切执行切换以及 TensorBoard 集成，用于调试 TensorFlow 应用程序。
- [Better performance with tf.function](https://www.tensorflow.org/guide/function) — TensorFlow Authors (2024)
  解释了 tf.function 和 AutoGraph 的机制，为理解和调试图执行问题提供了重要背景。
- [Automatic differentiation with tf.GradientTape](https://www.tensorflow.org/guide/autodiff) — TensorFlow Authors (2024)
  Publisher: TensorFlow
  涵盖了 tf.GradientTape 的原理和用法，包括 None 梯度等常见问题以及检查梯度值的方法。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125969/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  在使用 TensorFlow 构建和训练机器学习模型的背景下，提供了实用的调试建议和策略。
