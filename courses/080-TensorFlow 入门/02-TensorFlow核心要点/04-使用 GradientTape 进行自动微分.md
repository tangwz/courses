# 使用 GradientTape 进行自动微分

来源：[原文](https://apxml.com/zh/courses/getting-started-with-tensorflow/chapter-2-core-tensorflow-concepts/automatic-differentiation-gradienttape)

[返回章节目录](README.md) · [返回课程目录](../README.md)

训练机器学习 (machine learning)模型，特别是深度神经网络 (neural network)，很大程度上依赖于像梯度下降 (gradient descent)这样的优化算法。这些算法迭代地调整模型参数 (parameter)（如权重 (weight)和偏差），以最小化损失函数 (loss function)。为了知道*如何*调整参数，我们需要计算损失函数相对于每个参数的梯度。这个梯度表示损失函数最陡峭的上升方向，因此向相反方向移动有助于最小化损失。

手动推导具有数百万参数的复杂模型的梯度是不切实际且容易出错的。符号微分（如Mathematica或SymPy中使用的）可以生成精确的导数，但结果表达式可能变得极其复杂（“表达式膨胀”）且计算成本高昂。数值微分（使用有限差分近似梯度）实现起来更简单，但计算可能很慢，并存在数值精度问题。

TensorFlow 采用一种称为**自动微分**（AutoDiff）的强大技术，以高效准确地计算梯度。自动微分计算梯度的精确数值，而无需明确推导符号梯度表达式。它通过将计算分解为一系列基本操作（加法、乘法、激活函数 (activation function)等），并在计算图的遍历过程中系统地应用链式法则来实现这一点。

### 记录计算与 tf.GradientTape

TensorFlow 提供了用于自动微分的 `tf.GradientTape` API。它就像一个录音机：在 `tf.GradientTape` 上下文 (context)管理器范围内执行的操作被“记录”到磁带上。TensorFlow 随后使用这张磁带和链式法则，通过从输出（目标）到输入（源）反向遍历记录的操作来计算梯度。

默认情况下，`tf.GradientTape` 会自动“监视”在其上下文中访问的任何可训练 `tf.Variable`。当您请求梯度时，磁带会计算相对于这些被监视变量的梯度。

让我们看一个简单的例子。假设我们想计算当 $x=3$ 时，$y = x^2$ 相对于 $x$ 的梯度。

```python
import tensorflow as tf

# 创建一个标量 tf.Variable （微分需要浮点类型）
x = tf.Variable(3.0)

# 开始在磁带上记录操作
with tf.GradientTape() as tape:
  # 定义函数 y = x^2。此操作被记录。
  y = x * x # Or tf.square(x)

# 计算 y 相对于 x 的梯度
# dy_dx 将是当 x=3 时 2*x 的值，即 6.0
dy_dx = tape.gradient(y, x)

print(f"x: {x.numpy()}")
print(f"y = x^2: {y.numpy()}")
print(f"dy/dx: {dy_dx.numpy()}")

# 预期输出：
# x: 3.0
# y = x^2: 9.0
# dy/dx: 6.0
```

在这段代码中：

1. 我们将 `x` 定义为 `tf.Variable`。由于它默认是可训练的，`GradientTape` 将自动监视它。
2. 我们进入 `with tf.GradientTape() as tape:` 代码块。
3. 在代码块内部，执行 `y = x * x` 的计算。磁带会记录此操作，并知道 `y` 依赖于 `x`。
4. 在代码块之后，我们调用 `tape.gradient(y, x)`。磁带反向重播记录的操作，以计算目标（`y`）相对于源（`x`）的梯度，并应用链式法则。结果是 $\frac{dy}{dx} = 2x$，计算值为 $2 \times 3.0 = 6.0$。

### 非变量张量的梯度

如果您想计算相对于不是 `tf.Variable` 的普通 `tf.Tensor` 的梯度怎么办？默认情况下，磁带不监视张量。您需要使用 `tape.watch()` 明确告诉磁带监视它。

```python
import tensorflow as tf

# 创建一个常量张量
x0 = tf.constant(3.0)

with tf.GradientTape() as tape:
  # 手动告诉磁带监视此张量
  tape.watch(x0)
  # 定义计算
  y = x0 * x0

# 计算梯度
dy_dx0 = tape.gradient(y, x0)

print(f"x0: {x0.numpy()}")
print(f"y = x0^2: {y.numpy()}")
print(f"dy/dx0: {dy_dx0.numpy()}")

# 预期输出：
# x0: 3.0
# y = x0^2: 9.0
# dy/dx0: 6.0
```

尽管您*可以*监视常量，但通常梯度是相对于模型参数 (parameter)计算的，这些参数自然地表示为 `tf.Variable`。

### 相对于多个源的梯度

您可以通过向 `tape.gradient()` 传递源列表或元组，来计算目标相对于多个源（变量或被监视的张量）的梯度。它将返回一个梯度列表，其顺序与源相同。

```python
import tensorflow as tf

x = tf.Variable(2.0)
y = tf.Variable(3.0)

with tf.GradientTape() as tape:
  # z = x^2 + y^3
  z = tf.square(x) + tf.pow(y, 3)

# 计算 z 相对于 x 和 y 的梯度
# dz_dx = 2*x = 4.0
# dz_dy = 3*y^2 = 27.0
dz_dx, dz_dy = tape.gradient(z, [x, y])

print(f"x: {x.numpy()}, y: {y.numpy()}")
print(f"z = x^2 + y^3: {z.numpy()}")
print(f"dz/dx: {dz_dx.numpy()}")
print(f"dz/dy: {dz_dy.numpy()}")

# 预期输出：
# x: 2.0, y: 3.0
# z = x^2 + y^3: 31.0
# dz/dx: 4.0
# dz/dy: 27.0
```

### 控制监视对象

有时，您可能希望对监视哪些变量进行细粒度控制。可训练变量（`trainable=True`）默认被监视。您可以通过在创建磁带时设置 `watch_accessed_variables=False` 来阻止磁带监视它们。

```python
import tensorflow as tf

x0 = tf.Variable(2.0)
x1 = tf.Variable(2.0, trainable=False) # 不可训练
x2 = tf.Variable(2.0)
y = tf.Variable(3.0)

with tf.GradientTape(watch_accessed_variables=False) as tape:
  # 我们必须明确监视需要梯度的变量
  tape.watch(x0)
  tape.watch(x2)
  tape.watch(y)

  # z 依赖于 x0, x1, x2, y
  z = tf.square(x0) + tf.square(x1) * tf.square(x2) + tf.pow(y, 3) 

# 只计算被监视变量的梯度
# tape.gradient(z, x1) 将返回 None，因为 x1 未被监视。
dz_dx0, dz_dx2, dz_dy = tape.gradient(z, [x0, x2, y]) 

print(f"dz/dx0: {dz_dx0.numpy()}") # 2*x0 = 4.0
print(f"dz/dx2: {dz_dx2.numpy()}") # 2*x1^2*x2 = 2*(2^2)*2 = 16.0
print(f"dz/dy: {dz_dy.numpy()}")   # 3*y^2 = 3*(3^2) = 27.0

# 尝试获取未被监视的 x1 的梯度
dz_dx1 = tape.gradient(z, x1)
print(f"dz/dx1: {dz_dx1}") # 输出：None
```

### 持久化磁带

默认情况下，`GradientTape` 的资源在调用 `tape.gradient()` 方法后立即释放。这意味着每个磁带只能计算一次梯度。如果您需要计算多个梯度（例如，二阶导数，或不同目标相对于相同源的梯度），则必须通过设置 `persistent=True` 来创建一个**持久化**磁带。

```python
import tensorflow as tf

x = tf.Variable(3.0)

with tf.GradientTape(persistent=True) as tape:
  y = x * x      # y = x^2
  z = y * y      # z = y^2 = (x^2)^2 = x^4

# 计算 z 相对于 x 的梯度 (dz/dx = 4x^3 = 4*27 = 108)
dz_dx = tape.gradient(z, x)
print(f"dz/dx: {dz_dx.numpy()}")

# 计算 y 相对于 x 的梯度 (dy/dx = 2x = 6)
# 这之所以可行是因为磁带是持久化的
dy_dx = tape.gradient(y, x)
print(f"dy/dx: {dy_dx.numpy()}")

# 预期输出：
# dz/dx: 108.0
# dy/dx: 6.0

# 完成使用后，不要忘记删除磁带！
del tape
```

**重要提示：** 当使用持久化磁带时，您在使用完毕后**必须**手动使用 `del tape` 删除它，否则它所持有的资源（记录的操作）将不会被释放，可能导致内存泄漏。

### 梯度与控制流

`tf.GradientTape` 能正确处理像 `if` 语句和 `for` 循环这样的 Python 控制流。操作会按执行顺序被记录。

```python
import tensorflow as tf

x = tf.Variable(2.0)

with tf.GradientTape() as tape:
  if x > 0.0:
    y = tf.square(x) # 已执行，梯度为 2*x
  else:
    y = tf.negative(x) # 未执行

# dy/dx = 2*x = 4.0
dy_dx = tape.gradient(y, x)
print(f"dy/dx: {dy_dx.numpy()}") # 输出：4.0
```

### 不可微分的操作

如果您的计算涉及到相对于某个变量不可微分的操作（例如，使用 `tf.cast` 改变类型，或整数操作）会发生什么？`GradientTape` 对于此类梯度将返回 `None`。

```python
import tensorflow as tf

x = tf.Variable(2.0)

with tf.GradientTape() as tape:
  # 转换为整数使操作相对于 x 不可微分
  y = tf.cast(x, tf.int32) 
  # tf.GradientTape 无法通过离散操作（如转换为整数）计算梯度
  z = tf.cast(y, tf.float32) * 2.0 # 乘以浮点数以保持图连接

# 梯度路径被转换为 int32 的操作中断
dz_dx = tape.gradient(z, x)
print(f"dz/dx: {dz_dx}") # 输出：None
```

TensorFlow 无法通过本质上破坏微分所需连续连接的操作来计算梯度，例如转换为离散类型（`tf.int32`、`tf.bool`）或使用像 `tf.round` 这样的函数。

`tf.GradientTape` 是在 TensorFlow 中训练模型的根本。它允许框架自动计算更新模型参数 (parameter)所需的梯度，通过优化算法进行更新。理解它是如何记录操作和计算梯度的，对于构建、调试和自定义训练循环非常重要，我们将在后续章节中进一步讨论。

## 参考资料

- [tf.GradientTape](https://www.tensorflow.org/api_docs/python/tf/GradientTape) — TensorFlow Developers (2024)
  Publisher: TensorFlow (Google)
  `tf.GradientTape` 的官方 TensorFlow API 参考和使用指南，详细介绍了其功能和参数。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本实用指南，展示了 `tf.GradientTape` 在使用 TensorFlow 构建和训练各种机器学习模型时的应用。

---

[上一节](03-TensorFlow%20%E4%B8%AD%E7%9A%84%E5%8F%98%E9%87%8F.md) · [下一节](05-tf.function%20%E4%BB%8B%E7%BB%8D.md)
