---
course: "advanced-tensorflow"
chapter: "tensorflow-execution-graphs"
lesson: "automatic-differentiation-gradienttape"
sourceId: 2803
sourceUrl: "https://apxml.com/zh/courses/advanced-tensorflow/chapter-1-tensorflow-execution-graphs/automatic-differentiation-gradienttape"
title: "使用 tf.GradientTape 进行自动微分"
description: "掌握 tf.GradientTape 的用法，自动计算梯度，这对于模型训练非常重要。"
order: 5
plots: []
sourceHash: "b718f4b6c2348da65244341886cf40b8751e4f6afc24bce2e2ef600233439709"
sourceCorrections: []
---

优化机器学习 (machine learning)模型，特别是深度神经网络 (neural network)，取决于迭代调整模型参数 (parameter)以最小化损失函数 (loss function)。这一最小化过程通常依赖于基于梯度的优化算法，例如随机梯度下降 (gradient descent) (SGD) 及其变体 (Adam, RMSprop 等)。这些算法需要计算损失函数相对于模型参数的梯度。TensorFlow 为此提供了一个强大且灵活的机制：使用 `tf.GradientTape` 进行自动微分。

自动微分 (AD) 是一系列技术，用于数值计算由计算机程序指定的函数的导数。与符号微分（它操作数学表达式）或数值微分（它使用有限差分逼近导数）不同，AD 通过在程序执行期间，在基本操作级别系统地应用微积分链式法则，计算精确的导数。TensorFlow 主要使用 **反向模式 AD**，对于计算标量输出（如损失函数）相对于许多输入参数（如模型权重 (weight)）的梯度，这种方式效率极高，使其非常适合深度学习 (deep learning)。

## 使用 `tf.GradientTape` 记录操作

TensorFlow 中自动微分的核心 API 是 `tf.GradientTape`。它通过将其上下文 (context)内执行的 TensorFlow 操作“记录”到虚拟“磁带”上工作。TensorFlow 随后利用此磁带和链式法则计算梯度。

以下是基本使用模式：

1. 实例化一个 `tf.GradientTape` 并使用 `with` 块。
2. 在该块内，执行需要计算梯度的运算。TensorFlow 会自动追踪涉及可训练 `tf.Variable` 对象的运算。
3. 在该块之后，调用磁带的 `gradient()` 方法，传入目标张量（通常是损失）和源张量（通常是模型的可训练变量）来计算梯度。

让我们看一个简单的例子。假设我们有一个变量 $x$，并且想计算 $y = x^2$ 相对于 $x$ 的梯度：

```python
import tensorflow as tf

# 创建一个可训练变量
x = tf.Variable(3.0)

with tf.GradientTape() as tape:
  # 执行涉及该变量的运算
  y = x * x # or tf.square(x)

# 计算 y 相对于 x 的梯度
# dy/dx = 2*x = 2*3 = 6
dy_dx = tape.gradient(y, x)

print(dy_dx)
# tf.Tensor(6.0, shape=(), dtype=float32)
```

默认情况下，`tf.GradientTape` 只追踪涉及 `tf.Variable` 对象的运算。如果您需要计算相对于标准 `tf.Tensor` 的梯度，您必须使用 `tape.watch()` 明确告知磁带“观察”它：

```python
x = tf.constant(3.0) # 不是 Variable

with tf.GradientTape() as tape:
  # 明确观察这个张量
  tape.watch(x)
  y = x * x

dy_dx = tape.gradient(y, x)
print(dy_dx)
# tf.Tensor(6.0, shape=(), dtype=float32)
```

## “磁带”比喻

您可以将 `GradientTape` 上下文 (context)想象成记录了应用于被观察张量或变量的操作序列。当调用 `tape.gradient(target, sources)` 时，TensorFlow 会*反向*遍历此记录的序列（因此是反向模式 AD），在每一步应用链式法则，计算目标相对于指定源的梯度。

> `tf.GradientTape` 的流程：上下文中的操作被记录。调用 `gradient` 使用记录的磁带通过反向遍历计算导数。

## 为模型训练计算梯度

`tf.GradientTape` 的主要用途是在模型训练期间计算梯度。您根据模型的预测和真实标签计算损失，然后使用磁带来寻找该损失相对于模型可训练参数 (parameter)的梯度。

```python
# 假设一个简单的线性模型：y = W*x + b
# 创建一些模拟数据
x_input = tf.constant([[1.0, 2.0, 3.0]], dtype=tf.float32)
y_true = tf.constant([[10.0]], dtype=tf.float32)

# 定义模型变量（权重和偏置）
W = tf.Variable(tf.random.normal((3, 1)), name='weight')
b = tf.Variable(tf.zeros(1), name='bias')

# 在磁带上下文内定义模型和损失
with tf.GradientTape() as tape:
  # 正向传播：计算预测
  y_pred = tf.matmul(x_input, W) + b
  # 计算损失（例如，均方误差）
  loss = tf.reduce_mean(tf.square(y_pred - y_true))

# 计算损失相对于模型变量的梯度
# 源可以是变量列表或元组
gradients = tape.gradient(loss, [W, b])

print("Loss:", loss.numpy())
print("Gradient w.r.t. W:\n", gradients[0].numpy())
print("Gradient w.r.t. b:", gradients[1].numpy())
```

这些计算出的 `gradients` 正是 `tf.keras.optimizers.Adam` 或 `tf.keras.optimizers.SGD` 等优化器所需的，以正确的方向更新模型变量（`W` 和 `b`）来最小化 `loss`。这个过程构成了自定义训练循环的核心（您将在第 4 章中详细了解到）。

## 持久磁带与高阶梯度

默认情况下，`GradientTape` 在 `tape.gradient()` 被调用一次后，会立即释放保存已记录操作的资源。这使得它在每步计算一组梯度的常见情况下效率很高。

如果您需要在同一计算上计算多个梯度（例如，不同目标的梯度，或高阶导数），您可以通过设置 `persistent=True` 来创建一个**持久磁带**：

```python
x = tf.Variable(2.0)

with tf.GradientTape(persistent=True) as tape:
  y = x * x   # y = x^2
  z = y * y   # z = y^2 = (x^2)^2 = x^4

# 计算 z 相对于 x 的梯度 (dz/dx = 4x^3 = 4*2^3 = 32)
dz_dx = tape.gradient(z, x)
print("dz/dx:", dz_dx) # tf.Tensor(32.0, shape=(), dtype=float32)

# 计算 y 相对于 x 的梯度 (dy/dx = 2x = 2*2 = 4)
# 这是因为磁带是持久的
dy_dx = tape.gradient(y, x)
print("dy/dx:", dy_dx) # tf.Tensor(4.0, shape=(), dtype=float32)

# 记得在使用完持久磁带后手动删除它
# 以释放资源
del tape
```

持久磁带也支持计算**高阶梯度**。您可以通过嵌套 `GradientTape` 上下文 (context)或使用持久磁带，来计算梯度的梯度：

```python
x = tf.Variable(3.0)

with tf.GradientTape() as tape2:
  with tf.GradientTape() as tape1:
    y = x * x * x # y = x^3
  # 一阶导数：dy/dx = 3x^2
  dy_dx = tape1.gradient(y, x)
# 二阶导数：d/dx(dy/dx) = d/dx(3x^2) = 6x = 6*3 = 18
d2y_dx2 = tape2.gradient(dy_dx, x)

print("y:", y)         # tf.Tensor(27.0, shape=(), dtype=float32)
print("dy/dx:", dy_dx) # tf.Tensor(27.0, shape=(), dtype=float32)
print("d2y/dx2:", d2y_dx2) # tf.Tensor(18.0, shape=(), dtype=float32)
```

高阶导数在标准深度学习 (deep learning)训练中不那么常见，但在一些高级优化方法、物理信息神经网络 (neural network)和模型分析技术（如计算 Hessian 矩阵）中会用到。

## 梯度与控制流

`tf.GradientTape` 在即时执行模式下能自然地与 Python 控制流（如 `if`、`for`、`while`）配合工作。磁带会简单地记录下被执行的操作。

```python
x = tf.Variable(2.0)
y = tf.Variable(5.0)

with tf.GradientTape() as tape:
  if x > 1.0:
    result = x * y # 结果 = 2.0 * 5.0 = 10.0
  else:
    result = x + y

# 梯度将根据所采取的路径 (x * y) 计算
# d(结果)/dx = y = 5.0
# d(结果)/dy = x = 2.0
grads = tape.gradient(result, {'x': x, 'y': y})

print(grads['x']) # tf.Tensor(5.0, shape=(), dtype=float32)
print(grads['y']) # tf.Tensor(2.0, shape=(), dtype=float32)
```

当您使用 `tf.function` 装饰包含此类控制流的函数时，AutoGraph 会将 Python 控制流转换为 TensorFlow 图操作（如 `tf.cond` 和 `tf.while_loop`）。`tf.GradientTape` 也能够正确地与这些基于图的控制流机制配合使用，确保在优化图中通过条件分支和循环正确计算梯度。

## 处理不可微分操作

并非所有 TensorFlow 操作都是可微分的（例如，转换为整数类型的 `tf.cast`、`tf.round`、`tf.argmax`）。如果您尝试通过此类操作计算梯度，`tape.gradient()` 将对依赖于不可微分路径的梯度返回 `None`。

```python
x = tf.Variable(2.7)

with tf.GradientTape() as tape:
  # tf.round 是不可微分的
  y = tf.round(x)

grad = tape.gradient(y, x)
print(grad) # 输出: None
```

在构建模型或自定义计算时，了解这一点很重要。如果意外返回 `None` 梯度，回溯计算过程，以确保源和目标之间路径上的所有操作都是可微分的。对于 TensorFlow 不自动支持的操作需要定义梯度，或者需要覆盖现有梯度的情况，您可以使用 `tf.custom_gradient`。

`tf.GradientTape` 是 TensorFlow 2 中实现基于梯度学习的基础机制。它能够记录操作并通过反向模式自动微分计算梯度，与即时执行、`tf.function` 和控制流相结合，为简单和高度复杂的模型架构提供了所需的灵活性和性能。理解其工作原理对于调试训练过程以及构建自定义训练逻辑、层和模型非常重要，正如我们将在后续章节中看到的那样。

## 参考资料

- [tf.GradientTape guide](https://www.tensorflow.org/guide/autodiff) — TensorFlow Authors (2024)
  Publisher: TensorFlow
  TensorFlow官方指南，详细介绍了`tf.GradientTape`的API及其在自动微分中的用法，包括持久化tape和高阶梯度等细节。
- [Automatic Differentiation in Machine Learning: A Survey](https://www.jmlr.org/papers/volume18/17-468/17-468.pdf) — Atilim Gunes Baydin, Barak A. Pearlmutter, Alexey Andreyevich Radul, Jeffrey Mark Siskind, David Eugene Stewart, Zachary DeVito, James Bradbury (2018)
  Journal: Journal of Machine Learning Research; Publisher: Microtome Publishing; Volume: 18; Pages: 1-65; DOI: [10.5555/3342930.3342931](https://doi.org/10.5555/3342930.3342931)
  一篇全面的学术综述，详细介绍了自动微分方法、其发展历程及其在机器学习中的广泛应用，重点关注反向模式自动微分。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapters 6 & 8
  一本基础教材，全面解释了反向传播（应用于神经网络的反向模式自动微分的特定实例）和各种基于梯度的优化算法。其中第6章和第8章尤为相关。
