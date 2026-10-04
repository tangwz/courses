---
course: "getting-started-with-tensorflow"
chapter: "core-tensorflow-concepts"
lesson: "variables-tensorflow"
sourceId: 1694
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-tensorflow/chapter-2-core-tensorflow-concepts/variables-tensorflow"
title: "TensorFlow 中的变量"
description: "了解 `tf.Variable` 如何保存模型权重等可变状态。"
order: 3
plots: []
sourceHash: "0dc9fba43b4f56ddc99368afe1affa3130e43886539dee6718cbcc157a664915"
sourceCorrections: []
---

TensorFlow 使用 `tf.Tensor` 对象来表示数据。然而，这些对象有一个特性使其不适用于某些任务：它们是不可变的。一旦创建，`tf.Tensor` 的值就无法更改。可以将它们视为 Python 中的数字或字符串；执行 `a = a + 1` 这样的操作不会修改原始的 `a`，而是创建一个表示结果的*新*对象，并将变量名 `a` 重新指向它。

然而，在机器学习 (machine learning)中，我们训练模型时需要持续更新模型参数 (parameter)，例如权重 (weight)和偏置 (bias)。我们需要一种以可修改的方式保存状态的方法。这正是 `tf.Variable` 的作用。

`tf.Variable` 代表一个可变张量，其值可以通过运行操作进行更改。变量是表示模型操作的共享、持久状态的标准方式，主要是可训练参数。在底层，`tf.Variable` 存储一个持久张量，并使其值在 TensorFlow 图或即时执行环境中可用于操作。

### 创建变量

您可以通过提供一个初始值来创建 `tf.Variable`，该值可以是 Python 标量、列表、NumPy 数组或现有 `tf.Tensor`。TensorFlow 会从初始值推断数据类型 (`dtype`)，但您也可以显式指定它。

```python
import tensorflow as tf
import numpy as np

# 创建一个标量变量
scalar_var = tf.Variable(5.0, dtype=tf.float32)
print(f"Scalar Variable: {scalar_var}")

# 从列表中创建一个向量变量
vector_var = tf.Variable([1.0, 2.0, 3.0], name="my_vector") # 可选名称
print(f"Vector Variable: {vector_var}")

# 从 NumPy 数组创建一个矩阵变量
matrix_var = tf.Variable(np.array([[1, 2], [3, 4]], dtype=np.int32))
print(f"Matrix Variable:\n{matrix_var}")

# 从另一个张量创建一个变量
initial_tensor = tf.zeros((2, 2))
tensor_var = tf.Variable(initial_tensor)
print(f"Variable from Tensor:\n{tensor_var}")
```

您会注意到，打印 `tf.Variable` 会显示其形状、数据类型和名称（如果提供），类似于 `tf.Tensor`。`trainable` 参数 (parameter)（默认为 `True`）特别重要；它告诉 TensorFlow 在自动微分期间是否应考虑该变量的值，我们很快就会介绍它。不可训练变量可用于跟踪步数等统计信息。

### 在操作中使用变量

变量可以在 TensorFlow 操作中使用，很像 `tf.Tensor` 对象。对变量执行操作会隐式读取其当前值。

```python
# 在操作中使用变量
result_tensor = scalar_var * 2.0 + vector_var[0]
print(f"\nOperation result (Tensor): {result_tensor}")

# 变量可以被重塑（创建新的张量）
reshaped_tensor = tf.reshape(matrix_var, (4,))
print(f"Reshaped Variable (Tensor): {reshaped_tensor}")

# 访问底层张量值
print(f"\nVariable value as NumPy: {vector_var.numpy()}")
```

需要指出的是，涉及变量的操作通常返回新的 `tf.Tensor` 对象，而不是变量。原始变量保持不变，保存其当前状态。

### 修改变量值

`tf.Variable` 的决定性特征是其可变性。您可以使用多种方法更改变量所持有的值，最常用的是 `.assign()`、`.assign_add()` 和 `.assign_sub()`。

- `variable.assign(new_value)`: 替换变量的全部值。
- `variable.assign_add(increment)`: 向变量添加一个值（相当于 `+=`）。
- `variable.assign_sub(decrement)`: 从变量中减去一个值（相当于 `-=`）。

这些赋值操作它们本身就是 TensorFlow 操作，在使用 `tf.function` 时会成为计算图的一部分。

```python
print(f"Original scalar_var: {scalar_var.numpy()}")

# 赋值新值
scalar_var.assign(10.0)
print(f"After assign(10.0): {scalar_var.numpy()}")

# 增加值
scalar_var.assign_add(2.5)
print(f"After assign_add(2.5): {scalar_var.numpy()}")

# 减少值
scalar_var.assign_sub(1.0)
print(f"After assign_sub(1.0): {scalar_var.numpy()}")

# assign 可以与兼容形状的张量一起使用
new_vector_val = tf.constant([4.0, 5.0, 6.0])
vector_var.assign(new_vector_val)
print(f"\nVector variable after assign: {vector_var.numpy()}")
```

尝试以类似方式修改 `tf.Tensor` 会导致错误，这突出了它们预期用途的根本区别。

### 变量与自动微分

修改变量的能力与模型训练密切相关。在训练过程中，梯度下降 (gradient descent)等算法需要计算模型参数 (parameter)（变量）的微小变化如何影响模型的误差（损失函数 (loss function)）。TensorFlow 的自动微分引擎 `tf.GradientTape` 旨在自动跟踪涉及可训练 `tf.Variable` 对象的计算。

当您在 `tf.GradientTape` 上下文 (context)内执行操作时，TensorFlow 会记录涉及可训练变量的操作。然后可以使用该记录器计算某个目标（如损失）相对于这些变量的梯度。这些梯度信息随后被优化器用于更新变量值，使模型表现更好。

```python
# 梯度跟踪示例
var_a = tf.Variable(2.0)
var_b = tf.Variable(3.0)

with tf.GradientTape() as tape:
  # 涉及变量的操作会被跟踪
  y = var_a * var_a * var_b # y = a^2 * b

# 计算 y 相对于 var_a 和 var_b 的梯度
# dy/da = 2*a*b = 2*2*3 = 12
# dy/db = a^2 = 2^2 = 4
gradients = tape.gradient(y, [var_a, var_b])

print(f"\ny = {y.numpy()}")
print(f"Gradient dy/da: {gradients[0].numpy()}")
print(f"Gradient dy/db: {gradients[1].numpy()}")

# 如果变量被标记为不可训练，则不会跟踪梯度
non_trainable_var = tf.Variable(5.0, trainable=False)
with tf.GradientTape() as tape:
    z = non_trainable_var * 2.0
gradients_z = tape.gradient(z, [non_trainable_var])
print(f"\nGradient for non-trainable variable: {gradients_z[0]}") # 输出为 None
```

这种自动跟踪和梯度计算是 TensorFlow 中训练大多数深度学习 (deep learning)模型的驱动力。

### 总结：`tf.Tensor` 与 `tf.Variable`

| 特性 | `tf.Tensor` | `tf.Variable` |
| --- | --- | --- |
| **可变性** | 不可变 | 可变 |
| **用途** | 表示固定数据、中间计算结果 | 表示可训练参数 (parameter)、共享状态 |
| **修改** | 无法原地更改 | 使用 `.assign()`、`.assign_add()` 等方法修改 |
| **梯度** | 默认不被 `GradientTape` 跟踪 | 如果 `trainable=True` 则被 `GradientTape` 跟踪 |

本质上，`tf.Tensor` 用于输入数据和中间计算结果。`tf.Variable` 用于模型中需要持久化并在多个计算步骤中更新的任何参数或状态，特别是将在训练期间调整的模型权重 (weight)和偏置 (bias)。理解这种区别是有效构建和训练模型的基础。现在，我们将了解 TensorFlow 如何使用 `tf.GradientTape` 计算对优化这些变量有重要作用的梯度。

## 参考资料

- [Variables: Create, Update, and Save](https://www.tensorflow.org/guide/variables) — TensorFlow Authors (2024)
  这份官方指南解释了 `tf.Variable` 的创建、在操作中的使用、通过赋值方法进行修改，以及其在模型参数自动微分中的作用。
- [Tensors](https://www.tensorflow.org/guide/tensor) — TensorFlow Authors (2024)
  提供 `tf.Tensor` 对象的基础信息，`tf.Tensor` 作为与 `tf.Variable` 对比的不可变数据结构。
- [Intro to auto-differentiation](https://www.tensorflow.org/guide/autodiff) — TensorFlow Authors (2024)
  Publisher: Google
  详细说明了 TensorFlow 的自动微分引擎 `tf.GradientTape` 如何跟踪涉及 `tf.Variable` 对象进行梯度计算的运算，以用于训练。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本全面的教材，为深度学习模型中使用可变参数（变量）及其在训练期间的优化奠定了理论基础。
