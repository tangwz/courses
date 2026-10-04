# 从张量、NumPy 和生成器创建数据集

来源：[原文](https://apxml.com/zh/courses/getting-started-with-tensorflow/chapter-5-data-input-pipelines-tfdata/creating-datasets-sources)

[返回章节目录](README.md) · [返回课程目录](../README.md)

创建 `tf.data.Dataset` 对象是使用 `tf.data` API 的基本第一步。该 API 提供了便捷方法，可以从您可能已在使用的常见结构中读取数据，例如内存中的张量、NumPy 数组，甚至标准 Python 生成器。这种灵活性让您可以快速构建数据管道，无论数据初始格式如何。

### 从张量和 NumPy 数组创建数据集

创建数据集最直接的方式是当您的数据已作为 TensorFlow 张量或 NumPy 数组存在于内存中时。这对于能轻松放入机器内存中的较小数据集，或者使用其他库刚生成或加载的数据来说很常见。

为此，主要函数是 `tf.data.Dataset.from_tensor_slices()`。此函数接收张量作为输入，并创建一个数据集，其中每个元素对应于输入张量第一维度的*切片*。

我们来看一个使用 NumPy 数组的简单示例：

```python
import tensorflow as tf
import numpy as np

# NumPy 数组示例
numpy_data = np.arange(10)
print(f"Original NumPy array: {numpy_data}")

# 从 NumPy 数组创建数据集
dataset_from_numpy = tf.data.Dataset.from_tensor_slices(numpy_data)

print("\nDataset elements:")
# 迭代数据集以查看切片
for element in dataset_from_numpy:
  # 每个元素都是一个 tf.Tensor
  print(element.numpy())
```

**输出：**

```
Original NumPy array: [0 1 2 3 4 5 6 7 8 9]

Dataset elements:
0
1
2
3
4
5
6
7
8
9
```

如您所见，`from_tensor_slices` 将一维 NumPy 数组 `[0, 1, ..., 9]` 视为 10 个独立元素的集合，创建了一个逐个生成每个数字的数据集。

同样适用于 TensorFlow 张量：

```python
# TensorFlow 张量示例
tensor_data = tf.range(5, 10)
print(f"Original TensorFlow tensor: {tensor_data.numpy()}")

# 从张量创建数据集
dataset_from_tensor = tf.data.Dataset.from_tensor_slices(tensor_data)

print("\nDataset elements:")
for element in dataset_from_tensor:
    print(element.numpy())
```

**输出：**

```
Original TensorFlow tensor: [5 6 7 8 9]

Dataset elements:
5
6
7
8
9
```

`from_tensor_slices` 的一个重要用途是处理成对数据，例如特征和对应标签。您可以传入张量或 NumPy 数组的元组（或字典），`tf.data` 将它们一起切片，确保对齐 (alignment)。

```python
# 示例特征（例如，测量值）和标签（例如，类别）
features = np.array([[1, 2], [3, 4], [5, 6]])
labels = np.array([0, 1, 0])

print(f"Features:\n{features}")
print(f"Labels: {labels}")

# 从 NumPy 数组元组创建数据集
dataset_features_labels = tf.data.Dataset.from_tensor_slices((features, labels))

print("\nDataset elements (feature, label pairs):")
for feature_element, label_element in dataset_features_labels:
  print(f"Feature: {feature_element.numpy()}, Label: {label_element.numpy()}")
```

**输出：**

```
Features:
[[1 2]
 [3 4]
 [5 6]]
Labels: [0 1 0]

Dataset elements (feature, label pairs):
Feature: [1 2], Label: 0
Feature: [3 4], Label: 1
Feature: [5 6], Label: 0
```

请注意，数据集生成的每个元素现在都是一个元组，其中包含来自 `features` 数组的一个切片和来自 `labels` 数组的对应切片。这种结构正是监督学习 (supervised learning)任务中经常需要的。

区分 `from_tensor_slices` 和 `tf.data.Dataset.from_tensors()` 很重要。后者创建一个只包含一个*单个*元素的数据集，该元素就是输入张量本身，而不是对其进行切片。

```python
# 使用 from_tensors
single_element_dataset = tf.data.Dataset.from_tensors(features)

print("\nDataset created with from_tensors:")
for element in single_element_dataset:
    print("Element shape:", element.shape)
    print(element.numpy())
```

**输出：**

```
Dataset created with from_tensors:
Element shape: (3, 2)
[[1 2]
 [3 4]
 [5 6]]
```

当您希望将整个张量结构视为数据集中的一个项时，例如用于后续的批处理或处理，请使用 `from_tensors`。当您希望迭代数据的各个行（或沿着第一维度的切片）时，请使用 `from_tensor_slices`。

### 从 Python 生成器创建数据集

有时您的数据并非随时可用作张量或 NumPy 数组。它可能是在运行时动态生成，从 TensorFlow 不直接支持的源（如自定义文件格式或数据库）读取，或者需要复杂的 Python 逻辑进行创建或预处理，这些逻辑难以纯粹用 TensorFlow 操作来表达。在这种情况下，您可以使用 Python 生成器函数。

`tf.data.Dataset.from_generator()` 方法连接了 Python 代码执行和 TensorFlow 图之间的桥梁。它允许您包装一个 Python 生成器函数，并将其生成的项转换为 `tf.data.Dataset`。

以下是基本结构：

1. **定义 Python 生成器：** 创建一个标准的 Python 生成器函数（使用 `yield`），逐个生成您的数据项。
2. **指定输出签名：** 告知 TensorFlow 您的生成器生成的项的结构和数据类型。这必不可少，因为 TensorFlow 需要在运行 Python 代码*之前*构建图，并且必须知道预期的输出签名（类型和形状）才能做到这一点。这通过 `output_signature` 参数 (parameter)完成，通常使用 `tf.TensorSpec` 定义。
3. **创建数据集：** 调用 `tf.data.Dataset.from_generator()`，传入您的生成器函数和 `output_signature`。

我们来看一个生成器生成递增数字序列的示例：

```python
import tensorflow as tf
import itertools # For generating sequences

# 1. 定义 Python 生成器
def count_generator(stop):
  """生成序列 [0], [0, 1], [0, 1, 2], ... 直到 stop"""
  for i in range(1, stop + 1):
    # 生成序列作为列表或 NumPy 数组
    sequence = np.arange(i)
    yield sequence # 生成器生成 NumPy 数组

# 2. 定义输出签名
# 由于序列长度可变，形状维度使用 None
output_signature = tf.TensorSpec(shape=(None,), dtype=tf.int64)

# 3. 创建数据集
# 使用 lambda 表达式将 'stop' 参数传递给生成器
stop_value = 5
dataset_from_generator = tf.data.Dataset.from_generator(
    lambda: count_generator(stop_value), # 使用 lambda 传递参数
    output_signature=output_signature
)

print("\nDataset elements from generator:")
for element in dataset_from_generator:
  print(element.numpy())
```

**输出：**

```
Dataset elements from generator:
[0]
[0 1]
[0 1 2]
[0 1 2 3]
[0 1 2 3 4]
```

**关于 `from_generator` 的要点：**

- **灵活性：** 它允许将任意 Python 逻辑融入您的数据管道。
- **`output_signature`：** 这是强制且重要的。`tf.TensorSpec(shape=..., dtype=...)` 准确描述了每个生成的元素。对于大小可变的维度，使用 `None`。如果您的生成器生成多个项（如特征和标签），请提供一个 `tf.TensorSpec` 对象的元组或字典。
- **性能：** 由于 `from_generator` 涉及运行 Python 代码（这可能比原生 TensorFlow 操作慢，并受限于 Python 的全局解释器锁），它的性能可能不如 `from_tensor_slices` 或从 TFRecord 文件读取，尤其是对于简单的转换。然而，对于复杂的数据生成或从不支持的源读取数据，它是一个非常有用的工具。TensorFlow 内部在 `tf.py_function` 中运行生成器代码。

对内存中的数据使用 `from_tensor_slices`，对基于 Python 的自定义数据源使用 `from_generator`，这两种方式能让您创建 `tf.data.Dataset` 对象，并为我们接下来将研究的转换和优化做好了准备。

## 参考资料

- [tf.data.Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) — TensorFlow Team (2024)
  tf.data.Dataset的官方API参考，详细介绍了从张量、NumPy数组和Python生成器创建数据集的方法。
- [Better performance with the tf.data API](https://www.tensorflow.org/guide/data_performance) — TensorFlow Team (2024)
  官方指南，解释了tf.data API的架构和性能考量，为高效数据输入管道提供了背景信息。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning-with/9781098125974/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本关于使用TensorFlow进行机器学习的全面实践指南，包括对tf.data API构建数据输入管道的详细介绍。第四版。

---

[上一节](01-%E4%B8%BA%E4%BB%80%E4%B9%88%E9%80%89%E6%8B%A9%20tf.data%EF%BC%9F.md) · [下一节](03-%E4%BD%BF%E7%94%A8%20TFRecord%20%E6%96%87%E4%BB%B6.md)
