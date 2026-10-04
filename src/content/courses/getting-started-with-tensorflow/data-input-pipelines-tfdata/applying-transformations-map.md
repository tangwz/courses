---
course: "getting-started-with-tensorflow"
chapter: "data-input-pipelines-tfdata"
lesson: "applying-transformations-map"
sourceId: 1766
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-tensorflow/chapter-5-data-input-pipelines-tfdata/applying-transformations-map"
title: "应用转换：map()"
description: "使用 map 转换对数据集中的每个元素应用预处理函数。"
order: 4
plots: []
sourceHash: "5d1cf12205f5e39b01fc358faf158d8091fc875ff42a56d1ad6e6e1a6a8bb851"
sourceCorrections: []
---

`tf.data.Dataset` 对象是 TensorFlow 中数据输入管道的基础，可以从张量、NumPy 数组或 TFRecord 等文件等多种来源创建。获得此类数据集后，通常下一步是对数据进行预处理。这可能包括对数值特征进行归一化 (normalization)、调整图像大小、解码字节字符串，或者应用其他自定义逻辑来准备数据以供模型使用。为了实现这些预处理任务，`tf.data` API 提供了 `map()` 转换。

`map(map_func, num_parallel_calls=None)` 方法将给定函数 `map_func` 应用于数据集的每个元素，返回一个包含转换后元素的新数据集。这类似于标准 Python 中的 `map` 函数或 Pandas 等库中的操作，但其设计目的是在 TensorFlow 体系中高效运行。

### 基本使用

我们从一个简单示例开始。假设我们有一个数值数据集，并且我们希望通过对每个元素求平方来对其进行归一化 (normalization)。

```python
import tensorflow as tf
import numpy as np

# 从 NumPy 数组创建数据集
numeric_dataset = tf.data.Dataset.from_tensor_slices(np.arange(1, 6, dtype=np.float32))

# 定义一个简单的映射函数（可以是 lambda 或常规函数）
def square_element(x):
  return x * x

# 应用 map 转换
squared_dataset = numeric_dataset.map(square_element)

# 遍历转换后的数据集以查看结果
print("Original Dataset:")
for element in numeric_dataset:
  print(element.numpy())

print("\nSquared Dataset:")
for element in squared_dataset:
  print(element.numpy())
```

Output:

```
Original Dataset:
1.0
2.0
3.0
4.0
5.0

Squared Dataset:
1.0
4.0
9.0
16.0
25.0
```

传递给 `map()` 的函数接收一个参数 (parameter)，该参数表示数据集中的一个元素（它可能是一个张量，或一个张量元组/字典，具体取决于数据集的构建方式）。它应该返回转换后的元素。TensorFlow 会自动追踪此函数以构建图，从而实现高效执行。

### 应用 TensorFlow 操作

为了性能，应尽可能在映射函数内部使用 TensorFlow 操作 (`tf.*`)。这使得 TensorFlow 能够将预处理步骤直接集成到计算图中，可能会在 GPU 或 TPU 等加速器上运行它们，并避免 Python 和 TensorFlow 运行时之间代价高昂的数据传输。

考虑解码和调整存储为字节字符串的图像大小（这在从 TFRecord 文件读取时是一个常见情况）：

```python
# 假设我们有一个数据集，其中每个元素都是一个包含 JPEG 编码图像数据的标量字符串张量
# （这里我们进行模拟）
# 在实际情况中，这可能来自 tf.data.TFRecordDataset
jpeg_strings = [
    tf.random.uniform(shape=(), minval=0, maxval=255, dtype=tf.int32), # 模拟一些字节
    tf.random.uniform(shape=(), minval=0, maxval=255, dtype=tf.int32)
]
# 真实的图像字符串会更长且有结构。
# 我们使用虚拟字符串进行演示。
dummy_image_dataset = tf.data.Dataset.from_tensor_slices(["dummy_jpeg1", "dummy_jpeg2"])

def parse_and_resize_image(jpeg_string):
  # 假设 tf.io.decode_jpeg 适用于这些虚拟数据。
  # 实际上，您会有真实的 JPEG 字节。
  # image = tf.io.decode_jpeg(jpeg_string, channels=3)

  # 模拟解码到随机张量形状
  image = tf.random.uniform(shape=(200, 200, 3), maxval=256, dtype=tf.int32)
  image = tf.cast(image, tf.float32) # 转换为 float 类型以便处理

  # 调整图像大小
  image = tf.image.resize(image, [128, 128])

  # 归一化像素值（示例）
  image = image / 255.0
  return image

# 使用 map 应用预处理函数
processed_image_dataset = dummy_image_dataset.map(parse_and_resize_image)

# 检查第一个元素的形状
for img in processed_image_dataset.take(1):
    print("Processed image shape:", img.shape)
    # print("Sample pixel value:", img.numpy()[0,0,0]) # 值将在 0 到 1 之间
```

Output (shape will be consistent, value will vary):

```
处理后的图像形状: (128, 128, 3)
```

在这里，`tf.image.decode_jpeg`、`tf.image.resize` 和标准算术运算都是 TensorFlow 操作，确保高效的基于图的执行。

### 并行化转换

预处理有时计算量很大，特别是对于复杂操作或高分辨率图像等大数据元素。如果不同元素的预处理步骤是独立的，您可以并行化 `map` 转换以加快速度。

`num_parallel_calls` 参数 (parameter)控制同时处理的元素数量。将其设置为 `tf.data.AUTOTUNE` 允许 TensorFlow 在运行时根据可用的 CPU 资源动态调整并行级别，这通常是推荐的方法。

```python
# 使用并行调用应用图像处理函数
processed_image_dataset_parallel = dummy_image_dataset.map(
    parse_and_resize_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

# 迭代会产生相同的结果，但可能更快
# for img in processed_image_dataset_parallel.take(1):
#     print("Processed image shape (parallel):", img.shape)
```

使用 `tf.data.AUTOTUNE` 让 `tf.data` 运行时找到合适的并行级别，平衡吞吐量 (throughput)与资源使用，而无需手动调整。

> 该图显示了 `map` 如何通过 `num_parallel_calls` 参数使用提供的函数并行处理多个数据集元素。

### 使用 `tf.py_function` 处理 Python 逻辑

尽管 TensorFlow 操作是首选，但有时您需要在映射函数中使用任意 Python 代码，例如，调用没有 TensorFlow 等效项的库（比如某些音频处理或复杂字符串操作库）。在这种情况下，您可以使用 `tf.py_function` 包装您的 Python 函数。

`tf.py_function` 允许您将非 TensorFlow Python 代码嵌入 (embedding)到 TensorFlow 图中。但是，请注意以下影响：

1. **性能：** 由于在 TensorFlow 运行时和 Python 解释器之间切换，它会引入额外开销。它会破坏图的序列化（您无法轻松保存一个严重依赖 `py_function` 的模型图）。并行性可能会受到 Python 全局解释器锁 (GIL) 的限制。
2. **序列化：** 使用 `tf.py_function` 的模型可能无法轻松导出到没有 Python 解释器的环境中（例如某些配置下的 TensorFlow Lite 或 TensorFlow Serving）。

仅在必要时谨慎使用 `tf.py_function`。

```python
import cv2 # 示例：使用 OpenCV，一个非 TF 库

def python_based_processing(tensor_path):
  # 将张量路径（字节）转换为字符串
  path_str = tensor_path.numpy().decode('utf-8')
  # 使用 OpenCV（或任何 Python 库）
  # img = cv2.imread(path_str)
  # processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # 模拟一些处理，返回一个 numpy 数组
  processed_data = np.array([len(path_str)], dtype=np.int32) # 示例：路径长度
  return processed_data

# 假设数据集包含表示文件路径的字符串张量
paths_dataset = tf.data.Dataset.from_tensor_slices(["/path/to/file1.txt", "/path/to/file2.txt"])

def tf_wrapper_func(tensor_path):
  # 为 TensorFlow 定义输出类型和形状
  result = tf.py_function(
      func=python_based_processing,
      inp=[tensor_path],       # 输入张量
      Tout=tf.int32            # 输出 TensorFlow 数据类型
  )
  # 如果已知，显式设置形状，因为 py_function 会丢失形状信息
  result.set_shape([1]) # 示例输出形状
  return result

# 使用包装函数进行映射
processed_py_dataset = paths_dataset.map(tf_wrapper_func)

for item in processed_py_dataset:
    print(item.numpy())
```

Output:

```
[18]
[18]
```

`map` 转换是 `tf.data` 管道中准备数据的重要工具。通过高效地（可能并行地）应用 TensorFlow 操作，您可以创建整洁、高性能的预处理阶段，与模型训练顺畅结合。请记住，尽可能使用 TensorFlow 操作，并且仅在外部 Python 库必不可少时才使用 `tf.py_function`。

## 参考资料

- [Build TensorFlow input pipelines with tf.data](https://www.tensorflow.org/guide/data) — TensorFlow Authors (2023)
  一份官方指南，解释了如何使用`tf.data`构建高效的数据输入管道，包含`map()`等转换的实际示例和通用用法。
- [tf.data.Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) — TensorFlow Authors (2023)
  `tf.data.Dataset`的官方API文档，提供了其方法（包括`map()`）及其参数的详细信息。
- [Better performance with the tf.data API](https://www.tensorflow.org/guide/data_performance) — TensorFlow Authors (2023)
  本指南侧重于优化`tf.data`管道的性能，包括使用并行调用`map()`（`tf.data.AUTOTUNE`）的最佳实践以及`tf.py_function`的影响。
