---
course: "getting-started-with-tensorflow"
chapter: "data-input-pipelines-tfdata"
lesson: "working-tfrecord-files"
sourceId: 1762
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-tensorflow/chapter-5-data-input-pipelines-tfdata/working-tfrecord-files"
title: "使用 TFRecord 文件"
description: "使用 TensorFlow 推荐的 TFRecord 格式高效读写数据。"
order: 3
plots: []
sourceHash: "23528ab50ed1bd32f58307eaabcb2a6739fbef436f6a5a1e41f379cba9b56de5"
sourceCorrections: []
---

虽然直接从 NumPy 数组或 Python 生成器加载数据对于较小的数据集很有效，但当处理非常大的数据集时，如果它们无法完全放入内存或涉及从大量小文件缓慢加载数据，这种方式可能会变得效率低下。读取数据涉及文件打开、关闭和查找等额外开销，如果管理不当，这些开销可能会占据处理时间的大部分。

TensorFlow 提供了一种专门为其数据管道优化的文件格式：**TFRecord**。TFRecord 文件将你的数据存储为一系列二进制记录。将数据序列化为这种格式并顺序读回，可以大幅提高 I/O 性能，尤其是在处理存储在网络文件系统或机械硬盘上的大型数据集时。它是 TensorFlow 推荐的训练数据格式。

### TFRecord 文件的结构

TFRecord 文件（`.tfrecord` 或 `.tfrec`）本质上包含一系列变长二进制字符串。每个字符串代表一个数据记录（例如，一张图片及其标签）。TensorFlow 内部使用 Protocol Buffers (protobufs) 来组织这些记录。具体来说，每个记录通常是一个序列化的 `tf.train.Example` 消息。

`tf.train.Example` 是一个灵活的容器，设计用于存放键值对，键是字符串（特征名称），值是 `tf.train.Feature` 消息。

```
tf.train.Example {
  features: tf.train.Features {
    feature: map<string, tf.train.Feature> {
      "image_raw": tf.train.Feature { bytes_list: tf.train.BytesList {...} },
      "label":     tf.train.Feature { int64_list: tf.train.Int64List {...} },
      "height":    tf.train.Feature { int64_list: tf.train.Int64List {...} },
      "width":     tf.train.Feature { int64_list: tf.train.Int64List {...} },
      "caption":   tf.train.Feature { bytes_list: tf.train.BytesList {...} }
    }
  }
}
```

`tf.train.Feature` 消息本身可以包含以下三种列表类型之一：

1. `tf.train.BytesList`: 用于二进制字符串（如序列化图像数据）或 UTF-8 字符串。
2. `tf.train.FloatList`: 用于浮点数值（float32, float64）。
3. `tf.train.Int64List`: 用于整数值（布尔型、枚举型、int32、uint32、int64、uint64）。

这种结构允许你在单一记录格式中存储不同类型的数据。例如，一张图片可以存储为原始字节 (`BytesList`)，其标签存储为整数 (`Int64List`)，而边界框坐标则可能存储为浮点数 (`FloatList`)。

### 将数据写入 TFRecord 文件

要创建一个 TFRecord 文件，你需要将原始数据转换为 `tf.train.Example` 协议缓冲区，然后使用 `tf.io.TFRecordWriter` 将这些序列化的协议写入文件。

我们通过一个例子来说明，假设我们有图像数据（作为字节字符串）和相应的整数标签。

首先，我们需要辅助函数来轻松地从标准 Python/NumPy 类型创建 `tf.train.Feature` 消息：

```python
import tensorflow as tf
import numpy as np

# Helper functions to create tf.train.Feature
def _bytes_feature(value):
  """从字符串/字节返回一个 bytes_list。"""
  if isinstance(value, type(tf.constant(0))): # 如果它是一个张量
    value = value.numpy() # 获取其值
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """从浮点数/双精度数返回一个 float_list。"""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
  """从布尔型/枚举型/整型/无符号整型返回一个 int64_list。"""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

# 示例：假设你有图像字节和标签
# 替换为你的实际数据加载逻辑
image_bytes = tf.io.encode_jpeg(np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)).numpy()
label = 5 # 示例类别标签

# 创建一个将特征名称映射到 Feature 协议的字典
feature_dict = {
    'image_raw': _bytes_feature(image_bytes),
    'label': _int64_feature(label),
}

# 使用字典创建一个 Features 消息
features = tf.train.Features(feature=feature_dict)

# 使用 Features 消息创建一个 Example 消息
example_proto = tf.train.Example(features=features)

# 将 Example 消息序列化为二进制字符串
serialized_example = example_proto.SerializeToString()

# 现在，将此（以及可能许多其他）写入 TFRecord 文件
tfrecord_filename = 'data.tfrecord'

# 使用 tf.io.TFRecordWriter 写入序列化的示例
with tf.io.TFRecordWriter(tfrecord_filename) as writer:
  # 在循环中写入多个示例以处理真实数据集
  writer.write(serialized_example)
  # writer.write(another_serialized_example)
  # ...
```

在典型的工作流程中，你会迭代你的数据集（例如，图像文件路径和标签），加载每个样本，使用上面定义的结构将其转换为 `tf.train.Example`，然后将其序列化并写入 `TFRecordWriter`。你还可以将数据分割成多个 TFRecord 文件（分片），以便日后更好地并行读取。

### 从 TFRecord 文件读取数据

读回数据涉及使用 `tf.data.TFRecordDataset` 并解析序列化的 `tf.train.Example` 消息。

1. **创建 `TFRecordDataset`**：此数据集从一个或多个 TFRecord 文件中读取原始的、序列化的协议缓冲区字符串。

   ```python
   # 可以提供一个文件名列表，用于分片数据
   raw_dataset = tf.data.TFRecordDataset([tfrecord_filename])

   # raw_dataset 现在生成序列化的 tf.train.Example 字符串
   for raw_record in raw_dataset.take(1):
       print(repr(raw_record))
   ```
2. **定义解析函数**：由于数据集生成序列化的字符串，你需要将其解析回张量。这需要定义你期望在每个 `tf.train.Example` 中找到的数据结构。你创建一个 `feature_description` 字典，将特征名称映射到解析指令（`tf.io.FixedLenFeature` 用于固定大小的特征，或 `tf.io.VarLenFeature` 用于变长特征）。

   ```python
   # 定义你在写入时使用的特征结构
   feature_description = {
       'image_raw': tf.io.FixedLenFeature([], tf.string), # 0维字符串张量
       'label': tf.io.FixedLenFeature([], tf.int64),   # 0维 int64 张量
   }

   # 用于解析单个 tf.train.Example 协议的函数
   def _parse_function(example_proto):
     # 使用上述字典解析输入的 `tf.train.Example` 协议。
     return tf.io.parse_single_example(example_proto, feature_description)
   ```
3. **映射解析函数**：使用 `dataset.map()` 将解析函数应用于原始数据集中的每个元素。

   ```python
   parsed_dataset = raw_dataset.map(_parse_function)

   # parsed_dataset 现在生成张量字典
   for features in parsed_dataset.take(1):
     print(f"Label: {features['label'].numpy()}")
     # 图像仍是原始字节，需要解码
     image_tensor = tf.io.decode_jpeg(features['image_raw'])
     print(f"Image shape: {image_tensor.shape}")
   ```
4. **进一步处理（解码、预处理）**：通常，存储在 TFRecord 中的数据（如图像字节）需要进一步解码或预处理。你可以在 `map` 函数内部添加这些步骤，或者链接额外的 `map` 调用。

   ```python
   def _decode_and_preprocess(features):
       # 解码 JPEG 图像
       image = tf.io.decode_jpeg(features['image_raw'], channels=3)
       # 调整大小或应用其他预处理
       image = tf.image.resize(image, [128, 128])
       image = tf.cast(image, tf.float32) / 255.0 # 归一化
       label = features['label']
       return image, label

   # 应用解码和预处理
   processed_dataset = parsed_dataset.map(_decode_and_preprocess)

   # 现在数据集生成 (image_tensor, label_tensor) 元组
   for image, label in processed_dataset.take(1):
       print(f"Processed Image shape: {image.shape}, Label: {label.numpy()}")
   ```

### 与完整 `tf.data` 管道集成

真正的优势在于你将 TFRecord 读取与其他 `tf.data` 转换（如洗牌、批处理和预取）结合起来时：

```python
# 假设你有多个 TFRecord 文件（分片）
filenames = tf.data.Dataset.list_files("data_*.tfrecord")

# 交错读取多个文件以更好地洗牌
dataset = filenames.interleave(tf.data.TFRecordDataset,
                               cycle_length=tf.data.AUTOTUNE,
                               num_parallel_calls=tf.data.AUTOTUNE)

# 洗牌、解析、解码、预处理、批处理、预取
BUFFER_SIZE = 10000
BATCH_SIZE = 32

dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.map(_decode_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

print(dataset)

# 此数据集现在已准备好传递给 model.fit()
# model.fit(dataset, epochs=10)
```

使用 `tf.data.AUTOTUNE` 允许 TensorFlow 根据可用系统资源动态调整 map 操作的并行度以及预取缓冲区大小，从而简化性能优化。

### 何时使用 TFRecord

TFRecord 在以下情况下特别有益：

- 你的数据集太大，无法完全放入内存。
- 你正在通过网络或从较慢的存储读取数据，且顺序读取更快时。
- 你想为 TensorFlow 应用程序标准化你的数据存储格式。
- 你需要最大限度地提高输入管道吞吐量 (throughput)，以保持 GPU 或 TPU 繁忙。

虽然创建 TFRecord 文件会增加一个初始数据准备步骤，但训练期间潜在的性能提升，特别是对于大型数据集和长时间训练运行，通常值得付出。对于可以轻松放入内存的较小数据集，直接从 NumPy 数组加载可能更简单和足够。

## 参考资料

- [TFRecord and tf.Example](https://www.tensorflow.org/tutorials/load_data/tfrecord) — TensorFlow Authors (2024)
  TensorFlow关于在数据管道中使用TFRecord文件和`tf.train.Example`协议缓冲区进行高效数据存储的官方指南。
- [tf.data: Build TensorFlow input pipelines](https://www.tensorflow.org/guide/data) — TensorFlow Authors (2024)
  Publisher: TensorFlow
  全面的官方指南，涵盖`tf.data` API，包括`TFRecordDataset`、数据集转换、性能优化和`AUTOTUNE`等概念，用于构建稳健的数据输入管道。
- [Hands-On Machine Learning with Scikit-Learn, Keras, & TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media; Pages: Chapter 13
  一本广受认可的书籍，提供了TensorFlow数据管道的实用解释和代码示例，特别涵盖了TFRecord文件及其用法。
