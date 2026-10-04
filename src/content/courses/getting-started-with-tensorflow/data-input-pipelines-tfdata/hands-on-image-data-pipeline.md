---
course: "getting-started-with-tensorflow"
chapter: "data-input-pipelines-tfdata"
lesson: "hands-on-image-data-pipeline"
sourceId: 1782
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-tensorflow/chapter-5-data-input-pipelines-tfdata/hands-on-image-data-pipeline"
title: "动手实践：构建图像数据管道"
description: "练习创建用于加载、预处理和增强图像数据的完整 tf.data 数据管道。"
order: 9
plots: []
sourceHash: "3743990471f4194d8a21d29fcf850d777d4f96c5c926a4482690501c7bf739cc"
sourceCorrections: []
---

我们将使用 `tf.data` 构建一个完整的图像数据管道，从磁盘上的图像文件路径开始，最终得到准备好输入 Keras 模型的预处理图像张量批次。这个动手练习将帮助巩固对创建数据集、应用转换以及优化管道性能的理解。

我们假设您有一批按目录（可能按类别）组织的图像。例如，我们设想一个类似这样的结构：

```
data/
  train/
    class_a/
      image1.jpg
      image2.jpg
      ...
    class_b/
      image3.jpg
      image4.jpg
      ...
  validation/
    class_a/
      image101.jpg
      ...
    class_b/
      image102.jpg
      ...
```

我们的目标是创建一个 `tf.data.Dataset`，该数据集将读取这些 JPG 文件，解码它们，将它们调整为统一大小，应用一些基本的数据增强，打乱并分批处理它们，并使用预取。

### 准备工作

首先，请确保您已导入 TensorFlow。我们还将使用 `os` 模块，尽管 `tf.data.Dataset.list_files` 函数通常就足够了。

```python
import tensorflow as tf
import os
import matplotlib.pyplot as plt # 用于后续可视化

# 定义一些常量
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
BUFFER_SIZE = tf.data.AUTOTUNE # 对缓冲区大小使用 AUTOTUNE（自动调优）
```

### 步骤 1：创建文件路径数据集

第一步是获取所有要处理的图像文件列表。`tf.data.Dataset.list_files` 函数非常适合此目的。它接受文件模式（包括通配符 `*` 和 `**`），并返回匹配文件路径的数据集。对于训练集，将 `shuffle` 设置为 `True` 通常是个好主意。

```python
# 调整路径模式以匹配您的数据集结构
train_image_pattern = "data/train/*/*.jpg"

# 创建文件路径的初始数据集
list_ds = tf.data.Dataset.list_files(train_image_pattern, shuffle=True)

# 让我们查看几个文件路径
for f in list_ds.take(3):
  print(f.numpy())
```

> 注意：`list_files` 中的 `shuffle=True` 会在文件路径被处理 *之前* 打乱它们。我们将在图像处理后，稍后添加另一个打乱步骤。

### 步骤 2：定义处理函数

现在，我们需要函数来加载、解码、调整大小，并可选地增强图像。

**a) 加载和解码：**
此函数接收文件路径张量，读取文件内容，将其解码为 JPEG 图像，并调整其大小。我们还将像素值归一化 (normalization)到 `[0, 1]` 范围。

```python
def decode_img(img_path):
  # 读取原始文件内容
  img = tf.io.read_file(img_path)
  # 将 JPEG 文件解码为 uint8 张量
  img = tf.image.decode_jpeg(img, channels=3) # channels=3 表示 RGB
  # 将图像转换为 [0, 1] 范围的浮点数
  img = tf.image.convert_image_dtype(img, tf.float32)
  # 将图像调整为所需大小
  return tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
```

**b) (可选) 数据增强：**
让我们创建一个简单的增强函数，它随机水平翻转图像。更复杂的增强（亮度、对比度、旋转）可以在这里添加。

```python
def augment_img(image):
  image = tf.image.random_flip_left_right(image)
  # 如果需要，在这里添加更多增强
  # image = tf.image.random_brightness(image, max_delta=0.2)
  return image
```

**c) 组合处理函数：**
将解码和增强（如果使用）组合成一个单一的处理函数，然后将其映射到数据集上，通常很方便。如果您的数据集结构包含标签（例如，编码在路径中），您也将在此时提取标签。为了简单起见，我们将只关注图像管道。

```python
def process_path(file_path):
  img = decode_img(file_path)
  # 在训练期间应用增强
  # 对于验证/测试集，您可以跳过此步骤
  img = augment_img(img)
  return img # 在实际场景中，您会返回 (img, label)
```

### 步骤 3：映射处理函数

我们使用 `map()` 转换将 `process_path` 函数应用到数据集中的每个文件路径。使用 `num_parallel_calls=tf.data.AUTOTUNE` 允许 `tf.data` 自动调整并行处理级别，从而可能加快处理速度。

```python
# 将处理函数应用于数据集中的每个项
# 使用 AUTOTUNE 让 TensorFlow 优化并行处理
processed_ds = list_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
```

此时，`processed_ds` 是一个数据集，其中每个元素都是一个经过处理的图像张量（已调整大小、归一化 (normalization)和增强）。

### 步骤 4：配置性能和训练

为了准备用于模型训练的数据集，我们需要打乱已处理的图像，将它们分批，并进行预取。

- **`shuffle()`**: 随机化元素的顺序。缓冲区大小大于数据集大小可确保完全打乱，但会占用大量内存。常见的做法是使用一个相当大的缓冲区（例如 1000）。
- **`batch()`**: 将元素分组为批次。如果模型需要固定批次大小，`drop_remainder=True` 有时会很有用。
- **`prefetch()`**: 将数据预处理和模型执行重叠进行。它在模型处理当前批次时准备后续批次，从而显著减少 I/O 等待时间。`tf.data.AUTOTUNE` 允许 TensorFlow 选择最佳预取缓冲区大小。

顺序很重要：在分批 *之前* 打乱数据可确保每个 epoch 中元素在批次之间随机化。预取通常应该是 *最后* 一步。

```python
def configure_for_performance(ds):
  ds = ds.shuffle(buffer_size=BUFFER_SIZE) # 使用 AUTOTUNE 或像 1000 这样的固定大小
  ds = ds.batch(BATCH_SIZE)
  ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
  return ds

final_ds = configure_for_performance(processed_ds)
```

### 步骤 5：检查输出

让我们从最终数据集中取出一个批次，并检查其形状和值范围。

```python
# 从数据集中取出一个批次
image_batch = next(iter(final_ds))

# 检查形状 - 应该是 (BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, 3)
print(f"Batch shape: {image_batch.shape}")

# 检查值范围 - 应该大致在 [0, 1]
print(f"Min value: {tf.reduce_min(image_batch).numpy()}")
print(f"Max value: {tf.reduce_max(image_batch).numpy()}")

# 显示批次中的第一张图像
plt.figure(figsize=(6, 6))
plt.imshow(image_batch[0])
plt.title("Sample Image from Batch")
plt.axis("off")
plt.show()
```

如果形状是 `(32, 128, 128, 3)` (假设我们的常量设置) 并且值在 0 到 1 之间，那么我们的管道工作正常！显示的图像应该看起来像您的一张输入图像，可能由于增强而翻转了。

### 步骤 6：在 `model.fit` 中使用数据管道

这个 `final_ds` 对象现在可以直接传递给 Keras：

```python
# 假设 'model' 是一个已编译的 Keras 模型
# model.fit(final_ds, epochs=10)

# 同样用于评估：
# validation_ds = ... 为验证数据构建类似的管道 ...
# model.evaluate(validation_ds)
```

您已成功使用 `tf.data` 构建了一个高效的图像数据管道！这个管道处理文件发现、并行图像加载和预处理、增强、打乱、分批和预取，所有这些都与模型训练循环本身解耦。这种方法具有可伸缩性，对于在 TensorFlow 中处理大型图像数据集至关重要。如果监督学习 (supervised learning)任务需要，您可以根据目录结构或文件命名约定调整 `process_path` 函数以包含标签提取。

## 参考资料

- [Input pipelines with tf.data](https://www.tensorflow.org/guide/data) — TensorFlow Authors (2023)
  使用 tf.data API 构建高效数据输入管道的官方指南。
- [tf.image](https://www.tensorflow.org/api_docs/python/tf/image) — TensorFlow Authors (2023)
  TensorFlow 中图像处理和增强操作的官方 API 文档。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本内容丰富的书籍，提供 TensorFlow 的实际示例和详细解释，包含 tf.data 数据输入管道。
- [Better performance with the tf.data API](https://www.tensorflow.org/guide/data_performance) — TensorFlow Authors (2023)
  提供优化 tf.data 管道以提高训练性能的策略和实践的官方指南。
