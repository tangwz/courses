---
course: "getting-started-with-tensorflow"
chapter: "data-input-pipelines-tfdata"
lesson: "image-data-augmentation-tfdata"
sourceId: 1779
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-tensorflow/chapter-5-data-input-pipelines-tfdata/image-data-augmentation-tfdata"
title: "使用 tf.data 进行图像数据增强"
description: "在你的 tf.data 管道中执行常用图像增强技术。"
order: 8
plots: []
sourceHash: "05f307f81957ca0b3d8acea9e1b37c0c5fe89c2195a1c0f285843f4b15264130"
sourceCorrections: []
---

在训练图像模型，特别是深度神经网络 (neural network)时，拥有大量且多样的数据集对于获得良好的泛化能力和防止过拟合 (overfitting)通常很重要。过拟合是指模型对训练数据学习得太好，包括其噪声和特定模式，但在新的、未见过的数据上表现不佳。图像数据增强是一种常用方法，通过创建现有图像的修改版本来人工扩充训练数据集。

通过 `tf.data` 输入管道，可以即时进行图像数据增强。这避免了手动创建和存储大量修改后的图像（这会大幅增加存储需求）。这种方法节省内存，并允许增强操作与模型训练并行执行，从而利用CPU甚至GPU资源。

### 使用 `tf.image` 进行增强

TensorFlow 提供了 `tf.image` 模块，其中包含一组专门用于图像处理的函数，包括常用的增强方法。这些函数直接作用于张量，使它们可以自然地通过 `Dataset.map()` 转换集成到 `tf.data` 管道中。

让我们看看 `tf.image` 中一些常用的增强函数：

- **几何变换：**

  - `tf.image.random_flip_left_right(image)`: 随机地水平翻转图像（从左到右）。
  - `tf.image.random_flip_up_down(image)`: 随机地垂直翻转图像（从上到下）。注意：这可能不适用于所有数据集（例如，方向重要的图像，如数字）。
  - `tf.image.rot90(image, k)`: 将图像旋转90度 *k* 次。你可以使用 `tf.random.uniform` 选择一个随机的 *k* 值。更复杂的旋转通常需要外部库或自定义实现。
- **颜色和对比度调整：**

  - `tf.image.random_brightness(image, max_delta)`: 通过在 `[-max_delta, max_delta]` 范围内均匀选择的随机因子调整图像的亮度。`max_delta` 应为非负数。
  - `tf.image.random_contrast(image, lower, upper)`: 通过在 `[lower, upper]` 范围内均匀选择的随机因子调整图像的对比度。
  - `tf.image.random_saturation(image, lower, upper)`: 通过在 `[lower, upper]` 范围内均匀选择的随机因子调整RGB图像的饱和度。
  - `tf.image.random_hue(image, max_delta)`: 通过在 `[-max_delta, max_delta]` 范围内均匀选择的随机因子调整RGB图像的色调。`max_delta` 应在 `[0, 0.5]` 区间内。

这些操作需要输入图像张量，通常形状为 `[height, width, channels]` 或 `[batch, height, width, channels]`。当在 `Dataset.map()` 中使用时，它们会单独应用于流经管道的每个图像张量。

### 将增强集成到管道中

常用做法是 *只* 对训练数据集应用增强。验证和测试数据集应保持不变，以便对模型在未见过原始数据上的表现进行一致的衡量。

可以通过定义一个应用增强的函数，并 *只* 将其映射到训练数据集来实现。通常最好在图像解码和调整大小之后，但在批处理之前应用增强。

以下是包含随机增强的预处理函数的结构示例：

```python
import tensorflow as tf

# 假设 IMG_HEIGHT 和 IMG_WIDTH 已定义
IMG_HEIGHT = 180
IMG_WIDTH = 180

def decode_and_resize(image_bytes):
  """解码 JPEG，转换为浮点数，并调整大小。"""
  img = tf.io.decode_jpeg(image_bytes, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)
  img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
  return img

def augment_image(image, label):
  """对图像应用随机增强。"""
  image = tf.image.random_flip_left_right(image)
  image = tf.image.random_brightness(image, max_delta=0.2)
  image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
  # 根据需要添加更多 tf.image 增强
  # 确保图像值保持在有效范围，例如浮点图像的 [0, 1]
  image = tf.clip_by_value(image, 0.0, 1.0)
  return image, label

# 在 tf.data 管道中的使用示例：
# 假设 'train_files_ds' 是图像文件路径的 Dataset
# 假设 'load_and_preprocess_image' 读取文件路径、解码、调整大小
AUTOTUNE = tf.data.AUTOTUNE

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    # 为简化，假设标签可以推导或在其他地方配对
    # 实际上，你会适当地加载标签。
    label = 0 # 占位符标签
    return decode_and_resize(image), label

# 创建文件路径数据集（替换为你的实际数据源）
list_ds = tf.data.Dataset.list_files('path/to/images/*.jpg') # 示例

# 创建训练数据集
train_ds = list_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

# 仅对训练集应用增强
train_ds = train_ds.map(augment_image, num_parallel_calls=AUTOTUNE)

# 配置性能：打乱、批处理、预取
train_ds = train_ds.shuffle(buffer_size=1000)
train_ds = train_ds.batch(32) # 示例批处理大小
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

# 创建验证数据集（无增强）
# 假设 'val_files_ds' 是验证数据的来源
# val_ds = val_files_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
# val_ds = val_ds.batch(32)
# val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

print("Training Dataset Spec:", train_ds.element_spec)
# 输出可能如下所示：
# (TensorSpec(shape=(None, 180, 180, 3), dtype=tf.float32, name=None), TensorSpec(shape=(None,), dtype=tf.int32, name=None))
```

在此示例中：

1. 我们为基本预处理 (`decode_and_resize`) 和增强 (`augment_image`) 定义了独立的函数。
2. `load_and_preprocess_image` 函数处理读取、解码和调整大小。
3. `train_ds` 首先映射 `load_and_preprocess_image` 以获取处理后的图像。
4. 重要的是，`train_ds` 随后映射 `augment_image` 函数。
5. `val_ds` 将跳过 `augment_image` 映射步骤。
6. 之后应用了标准性能优化（`shuffle`、`batch`、`prefetch`）。在 `map` 函数中使用 `num_parallel_calls=tf.data.AUTOTUNE` 允许 `tf.data` 自动调整数据转换的并行级别，从而提高效率。

### 增强的考量

- **相关性：** 选择能够反映数据真实变化的增强。翻转手写数字“6”的图像可能会将其变成“9”，这通常是不希望的结果。然而，水平翻转猫的图像通常是可以接受的。调整亮度和对比度通常能模拟不同的光照条件。
- **顺序：** 应用增强的顺序有时很重要。例如，先旋转图像再裁剪，可能与先裁剪再旋转裁剪后的部分得到不同的结果。通过实验找到最佳方法。
- **强度：** 注意增强的强度（例如，亮度的 `max_delta`）。过于激进的增强有时会过度扭曲图像，使其失去代表性并可能损害训练。
- **性能：** 尽管 `tf.image` 操作已优化，但复杂的增强序列仍然会占用大量CPU资源。通过使用 `prefetch` 并监控输入管道性能（例如使用 TensorFlow Profiler），确保包含增强的数据管道能够跟上 GPU 的训练速度。对于计算密集型的增强，可以研究 KerasCV 等库，它们提供 GPU 加速的增强层，可以直接包含在模型定义中。

通过使用 `tf.image` 函数将图像数据增强直接集成到 `tf.data` 管道中，可以高效地准备和扩充训练数据，从而得到对新数据泛化能力更好的模型。请记住，要周全地应用这些转换，并考虑它们与你的特定任务和数据集的相关性。

## 参考资料

- [Build TensorFlow input pipelines](https://www.tensorflow.org/guide/data) — TensorFlow Developers (2023)
  Publisher: TensorFlow
  `tf.data` 创建高性能数据输入管道的官方指南，对于集成即时数据增强至关重要。
- [tf.image module](https://www.tensorflow.org/api_docs/python/tf/image) — TensorFlow Developers (2023)
  `tf.image` 函数的官方API文档，详细介绍了所使用的图像处理和增强操作。
- [A Survey on Data Augmentation for Deep Learning](https://doi.org/10.1016/j.neucom.2021.04.053) — Chen Long, Xiufang Wang, Qingyu Wang, Wei Li, and Yuan Xu (2021)
  Journal: Neurocomputing; Publisher: Elsevier; Volume: 450; Pages: 477-497; DOI: [10.1016/j.neucom.2021.04.053](https://doi.org/10.1016/j.neucom.2021.04.053)
  深度学习中数据增强技术的全面概述，为其应用和优点提供了广泛背景。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning-with-scikit-learn-keras-and-tensorflow/9781098122478/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media; Pages: 864
  一本关于使用 TensorFlow 进行深度学习的实用指南，提供了实现数据增强和其他重要技术的实际示例。
