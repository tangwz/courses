---
course: "getting-started-with-tensorflow"
chapter: "training-evaluating-models"
lesson: "training-model-fit"
sourceId: 1737
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-tensorflow/chapter-4-training-evaluating-models/training-model-fit"
title: "使用 model.fit() 进行训练"
description: "使用 fit 方法训练模型，理解轮次、批量大小和验证分割。"
order: 4
plots: []
sourceHash: "e77c438c76a7680e9c1c819fde8d3ceaeffa167004cdd7dea3e4861ad7efe4de"
sourceCorrections: []
---

`model.fit()` 方法用于在数据上训练机器学习 (machine learning)模型。此过程通常涉及一个已使用优化器、损失函数 (loss function)和评估指标配置的模型。该方法通过迭代数据集、计算损失、计算梯度，并使用选择的优化器更新模型的权重 (weight)来组织训练过程。

### `model.fit()` 方法

`model.fit()` 函数是 Keras 训练的核心功能。它接收训练数据、目标标签以及各种配置参数 (parameter)来管理学习过程。其核心在于，`fit` 会在指定的迭代次数（轮次）内重复执行以下步骤：

1. 将数据分成批次。
2. 对于每个批次：
   - 执行前向传播以生成预测。
   - 使用编译好的损失函数 (loss function)计算预测值与真实标签之间的损失。
   - 计算损失相对于模型可训练权重 (weight)的梯度。
   - 根据计算出的梯度，使用编译好的优化器更新权重。
3. 在训练数据上计算指定的评估指标。
4. （可选）在每个轮次结束时，在单独的验证数据集上评估评估指标（和损失）。

### 向 `fit()` 提供数据

你可以通过以下几种方式向 `model.fit()` 提供训练数据：

1. **NumPy 数组：** 对于能完全载入内存的小型数据集，最直接的方式是为输入特征 (`x`) 和目标标签 (`y`) 提供 NumPy 数组。
2. **TensorFlow 张量：** 类似于 NumPy 数组，你可以直接传递 TensorFlow 张量。
3. **`tf.data.Dataset` 对象：** 对于可能无法完全载入内存的大型数据集，或者当你需要复杂的输入处理（如预取、缓存或复杂转换）时，推荐使用 `tf.data.Dataset` 对象。我们将在下一章详细介绍 `tf.data`。

### 训练参数 (parameter)

我们来看看 `model.fit()` 中最重要的参数：

- `x`：输入数据。可以是 NumPy 数组、TensorFlow 张量或 `tf.data.Dataset`。如果它是一个数据集，则不应提供 `y`（因为标签预计会包含在数据集中）。
- `y`：目标数据（标签）。如果 `x` 是数组/张量，则 `y` 应为 NumPy 数组或 TensorFlow 张量。如果 `x` 是生成 `(features, labels)` 元组的 `tf.data.Dataset`，则省略此项。
- `batch_size`：一个整数，指定每次梯度更新的样本数量。训练通常以小批量方式进行，而不是一次性处理整个数据集。这提高了计算效率，并有助于优化过程更好地泛化。常见的批量大小范围从 32 到 256，但最佳值取决于数据集大小、模型复杂度和可用内存（更大的批量需要更多内存）。如果将数据作为 `tf.data.Dataset` 提供，批量处理理想情况下应由数据集本身处理，并且可以在 `fit` 中将 `batch_size` 设为 `None`。
- `epochs`：一个整数，定义学习算法将遍历*整个*训练数据集的次数。一个轮次意味着训练数据集中的每个样本都有机会更新内部模型参数。训练通常需要多个轮次。
- `validation_data`：用于在每个轮次结束时评估损失和任何模型指标的数据。这通常应是一个模型不参与训练的独立验证集。提供验证数据有助于你监测过拟合 (overfitting)。它通常作为 NumPy 数组或张量的元组 `(x_val, y_val)`，或作为 `tf.data.Dataset` 对象传递。
- `validation_split`：`validation_data` 的替代选项。一个介于 0 和 1 之间的浮点数。如果指定，`fit` 将自动从*训练*数据中保留此部分用于验证，并且不会在此部分数据上进行训练。分割发生在洗牌*之前*。这便于快速验证，但不如使用专用验证集那样可靠，特别是如果你的数据具有固有顺序。不能同时使用 `validation_data` 和 `validation_split`。
- `shuffle`：一个布尔值（当使用数组/张量数据时默认为 `True`），指示是否在每个轮次*之前*打乱训练数据。打乱有助于防止模型学习数据的顺序并提高泛化能力。当使用 `tf.data.Dataset` 时，打乱操作理想情况下应在数据集管道内处理（例如，使用 `dataset.shuffle()`）。
- `callbacks`：`keras.callbacks.Callback` 实例的列表。回调是在训练过程的不同阶段（例如，轮次结束时、批量开始时）调用的实用程序，用于执行保存模型、提前停止训练或记录到 TensorBoard 等操作。我们将在本章稍后更详细地讨论回调。

### 例子：使用 NumPy 数据进行训练

假设你有一个已编译的 `model`，训练数据 `x_train`、`y_train`，以及验证数据 `x_val`、`y_val`，它们都是 NumPy 数组。你可以这样开始训练：

```python
import tensorflow as tf
# 假设模型已经构建并编译
# 假设 x_train, y_train, x_val, y_val 都是 NumPy 数组

print("开始训练...")
history = model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=20,
    validation_data=(x_val, y_val)
)
print("训练完成。")

# 'history' 对象包含训练日志
print("每轮验证准确率:", history.history['val_accuracy'])
```

在这个例子中，模型将使用 64 个样本的小批量进行 20 个轮次的训练。训练数据将在每个轮次之前被打乱。每个轮次结束后，模型的损失和评估指标将在 `(x_val, y_val)` 上进行评估。

### 例子：使用 `tf.data.Dataset` 进行训练

如果你使用 `tf.data` 准备了数据，假设 `train_dataset` 生成 `(features, labels)` 元组并且已经批处理，并且你有一个类似的 `val_dataset`：

```python
import tensorflow as tf
# 假设模型已经构建并编译
# 假设 train_dataset 和 val_dataset 是 tf.data.Dataset 对象
# 其中 train_dataset.element_spec 是 (tf.TensorSpec(shape=(None, ...), dtype=tf.float32),
#                                     tf.TensorSpec(shape=(None, ...), dtype=tf.int32))
# 并且 train_dataset 已经进行了批处理和打乱。

print("开始使用 tf.data 进行训练...")
history = model.fit(
    train_dataset, # 无需 y 参数
    epochs=20,
    validation_data=val_dataset
    # 这里通常省略 batch_size，因为数据集处理了批处理
    # 这里通常省略 shuffle，因为数据集处理了打乱
)
print("训练完成。")

# 类似地访问历史记录
print("每轮验证损失:", history.history['val_loss'])
```

请注意，当使用 `tf.data.Dataset` 时，你通常不需要向 `fit` 提供 `y`、`batch_size` 或 `shuffle`，因为这些处理都在数据集管道内部进行。

### 返回值：History 对象

`model.fit()` 方法返回一个 `History` 对象。此对象有一个 `history` 属性，它是一个字典，包含每个轮次记录的损失和评估指标值。键是评估指标的名称（例如 `'loss'`、`'accuracy'`、`'val_loss'`、`'val_accuracy'`），值是包含每个轮次结束时评估指标值的列表。

这个历史记录对于分析训练过程非常有用，例如绘制学习曲线以诊断过拟合 (overfitting)或欠拟合 (underfitting)等问题。

```python
import matplotlib.pyplot as plt

# 假设 'history' 是 model.fit() 返回的对象
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(train_loss) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs_range, train_loss, label='训练损失', color='#1c7ed6') # 蓝色
plt.plot(epochs_range, val_loss, label='验证损失', color='#f76707') # 橙色
plt.legend(loc='upper right')
plt.title('训练和验证损失')
plt.xlabel('轮次')
plt.ylabel('损失')
plt.show()
```

> 一张图表显示训练损失稳定下降，而验证损失最初下降但随后开始增加，这是过拟合的典型迹象。

使用 `model.fit()` 是训练 Keras 模型的标准方式。通过理解其参数 (parameter)以及如何提供数据，你可以有效地管理各种机器学习 (machine learning)任务的训练循环。请记住通过 `History` 对象或回调函数密切关注验证指标，以构建高效的模型。

## 参考资料

- [Model.fit method](https://keras.io/api/models/model_training_apis/#fit-method) — Keras Team (2024)
  Keras中`model.fit()`方法的官方API文档，提供了所有参数的详细解释和使用示例。
- [Build TensorFlow input pipelines with tf.data](https://www.tensorflow.org/guide/data) — TensorFlow Authors (2024)
  Publisher: TensorFlow
  使用`tf.data.Dataset`构建高效数据输入管道的官方指南，这是为大型数据集的`model.fit()`提供数据的推荐方法。
- [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python-second-edition) — François Chollet (2021)
  Publisher: Manning Publications
  Keras创建者撰写的深度学习指南，涵盖了训练循环、`model.fit()`和相关概念。
- [Training and evaluation with Keras](https://www.tensorflow.org/guide/keras/train_and_evaluate) — fchollet (2023)
  Publisher: TensorFlow
  TensorFlow官方指南，概述了使用Keras的训练和评估工作流程，包括`model.fit()`的作用和用法。
