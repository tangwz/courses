---
course: "getting-started-with-tensorflow"
chapter: "data-input-pipelines-tfdata"
lesson: "integrating-tfdata-modelfit"
sourceId: 1775
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-tensorflow/chapter-5-data-input-pipelines-tfdata/integrating-tfdata-modelfit"
title: "将 tf.data 与 model.fit() 结合使用"
description: "如何将 tf.data.Dataset 对象直接传递给 Keras 的 fit、evaluate 和 predict 方法。"
order: 7
plots: []
sourceHash: "27b0a71f8235e098179e6f0e1ead7b7aa02bd1b79b9a48be0d0f1ded051aa928"
sourceCorrections: []
---

`tf.data` API 构建高效的输入管道，将原始数据源转换为经过混洗、批处理和预取的数据流，以供使用。这些管道与 Keras 训练架构集成。Keras API，特别是 `model.fit()`、`model.evaluate()` 和 `model.predict()` 方法，设计为直接与 `tf.data.Dataset` 对象配合使用，这使得集成过程简单高效。

当您将 `tf.data.Dataset` 对象传递给 `model.fit()` 时，Keras 会自动遍历数据集，为每个训练步骤获取数据批次。这避免了手动批处理迭代循环的需要，并与 Keras 的回调等功能良好结合。

### 组织用于训练和评估的数据集

对于使用 `model.fit()` 进行训练，您的数据集通常会生成形式为 `(inputs, targets)` 的元组。Keras 期望数据集迭代器生成的每个元素代表一个数据批次。

- **`inputs`**: 这可以是一个单张量（用于单输入模型）或张量元组/字典（用于多输入模型）。其结构必须匹配模型的输入签名。
- **`targets`**: 类似地，这可以是一个单张量或张量元组/字典，对应于模型的输出和正在使用的损失函数 (loss function)。

如果您的数据集生成诸如 `(feature_batch, label_batch)` 的批次，Keras 会将 `feature_batch` 正确映射到模型的输入，并将 `label_batch` 映射到预期输出以计算损失。

考虑一个使用 `tf.data.Dataset.from_tensor_slices((features, labels))` 等方法创建的数据集 `train_dataset`，随后进行 `.shuffle()`、`.batch()` 和 `.prefetch()` 操作。您可以直接将此数据集传递给 `model.fit()`：

```python
# 假设 'model' 是一个已编译的 Keras 模型
# 假设 'train_dataset' 生成 (features_batch, labels_batch) 元组
# 假设 'val_dataset' 生成 (features_batch, labels_batch) 元组

history = model.fit(train_dataset, epochs=10, validation_data=val_dataset)
```

Keras 会自动处理迭代，将批次数据馈送给训练过程。同样适用于 `model.evaluate()`：

```python
loss, accuracy = model.evaluate(val_dataset)
print(f"验证损失: {loss}, 验证准确率: {accuracy}")
```

对于 `model.predict()`，数据集应只生成输入特征。如果数据集生成 `(inputs, targets)` 元组，Keras 在预测时会简单地忽略 `targets` 部分。

```python
# 假设 'test_dataset' 只生成特征批次，或 (features, ...) 元组
predictions = model.predict(test_dataset)
```

### 处理数据集大小：`steps_per_epoch` 和 `steps` 参数 (parameter)

使用 `tf.data` 时，您经常会处理长度可能不易预先确定的数据集，特别是如果您使用诸如 `repeat()`（用于无限循环数据）的转换，或者数据集源自生成器。

- **有限数据集：** 如果 Keras 可以确定数据集的基数（批次数量）（例如，从 NumPy 数组或不带 `repeat()` 的 TFRecord 文件创建），它将自动在每个 epoch 中运行完整个数据集一次。您不需要指定步数。
- **无限数据集或未知基数：** 如果您的数据集是无限的（例如，使用 `.repeat()`）或其大小无法确定，Keras 不知道一个 epoch 何时结束。在这种情况下，您**必须**为 `model.fit()` 提供 `steps_per_epoch` 参数。这个整数值告诉 Keras 从数据集中抽取多少批次来构成一个训练 epoch。

```python
# 创建一个数据集并无限重复
train_dataset_repeated = train_dataset.repeat()

# 定义构成一个 epoch 的批次数量
STEPS_PER_EPOCH = num_training_samples // BATCH_SIZE # 示例计算

history = model.fit(train_dataset_repeated,
                    epochs=10,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_data=val_dataset) # val_dataset 通常是有限的
```

类似地，`model.evaluate()` 和 `model.predict()` 接受 `steps` 参数。如果您向这些方法传递一个未知基数或无限的数据集，您必须指定 `steps` 参数，以指明应该使用多少批次进行评估或预测。如果数据集是有限的且未提供 `steps`，它们将运行直到数据集耗尽。

```python
# 在验证集上评估指定数量的批次
EVALUATION_STEPS = num_validation_samples // BATCH_SIZE # 示例计算

loss, accuracy = model.evaluate(val_dataset, steps=EVALUATION_STEPS)

# 在测试集上预测指定数量的批次
PREDICTION_STEPS = num_test_samples // BATCH_SIZE # 示例计算

predictions = model.predict(test_dataset, steps=PREDICTION_STEPS)
```

为 `steps_per_epoch` 选择正确的值很重要。一种常用做法是将其设置成模型在每个 epoch 大致能看到相当于整个训练数据集一次的量：`steps_per_epoch = total_training_samples // batch_size`。

### 综合应用：一个实用例子

让我们用一个使用 NumPy 数据的简单例子来说明。

```python
import tensorflow as tf
import numpy as np

# 1. 生成一些虚拟数据
num_samples = 1000
input_dim = 10
num_classes = 2
batch_size = 32

X_train = np.random.rand(num_samples, input_dim).astype(np.float32)
y_train = np.random.randint(0, num_classes, size=num_samples).astype(np.int32)

X_val = np.random.rand(200, input_dim).astype(np.float32)
y_val = np.random.randint(0, num_classes, size=200).astype(np.int32)

# 2. 创建 tf.data 数据集
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=num_samples).batch(batch_size).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# 3. 构建一个简单的 Keras 模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax') # 多分类问题使用 softmax
])

# 4. 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', # 整数标签使用 sparse
              metrics=['accuracy'])

# 5. 使用数据集训练模型
print("正在使用 tf.data.Dataset 训练模型...")
# 数据集是有限的，因此不需要 steps_per_epoch
history = model.fit(train_dataset, epochs=5, validation_data=val_dataset)
print("训练完成。")

# 6. 评估模型
print("\n正在评估模型...")
loss, accuracy = model.evaluate(val_dataset) # 这里也不需要 steps
print(f"验证损失: {loss:.4f}, 验证准确率: {accuracy:.4f}")

# 7. 进行预测（为简单起见，使用派生自验证数据的数据集）
pred_dataset = tf.data.Dataset.from_tensor_slices(X_val).batch(batch_size)
print("\n正在进行预测...")
predictions = model.predict(pred_dataset)
print(f"预测结果形状: {predictions.shape}") # 形状: (验证样本数, 类别数)
```

这个例子演示了 `tf.data.Dataset` 对象如何融入标准 Keras 工作流。`shuffle`、`batch` 和 `prefetch` 操作确保数据得到高效准备并馈送给模型，从而最大限度地提高硬件利用率，特别是在与 GPU 或 TPU 加速结合时。这种集成是 TensorFlow 中构建可扩展机器学习 (machine learning)工作流的一个重要方面。

## 参考资料

- [Better performance with the tf.data API](https://www.tensorflow.org/guide/data_performance) — TensorFlow Authors (2024)
  解释如何使用 `tf.data` 构建高效的输入管道，涵盖 `shuffle`、`batch` 和 `prefetch` 等关键转换。
- [tf.keras.Model.fit](https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit) — TensorFlow Authors (2023)
  提供 `tf.keras.Model.fit()` 方法的详细 API 文档，包括其参数以及如何处理 `tf.data.Dataset` 对象。
- [Load TensorFlow data](https://www.tensorflow.org/guide/data) — TensorFlow Authors (2023)
  关于如何使用 `tf.data` 加载和预处理各种类型数据的综合指南，并包含将这些数据集集成到 Keras 模型中的示例。
