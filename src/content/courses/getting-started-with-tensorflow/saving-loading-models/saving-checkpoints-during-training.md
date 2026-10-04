---
course: "getting-started-with-tensorflow"
chapter: "saving-loading-models"
lesson: "saving-checkpoints-during-training"
sourceId: 1787
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-tensorflow/chapter-6-saving-loading-models/saving-checkpoints-during-training"
title: "训练期间保存检查点"
description: "使用 ModelCheckpoint 回调函数定期保存模型权重或整个模型。"
order: 2
plots: []
sourceHash: "c5c92b5f2d22117b761d766fc32737c6dedc92509c915b6dfd3975312b0be340"
sourceCorrections: []
---

训练深度学习 (deep learning)模型常常是耗时的过程，有时需要数小时、数天甚至数周，这取决于模型的复杂程度和数据集的大小。在如此漫长的训练过程中，可能会出现各种问题：断电、系统崩溃，或者仅仅是需要暂停并在之后恢复。如果没有保存进度的机制，你可能会丢失宝贵的计算时间，以及训练期间可能获得的模型最佳版本。

保存*检查点*在深度学习模型训练中必不可少。检查点保存是指在训练过程中，定期或在满足特定条件时保存模型的状态。TensorFlow 的 Keras API 提供了一种便捷的方法，可以使用回调来实现这一点。

### ModelCheckpoint 回调函数

Keras 中用于保存检查点的主要工具是 `tf.keras.callbacks.ModelCheckpoint` 回调函数。回调函数是一个对象，可以在训练的不同阶段执行操作（例如，在周期开始或结束时，在处理批次之前或之后）。`ModelCheckpoint` 回调函数专门监控训练过程，并根据配置的条件保存模型。

你可以通过创建 `ModelCheckpoint` 回调函数的一个实例，并将其作为 `model.fit()` 方法中的 `callbacks` 列表参数 (parameter)传递，来使用它。

我们来看一下它的主要配置选项：

- `filepath`: 这是检查点文件将保存的路径。你可以在文件名中包含格式选项，使每次保存的文件名都是独立的，并包含周期数和被监控指标的值。例如：
  - `'model_checkpoint.weights.h5'` : 保存到单个文件（每次都会覆盖，除非 `save_best_only=True`）。
  - `'checkpoints/epoch_{epoch:02d}-val_loss_{val_loss:.2f}.weights.h5'`: 创建类似 `epoch_01-val_loss_0.54.weights.h5`、`epoch_02-val_loss_0.51.weights.h5` 等文件。这会保存多个周期的检查点。
- `monitor`: 指定要监控的指标。常见选择包括 `'val_loss'`（验证损失）或 `'val_accuracy'`（验证准确率）。回调函数将使用此指标的值来判断当前模型是否比之前最佳的模型“更好”。如果未指定，回调函数将在不考虑性能指标的情况下运行（例如，无论性能如何，每个周期都保存）。
- `save_best_only`: 如果为 `True`，回调函数只在被监控的指标相较于训练过程中目前为止的最佳值有所改进时保存检查点。这对于只保留性能最好的模型检查点非常有帮助。如果为 `False`，它会在由 `save_freq` 定义的每个时间段结束时保存模型。
- `save_weights_only`: 如果为 `True`，只保存模型的权重 (weight)（可学习参数的值）。这会使检查点文件更小。如果为 `False`，则保存整个模型，包括其架构、权重和优化器的状态。保存整个模型可以让你重新创建模型，并在中断的地方精确地恢复训练。
- `mode`: 决定改进是指最小化还是最大化被监控的指标。选项包括 `'min'`、`'max'` 或 `'auto'`。如果 `monitor` 设置为 `'val_loss'`，`'auto'` 会正确推断为 `'min'`。如果设置为 `'val_accuracy'`，`'auto'` 会推断为 `'max'`。明确设置它可以避免不确定性。
- `save_freq`: 定义保存检查点的频率。
  - `'epoch'`（默认）：在每个周期结束时保存。
  - 整数值（例如，`1000`）：在每指定数量的批次后保存。

### 示例：根据验证损失保存最佳模型

我们来演示如何使用 `ModelCheckpoint`，根据验证损失只保存目前为止观察到的最佳模型的权重 (weight)。

```python
import tensorflow as tf
import numpy as np

# 假设 'model' 是一个已编译的 Keras 模型
# 假设 'x_train'、'y_train'、'x_val'、'y_val' 是你的训练和验证数据

# 定义保存检查点的路径
checkpoint_filepath = 'best_model.weights.h5'

# 创建 ModelCheckpoint 回调函数
# 监控验证损失，只保存最佳权重
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

# 训练模型，包含回调函数
print("开始使用检查点回调函数进行训练...")
history = model.fit(x_train, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_data=(x_val, y_val),
                    callbacks=[model_checkpoint_callback]) # 将回调函数传递给训练

print(f"训练完成。最佳模型权重已保存到 {checkpoint_filepath}")

# 稍后，你可以将这些权重加载到具有相同架构的模型中
# model.load_weights(checkpoint_filepath)
# print("模型权重已从检查点加载。")
```

在此示例中：

1. 我们定义了 `checkpoint_filepath`，最佳权重将存储在此处。
2. 我们创建一个 `ModelCheckpoint` 实例。
   - `save_weights_only=True`：我们只保存权重。
   - `monitor='val_loss'`：我们监控验证损失。
   - `mode='min'`：改进意味着验证损失减小。
   - `save_best_only=True`：只有迄今为止观察到的最低 `val_loss` 对应的检查点才会被保存/覆盖。
3. 我们将此回调函数作为列表传递给 `model.fit()` 的 `callbacks` 参数 (parameter)。

在训练期间，Keras 会在每个周期结束时评估验证损失。如果损失相对于所有之前的周期有所改进（减小），当前模型权重将被保存到 `best_model.weights.h5`，覆盖任何之前的版本。如果损失没有改进，该周期不会保存任何文件。在训练结束时，`best_model.weights.h5` 将包含获得最低验证损失的周期中的权重。

> 流程图演示了当 `save_best_only=True` 且 `monitor='val_loss'` 时 `ModelCheckpoint` 的行为。只有当验证损失有所改进时才保存检查点。

### 保存整个模型

如果你设置 `save_weights_only=False`（默认值），Keras 会保存整个模型：

- 模型的架构（层、连接）。
- 模型的权重 (weight)。
- 在 `model.compile()` 中指定的训练配置（损失、优化器、指标）。
- 优化器的状态（允许你从中断的地方精确地恢复训练）。

这会以 TensorFlow 的 `SavedModel` 格式保存（如果文件路径不以 `.h5` 结尾），或以旧的 Keras HDF5 格式保存（如果文件路径以 `.h5` 结尾）。`SavedModel` 格式通常更值得推荐。

```python
# 定期保存整个模型的示例
checkpoint_filepath_full = 'checkpoints/model_epoch_{epoch:02d}.keras' # 使用 .keras 格式用于 SavedModel

full_model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath_full,
    save_weights_only=False, # 保存整个模型
    monitor='val_accuracy',  # 监控验证准确率
    mode='max',              # 改进意味着准确率提高
    save_best_only=False,    # 在每个周期结束时保存
    save_freq='epoch'        # 明确指定保存频率
)

# 使用此回调函数训练模型
# history = model.fit(..., callbacks=[full_model_checkpoint_callback])
```

此配置会在每个周期结束时保存完整的模型，到根据周期数命名的目录结构中（如果使用像 `.keras` 这样的 SavedModel 格式）或单个文件（如果使用 `.h5`）。这会占用更多磁盘空间，但会捕获恢复模型或恢复训练所需的所有内容。

只保存权重还是保存整个模型，这取决于你的需求。如果你只需要学习到的参数 (parameter)用于推理 (inference)或微调 (fine-tuning)，保存权重就足够了，而且效率更高。如果你需要恢复训练或部署具有完整配置的模型，保存整个模型更受欢迎。

有效使用检查点能确保你的训练成果得以保留，让你能够从中断中恢复，并保留模型性能最佳的版本。接下来的部分将介绍如何加载这些已保存的权重和模型。

## 参考资料

- [tf.keras.callbacks.ModelCheckpoint](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint) — TensorFlow Developers (2024)
  详细介绍 `tf.keras.callbacks.ModelCheckpoint` 回调函数、其参数及用法的官方文档。
- [Save and load Keras models](https://www.tensorflow.org/tutorials/keras/save_and_load) — TensorFlow Developers (2024)
  Publisher: TensorFlow
  解释保存和加载 Keras 模型及权重（包括 SavedModel 等不同格式）的各种方法的官方指南。
- [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python-second-edition) — François Chollet (2021)
  Publisher: Manning Publications
  一本使用 Keras 和 TensorFlow 进行深度学习的实践指南，提供模型训练和保存技术的解释。
