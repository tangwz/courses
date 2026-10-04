---
course: "getting-started-with-tensorflow"
chapter: "training-evaluating-models"
lesson: "using-callbacks-training"
sourceId: 1746
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-tensorflow/chapter-4-training-evaluating-models/using-callbacks-training"
title: "训练时使用回调"
description: "使用 ModelCheckpoint、EarlyStopping 和 TensorBoard 等回调，以更好地控制训练。"
order: 7
plots: []
sourceHash: "830a247bc02d705d4ccdea3a0ea24d06337c39aa9fc788c65aa84322d16532f7"
sourceCorrections: []
---

使用 `model.fit()` 训练模型可能需要相当长的时间，尤其是对于大型数据集或复杂架构，运行多个周期时更是如此。在这个过程中，你可能想在特定时点自动执行一些操作，比如定期保存模型、在性能不再提升时提前停止训练，或者调整学习率。Keras 回调函数在此发挥作用。

回调函数是可以传递给 `model.fit()` 的对象（在 `callbacks` 参数 (parameter)列表中），它们在训练的不同阶段（例如，在一个周期的开始或结束时，在一个批次之前或之后）执行预设的操作。它们提供了一种有效的方法来定制和控制训练循环，而无需修改核心的 `fit` 方法。

让我们来看看 `tf.keras.callbacks` 提供的一些最常用的回调函数。

### 自动保存最佳模型：ModelCheckpoint

设想训练一个模型100个周期。也许在验证集上的最佳表现出现在第75个周期，但到了第100个周期，模型开始过拟合 (overfitting)，验证表现下降了。如果你只在最后保存模型，你就会错过最好的版本！

`ModelCheckpoint` 回调函数通过在训练期间定期保存模型来解决这个问题。你可以配置它只保存模型的权重 (weight)，或者保存整个模型（架构、权重和优化器状态）。重要的一点是，你可以指示它只保存迄今为止观察到的*最佳*模型，这依据的是所监控的指标，例如验证损失或准确率。

以下是你可以如何配置它，以便根据验证损失保存最佳模型：

```python
import tensorflow as tf

# 假设 'model' 是一个已编译的 Keras 模型

# 定义回调函数
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='best_model_{epoch:02d}_{val_loss:.2f}.keras', # 文件路径，包含格式选项
    save_weights_only=False,  # 保存整个模型
    monitor='val_loss',       # 要监控的指标
    mode='min',               # 我们希望最小化损失
    save_best_only=True)      # 仅在 'val_loss' 改进时保存

# 现在，将其传递给 model.fit()（假设你有训练和验证数据）
# history = model.fit(train_dataset, epochs=100, validation_data=validation_dataset,
#                     callbacks=[model_checkpoint_callback])
```

**`ModelCheckpoint` 的参数 (parameter)：**

- `filepath`: 保存模型文件的路径。你可以使用 `{epoch:02d}` 等格式选项来包含周期数（用零填充）或使用 `{val_loss:.2f}` 来包含监控指标的值（格式化为两位小数）。保存整个模型的推荐格式是 `.keras`。如果 `save_weights_only=True`，则使用 `.weights.h5`。
- `monitor`: 要监控的数量（例如，`'val_loss'`、`'val_accuracy'`）。训练指标（例如 `'loss'`、`'accuracy'`）也可以监控，但为了选择泛化能力最佳的模型，通常更推荐监控验证指标。
- `mode`: `{'auto', 'min', 'max'}` 之一。如果 `monitor` 是 `val_loss`，模式应为 `'min'`。如果 `monitor` 是 `val_accuracy`，模式应为 `'max'`。在 `'auto'` 模式下，Keras 会根据指标名称推断方向。
- `save_best_only`: 如果为 `True`，则仅当监控量相对于迄今为止观察到的最佳值有所改进时才保存模型。
- `save_weights_only`: 如果为 `True`，则只保存模型的权重（`model.save_weights()`）。如果为 `False`，则保存整个模型（`model.save()`），包括架构和优化器状态。

### 使用 EarlyStopping 防止过拟合 (overfitting)

训练时间过长可能导致过拟合，模型在训练数据上表现良好，但在未见过的验证数据上表现不佳。`EarlyStopping` 回调函数监控一个指定的指标（通常是验证指标），如果该指标在设定的周期数内没有改善，则停止训练过程。

```python
# 定义回调函数
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',     # 要监控的指标
    patience=10,            # 在没有改进的情况下，多少个周期后训练会停止
    mode='min',             # 我们希望最小化损失
    restore_best_weights=True # 从 'val_loss' 最佳的周期恢复模型权重
)

# 将其传递给 model.fit()，可能与 ModelCheckpoint 一起使用
# history = model.fit(train_dataset, epochs=100, validation_data=validation_dataset,
#                     callbacks=[model_checkpoint_callback, early_stopping_callback])
```

**`EarlyStopping` 的参数 (parameter)：**

- `monitor`: 要监控的数量（例如，`'val_loss'`）。
- `patience`: 在没有改进的情况下，训练将停止的周期数。例如，如果 `patience=10`，如果监控的指标连续10个周期没有改进，训练将停止。
- `min_delta`: 监控数量的最小变化量，才算作改进。默认为0。设置一个小的正值可以避免因微不足道的改进而停止。
- `mode`: `{'auto', 'min', 'max'}` 之一。确定改进是表示下降（`'min'`）还是上升（`'max'`）。
- `restore_best_weights`: 如果为 `True`，训练停止后，模型权重 (weight)会回滚到在监控数量上取得最佳值的周期时的权重。这是非常推荐的，因为训练可能会在观察到最佳表现的几个周期*之后*才停止。

### 用于可视化的日志记录：TensorBoard

当 `ModelCheckpoint` 保存你的模型，`EarlyStopping` 控制训练时长时，`TensorBoard` 回调函数在训练期间记录指标、图表和其他信息。这些记录的数据随后可以使用 TensorBoard 工具进行可视化（我们将在下一节中查看），以此理解训练过程，诊断问题，并比较不同的运行。

```python
import datetime

# 定义回调函数
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1 # 每1个周期记录一次直方图可视化
)

# 在 fit 期间传递它
# history = model.fit(train_dataset, epochs=100, validation_data=validation_dataset,
#                     callbacks=[model_checkpoint_callback,
#                                early_stopping_callback,
#                                tensorboard_callback])
```

这里的主要参数 (parameter)是 `log_dir`，它指定了 TensorBoard 日志将被写入的目录。在目录名中使用时间戳有助于将不同运行的日志分开。我们将在下一节介绍如何使用这些日志。

### 使用多个回调函数

你可以通过将多个回调函数作为列表传递给 `model.fit()` 中的 `callbacks` 参数 (parameter)来同时使用它们。Keras 将在训练循环的适当阶段执行每个回调函数。

```python
# 使用所有三个已讨论回调的例子
callbacks_list = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath='best_model.keras', monitor='val_loss', save_best_only=True
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True
    ),
    tf.keras.callbacks.TensorBoard(
        log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
]

# history = model.fit(train_dataset, epochs=100, validation_data=validation_dataset,
#                     callbacks=callbacks_list)
```

存在其他有用的回调函数，例如用于自定义学习率调度的 `LearningRateScheduler` 或在指标停止改善时降低学习率的 `ReduceLROnPlateau`。回调函数提供了一种灵活的方式，可以为你的 Keras 训练流程增加自定义行为和控制，帮助你管理长时间的训练运行、防止过拟合 (overfitting)，并保存最佳模型状态。

## 参考资料

- [Keras Callbacks](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks) — TensorFlow Team (2024)
  `tf.keras.callbacks`的官方文档，详细说明其用法、可用类型和自定义回调的创建。
- [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python-second-edition) — François Chollet (2021)
  Publisher: Manning Publications
  由Keras框架的创建者撰写，本书全面解释了该框架，包括如何使用和实现回调以进行有效的模型训练。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一份实用指南，提供了使用TensorFlow和Keras训练深度学习模型的示例和最佳实践，包括使用回调的策略。
