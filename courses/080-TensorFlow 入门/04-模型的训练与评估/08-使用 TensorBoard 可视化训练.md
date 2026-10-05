# 使用 TensorBoard 可视化训练

来源：[原文](https://apxml.com/zh/courses/getting-started-with-tensorflow/chapter-4-training-evaluating-models/visualizing-tensorboard)

[返回章节目录](README.md) · [返回课程目录](../README.md)

TensorBoard 是 TensorFlow 专门的可视化工具集，旨在帮助您跟踪和可视化机器学习 (machine learning)实验的不同方面。要完全理解模型的学习动态，需要比基本控制台输出更多的细节。可以将其看作是训练过程的仪表板，让您能够查看趋势、比较运行结果，并比单纯依赖文本输出更高效地调试可能的问题。

TensorBoard 通过读取训练期间生成的日志文件来运行。您可以跟踪随时间变化的标量值，例如损失和准确率，可视化模型的计算图，查看权重 (weight)和梯度的直方图，显示图像等等。这种视觉上的理解对于诊断过拟合 (overfitting)等问题、理解模型收敛情况以及评估不同超参数 (parameter) (hyperparameter)的影响都非常有帮助。

### 通过 Keras 回调集成 TensorBoard

在训练 Keras 模型时，使用 TensorBoard 最直接的方法是利用 `tf.keras.callbacks.TensorBoard` 回调。回调是传递给 `model.fit()` 的对象，它们可以在训练的不同阶段（例如，在每个训练周期或批次的开始/结束时）执行操作。

要使用 TensorBoard 回调，您首先需要实例化它，并指定一个 `log_dir`。这个目录是 TensorFlow 将写入 TensorBoard 读取的日志文件的位置。为不同的实验运行创建唯一的日志目录是一个好的做法，通常可以使用时间戳或描述性名称。

```python
import tensorflow as tf
import datetime

# 假设 'model' 是您已编译的 Keras 模型

# 定义日志目录路径，通常包含时间戳
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# 创建 TensorBoard 回调实例
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1  # 每 1 个训练周期记录直方图可视化数据
)

# 如果需要，定义其他回调，例如 EarlyStopping
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# 训练模型，将回调以列表形式传入
# 假设 X_train, y_train, X_val, y_val 是您的数据集
# history = model.fit(X_train, y_train,
#                     epochs=50,
#                     validation_data=(X_val, y_val),
#                     callbacks=[tensorboard_callback, early_stopping_callback])
```

在这个例子中：

- 我们使用当前时间戳创建一个唯一的 `log_dir`。这可以避免不同运行的日志互相覆盖。
- 我们实例化 `tf.keras.callbacks.TensorBoard`，并提供 `log_dir`。
- `histogram_freq=1` 告诉 TensorBoard 每隔一个训练周期计算并记录层激活和权重 (weight)的直方图。这对于更细致的分析很有用，但会消耗更多资源。将其设置为 0 会禁用直方图。
- `tensorboard_callback` 被包含在传递给 `model.fit()` 的 `callbacks` 参数 (parameter)的列表中。TensorFlow 现在将自动记录训练和验证指标（损失以及在 `model.compile` 期间指定的任何其他指标）到指定的 `log_dir`。

其他适用于 `TensorBoard` 回调的参数包括：

- `update_freq`：控制指标写入的频率。'epoch'（默认值）在每个训练周期结束后写入。'batch' 在每个批次结束后写入。整数值表示每 N 个批次写入。按批次写入提供更精细的细节，但会生成更大的日志文件。
- `profile_batch`：允许分析特定批次以分析性能瓶颈（一个高级功能）。

### 启动和使用 TensorBoard 用户界面

一旦训练开始且日志文件正在写入，您就可以启动 TensorBoard 界面。打开您的终端或命令提示符，导航到包含您顶级日志目录的目录（例如，我们例子中包含 `logs` 文件夹的目录），并运行以下命令：

```bash
tensorboard --logdir logs
```

TensorBoard 将启动一个本地 Web 服务器并打印访问它的 URL，通常是 `http://localhost:6006/`。在您的网页浏览器中打开此 URL。

您将看到 TensorBoard 仪表板。以下是一些最常用的选项卡：

- **标量：** 这通常是访问最多的选项卡。它显示训练期间记录的标量指标（例如损失和准确率）的图表，这些图表以训练周期或步数为X轴。您可以选择不同的运行（如果您的 `--logdir` 中有多个子目录）进行比较。此视图对于监控收敛和检测过拟合 (overfitting)（当验证损失开始增加而训练损失持续减少时）非常重要。



[交互图表：训练和验证损失](https://apxml.com/zh/courses/getting-started-with-tensorflow/chapter-4-training-evaluating-models/visualizing-tensorboard#plot-so2qcs)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "x": [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10
      ],
      "y": [
        1.2,
        0.8,
        0.6,
        0.5,
        0.45,
        0.4,
        0.38,
        0.36,
        0.35,
        0.34
      ],
      "mode": "lines+markers",
      "name": "训练",
      "line": {
        "color": "#4263eb"
      }
    },
    {
      "x": [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10
      ],
      "y": [
        1.5,
        1.1,
        0.9,
        0.8,
        0.75,
        0.72,
        0.71,
        0.73,
        0.75,
        0.78
      ],
      "mode": "lines+markers",
      "name": "验证",
      "line": {
        "color": "#fd7e14"
      }
    }
  ],
  "layout": {
    "title": "训练和验证损失",
    "xaxis": {
      "title": "训练周期"
    },
    "yaxis": {
      "title": "损失"
    },
    "legend": {
      "title": "数据集"
    }
  }
}
```

</details>



> 示例图表显示训练损失减少，而验证损失在第 7 个训练周期后开始增加，表明可能出现过拟合。

- **图表：** 此选项卡可视化您的 TensorFlow 模型结构。它将层和操作显示为图。这有助于确认您的模型架构（特别是对于使用函数式 API 构建的复杂模型），并理解张量的流动。

> 简单的图表显示了从输入层通过两个全连接层到输出层的流动。

- **分布和直方图：** 如果您启用了 `histogram_freq`，这些选项卡会显示权重 (weight)、偏差或激活的分布在训练过程中如何变化。直方图显示在特定训练周期（或步数）的分布情况，而分布则提供类似热图的视图，显示这些分布随时间如何演变。它们有时有助于诊断梯度消失或梯度爆炸等问题，在这些问题中，权重会持续变得非常小或非常大。

### 解释可视化结果以优化模型

TensorBoard 可视化不仅仅用于观察；它们是用于采取行动的工具。通过解释这些图表，您可以做出明智的决定：

- **收敛性：** 损失曲线是否稳定下降并趋于平稳？如果它们过早趋于平稳，可能是学习率过低，或者模型容量不足。如果训练停止时它们仍在急剧下降，请考虑进行更多训练周期。
- **过拟合 (overfitting)：** 验证损失/指标是否明显差于训练损失/指标，或者验证损失在训练损失减少时开始增加（如 Plotly 示例所示）？这表明存在过拟合。您可能需要正则化 (regularization)（例如 Dropout、L1/L2）、数据增强或早期停止（我们已在代码示例中包含）等方法。
- **学习率：** 非常嘈杂的损失曲线可能表明学习率过高。极度缓慢的收敛可能表明学习率过低。
- **初始化/梯度：** 权重 (weight)/梯度直方图中不寻常的模式（例如，分布收缩到零或爆炸）可能指向权重初始化或梯度流动的问题，建议更改激活函数 (activation function)或归一化 (normalization)层。

通过使用 `TensorBoard` 回调将 TensorBoard 整合到您的工作流程中，您可以更好地观察训练过程。它超越了简单的最终指标，提供了模型学习方式的动态图像，从而实现更系统化的调试和改进。

## 参考资料

- [TensorBoard: Visualizing your Machine Learning Experiments](https://www.tensorflow.org/tensorboard/get_started) — TensorFlow Authors (2023)
  使用TensorFlow理解和操作TensorBoard的全面指南，涵盖了其功能和设置。
- [\`tf.keras.callbacks.TensorBoard\`](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard) — TensorFlow Authors (2023)
  Publisher: TensorFlow
  Keras TensorBoard回调函数的官方API文档，详细说明了其参数和用法，以便与Keras模型训练集成。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media; Pages: Chapter 10: Training Deep Neural Networks
  一本备受推崇的书，其中包含一个实用章节，介绍如何在TensorFlow/Keras生态系统中使用TensorBoard监控和显示神经网络训练。

---

[上一节](07-%E8%AE%AD%E7%BB%83%E6%97%B6%E4%BD%BF%E7%94%A8%E5%9B%9E%E8%B0%83.md) · [下一节](09-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E8%AE%AD%E7%BB%83%E4%B8%8E%E7%9B%91%E6%8E%A7.md)
