# 训练循环：`fit()` 方法

来源：[原文](https://apxml.com/zh/courses/deep-learning-fundamentals-keras/chapter-3-training-deep-neural-networks/training-loop-fit-method)

[返回章节目录](README.md) · [返回课程目录](../README.md)

神经网络 (neural network)的学习过程始于其架构的定义和编译，其中需指定优化器、损失函数 (loss function)和评估指标。模型训练在 Keras 中主要通过 `fit()` 方法完成，此方法巧妙地包含了从数据中学习的复杂迭代过程。

可以把 `fit()` 方法看作驱动模型训练的引擎。它接收训练数据，将其送入网络，计算误差（损失），使用反向传播 (backpropagation)和所选优化器计算如何调整内部参数 (parameter)（权重 (weight)和偏置 (bias)），并多次重复此过程。

### 调用 `fit()` 方法

其核心是，你在已编译的模型对象上调用 `fit()`，提供训练数据和对应的目标标签。基本语法如下：

```python
# 假设 'model' 是你已编译的 Keras 模型
# 假设 'x_train' 包含你的训练特征（例如，图像、文本序列）
# 假设 'y_train' 包含你的训练标签（例如，图像类别、情感分数）

history = model.fit(x_train, y_train, epochs=10, batch_size=32)
```

让我们分解一下主要的参数 (parameter)：

1. **`x` (或 `x_train`)**：这是你的输入训练数据。它通常是 NumPy 数组或兼容格式（例如，当使用 Keras 3 搭配不同后端时，可以是 TensorFlow `Dataset` 或 PyTorch `DataLoader`）。此数据的形状必须与模型第一层中指定的输入形状一致。
2. **`y` (或 `y_train`)**：这些是与输入数据对应的目标标签。对于分类任务，它们可能是整数类索引或独热编码向量 (vector)。对于回归任务，它们将是连续值。格式必须与模型的输出层和所选损失函数 (loss function)对齐 (alignment)。
3. **`epochs`**：如前所述，一个 epoch 表示对*整个*训练数据集的一次完整遍历。`epochs` 参数告诉 `fit()` 对完整数据集迭代多少次。模型通常需要多个 epoch 才能有效学习。
4. **`batch_size`**：这决定了在一个 epoch 内，每次迭代（梯度更新步骤）中处理的样本数量。`fit()` 不会一次性处理整个数据集（那样计算开销大且内存密集），而是分批处理数据。模型的权重 (weight)在每个批次处理后更新。`batch_size` 为 32 意味着在更新权重之前，使用 32 个样本计算梯度。

### `fit()` 内部发生了什么？

当你调用 `model.fit()` 时，Keras 会执行训练循环：

1. **Epoch 迭代**：外部循环按照指定的 `epochs` 数量进行迭代。
2. **批次迭代**：在每个 epoch 内，Keras 根据 `batch_size` 将训练数据（`x_train`，`y_train`）分成多个批次。然后遍历这些批次。
3. **前向传播**：对于每个批次，输入数据通过网络层进行前向传播以生成预测。
4. **损失计算**：所选的损失函数 (loss function)（在 `model.compile()` 时指定）将批次的预测与真实目标标签（该批次的 `y_train` 部分）进行比较，并计算一个损失值，量化 (quantization)该批次中模型的误差。
5. **反向传播 (backpropagation)**：计算损失相对于模型可训练参数 (parameter)（权重 (weight)和偏置 (bias)）的梯度。这包括通过网络向后传播误差。
6. **优化器步骤**：所选的优化器（例如 Adam、SGD、RMSprop，同样在 `compile()` 时指定）使用计算出的梯度更新模型参数，目标是最小化损失。
7. **指标计算**：处理完每个批次后（或者有时在每个 epoch 结束时），会对该 epoch 中目前为止的训练数据计算指定的评估指标（如准确率）。
8. **日志记录**：Keras 会打印每个 epoch 的进度，通常显示训练数据的损失和评估指标。

### 使用验证数据

仅凭训练数据监测模型表现可能会有误导性，因为模型可能只是记住了训练样本（过拟合 (overfitting)）。为了更实际地评估模型对未见数据的泛化能力，你应该使用 `validation_data` 参数 (parameter)向 `fit()` 方法提供验证数据：

```python
# 假设 'x_val' 和 'y_val' 是你的验证特征和标签
history = model.fit(x_train,
                    y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))
```

当提供了 `validation_data` 时，Keras 会在每个 epoch 结束时执行一个附加步骤：

- **验证评估**：模型会在提供的验证数据集（`x_val`，`y_val`）上评估其表现（计算损失和评估指标）。重要的一点是，模型*不会*从这些数据中学习；其参数不会根据验证结果进行更新。此评估纯粹用于监测泛化表现。

日志现在将包含验证损失（`val_loss`）和验证指标（例如，`val_accuracy`）。比较训练损失/指标与验证损失/指标对于诊断过拟合等问题非常有帮助。

### `History` 对象

`fit()` 方法返回一个 `History` 对象。该对象作用类似于一个字典，包含每个 epoch 记录的损失和评估指标值，既有训练集的，也有验证集的（如果提供）。

```python
print(history.history.keys())
# 输出可能类似：dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])

# 获取每个 epoch 的训练损失值
training_loss = history.history['loss']

# 获取每个 epoch 的验证准确率值
validation_accuracy = history.history['val_accuracy']
```

这个 `History` 对象对于可视化训练过程非常有用。例如，你可以绘制训练和验证损失随 epoch 的变化图，以观察模型学习和泛化的情况。



[交互图表：训练和验证损失随 Epoch 变化](https://apxml.com/zh/courses/deep-learning-fundamentals-keras/chapter-3-training-deep-neural-networks/training-loop-fit-method#plot-7mv4q0)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "训练和验证损失随 Epoch 变化",
    "xaxis": {
      "title": "Epoch"
    },
    "yaxis": {
      "title": "损失"
    },
    "legend": {
      "title": "数据集"
    },
    "template": "plotly_white",
    "width": 600,
    "height": 400
  },
  "data": [
    {
      "name": "训练损失",
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
        0.65,
        0.45,
        0.35,
        0.3,
        0.27,
        0.25,
        0.23,
        0.22,
        0.21,
        0.2
      ],
      "type": "scatter",
      "mode": "lines+markers",
      "marker": {
        "color": "#228be6"
      }
    },
    {
      "name": "验证损失",
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
        0.55,
        0.42,
        0.38,
        0.36,
        0.35,
        0.34,
        0.345,
        0.35,
        0.36,
        0.37
      ],
      "type": "scatter",
      "mode": "lines+markers",
      "marker": {
        "color": "#fd7e14"
      }
    }
  ]
}
```

</details>



> 训练损失（蓝色）通常会下降，而验证损失（橙色）最初会下降，但如果发生过拟合 (overfitting)，可能会开始趋于平稳或增加。

总之，`fit()` 方法是 Keras 中模型训练的核心操作。它自动化了前向传播、损失计算、反向传播 (backpropagation)和权重 (weight)更新的复杂循环，让你只需几行代码就能训练复杂的深度学习 (deep learning)模型，同时通过验证数据和返回的 `History` 对象，提供了监测性能的重要机制。

## 参考资料

- [Model.fit() method](https://keras.io/api/models/model_training_apis/model/#fit-method) — Keras team (2024)
  Publisher: TensorFlow
  提供 Keras `fit()` 方法的官方 API 规范和使用指南。
- [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python-second-edition) — François Chollet (2021)
  Publisher: Manning Publications
  一本使用 Keras 进行深度学习的实践指南，说明模型训练、周期、批次和验证。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本详细介绍深度学习模型训练背后的算法和数学的教科书。

---

[上一节](04-%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD.md) · [下一节](06-%E6%89%B9%E6%AC%A1%E4%B8%8E%E5%91%A8%E6%9C%9F.md)
