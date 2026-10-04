---
course: "introduction-autoencoders-feature-learning"
chapter: "building-basic-autoencoder"
lesson: "executing-training-process"
sourceId: 6434
sourceUrl: "https://apxml.com/zh/courses/introduction-autoencoders-feature-learning/chapter-5-building-basic-autoencoder/executing-training-process"
title: "执行训练过程"
description: "如何使用Keras的'fit'方法训练你的自编码器模型并监控其进展。"
order: 7
plots: []
sourceHash: "017e70400119d869b161bdac744ce8062b0059ebb3618cff33e041488770fbf2"
sourceCorrections: []
---

自编码器模型在定义并配置了优化器和损失函数 (loss function)之后，现在已准备好进入训练阶段。训练是自编码器学习执行其任务的阶段，即尽可能准确地重构输入数据。这种学习是通过根据模型产生的重构误差，迭代调整模型内部参数 (parameter)（权重 (weight)）来实现的。在 Keras 中，主要工具是 `fit` 方法。

### 使用 `model.fit()` 启动训练

`fit` 方法是你指示 Keras 模型从数据中学习的主要界面。你将提供训练数据，指定一些训练参数 (parameter)，Keras 将处理迭代和权重 (weight)调整的复杂底层过程。

下面是调用 `model.fit()` 来训练自编码器的一种常见方式：

```python
# 假设 'autoencoder' 是你已编译的 Keras 模型
# 'x_train_processed' 是你预处理过的训练数据
# 'x_val_processed' 是你预处理过的验证数据（未直接用于训练的数据子集）

# 定义一些训练设置
num_epochs = 50           # 遍历整个训练数据集的次数
batch_size_val = 128      # 在更新模型权重之前处理的样本数量

# 让我们来训练模型！
history = autoencoder.fit(
    x_train_processed,    # 训练的输入数据
    x_train_processed,    # 目标数据：对于自编码器，这与输入相同
    epochs=num_epochs,
    batch_size=batch_size_val,
    shuffle=True,         # 在每个 epoch 开始时打乱训练数据是良好的做法
    validation_data=(x_val_processed, x_val_processed) # 在每个 epoch 结束时用于评估损失的数据
)
```

这些参数的含义是：

- **第一个参数（`x_train_processed`）：** 这是你的**输入训练数据**。自编码器会接收这些数据，通过编码器获得压缩表示，然后通过解码器尝试重构它。
- **第二个参数（`x_train_processed`）：** 这是**目标数据**。由于自编码器的目标是重构其输入，输入数据本身就作为期望的输出。
- `epochs`: 一个**epoch**表示对整个训练数据集的一次完整遍历。训练多个 epoch 使模型能够反复查看数据并调整其权重。
- `batch_size`: 将数据分成更小的段，称为**批次**（batches）。`batch_size` 定义了每个批次中的样本数量。
- `shuffle`: 指示 Keras 在每个 epoch 开始前**打乱训练样本的顺序**，有助于防止模型学习到与数据呈现顺序相关的模式。
- `validation_data`: **验证数据**，模型不在此上训练。用于监控模型对训练期间未见过的数据的泛化能力。

### `fit` 过程中发生了什么？

当你调用 `model.fit()` 时，Keras 会启动并管理学习循环。对于每个 epoch，以及该 epoch 内的每个批次：

1. **前向传播**：输入数据流经编码器网络，被压缩成瓶颈表示，然后流经解码器网络以生成重构输出。
2. **损失函数 (loss function)**：计算重构输出与原始输入批次之间的差异，即“重构误差”。
3. **优化器**：使用此误差值来调整模型的内部权重 (weight)。
4. 这个循环在当前 epoch 的所有批次中重复进行，然后进入下一个 epoch，直到所有指定的 epoch 完成。

### 使用 `History` 对象监控训练

`model.fit()` 方法会返回一个 `History` 对象，它记录了训练过程中发生的一切，特别是每个 epoch 结束时的损失值（和任何其他指标）。你可以从 `history.history` 属性中访问这些信息：

```python
# 'history' 是 autoencoder.fit() 返回的对象

training_loss_values = history.history['loss']
validation_loss_values = history.history['val_loss']

# 让我们打印第一个和最后一个 epoch 后的损失
print(f"第一个 epoch 后训练数据的损失: {training_loss_values[0]:.4f}")
print(f"第一个 epoch 后验证数据的损失: {validation_loss_values[0]:.4f}")
print(f"最后一个 epoch 后训练数据的损失: {training_loss_values[-1]:.4f}")
print(f"最后一个 epoch 后验证数据的损失: {validation_loss_values[-1]:.4f}")
```

### 可视化训练进程

绘制训练和验证损失随 epoch 变化的图表是评估训练进程的常用方法。

**解释损失曲线：**

- **一个好的迹象：** 训练损失和验证损失都随 epoch 的增加而下降并趋于稳定。验证损失通常会略高于训练损失，这表明模型正在学习和泛化。
- **过拟合 (overfitting)：** 训练损失持续下降，但验证损失趋于平坦或开始上升。这表明模型在训练数据上表现良好，但无法泛化到新数据。
- **欠拟合 (underfitting)：** 训练和验证损失都保持较高水平，或下降非常缓慢，这可能表示模型过于简单或训练不足。

一旦 `model.fit()` 完成，你的 `autoencoder` 模型的权重 (weight)就通过学习过程得到了更新。它现在已经学会了如何将输入数据压缩到瓶颈层然后进行重构。下一步是评估它学习这项任务的成效。

## 参考资料

- [model.fit()](https://keras.io/api/models/model_training_apis/#fit-method) — Keras Team (2024)
  官方文档，提供训练 Keras 模型的详细 API 规范和使用示例，包括 epoch、batch_size 和 validation_data 等参数。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 7: Regularization for Deep Learning, Chapter 8: Optimization for Training Deep Models
  一本基础教科书，对神经网络训练提供了全面的理论解释，包括优化算法、损失函数、批处理、周期以及过拟合和欠拟合等泛化概念。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125965/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media; Pages: Chapter 12: Training Deep Neural Networks
  一本实用指南，演示如何使用 Keras 训练深度学习模型，详细说明了 `model.fit()` 的参数，并解释了如何监控训练进度和解决过拟合等常见问题。（第4版）
