---
course: "rnns-and-sequence-modeling"
chapter: "building-simple-rnns"
lesson: "practice-simple-sequence-prediction"
sourceId: 2551
sourceUrl: "https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-3-building-simple-rnns/practice-simple-sequence-prediction"
title: "动手实践：简单序列预测"
description: "用于训练基本RNN的合成序列预测任务代码示例。"
order: 7
plots: ["plots/2551-0.json"]
sourceHash: "50dd534b668f70363ff5fb48d69f17be48c0d17cdd8ee1aa05f9881a76fb20eb"
sourceCorrections: []
---

这个实践例子展示了如何使用框架API应用`SimpleRNN`层及其输入/输出形状要求。我们将处理一个直接的序列预测任务：预测简单算术序列中的下一个数字。这个练习将巩固你对构建和训练基本RNN的理解。

我们的目标是训练一个RNN，使其学会`[10, 20, 30]`、`[25, 35, 45]`等序列中的规律，并预测后续的数字。例如，给定输入序列`[10, 20]`，模型应该学会预测`30`。

### 1. 生成数据

首先，我们需要一些数据。我们将创建合成序列，每个数字都比前一个大10。我们将生成`(input_sequence, target_value)`对。我们将使用长度为3的序列作为输入，以预测第四个元素。

```python
import numpy as np

def generate_sequences(n_sequences=1000, sequence_length=4):
    """生成算术序列（步长为10），并将其分为X和y。"""
    X, y = [], []
    for i in range(n_sequences):
        start_value = np.random.randint(0, 100)
        sequence = [start_value + j * 10 for j in range(sequence_length)]
        X.append(sequence[:-1]) # 输入序列（前3个元素）
        y.append(sequence[-1])  # 目标值（最后一个元素）
    return np.array(X), np.array(y)

# 生成1000个样本，每个序列总共有4个数字
# 输入序列长度（时间步）将为3
X_train_raw, y_train = generate_sequences(n_sequences=1000, sequence_length=4)

print("样本输入序列 (X):", X_train_raw[0])
print("样本目标值 (y):", y_train[0])
print("X_train_raw的形状:", X_train_raw.shape)
print("y_train的形状:", y_train.shape)

# 预期输出:
# 样本输入序列 (X): [start_val, start_val+10, start_val+20] (e.g., [42 52 62])
# 样本目标值 (y): start_val+30 (e.g., 72)
# X_train_raw的形状: (1000, 3)
# y_train的形状: (1000,)
```

### 2. 为RNN准备数据

如前所述，TensorFlow或PyTorch等框架中的标准RNN层需要特定3D格式的输入数据：`(批大小, 时间步, 特征)`。

- `batch_size`（批大小）：训练时一次处理的序列数量。我们将让框架在训练过程中处理此项，因此我们首先关注单个样本的形状。
- `time_steps`（时间步）：输入序列的长度。在此例中，其长度为3（例如，`[10, 20, 30]`）。
- `features`（特征）：每个时间步的特征数量。由于序列中的每个数字都是单个值，因此特征数量为1。

我们当前的`X_train_raw`形状为`(1000, 3)`。我们需要将其重塑为`(1000, 3, 1)`。

```python
# 将X重塑为[样本数, 时间步, 特征数]
n_samples = X_train_raw.shape[0]
n_time_steps = X_train_raw.shape[1]
n_features = 1 # 每个时间步只有一个特征（数字本身）

X_train = X_train_raw.reshape((n_samples, n_time_steps, n_features))

print("重塑后的X_train形状:", X_train.shape)

# 预期输出:
# 重塑后的X_train形状: (1000, 3, 1)
```

我们也常对数据进行归一化 (normalization)，以提高训练的稳定性，尽管对于这个简单的算术任务，即使不进行归一化也可能收敛。为了演示，我们通过除以一个常数（例如100）进行简单缩放。

```python
# 归一化数据（可选但推荐）
X_train = X_train / 100.0
y_train = y_train / 100.0
```

### 3. 构建简单RNN模型

现在，我们使用TensorFlow的Keras API构建模型。我们将使用一个包含以下层的`Sequential`模型：

1. 一个`SimpleRNN`层：这是核心循环层。我们需要指定隐藏状态中的单元（神经元）数量。我们从5个单元开始。我们还需要提供`input_shape`，它对应于`(time_steps, features)`，不包括批大小。
2. 一个`Dense`层：这是一个标准的完全连接层，将生成最终的输出预测。由于我们预测的是单个数字，它将有1个单元。

```python
# 导入TensorFlow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# 定义模型
model = Sequential([
    # 具有5个隐藏单元的SimpleRNN层。
    # input_shape是(时间步, 特征) -> (3, 1)
    SimpleRNN(5, input_shape=(n_time_steps, n_features)),

    # 输出层：具有1个单元的Dense层，用于预测单个后续值
    Dense(1)
])

# 显示模型架构
model.summary()
```

`model.summary()`的输出将显示各层、它们的输出形状以及参数 (parameter)数量。注意`SimpleRNN`层如何处理序列并输出`(None, 5)`的形状，其中`None`代表批大小，`5`是隐藏单元的数量。此输出（最终隐藏状态）随后被送入`Dense`层。

### 4. 编译模型

在训练之前，我们需要编译模型。这包括指定：

- `optimizer`（优化器）：用于更新模型权重 (weight)的算法（例如，'adam'、'rmsprop'）。Adam是一个常用且有效的选择。
- `loss`（损失函数 (loss function)）：衡量模型预测与实际目标值之间差异的函数。由于这是一个回归任务（预测连续值），均方误差（`'mse'`）是合适的。

```python
# 编译模型
model.compile(optimizer='adam', loss='mse')
```

### 5. 训练模型

我们现在可以使用`fit`方法训练模型。我们提供训练数据（`X_train`, `y_train`），指定`epochs`（遍历整个数据集的次数），并可选地设置`batch_size`。我们还将使用一小部分数据进行训练期间的验证，以监测模型在未见过的样本上的表现。

```python
# 训练模型
# 使用一部分数据进行验证（例如，20%）
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2, verbose=1)

# verbose=1显示进度条
# verbose=2每轮显示一行
# verbose=0不显示任何内容
```

在训练过程中，你会看到训练集和验证集的损失随轮次逐渐降低。这表明模型正在学习规律。

我们可以通过绘制损失来可视化训练过程：

```python
import matplotlib.pyplot as plt

# 绘制训练和验证损失值
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], color='#1c7ed6', label='训练损失')
plt.plot(history.history['val_loss'], color='#fd7e14', linestyle='--', label='验证损失')
plt.title('模型训练损失')
plt.ylabel('损失 (均方误差)')
plt.xlabel('轮次')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
```



![模型训练损失](plots/2551-0.json)



> 训练和验证损失随轮次降低，表明模型学习成功。

### 6. 进行预测

现在，让我们测试训练好的模型。我们将创建一个新的输入序列，像训练数据一样对其进行预处理（重塑和归一化 (normalization)），然后使用`model.predict`获取输出。记住将预测结果缩放回原始范围。

```python
# 示例：预测[50, 60, 70]之后的数字
# 预期预测值：80

input_sequence_raw = np.array([50, 60, 70])

# 1. 归一化
input_sequence_normalized = input_sequence_raw / 100.0

# 2. 重塑为(1, 时间步, 特征) -> (1, 3, 1)
input_sequence_reshaped = input_sequence_normalized.reshape((1, n_time_steps, n_features))

# 3. 预测
predicted_normalized = model.predict(input_sequence_reshaped)

# 4. 反归一化预测值
predicted_value = predicted_normalized[0, 0] * 100.0

print(f"输入序列: {input_sequence_raw}")
print(f"预测的下一个值: {predicted_value:.2f}")

# 预期输出:
# 输入序列: [50 60 70]
# 预测的下一个值: ~80.00 (可能略有偏差，例如79.85)
```

模型应该预测一个非常接近80的值，这表明它已经从训练数据中学到了简单的算术序列规律。

这个实践例子展示了使用`SimpleRNN`完成基本序列任务的基本步骤：生成数据、将其预处理成正确形状、构建模型架构、训练以及进行预测。虽然这个任务很简单，但其工作流程为处理更复杂的序列建模问题提供了基本方法，例如使用RNN、LSTM和GRU，我们将在接下来进行介绍。在处理更长序列时，你会遇到诸如梯度消失等挑战，这促使我们需要更先进的架构，如后续章节中将介绍的LSTM和GRU。

## 参考资料

- [Finding Structure in Time](https://onlinelibrary.wiley.com/doi/10.1207/s15516709cog1402_1) — Jeffrey L. Elman (1990)
  Journal: Cognitive Science; Publisher: John Wiley & Sons on behalf of the Cognitive Science Society; Volume: 14; Pages: 179-211; DOI: [10.1207/s15516709cog1402_1](https://doi.org/10.1207/s15516709cog1402_1)
  介绍简单循环网络（SRN，也称Elman网络）的开创性论文，它是现代深度学习框架中SimpleRNN层的基础。该文解释了这些网络如何通过维护上下文层来学习序列模式。
- [tf.keras.layers.SimpleRNN](https://www.tensorflow.org/api_docs/python/tf/keras/layers/SimpleRNN) — TensorFlow Developers (2023)
  TensorFlow Keras中SimpleRNN层的官方API文档。它详细说明了该层的参数、输入/输出形状要求及使用方法，直接支持了实践示例中的实现。
- [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python-second-edition) — François Chollet (2021)
  Publisher: Manning Publications
  Keras创建者撰写的深度学习实用指南。第六章侧重于循环神经网络，提供了与构建和训练用于序列任务的RNN相关的示例和解释。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本关于深度学习的综合性教科书。第10章讨论了循环和递归网络，为理解序列建模、RNN架构和训练挑战提供了理论基础。
