---
course: "rnns-and-sequence-modeling"
chapter: "sequence-modeling-applications"
lesson: "practice-time-series-forecasting"
sourceId: 2644
sourceUrl: "https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-9-sequence-modeling-applications/practice-time-series-forecasting"
title: "实战：时间序列预测"
description: "实现LSTM或GRU模型，用于解决时间序列预测的实际问题。"
order: 9
plots: ["plots/2644-0.json", "plots/2644-1.json", "plots/2644-2.json"]
sourceHash: "be225afa36e81d7b3148e93ea92e13d91d7c05453c23a8bbb3593fdd9548a269"
sourceCorrections: []
---

对于一个实际的时间序列预测问题，LSTM网络预测合成正弦波的未来数值。此练习展现了核心工作流程：准备序列数据、构建循环模型、训练模型，并评估其预测结果。虽然使用合成数据集以求清晰和可复现，但这些原理直接适用于天气模式、股票价格或传感器读数等时间序列。

本例中我们将使用带Keras API的TensorFlow。请确保您已安装TensorFlow和NumPy。

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt # 如果需要内部可视化，暂时使用Matplotlib，输出将使用Plotly

# 设置随机种子以保证结果可复现
tf.random.set_seed(42)
np.random.seed(42)
```

### 1. 生成和准备数据

首先，我们创建时间序列数据，一个带有噪声的正弦波，使其略具挑战性。

```python
# 生成正弦波数据
time = np.arange(0, 100, 0.1)
amplitude = np.sin(time) + np.random.normal(scale=0.15, size=len(time))

# 可视化生成的数据
```



![合成正弦波数据](plots/2644-0.json)



> 已生成带有随机噪声的正弦波。我们将使用此时间序列进行预测任务。

在将数据输入神经网络 (neural network)之前，通常需要进行缩放，通常缩放到0到1或-1到1的范围。这有助于优化算法更有效地收敛。我们将使用`MinMaxScaler`。

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
# 为缩放器重塑数据：(样本数, 特征数)
amplitude_scaled = scaler.fit_transform(amplitude.reshape(-1, 1))
```

现在，我们需要将这个单一时间序列转换为适合监督学习 (supervised learning)的输入/输出对。我们将使用滑动窗口方法：将过去值序列（`window_size`）作为输入（$X$），下一个单一值作为输出（$y$）。

```python
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size), 0])
        y.append(data[i + window_size, 0])
    return np.array(X), np.array(y)

window_size = 50 # 用于作为输入特征的过去时间步长数量
X, y = create_sequences(amplitude_scaled, window_size)

# 将数据拆分为训练集和测试集（例如，80%训练，20%测试）
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"原始数据长度: {len(amplitude)}")
print(f"缩放后数据长度: {len(amplitude_scaled)}")
print(f"序列数量 (X): {len(X)}")
print(f"训练序列: {len(X_train)}, 测试序列: {len(X_test)}")
```

LSTM（以及TensorFlow/Keras中的其他RNN层）期望输入数据为特定的3D形状：`(batch_size, timesteps, features)`。

- `batch_size`: 训练时一次处理的序列数量。
- `timesteps`: 每个输入序列的长度（即我们的`window_size`）。
- `features`: 每个时间步观察到的特征数量。由于我们是单变量时间序列（只有振幅），此值为1。

```python
# 为LSTM输入重塑X：[样本数, 时间步长, 特征数]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

print(f"X_train shape: {X_train.shape}") # 应为 (训练样本数, 窗口大小, 1)
print(f"X_test shape: {X_test.shape}")   # 应为 (测试样本数, 窗口大小, 1)
print(f"y_train shape: {y_train.shape}") # 应为 (训练样本数,)
print(f"y_test shape: {y_test.shape}")   # 应为 (测试样本数,)
```

### 2. 构建LSTM模型

我们将构建一个简单的LSTM模型。它将包含一个LSTM层，后接一个Dense输出层。

- **LSTM层：** 我们使用`layers.LSTM`。第一个参数 (parameter)是LSTM层中的单元（神经元）数量。这决定了隐藏状态和单元状态的维度。`input_shape=(window_size, 1)`告知该层预期接收的输入序列形状（此处无需指定批量大小）。
- **Dense层：** 一个标准的、带有一个神经元的全连接层，因为我们正在预测一个单一输出值（时间序列中的下一个点）。对于此类回归任务，激活函数 (activation function)通常是线性的。

```python
model = keras.Sequential(
    [
        layers.LSTM(units=64, input_shape=(window_size, 1)),
        layers.Dropout(0.2), # 添加Dropout进行正则化
        layers.Dense(units=1) # 输出层：预测下一个单一值
    ]
)

model.summary()
```

我们需要编译模型，指定优化器和损失函数 (loss function)。Adam是一种常见且有效的优化器。由于这是一个回归问题（预测连续值），均方误差（MSE）是一个合适的损失函数。


$$
\text{均方误差} = \frac{1}{N} \sum_{i=1}^{N} (y_{\text{真实}, i} - y_{\text{预测}, i})^2
$$


```python
model.compile(optimizer='adam', loss='mean_squared_error')
```

### 3. 训练模型

现在，我们使用准备好的训练数据来训练模型。我们指定训练轮数（遍历整个数据集的次数）和批量大小。我们还使用一部分训练数据作为验证集，以在训练期间监测模型在未见数据上的表现。

```python
history = model.fit(
    X_train,
    y_train,
    epochs=25,
    batch_size=32,
    validation_split=0.1, # 使用10%的训练数据作为验证
    verbose=1 # 显示进度
)
```

我们可以可视化训练和验证损失以检查收敛和过拟合 (overfitting)情况。



![模型训练历史](plots/2644-1.json)



> 训练和验证损失曲线随训练轮数的变化。理想情况下，两者都应降低并收敛。它们之间的大差距可能表明过拟合。

### 4. 进行预测和评估

模型训练完成后，让我们评估其在保留测试集上的表现。

```python
# 在测试数据上评估模型
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f"测试损失 (MSE): {test_loss:.6f}")

# 对测试数据进行预测
predictions_scaled = model.predict(X_test)

# 将预测结果和真实值反向缩放回原始比例
predictions = scaler.inverse_transform(predictions_scaled)
y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))
```

最后，让我们可视化模型预测值与测试集中实际值的匹配程度。



![时间序列预测结果](plots/2644-2.json)



> 实际时间序列值（测试集）与LSTM模型预测值的比较。

### 讨论

这个实践案例演示了使用简单LSTM模型预测正弦波。我们涵盖了数据生成、缩放、序列创建、模型构建（LSTM + Dense）、训练、评估（MSE）以及预测可视化。

结果显示LSTM能相当好地捕捉到潜在模式。提升表现的潜在后续步骤可能包括：

- 调整超参数 (parameter) (hyperparameter)（LSTM单元数量、窗口大小、学习率、批量大小）。
- 尝试更深的模型（堆叠LSTM层）。
- 使用GRU而非LSTM以比较表现和效率。
- 尝试不同的优化器或添加更多正则化 (regularization)。
  "\* 将此工作流程应用于更复杂的时间序列数据。"

此练习为处理各类时间序列预测问题奠定了基础，使用循环神经网络 (neural network) (RNN)。

## 参考资料

- [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735) — Sepp Hochreiter and Jürgen Schmidhuber (1997)
  Journal: Neural Computation; Publisher: MIT Press; Volume: 9; Pages: 1735-1780; DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
  介绍了LSTM架构，它是用于序列建模的循环神经网络的核心组成部分。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: 367-414 (Chapter 10)
  包含关于循环和递归神经网络的一章，详细解释了LSTM及其在序列建模中的应用。
- [Recurrent Neural Networks (RNNs) with Keras](https://www.tensorflow.org/guide/keras/rnn) — Scott Zhu, Francois Chollet (2023)
  Publisher: TensorFlow
  使用TensorFlow中Keras API实现循环神经网络（包括LSTM）的官方指南。
- [Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow](https://www.layitflat.com/products/hands-on-machine-learning-with-scikit-learn-keras-and-tensorflow) — Aurélien Géron (2022)
  Publisher: O'Reilly Media; Pages: 529-576 (Chapter 15, 4th edition)
  包含循环神经网络（包括LSTM）的实际示例和说明，使用TensorFlow和Keras应用于时间序列预测。
