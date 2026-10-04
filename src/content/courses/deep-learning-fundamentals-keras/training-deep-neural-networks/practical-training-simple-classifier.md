---
course: "deep-learning-fundamentals-keras"
chapter: "training-deep-neural-networks"
lesson: "practical-training-simple-classifier"
sourceId: 4928
sourceUrl: "https://apxml.com/zh/courses/deep-learning-fundamentals-keras/chapter-3-training-deep-neural-networks/practical-training-simple-classifier"
title: "动手实践：训练一个简单分类器"
description: "在一个数据集上训练一个基本分类模型，监测其性能，并评估结果。"
order: 9
plots: []
sourceHash: "19509d7ed1baa8535c9c253fe011913c08a3cf72837646127ddd07c8ed317c9a"
sourceCorrections: []
---

构建一个简单神经网络 (neural network)包括多个步骤：使用恰当的设置对其进行编译，在合成数据上进行训练，观察其学习过程，并最终评估其性能。这个实践示例展示了定义、编译、训练和评估Keras模型的完整流程。

### 目标

我们的目标是训练一个二分类器。我们将生成一个非线性可分的人工数据集，构建一个简单的前馈神经网络 (neural network)，使用`fit()`方法进行训练，观察其训练进展，并使用`evaluate()`方法评估其学习效果。

### 设置

首先，请确保已安装所需库。我们将使用`keras`，`numpy`进行数值运算，`scikit-learn`用于生成和划分人工数据集，以及`plotly`用于可视化（如果您想运行绘图代码）。

```python
import numpy as np
import keras
from keras import layers
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.io as pio

# 可选：设置Plotly图表的默认主题
pio.templates.default = "plotly_white"
```

### 1. 数据生成和准备

我们需要数据来训练分类器。scikit-learn中的`make_moons`函数非常适合此目的，因为它会生成两个交错的半圆，这样的数据集需要非线性决策边界，因此非常适合神经网络 (neural network)。

```python
# 生成人工数据
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)

# 将数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 对特征进行标准化（对许多神经网络任务都很重要）
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"训练数据形状: {X_train.shape}")
print(f"训练标签形状: {y_train.shape}")
print(f"测试数据形状: {X_test.shape}")
print(f"测试标签形状: {y_test.shape}")
print(f"样本数据点（标准化后）: {X_train[0]}")
print(f"样本标签: {y_train[0]}")
```

此代码生成1000个数据点，每个点有2个特征（例如二维平面上的坐标）。它添加了一些噪声，使分类略有难度。然后我们将其划分为80%用于训练和20%用于测试。最后，我们使用`StandardScaler`对特征进行标准化。标准化可确保所有特征在网络内的距离计算中更均匀地发挥作用，并有助于优化算法更快收敛。

### 2. 构建神经网络 (neural network)模型

现在，我们使用Keras API定义一个简单的序贯模型，类似于在第2章中学习的内容。我们的网络将包含：

- 在第一个`Dense`层中隐式定义的输入层，预期有2个特征。
- 一个包含8个神经元和ReLU激活函数 (activation function)的隐藏`Dense`层。ReLU是隐藏层常用的选择。
- 一个包含1个神经元（用于二分类）和Sigmoid激活函数的输出`Dense`层。Sigmoid输出0到1之间的值，可以解释为正类别的概率。

```python
# 使用序贯API定义模型架构
model = keras.Sequential(
    [
        layers.Input(shape=(2,)), # 显式指定输入形状
        layers.Dense(8, activation="relu", name="hidden_layer"),
        layers.Dense(1, activation="sigmoid", name="output_layer")
    ]
)

# 显示模型架构
model.summary()
```

`model.summary()`提供了各层、其输出形状以及每层中可训练参数 (parameter)（权重 (weight)和偏差）数量的快速概览。

### 3. 编译模型

在训练之前，我们必须编译模型。此步骤配置学习过程。如本章所述，我们需要指定：

- **优化器：** 用于更新网络权重 (weight)的算法。`Adam`是一个流行且通常有效的选择。
- **损失函数 (loss function)：** 衡量模型在训练数据上的表现。由于这是一个二分类问题，并且我们的输出层使用Sigmoid激活，因此`binary_crossentropy`是合适的损失函数。
- **评估指标：** 用于监测训练和测试步骤。我们将使用`accuracy`来查看正确分类样本的百分比。

```python
# 编译模型
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("模型编译成功。")
```

### 4. 训练模型

现在，我们使用`fit()`方法训练模型。我们提供训练数据（`X_train`，`y_train`），指定`epochs`数量（遍历整个数据集的次数），以及`batch_size`（模型权重 (weight)更新前处理的样本数量）。我们还提供测试集作为`validation_data`，以便在训练期间监测模型在未见数据上的表现。这有助于我们识别潜在的过拟合 (overfitting)。

`fit()`方法返回一个`History`对象，其中包含每个周期的训练损失、验证损失、训练准确率和验证准确率的记录。

```python
# 训练模型
print("开始训练...")
history = model.fit(X_train,
                    y_train,
                    epochs=50,           # 遍历整个数据集的次数
                    batch_size=32,       # 每次梯度更新的样本数量
                    validation_data=(X_test, y_test), # 每个周期结束时评估损失和指标的数据
                    verbose=1)           # 将verbose设置为1以查看每个周期的进展，0为静默模式
print("训练结束。")

# 训练历史记录
print("\n训练历史记录键：", history.history.keys())
```

将`verbose=1`可显示每个周期的进展，包括训练集和验证集上的损失和准确率。

### 5. 监测训练性能

`history`对象非常有用。让我们绘制训练和验证损失以及准确率随周期的变化，以直观查看学习过程。

````python
# 从history对象中提取数据
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
epochs_range = range(1, len(loss_values) + 1)

# 创建损失图
fig_loss = go.Figure()
fig_loss.add_trace(go.Scatter(x=list(epochs_range), y=loss_values,
                             mode='lines+markers', name='训练损失',
                             line=dict(color='#4263eb'), marker=dict(color='#4263eb')))
fig_loss.add_trace(go.Scatter(x=list(epochs_range), y=val_loss_values,
                             mode='lines+markers', name='验证损失',
                             line=dict(color='#f76707'), marker=dict(color='#f76707')))
fig_loss.update_layout(title='训练和验证损失',
                       xaxis_title='周期',
                       yaxis_title='损失',
                       width=700, height=400)

# 创建准确率图
fig_acc = go.Figure()
fig_acc.add_trace(go.Scatter(x=list(epochs_range), y=acc_values,
                            mode='lines+markers', name='训练准确率',
                            line=dict(color='#1c7ed6'), marker=dict(color='#1c7ed6')))
fig_acc.add_trace(go.Scatter(x=list(epochs_range), y=val_acc_values,
                            mode='lines+markers', name='验证准确率',
                            line=dict(color='#fd7e14'), marker=dict(color='#fd7e14')))
fig_acc.update_layout(title='训练和验证准确率',
                      xaxis_title='周期',
                      yaxis_title='准确率',
                      width=700, height=400)

# 显示图表（这需要您的环境中安装并配置好Plotly）
# 在Jupyter notebook中，这些图表可能会自动显示。否则请使用fig.show()。
# 用于网页渲染：
print("```plotly")
print(fig_loss.to_json(pretty=False))
print("```")
print("> 训练和验证损失随周期变化。")
print("```plotly")
print(fig_acc.to_json(pretty=False))
print("```")
print("> 训练和验证准确率随周期变化。")

# 如果不在自动显示Plotly的环境中运行：
# fig_loss.show()
# fig_acc.show()
````

观察这些图表：

- **损失图：** 通常我们希望看到训练和验证损失都下降并收敛。如果验证损失开始增加而训练损失持续下降，这是过拟合 (overfitting)的迹象。
- **准确率图：** 我们希望看到训练和验证准确率都增加并收敛。训练准确率和验证准确率之间的较大差距也表明存在过拟合。

这些图表直观地反映了训练参数 (parameter)（如周期数）的选择是否恰当，以及模型对未见数据是否具备良好的泛化能力。

### 6. 评估模型

最后，让我们使用`evaluate()`方法评估训练好的模型在测试集上的表现。这会提供模型在训练更新期间从未见过的数据上的最终损失和准确率。

```python
# 在测试集上评估模型
loss, accuracy = model.evaluate(X_test, y_test, verbose=0) # verbose=0 表示静默评估

print(f"\n测试损失: {loss:.4f}")
print(f"测试准确率: {accuracy:.4f}")
```

输出显示了模型在测试集上的最终性能指标。准确率接近1.0表明模型在此数据集上表现良好。

### 总结

在本次实践练习中，我们回顾了使用Keras训练一个简单神经网络 (neural network)分类器的完整流程：

1. 我们准备了一个适合二分类的人工数据集。
2. 我们定义了一个包含`Dense`层的神经网络架构。
3. 我们编译了模型，指定了`adam`优化器、`binary_crossentropy`损失函数 (loss function)和`accuracy`指标。
4. 我们使用`fit()`方法训练了模型，提供了训练和验证数据，并设置了周期数和批量大小。
5. 我们通过绘制损失和准确率曲线来监测训练过程，并检查了收敛情况和过拟合 (overfitting)。
6. 我们使用`evaluate()`方法评估了模型在未见测试集上的最终表现。

此流程构成了在Keras中训练更复杂深度学习 (deep learning)模型的基础，结合了本章所讲的主要内容。您现在已亲身体验了使神经网络从数据中学习所需的基本步骤。

## 参考资料

- [Keras documentation](https://keras.io) — Keras team (2024)
  Keras深度学习库的官方文档，提供构建、训练和评估神经网络的全面指南和API参考。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; DOI: [10.7551/mitpress/9780262035613.001.0001](https://doi.org/10.7551/mitpress/9780262035613.001.0001)
  一本关于深度学习的综合性教材，涵盖神经网络架构、训练算法、优化和正则化技术。
- [scikit-learn: Machine Learning in Python](https://scikit-learn.org/stable/) — Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel, et al. (2024)
  scikit-learn库的官方文档，提供了机器学习中数据生成、预处理（缩放）和模型选择工具的详细信息。
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — Diederik P. Kingma, Jimmy Ba (2015)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980)
  介绍了Adam优化算法，该算法广泛用于训练深度神经网络，详细阐述了其理论基础和实际优势。
