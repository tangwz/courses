---
course: "introduction-to-neural-networks"
chapter: "building-first-neural-network"
lesson: "train-simple-classifier-practical"
sourceId: 2021
sourceUrl: "https://apxml.com/zh/courses/introduction-to-neural-networks/chapter-5-building-first-neural-network/train-simple-classifier-practical"
title: "动手实践：训练一个简单分类器"
description: "使用 Python 构建并训练一个用于分类任务的基础神经网络。"
order: 7
plots: ["plots/2021-0.json", "plots/2021-1.json", "plots/2021-2.json"]
sourceHash: "25c62e12aa459b5d3bd2f7f8525b2528715cd09f457e1fc31248d75ceab6fe93"
sourceCorrections: []
---

构建和训练神经网络 (neural network)涉及定义网络结构、初始化参数 (parameter)、通过前向传播计算预测、使用损失函数 (loss function)衡量误差、通过反向传播 (backpropagation)计算梯度，以及使用梯度下降 (gradient descent)更新参数。整合这些步骤，即可使用 Python 和 NumPy 从头开始构建并训练一个简单的神经网络分类器。

我们的目标是训练一个网络，使其能够根据两个输入特征将数据点分为两类之一。这是一个典型的二元分类任务，非常适合说明核心训练流程。

### 准备工作：导入和数据

首先，我们需要用于数值计算的主要工具 NumPy，以及一个用于可视化的库，比如 Plotly，以便查看我们的数据和结果。

```python
import numpy as np
import json # 用于嵌入 Plotly JSON

# 设置随机种子以保证结果可重现
np.random.seed(42)
```

接下来，生成一些简单的合成数据。我们将在二维平面中创建两个不同的点群，代表我们的两个类别（标记 (token)为 0 和 1）。

```python
def generate_data(n_samples=100, noise=0.1):
    """生成两个不同的数据点群。"""
    # 类别 0：围绕 (1, 1) 分布
    X0 = np.random.randn(n_samples // 2, 2) * noise + np.array([1, 1])
    Y0 = np.zeros((n_samples // 2, 1))

    # 类别 1：围绕 (-1, -1) 分布
    X1 = np.random.randn(n_samples // 2, 2) * noise + np.array([-1, -1])
    Y1 = np.ones((n_samples // 2, 1))

    X = np.vstack((X0, X1))
    Y = np.vstack((Y0, Y1))

    # 打乱数据
    permutation = np.random.permutation(n_samples)
    X = X[permutation]
    Y = Y[permutation]
    return X, Y

# 生成数据
X_train, Y_train = generate_data(n_samples=200, noise=0.2)

# 可视化数据
trace0 = {
    "type": "scatter", "mode": "markers",
    "x": X_train[Y_train.flatten() == 0, 0].tolist(),
    "y": X_train[Y_train.flatten() == 0, 1].tolist(),
    "name": "类别 0", "marker": {"color": "#fa5252", "size": 8} # 红色
}
trace1 = {
    "type": "scatter", "mode": "markers",
    "x": X_train[Y_train.flatten() == 1, 0].tolist(),
    "y": X_train[Y_train.flatten() == 1, 1].tolist(),
    "name": "类别 1", "marker": {"color": "#4c6ef5", "size": 8} # 蓝色
}
layout = {
    "title": {"text": "合成分类数据"},
    "xaxis": {"title": "特征 1"}, "yaxis": {"title": "特征 2"},
    "width": 600, "height": 400, "showlegend": True,
    "plot_bgcolor": "#e9ecef"
}
```



![合成分类数据](plots/2021-0.json)



> 合成数据集包含两个类别，它们在二维特征空间中视觉上是分离的。我们的网络应该学会在这两个类别之间划定界限。

### 网络结构

我们将定义一个简单的前馈网络：

- **输入层：** 2 个神经元（与我们的 2 个特征匹配）。
- **隐藏层：** 4 个神经元，使用 ReLU 激活函数 (activation function)。
- **输出层：** 1 个神经元，使用 Sigmoid 激活函数（适用于二元分类，输出 0 到 1 之间的概率）。

定义层的大小：

```python
n_input = X_train.shape[1]  # 特征数量 = 2
n_hidden = 4
n_output = 1
```

### 参数 (parameter)初始化

我们需要连接输入层和隐藏层（$W_1, b_1$）以及隐藏层和输出层（$W_2, b_2$）的权重 (weight)（$W$）和偏置 (bias)（$b$）。我们将用小的随机数初始化权重（乘以 0.01 以避免初始值过大），并将偏置设为零。

```python
def initialize_parameters(n_in, n_hid, n_out):
    """初始化权重和偏置。"""
    W1 = np.random.randn(n_in, n_hid) * 0.01
    b1 = np.zeros((1, n_hid))
    W2 = np.random.randn(n_hid, n_out) * 0.01
    b2 = np.zeros((1, n_out))

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters

parameters = initialize_parameters(n_input, n_hidden, n_output)
print("初始 W1 形状:", parameters["W1"].shape)
print("初始 b1 形状:", parameters["b1"].shape)
print("初始 W2 形状:", parameters["W2"].shape)
print("初始 b2 形状:", parameters["b2"].shape)
```

### 构成要素：激活、前向/反向传播 (backpropagation)、损失

现在，我们来实现前几节中讨论的核心函数。

**激活函数 (activation function)：**

```python
def sigmoid(Z):
    """Sigmoid 激活函数。"""
    A = 1 / (1 + np.exp(-Z))
    return A

def relu(Z):
    """ReLU 激活函数。"""
    A = np.maximum(0, Z)
    return A
```

**前向传播：**
该函数接收输入数据 $X$ 和网络参数 (parameter)，逐层进行线性变换并应用激活函数，返回最终预测 $A_2$ 和反向传播所需的中间值（缓存）。

```python
def forward_propagation(X, parameters):
    """执行前向传播。"""
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # 第 1 层（隐藏层）
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1) # 隐藏层使用 ReLU 激活函数

    # 第 2 层（输出层）
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2) # 输出层使用 Sigmoid 激活函数（二元分类）

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache
```

**损失函数 (loss function)：**
我们将使用二元交叉熵损失，它适用于输出为概率的二元分类问题。


$$
L = - \frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(a^{(i)}) + (1 - y^{(i)}) \log(1 - a^{(i)})]
$$


```python
def compute_loss(A2, Y):
    """计算二元交叉熵损失。"""
    m = Y.shape[0] # 样本数量
    # 添加一个小的 epsilon 以防止 log(0)
    epsilon = 1e-8
    loss = - (1 / m) * np.sum(Y * np.log(A2 + epsilon) + (1 - Y) * np.log(1 - A2 + epsilon))
    loss = np.squeeze(loss) # 确保损失是一个标量
    return loss
```

**反向传播：**
在这里，我们利用链式法则，从输出层反向计算梯度（$\frac{\partial L}{\partial W_1}, \frac{\partial L}{\partial b_1}, \frac{\partial L}{\partial W_2}, \frac{\partial L}{\partial b_2}$）。

```python
def backward_propagation(parameters, cache, X, Y):
    """执行反向传播以计算梯度。"""
    m = X.shape[0]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    Z1 = cache["Z1"]

    # 输出层梯度
    dZ2 = A2 - Y # BCE 损失对 Z2 的导数
    dW2 = (1 / m) * np.dot(A1.T, dZ2)
    db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)

    # 隐藏层梯度
    dA1 = np.dot(dZ2, W2.T)
    # ReLU 的梯度：如果 Z1 > 0 则为 1，否则为 0
    dZ1 = dA1 * (Z1 > 0)
    dW1 = (1 / m) * np.dot(X.T, dZ1)
    db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

    gradients = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return gradients
```

**参数更新：**
应用梯度下降 (gradient descent)规则：$W = W - \alpha \frac{\partial L}{\partial W}$，$b = b - \alpha \frac{\partial L}{\partial b}$。

```python
def update_parameters(parameters, gradients, learning_rate):
    """使用梯度下降更新参数。"""
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = gradients["dW1"]
    db1 = gradients["db1"]
    dW2 = gradients["dW2"]
    db2 = gradients["db2"]

    # 更新规则
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters
```

### 训练循环

现在我们将所有部分组合到主训练循环中。我们将对整个数据集进行多次迭代（训练周期），每次迭代执行前向传播、计算损失、执行反向传播 (backpropagation)并更新参数 (parameter)。

```python
def train_network(X, Y, n_hidden, num_epochs=10000, learning_rate=0.1, print_loss=True):
    """构建和训练神经网络。"""
    n_input = X.shape[1]
    n_output = Y.shape[1]
    m = X.shape[0]
    losses = []

    # 1. 初始化参数
    parameters = initialize_parameters(n_input, n_hidden, n_output)

    # 2. 训练循环（梯度下降）
    for i in range(num_epochs):
        # 3. 前向传播
        A2, cache = forward_propagation(X, parameters)

        # 4. 计算损失
        loss = compute_loss(A2, Y)
        losses.append(loss)

        # 5. 反向传播
        gradients = backward_propagation(parameters, cache, X, Y)

        # 6. 更新参数
        parameters = update_parameters(parameters, gradients, learning_rate)

        # 每 1000 个训练周期打印一次损失
        if print_loss and i % 1000 == 0:
            print(f"训练周期 {i} 后的损失: {loss:.4f}")

    if print_loss:
         print(f"训练周期 {num_epochs} 后的最终损失: {loss:.4f}")

    return parameters, losses

# --- 训练模型 ---
trained_parameters, training_losses = train_network(
    X_train, Y_train, n_hidden, num_epochs=20000, learning_rate=0.5
)
```

### 监测训练进度

监测训练的常用方法是绘制损失随训练周期的变化图。我们预期随着网络的学习，损失会降低。

```python
# 绘制损失曲线
epochs = list(range(len(training_losses)))
loss_trace = {
    "type": "scatter", "mode": "lines",
    "x": epochs, "y": training_losses,
    "name": "训练损失", "line": {"color": "#7048e8"} # 紫色
}
loss_layout = {
    "title": {"text": "训练损失随训练周期的变化"},
    "xaxis": {"title": "训练周期"}, "yaxis": {"title": "二元交叉熵损失"},
    "width": 600, "height": 400, "showlegend": False,
    "yaxis_range": [0, max(training_losses)*1.1] # Adjust y-axis slightly
}
```



![训练损失随训练周期的变化](plots/2021-1.json)



> 训练损失在训练周期中显著降低，表明网络正在学习最小化预测误差。

### 评估模型

我们通过计算训练集上的准确度来评估性能。我们将使用最终训练好的参数 (parameter)进行预测，并将其与真实标签进行比较。对于 Sigmoid 输出的二元分类，通常使用 0.5 作为阈值。

```python
def predict(parameters, X):
    """使用训练好的参数进行预测。"""
    A2, _ = forward_propagation(X, parameters)
    predictions = (A2 > 0.5).astype(int) # 阈值为 0.5
    return predictions

# 在训练集上进行预测
predictions = predict(trained_parameters, X_train)

# 计算准确度
accuracy = np.mean(predictions == Y_train) * 100
print(f"训练准确度: {accuracy:.2f}%")
```

你应该会看到较高的准确度（对于这个简单数据集可能接近 100%），这验证了网络正确地学会了分类数据点。

### 可视化决策边界（可选但有启发性）

为了更好地理解网络所学到的内容，我们可以可视化决策边界。这涉及创建一个覆盖特征空间的点网格，预测每个点的类别，并绘制与每个预测类别对应的区域。

```python
def plot_decision_boundary(pred_func, X, Y, parameters):
    """绘制模型学习到的决策边界。"""
    # 设置最小值和最大值并添加一些边距
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.01 # 网格步长

    # 生成点网格，点之间距离为 h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # 预测整个网格的函数值
    Z = pred_func(parameters, np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 绘制等高线
    contour_trace = {
        "type": 'contour',
        "x": xx[0,:].tolist(), "y": yy[:,0].tolist(), "z": Z.tolist(),
        "colorscale": [[0, '#ffc9c9'], [1, '#a5d8ff']], # 红色到蓝色
        "opacity": 0.4, "showscale": False, "name": "决策边界"
    }

    # 绘制原始数据点
    trace0 = {
        "type": "scatter", "mode": "markers",
        "x": X[Y.flatten() == 0, 0].tolist(), "y": X[Y.flatten() == 0, 1].tolist(),
        "name": "类别 0", "marker": {"color": '#fa5252', "size": 8} # 红色
    }
    trace1 = {
        "type": "scatter", "mode": "markers",
        "x": X[Y.flatten() == 1, 0].tolist(), "y": X[Y.flatten() == 1, 1].tolist(),
        "name": "类别 1", "marker": {"color": '#4c6ef5', "size": 8} # 蓝色
    }

    layout = {
        "title": {"text": "决策边界"},
        "xaxis": {"title": "特征 1"}, "yaxis": {"title": "特征 2"},
        "width": 600, "height": 450, "showlegend": True,
        "plot_bgcolor": "#e9ecef"
    }

    fig_data = [contour_trace, trace0, trace1]
    return {"layout": layout, "data": fig_data}

# 生成绘图 JSON
boundary_plot_json = plot_decision_boundary(predict, X_train, Y_train, trained_parameters)

# （可选）如果 Plotly 库可用，则显示图形，或仅显示 JSON
# import plotly.graph_objects as go
# fig = go.Figure(data=boundary_plot_json['data'], layout=boundary_plot_json['layout'])
# fig.show()

# 嵌入 JSON 以供网页显示
```



![Interactive chart](plots/2021-2.json)

## 参考资料

- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) — Michael Nielsen (2015)
  Publisher: Determination Press
  一本免费在线书籍，详细解释了神经网络，包括反向传播和梯度下降，侧重于使用 Python 和 NumPy 从零开始实现。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本权威教材，提供了深度学习的数学和概念基础，涵盖了网络架构、激活函数、损失函数、反向传播和优化算法。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://books.google.com/books/about/Hands_On_Machine_Learning_with_Scikit_Le.html?id=z17uEAAAQBAJ) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本实用指南，帮助读者理解并提供了机器学习和深度学习的实现示例，包括神经网络训练原则，可以补充从零开始的实现。
- [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/publication/pattern-recognition-and-machine-learning/) — Christopher M. Bishop (2006)
  Publisher: Springer; Pages: 778
  一本关于模式识别和机器学习理论基础的教材，提供了分类线性模型、神经网络和优化等主题的数学处理。
