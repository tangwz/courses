---
course: "deep-learning-regularization-optimization"
chapter: "generalization-challenge"
lesson: "practice-visualizing-overfitting"
sourceId: 4889
sourceUrl: "https://apxml.com/zh/courses/deep-learning-regularization-optimization/chapter-1-generalization-challenge/practice-visualizing-overfitting"
title: "动手：过拟合的可视化"
description: "动手实践，使用学习曲线或模型预测来可视化过拟合。"
order: 8
plots: []
sourceHash: "1e155e1c76e68421d55c1f2296e396a4f82f14f62d0330534a68a584533b9d13"
sourceCorrections: []
---

在前面的章节中，我们讲解了泛化、过拟合 (overfitting)和欠拟合 (underfitting)的思想。你学习了学习曲线如何作为一种诊断工具，显示训练表现和验证表现之间的差异。现在，让我们将这些知识付诸实践，生成一个可能出现过拟合的情境并直接将其可视化。这项练习将巩固你对如何运用性能指标和模型行为来发现过拟合的理解。

我们将使用一个简单的回归问题：将模型拟合到从已知函数生成并添加了噪声的数据。通过比较一个简单模型和一个复杂模型，我们可以观察到复杂模型如何学习训练数据中的噪声，从而导致在未见过的数据上表现不佳。

### 场景设置

为了顺利进行，请准备好Python环境，并安装PyTorch、NumPy和Matplotlib（或其他绘图库，如Plotly或Seaborn）。

“我们将基于一个简单的底层函数（如正弦波或低次多项式）创建一个小型合成数据集，并添加一些随机噪声。这种噪声模拟了数据中存在的不完美和随机变化。”

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 生成合成数据
torch.manual_seed(42) # 为了结果可复现
n_samples = 30
X = torch.linspace(-np.pi, np.pi, n_samples).unsqueeze(1)
# 真实函数：sin(x) + 小线性趋势
y_true = torch.sin(X) + 0.1 * X
# 添加噪声
y = y_true + torch.randn(X.size()) * 0.2

# 划分为训练集和验证集（简单划分仅为演示）
n_train = 20
X_train, y_train = X[:n_train], y[:n_train]
X_val, y_val = X[n_train:], y[n_train:]

# 绘图函数
def plot_data(X_train, y_train, X_val, y_val, X_full=None, y_pred=None, model_name=None):
    plt.figure(figsize=(8, 5))
    plt.scatter(X_train.numpy(), y_train.numpy(), label='Training Data', c='#1f77b4', s=50, alpha=0.7)
    plt.scatter(X_val.numpy(), y_val.numpy(), label='Validation Data', c='#ff7f0e', s=50, alpha=0.7, marker='x')
    if X_full is not None and y_pred is not None:
        plt.plot(X_full.numpy(), y_pred.detach().numpy(), label=f'{model_name} Prediction', c='#2ca02c', linewidth=2)
    plt.xlabel("Feature (X)")
    plt.ylabel("Target (y)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title("Synthetic Data and Model Fit")
    plt.show()

# 绘制生成的数据
# plot_data(X_train, y_train, X_val, y_val) # 取消注释以查看原始数据
```

### 模型定义

现在，让我们使用PyTorch定义两个神经网络 (neural network)模型。一个将相对简单，另一个将明显更复杂（具有更多层或神经元），使其在我们的小型数据集上容易过拟合 (overfitting)。

```python
# 简单模型（可能欠拟合或拟合良好）
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 一个简单的线性层可能欠拟合，我们尝试一个小型的多层感知器
        self.layer1 = nn.Linear(1, 10)
        self.activation = nn.ReLU()
        self.output_layer = nn.Linear(10, 1)

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.output_layer(x)
        return x

# 复杂模型（容易过拟合）
class ComplexModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 更多层和神经元
        self.layer1 = nn.Linear(1, 128)
        self.activation1 = nn.ReLU()
        self.layer2 = nn.Linear(128, 128)
        self.activation2 = nn.ReLU()
        self.layer3 = nn.Linear(128, 64)
        self.activation3 = nn.ReLU()
        self.output_layer = nn.Linear(64, 1)

    def forward(self, x):
        x = self.activation1(self.layer1(x))
        x = self.activation2(self.layer2(x))
        x = self.activation3(self.layer3(x))
        x = self.output_layer(x)
        return x

simple_model = SimpleModel()
complex_model = ComplexModel()
```

### 模型训练

我们将使用标准的训练循环来训练这两个模型。我们将使用均方误差（MSE）作为损失函数 (loss function)，并使用Adam优化器（我们稍后会详细介绍，但这是一个常见默认设置）。我们将跟踪每个周期的训练损失和验证损失。

```python
def train_model(model, X_train, y_train, X_val, y_val, epochs=2000, lr=0.005):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train() # 将模型设置为训练模式

        # 前向传播
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 在验证集上评估
        model.eval() # 将模型设置为评估模式
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        if (epoch + 1) % 200 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

    return train_losses, val_losses

print("训练简单模型...")
simple_train_losses, simple_val_losses = train_model(simple_model, X_train, y_train, X_val, y_val)

print("\n训练复杂模型...")
complex_train_losses, complex_val_losses = train_model(complex_model, X_train, y_train, X_val, y_val)
```

### 可视化 1：学习曲线

学习曲线绘制了训练损失和验证损失随周期变化的情况。它们是诊断拟合问题的主要工具。

- **良好拟合：** 训练损失和验证损失下降并收敛。
- **欠拟合 (underfitting)：** 两种损失均保持高位或过早稳定。
- **过拟合 (overfitting)：** 训练损失持续下降，而验证损失开始增加或在明显更高的水平上稳定。

让我们绘制两个模型的学习曲线。

```plotly
{"layout": {"title": "学习曲线：简单模型 vs. 复杂模型", "xaxis": {"title": "周期"}, "yaxis": {"title": "均方误差损失", "type": "log"}, "legend": {"title": "图例"}, "width": 700, "height": 450}, "data": [{"x": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000], "y": [0.15, 0.10, 0.08, 0.06, 0.05, 0.045, 0.04, 0.038, 0.037, 0.036, 0.035, 0.034, 0.033, 0.032, 0.031, 0.030, 0.030, 0.029, 0.029, 0.028], "mode": "lines", "name": "简单 - 训练损失", "line": {"color": "#1f77b4"}}, {"x": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000], "y": [0.18, 0.14, 0.12, 0.10, 0.09, 0.08, 0.075, 0.07, 0.068, 0.067, 0.066, 0.065, 0.064, 0.063, 0.062, 0.061, 0.061, 0.060, 0.060, 0.059], "mode": "lines", "name": "简单 - 验证损失", "line": {"color": "#aec7e8", "dash": "dash"}}, {"x": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000], "y": [0.05, 0.02, 0.01, 0.005, 0.003, 0.002, 0.0015, 0.001, 0.0008, 0.0006, 0.0005, 0.0004, 0.0003, 0.00025, 0.0002, 0.00018, 0.00016, 0.00015, 0.00014, 0.00013], "mode": "lines", "name": "复杂 - 训练损失", "line": {"color": "#ff7f0e"}}, {"x": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000], "y": [0.10, 0.08, 0.09, 0.11, 0.13, 0.15, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.41, 0.42], "mode": "lines", "name": "复杂 - 验证损失", "line": {"color": "#ffbb78", "dash": "dash"}}]
```

> 简单模型和复杂模型的学习曲线比较。注意y轴采用对数刻度。复杂模型显示训练损失（下降）和验证损失（增加）之间存在明显背离，表明过拟合。简单模型的损失收敛得更紧密。（注意：实际损失值取决于具体的运行；这些仅作说明。）

观察学习曲线。复杂模型的训练损失可能降得很低，表明它很好地拟合了训练数据。然而，它的验证损失在最初下降后，可能会开始增加或在远高于训练损失的水平上稳定。这种不断扩大的差异是经典的过拟合迹象。简单模型可能差异较小，或者两种损失都可能稳定，这表明其复杂度不足以导致过拟合（或者甚至有点欠拟合）。

### 可视化 2：模型预测

另一种可视化过拟合 (overfitting)的方法，尤其是在回归问题中，是绘制模型的预测与实际数据点的对比图。过拟合的模型通常会显示过度的波动，试图穿过每个训练数据点，包括噪声。

```python
# 绘制预测结果
simple_model.eval()
complex_model.eval()

with torch.no_grad():
    # 使用完整的X范围来绘制曲线
    y_pred_simple = simple_model(X) 
    y_pred_complex = complex_model(X)

# 绘制简单模型的拟合
plot_data(X_train, y_train, X_val, y_val, X, y_pred_simple, "Simple Model")

# 绘制复杂模型的拟合
plot_data(X_train, y_train, X_val, y_val, X, y_pred_complex, "Complex Model")

# 可选：绘制真实的底层函数
plt.figure(figsize=(8, 5))
plt.scatter(X_train.numpy(), y_train.numpy(), label='Training Data', c='#1f77b4', s=50, alpha=0.7)
plt.scatter(X_val.numpy(), y_val.numpy(), label='Validation Data', c='#ff7f0e', s=50, alpha=0.7, marker='x')
plt.plot(X.numpy(), y_true.numpy(), label='True Function', c='black', linestyle=':', linewidth=2)
plt.plot(X.numpy(), y_pred_complex.detach().numpy(), label='Complex Model Prediction', c='#d62728', linewidth=2) # 过拟合模型使用红色
plt.xlabel("Feature (X)")
plt.ylabel("Target (y)")
plt.legend()
plt.ylim(y.min()-0.5, y.max()+0.5) # 调整y轴限制以获得更好的视图
plt.grid(True, linestyle='--', alpha=0.5)
plt.title("Complex Model vs. True Function")
plt.show()
```

当你查看这些图时：

1. **简单模型：** 预测曲线应相对平滑。它可能无法捕捉训练数据的所有细节，但它可能代表了总体趋势，没有剧烈波动。
2. **复杂模型：** 该模型的预测曲线可能会剧烈摆动，尤其是在训练点之间。它过于努力地适应每个训练数据点，包括噪声。这种复杂的曲线通常会导致验证点（橙色“x”标记 (token)）的预测效果不佳，并且很可能在从真实底层函数中抽取的任何新数据上表现不佳。与真实函数（如果绘制）的比较凸显了过拟合模型偏离的程度。

### 要点总结

这项实践练习体现了两种识别过拟合 (overfitting)的可视化方法：

1. **学习曲线：** 观察训练损失和验证损失随周期变化的差异。
2. **模型预测：** 看到模型拟合了训练数据中的噪声，导致生成一个过于复杂的函数，与底层模式不符。

识别过拟合是第一步。后续章节将介绍正则化 (regularization)（L1/L2、Dropout、批量归一化 (normalization)）等方法以及复杂的优化算法，它们旨在解决这个问题，并帮助我们的模型更好地泛化到未见过的数据。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  对深度学习的全面介绍，涵盖了泛化、过拟合和模型复杂性等基础概念。
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) — PyTorch Contributors (2024)
  Publisher: PyTorch Foundation
  PyTorch 深度学习框架的官方文档，提供了实践中使用的模块、优化器和数据处理的详细信息。
- [Machine Learning Yearning](https://www.deeplearning.ai/machine-learning-yearning/) — Andrew Ng (2018)
  Publisher: deeplearning.ai
  构建机器学习系统的实用指南，提供了使用学习曲线诊断和解决过拟合和欠拟合的策略。
