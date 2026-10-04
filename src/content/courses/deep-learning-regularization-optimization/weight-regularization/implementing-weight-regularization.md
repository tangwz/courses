---
course: "deep-learning-regularization-optimization"
chapter: "weight-regularization"
lesson: "implementing-weight-regularization"
sourceId: 4915
sourceUrl: "https://apxml.com/zh/courses/deep-learning-regularization-optimization/chapter-2-weight-regularization/implementing-weight-regularization"
title: "权重正则化的实现"
description: "演示如何在常见的深度学习框架中添加 L1/L2 正则化。"
order: 8
plots: []
sourceHash: "ec32f44e53c88f4485f51a39c1c836273b7474611e9f53d9b0313da8ec6acd53"
sourceCorrections: []
---

L1、L2 和弹性网络正则化 (regularization)是用于防止深度学习 (deep learning)模型过拟合 (overfitting)的技术。在流行的深度学习框架（如 PyTorch）中实现这些正则化技术，需要调整标准训练循环，以考虑损失函数 (loss function)中增加的惩罚项。

### 实际中添加正则化 (regularization)

大多数深度学习 (deep learning)框架都提供了方便的方法来添加权重 (weight)正则化。L2（权重衰减）和 L1 正则化的实现策略通常略有不同，这是因为它们梯度特性不同。

#### L2 正则化（权重衰减）

L2 正则化非常普遍，以至于大多数优化器都直接将其作为参数 (parameter)实现，该参数通常称为 `weight_decay`。回顾一下 L2 正则化损失函数 (loss function)：


$$
L_{\text{总}} = L_{\text{原始}} + \frac{\lambda}{2} \sum_{w} w^2
$$


计算相对于权重 $w_i$ 的梯度时，L2 惩罚项的导数就是 $\lambda w_i$。在权重更新步骤（例如，SGD 中），更新规则变为：


$$
w_i \leftarrow w_i - \eta \left( \frac{\partial L_{\text{原始}}}{\partial w_i} + \lambda w_i \right)
$$


这可以改写为：


$$
w_i \leftarrow (1 - \eta \lambda) w_i - \eta \frac{\partial L_{\text{原始}}}{\partial w_i}
$$


这个更新规则表示，在每一步中，权重在应用标准梯度更新之前会首先按 $(1 - \eta \lambda)$ 的因子进行收缩。这就是为什么 L2 正则化常被称为“权重衰减”。

在 PyTorch 中，通过在初始化优化器时指定 `weight_decay` 参数，可以轻松添加权重衰减。这个参数对应于 L2 惩罚公式中的 $\lambda$ 超参数 (hyperparameter)。

```python
import torch
import torch.optim as optim
from torch import nn

# 假设 'model' 是你定义的神经网络
# Example: model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 1))

# 定义原始损失函数
criterion = nn.MSELoss() 

# 定义带权重衰减（L2 正则化）的优化器（例如 Adam）
# lambda_l2 是正则化强度
lambda_l2 = 0.01 
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=lambda_l2)

# --- 训练循环内部 ---
# 假设输入和目标已准备好
# outputs = model(inputs)
# loss = criterion(outputs, targets)

# 权重衰减由优化器在 step() 期间自动处理
# optimizer.zero_grad()
# loss.backward() 
# optimizer.step() 
# --- 训练循环代码片段结束 ---

print(f"优化器已使用权重衰减（L2 lambda）：{lambda_l2}")
```

在 PyTorch 中，使用 `weight_decay` 参数是实现 L2 正则化的标准且最有效的方法。

#### L1 正则化

L1 正则化添加一个与权重绝对值之和成比例的惩罚项：


$$
L_{\text{总}} = L_{\text{原始}} + \lambda_1 \sum_{w} |w|
$$


L1 惩罚项的梯度是 $\lambda_1 \text{sign}(w)$，这会带来问题，因为符号函数在 $w=0$ 处未定义，其导数（某些更高级的优化器需要）在 $w=0$ 之外的任何地方都是零，而在 $w=0$ 处是无穷大。因此，L1 正则化通常不会像权重衰减那样直接在优化器中实现。

相反，我们通常在执行反向传播 (backpropagation)之前，手动将 L1 惩罚项添加到损失中。

```python
import torch
import torch.optim as optim
from torch import nn

# 假设 'model' 是你定义的神经网络
# model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 1))

# 定义原始损失函数
criterion = nn.MSELoss() 

# 定义不带内置 L1 的优化器（例如 SGD）
optimizer = optim.SGD(model.parameters(), lr=0.001) 

# L1 正则化强度
lambda_l1 = 0.005 

# --- 训练循环内部 ---
# 假设输入和目标已准备好
# outputs = model(inputs)
# original_loss = criterion(outputs, targets)

# 计算 L1 惩罚项
l1_penalty = 0
for param in model.parameters():
    # 确保我们只惩罚权重，不惩罚偏置（可选但常见）
    if param.dim() > 1: 
        l1_penalty += torch.sum(torch.abs(param))

# 将 L1 惩罚项添加到原始损失中
total_loss = original_loss + lambda_l1 * l1_penalty

# 反向传播总损失
# optimizer.zero_grad()
# total_loss.backward() 
# optimizer.step()
# --- 训练循环代码片段结束 ---

print(f"手动添加 L1 惩罚项，lambda 值为：{lambda_l1}")
# print(f"原始损失示例: {original_loss.item():.4f}") # 占位符值
# print(f"L1 惩罚项示例: {l1_penalty.item():.4f}")       # 占位符值
# print(f"总损失示例: {total_loss.item():.4f}")       # 占位符值
```

在上面的代码片段中，我们遍历模型的参数，计算它们的绝对值之和（即 $L_1$ 范数），然后通过超参数 $\lambda_1$ 进行缩放，并将其添加到由准则函数计算的原始损失中。反向传播接着根据这个组合的 `total_loss` 计算梯度。请注意，通常只对权重矩阵/张量（当 `param.dim() > 1` 时）进行惩罚，而不对偏置 (bias)向量 (vector)进行惩罚，尽管这是一个设计选择。

#### 弹性网络正则化

弹性网络结合了 L1 和 L2 惩罚项：


$$
L_{\text{总}} = L_{\text{原始}} + \lambda_1 \sum_{w} |w| + \frac{\lambda_2}{2} \sum_{w} w^2
$$


要实现弹性网络，我们只需结合上述两种方法：使用优化器的 `weight_decay` 参数处理 L2 部分（$\lambda_2$），并手动将 L1 惩罚项（$\lambda_1$）添加到损失中。

```python
import torch
import torch.optim as optim
from torch import nn

# 假设 'model' 是你定义的神经网络
# model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 1))

# 定义原始损失函数
criterion = nn.MSELoss() 

# 正则化强度
lambda_l1 = 0.005 
lambda_l2 = 0.01 

# 定义带权重衰减以处理 L2 部分的优化器
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=lambda_l2) 

# --- 训练循环内部 ---
# 假设输入和目标已准备好
# outputs = model(inputs)
# original_loss = criterion(outputs, targets)

# 计算 L1 惩罚项
l1_penalty = 0
for param in model.parameters():
    if param.dim() > 1:
        l1_penalty += torch.sum(torch.abs(param))

# 将 L1 惩罚项添加到原始损失中
# L2 惩罚项由优化器的 weight_decay 处理
total_loss = original_loss + lambda_l1 * l1_penalty

# 反向传播总损失（L1 部分 + 原始损失）
# optimizer.zero_grad()
# total_loss.backward() 
# optimizer.step() # 优化器在更新期间应用 L2 衰减
# --- 训练循环代码片段结束 ---

print(f"实现弹性网络，L1 lambda: {lambda_l1}, L2 lambda (weight_decay): {lambda_l2}")
```

### 选择正则化 (regularization)强度（$\lambda$）

正则化强度 $\lambda$、$\lambda_1$ 和 $\lambda_2$ 都是超参数 (parameter) (hyperparameter)，它们控制着惩罚项的影响。

- 如果 $\lambda$ 过小，正则化效果会很弱，模型可能仍然过拟合 (overfitting)。
- 如果 $\lambda$ 过大，模型可能会受到过多限制（欠拟合 (underfitting)），这将优先考虑小权重 (weight)而非很好地拟合数据。

找到这些超参数的合适值是模型调优过程的一部分。常见的值范围通常是对数尺度的，例如从 $10^{-6}$ 到 $10^{-1}$。通常使用网格搜索或随机搜索等方法在验证集上寻找 $\lambda$ 的良好取值。我们将在第 7 章更详细地讨论超参数调优。

借助这些实现方法，你现在可以有效地将 L1、L2 或弹性网络正则化应用于你的神经网络 (neural network)模型，为你提供了强大的工具来对抗过拟合并提高泛化能力。下一节将提供一个实践练习，以巩固这些知识点。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  全面涵盖了深度学习中的正则化技术，包括L1、L2和权重衰减的理论和实践。
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) — Ilya Loshchilov, Frank Hutter (2019)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1711.05101](https://doi.org/10.48550/arXiv.1711.05101)
  详细分析了自适应优化器中的权重衰减，将其与标准L2正则化区分开来，并解释了其有效性。
- [Regularization and Variable Selection via the Elastic Net](https://academic.oup.com/jrssb/article-abstract/67/2/301/7086088) — Hui Zou, Trevor Hastie (2005)
  Journal: Journal of the Royal Statistical Society: Series B (Statistical Methodology); Publisher: Wiley-Blackwell Publishing Ltd; Volume: 67; Pages: 301-320; DOI: [10.1111/j.1467-9868.2005.00503.x](https://doi.org/10.1111/j.1467-9868.2005.00503.x)
  介绍了弹性网络方法，该方法结合L1和L2正则化以提高高维数据中的性能。
- [torch.optim package - Optimizers](https://pytorch.org/docs/stable/optim.html) — PyTorch Developers (2025)
  Publisher: PyTorch Foundation
  PyTorch优化器的官方文档，详细说明了`weight_decay`参数及其在L2正则化中的用法。
