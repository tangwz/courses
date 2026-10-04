---
course: "introduction-to-deep-learning"
chapter: "activation-functions-architecture"
lesson: "practice-implementing-activations"
sourceId: 5033
sourceUrl: "https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-2-activation-functions-architecture/practice-implementing-activations"
title: "动手实践：实现不同的激活函数"
description: "使用 Python 可视化并实现各种激活函数。"
order: 9
plots: ["plots/5033-0.json"]
sourceHash: "9238a487b26a26ae1dfd8f7a9df09177d82adc875ee5dfbab579cc40d22d45e7"
sourceCorrections: []
---

Sigmoid、Tanh、ReLU 及其变体等多种激活函数 (activation function)在 Python 中实现，并对其行为进行可视化。绘制这些函数有助于理解它们的输出范围、饱和点和非线性等特性，这些在设计神经网络 (neural network)层时是重要的考量。

我们将使用 NumPy 进行数学计算，并使用 Plotly 创建交互式可视化。

首先，我们导入必要的库：

```python
import numpy as np
```

现在，我们根据它们的数学公式定义激活函数：

```python
# Sigmoid 函数
def sigmoid(x):
  """计算 Sigmoid 激活值。"""
  return 1 / (1 + np.exp(-x))

# 双曲正切 (Tanh) 函数
def tanh(x):
  """计算 Tanh 激活值。"""
  return np.tanh(x)

# 整流线性单元 (ReLU) 函数
def relu(x):
  """计算 ReLU 激活值。"""
  return np.maximum(0, x)

# 泄漏整流线性单元 (Leaky ReLU) 函数
def leaky_relu(x, alpha=0.01):
  """计算 Leaky ReLU 激活值。"""
  return np.maximum(alpha * x, x)
```

接下来，我们将生成一系列输入值，这些值通常以零为中心，以便观察函数在不同输入下的表现。

```python
# 生成从 -5 到 5 的输入值
x = np.linspace(-5, 5, 100)

# 计算每个激活函数的输出
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_relu = relu(x)
y_leaky_relu = leaky_relu(x)
```

在计算出每个函数的输入值和对应的输出后，我们现在可以绘制它们。这种可视化方法能够直接比较它们的形状和特点。



![常见激活函数](plots/5033-0.json)



> Sigmoid、Tanh、ReLU 和 Leaky ReLU 激活函数的比较。注意它们不同的输出范围：Sigmoid（0 到 1），Tanh（-1 到 1），ReLU（0 到 无穷大），以及 Leaky ReLU（负无穷大到正无穷大，对负输入具有小斜率）。

观察这个图，我们可以清楚地看到前面讨论的特点：

- **Sigmoid** 和 **Tanh** 呈 S 形，对于大量正或负输入会饱和（变平）。Tanh 是零中心化的，这在训练时通常比 Sigmoid 有益。
- **ReLU** 对正输入是线性的，对负输入是零。这种简单性使其计算效率高，但可能导致“ReLU 死亡”问题，即神经元输出始终为零。
- **Leaky ReLU** 通过允许负输入有一个小的非零梯度来解决“ReLU 死亡”问题，这在图中表现为轻微的负斜率。

这种实际实现有助于将激活函数的数学定义与其实际行为联系起来。理解这些差异是决定在神经网络的隐藏层和输出层中使用哪种激活函数的基础，这取决于具体的任务（例如，二分类任务通常在输出层使用 Sigmoid，多分类任务使用 Softmax，回归任务使用线性函数，而隐藏层常使用 ReLU 或其变体）。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本全面的教材，涵盖深度学习的理论基础和实践方面，包括对各种激活函数及其特性的详细讨论。
- [Deep Sparse Rectifier Networks](https://proceedings.mlr.press/v15/glorot11a.html) — Xavier Glorot, Antoine Bordes, and Yoshua Bengio (2011)
  Journal: Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics (AISTATS); Publisher: PMLR; Volume: 15; Pages: 315-323
  这篇有影响力的论文介绍并展示了修正线性单元（ReLU）作为深度神经网络中激活函数的好处。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  第3版。一本实用的指南，提供代码示例和关于实现深度学习模型的见解，包括对激活函数及其在流行框架中用例的详细分析。
