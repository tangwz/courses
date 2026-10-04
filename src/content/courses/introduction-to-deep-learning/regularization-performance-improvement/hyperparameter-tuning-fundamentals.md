---
course: "introduction-to-deep-learning"
chapter: "regularization-performance-improvement"
lesson: "hyperparameter-tuning-fundamentals"
sourceId: 5068
sourceUrl: "https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-6-regularization-performance-improvement/hyperparameter-tuning-fundamentals"
title: "超参数调整基本原理"
description: "介绍超参数及其调整的必要性 (例如：学习率、层大小)。"
order: 7
plots: ["plots/5068-0.json"]
sourceHash: "60d852a9d25efe3495f62abebb15048107cac3333b45a12361b2b2bface0f5a8"
sourceCorrections: []
---

在本课程中，我们已经讨论了神经网络 (neural network)的各个组成部分：层、激活函数 (activation function)、损失函数 (loss function)、优化器，以及现在的正则化 (regularization)方法。在构建和训练这些模型时，我们区分两种类型的设置：

1. **参数 (parameter)：** 它们是模型在训练过程中学习到的值。主要包括网络层内的权重 (weight)和偏置 (bias)。它们的值通过反向传播 (backpropagation)和梯度下降 (gradient descent)迭代调整，以最小化损失函数。
2. **超参数 (hyperparameter)：** 它们是在训练过程开始*前*指定的配置设置。它们不是直接从数据中学习到的，而是定义模型的高级结构或控制学习过程本身。

可以把参数视为模型获得的内部知识，而超参数则是*你*在外部做出的选择，关于模型应如何构建以及如何学习。

正确选择这些超参数对模型性能影响很大。正如正则化有助于防止过拟合 (overfitting)一样，选择合适的超参数会影响：

- **模型容量：** 模型能学习多复杂的函数？(例如：层数、每层神经元数量)。
- **训练速度和收敛性：** 模型学习得有多快、多可靠？(例如：学习率、优化器选择、批量大小)。
- **泛化能力：** 模型在未见过的数据上表现如何？(例如：正则化强度、Dropout比率)。

不佳的超参数选择可能导致模型训练过慢、陷入次优解、严重过拟合，或者根本无法有效学习。

### 常见需调整的超参数 (parameter) (hyperparameter)

根据我们已介绍的内容，以下是你将遇到并需要考虑调整的一些最常见的超参数：

- **学习率：** 可能是影响最大的超参数。它控制梯度下降 (gradient descent)时的步长。过高，训练可能会发散；过低，则可能耗时过长或陷入不佳的局部最小值。(在第3章介绍)
- **隐藏层数量：** 决定网络的深度。更深的网络可能可以模拟更复杂的函数，但更难训练且容易过拟合 (overfitting)。(在第2章和第5章介绍)
- **每隐藏层神经元数量：** 控制网络层的宽度，从而控制每个阶段的表示能力。(在第2章和第5章介绍)
- **激活函数 (activation function)：** 尽管通常根据层类型选择(例如，隐藏层用ReLU，输出层用Sigmoid/Softmax)，但有时替代方案(Leaky ReLU, Tanh)也可能被视为超参数。(在第2章介绍)
- **优化器：** 用于梯度下降的算法(例如：SGD、Adam、RMSprop)。不同的优化器有不同的收敛特性以及对其他超参数(如学习率)的敏感度。(在第4章介绍)
- **批量大小：** 在模型权重 (weight)更新前处理的样本数量。影响训练速度、内存使用和梯度估计的稳定性。(在第3章和第5章介绍)
- **正则化 (regularization)强度：** 对于L1/L2正则化，这是控制权重幅值惩罚的系数($\lambda$)。(在本章介绍)
- **Dropout比率：** 在Dropout层训练期间随机设置为零的神经元比例。(在本章介绍)

### 调整的挑战

选择最佳的超参数 (parameter) (hyperparameter)组合通常更像是一门艺术而非科学，需要通过实验来完成。挑战在于以下几个因素：

- **搜索空间大：** 随着超参数增多，可能的组合数量会呈指数级增长。
- **相互依赖性：** 一个超参数的最佳值通常取决于其他超参数的值(例如：如果你更换优化器或改变批量大小，最佳学习率可能会改变)。
- **计算成本高：** 训练深度学习 (deep learning)模型可能非常耗时。评估许多不同的超参数组合可能需要大量的计算资源和时间。
- **数据依赖性：** 最佳超参数可能因具体数据集和任务而异。

下面看看不同学习率如何影响训练损失：



![学习率对训练损失的影响](plots/5068-0.json)



> 一张图表，显示了不同学习率下训练损失随周期变化的曲线。良好的学习率表现出稳定收敛，学习率过低则收敛缓慢，而学习率过高则可能导致损失剧烈波动或发散。

超参数是深度学习模型中需要在训练前手动设置的配置值。这些值通常在定义模型架构或配置优化器时进行设置。例如，在PyTorch中：

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 示例超参数
input_size = 784
hidden_size = 128
output_size = 10
learning_rate = 0.001
dropout_prob = 0.5 # 用于Dropout层

# 使用超参数定义模型
model = nn.Sequential(
    nn.Linear(input_size, hidden_size),
    nn.ReLU(),
    nn.Dropout(dropout_prob), # 使用Dropout比率
    nn.Linear(hidden_size, output_size)
)

# 使用超参数定义优化器
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # 设置学习率

print(f"模型架构:\n{model}")
print(f"\n优化器:\n{optimizer}")
```

找到好的超参数是实现深度学习模型高性能的一个重要部分。由于通过试错进行手动调整效率低下且通常效果不佳，尤其是在超参数很多时，因此需要更系统的方法。下一节将介绍自动化此搜索过程的常用策略。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本基础教材，涵盖了深度学习的理论和实践方面，包括参数和超参数的区别、各种超参数类型以及它们对模型训练和性能的影响。
- [Adam: A Method for Stochastic Optimization](https://doi.org/10.48550/arXiv.1412.6980) — Diederik P. Kingma and Jimmy Ba (2015)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980)
  介绍了 Adam 优化器，这是一种广泛采用的自适应学习率优化算法。这篇论文对于理解最常见且具影响力的超参数（优化器选择）及其与学习率的关系至关重要。
- [Dropout: A Simple Way to Prevent Overfitting](http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf) — Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov (2014)
  Journal: Journal of Machine Learning Research; Publisher: Journal of Machine Learning Research; Volume: 15; Pages: 1929-1958; DOI: [10.5555/2620392.2620461](https://doi.org/10.5555/2620392.2620461)
  提出了 Dropout 作为一种有效的正则化技术。这篇论文有助于理解 Dropout 率，这是一个提高模型泛化能力和防止过拟合的关键超参数。
- [Random Search for Hyper-Parameter Optimization](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf) — James Bergstra and Yoshua Bengio (2012)
  Journal: Journal of Machine Learning Research; Volume: 13; Pages: 281-305
  这篇论文的引言全面概述了超参数调整面临的挑战，例如巨大的搜索空间和相互依赖性，有助于理解该领域的基础问题。
