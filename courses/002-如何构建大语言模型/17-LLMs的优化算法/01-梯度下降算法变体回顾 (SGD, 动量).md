# 梯度下降算法变体回顾 (SGD, 动量)

来源：[原文](https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-17-optimization-algorithms-llms/review-gradient-descent-variants)

[返回章节目录](README.md) · [返回课程目录](../README.md)

理解基础的梯度下降 (gradient descent)算法对于掌握现代大型语言模型训练中常用的自适应方法是必不可少的。分析其机制和局限，有助于阐明为何在处理LLM的复杂优化问题时，经常需要AdamW等更高级的优化器。

### 随机梯度下降 (gradient descent) (SGD)

其核心是，深度学习 (deep learning)中的优化涉及调整模型参数 (parameter)（权重 (weight)和偏置 (bias)，统称为$\theta$）以最小化损失函数 (loss function)$L$。梯度下降通过迭代地将参数沿与损失函数对参数的梯度相反的方向移动来实现此目的。

全批量梯度下降使用整个训练数据集计算梯度，这对于LLM预训练 (pre-training)中使用的庞大数据集来说，计算上是不可行的。随机梯度下降（SGD）通过在每一步仅使用一小部分随机数据子集（称为小批量）来近似梯度，从而处理此问题。

SGD的更新规则是：


$$
\theta \leftarrow \theta - \eta \nabla_{\theta} L(\theta; x^{(i:i+b)}, y^{(i:i+b)})
$$


这里：

- $\theta$ 代表模型参数。
- $\eta$ 是学习率，一个控制步长的超参数 (hyperparameter)。
- $L$ 是损失函数。
- $x^{(i:i+b)}, y^{(i:i+b)}$ 代表一个包含 $b$ 个输入样本及其对应目标的小批量数据。
- $\nabla_{\theta} L(\cdot)$ 是损失函数对参数的梯度，仅使用当前小批量数据计算。

SGD的主要优点是其每一步的计算效率。处理一个小批量数据比处理整个数据集要快得多。更新的随机性（由于随机小批量采样）也会引入噪声，这有时可以帮助优化器摆脱不好的局部最小值。

然而，这种噪声也可能是一个缺点。更新可能显著震荡，导致锯齿状的收敛路径。此外，SGD在具有高曲率或山谷（表面在一个维度上比另一个维度弯曲得多）的优化面中可能表现不佳，可能需要许多步骤才能抵达最小值。SGD也对学习率 $\eta$ 的选择相当敏感。

在PyTorch中，使用SGD很简单：

```python
import torch

# 假设 'model' 是你定义的神经网络
# 假设 'loss_fn' 是你的损失函数
# 假设 'data_loader' 提供小批量数据

learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 在你的训练循环中：
for inputs, targets in data_loader:
    optimizer.zero_grad()     # 重置上一步的梯度
    outputs = model(inputs)   # 前向传播
    loss = loss_fn(outputs, targets) # 计算损失
    loss.backward()           # 反向传播（计算梯度）
    optimizer.step()          # 更新参数
```

### 带有动量的SGD

为了减少SGD固有的震荡并加速收敛，尤其是在山谷中，引入了动量技术。它添加了一个“速度”项 $v$，该项积累了过去梯度的指数衰减移动平均。参数 (parameter)更新随后会包含此速度项。

更新规则通常表述为：

1. 计算当前小批量的梯度：$g_t = \nabla_{\theta} L(\theta_t)$
2. 更新速度：$v_t \leftarrow \beta v_{t-1} + g_t$
3. 更新参数：$\theta_{t+1} \leftarrow \theta_t - \eta v_t$

这里：

- $v_t$ 是在步长 $t$ 时的速度向量 (vector)。
- $\beta$ 是动量系数，通常是一个接近1的值（例如0.9）。它控制过去梯度对当前更新的影响程度。较高的 $\beta$ 表示过去梯度贡献更多。
- $\eta$ 是学习率。

速度项 $v_t$ 有助于平滑更新。如果连续的梯度指向相似的方向，速度会累积，导致更大的步长和更快的收敛。如果梯度震荡，动量项通过平均它们来帮助抑制这些震荡。可以将其想象成一个重球滚下山坡；它在当前方向上保持动量，并且较少受到小颠簸（噪声梯度）的影响。

虽然动量通常比普通的SGD在收敛速度和稳定性上有所改进，但它仍然依赖于所有参数的单一学习率 $\eta$，并且需要仔细调整 $\eta$ 和 $\beta$。

在PyTorch中使用动量，只需向`SGD`优化器添加`momentum`参数：

```python
import torch

# 假设 'model' 是你定义的神经网络
learning_rate = 0.01
momentum_beta = 0.9

# 使用带有动量的SGD优化器
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=learning_rate,
    momentum=momentum_beta
)

# 训练循环与SGD示例相同...
# 在你的训练循环中：
# for inputs, targets in data_loader:
#    optimizer.zero_grad()
#    outputs = model(inputs)
#    loss = loss_fn(outputs, targets)
#    loss.backward()
#    optimizer.step()
```

虽然SGD和动量构成了许多优化策略的依据，但训练大型语言模型通常涉及应对具有复杂损失表面的极高维参数空间。这些基础方法可能收敛缓慢或停滞不前。这促使人们使用像Adam和AdamW这样的自适应优化算法，它们会根据每个参数调整学习率，并且在实际中通常能使这些大型模型更快地收敛。我们将在以下部分考察这些自适应方法。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  涵盖深度学习理论和实践方面的综合教材，在第八章详细解释了梯度下降、SGD和动量。
- [On the importance of initialization and momentum in deep learning](http://proceedings.mlr.press/v28/sutskever13.pdf) — Ilya Sutskever, James Martens, George Dahl, and Geoffrey Hinton (2013)
  Journal: Proceedings of the 30th International Conference on Machine Learning (ICML); Publisher: PMLR; Volume: 28; Pages: 1139-1147; DOI: [10.55982/annals.v28i1.9213](https://doi.org/10.55982/annals.v28i1.9213)
  一篇基础论文，展示了动量（特别是Nesterov加速梯度）在训练深度神经网络中的有效性。
- [torch.optim.SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html) — PyTorch Core Team (2024)
  PyTorch中随机梯度下降优化器的官方文档，详细说明了包括`momentum`在内的参数及其使用方法。
- [Lecture Notes: Optimization Algorithms](http://cs231n.github.io/optimization-1/) — Stanford University (2023)
  高质量的教育资源，在深度学习背景下提供了梯度下降、SGD和动量的易懂解释。

---

[上一节](../16-%E5%AE%9E%E7%8E%B0%E5%88%86%E5%B8%83%E5%BC%8F%E8%AE%AD%E7%BB%83%E6%A1%86%E6%9E%B6/06-%E7%BB%93%E5%90%88%E6%A1%86%E6%9E%B6%E4%B8%8E%E7%AD%96%E7%95%A5.md) · [下一节](02-%E8%87%AA%E9%80%82%E5%BA%94%E4%BC%98%E5%8C%96%E5%99%A8%EF%BC%9AAdam%E5%92%8CAdamW.md)
