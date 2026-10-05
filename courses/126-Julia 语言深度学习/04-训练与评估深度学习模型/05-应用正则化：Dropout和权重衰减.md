# 应用正则化：Dropout和权重衰减

来源：[原文](https://apxml.com/zh/courses/julia-deep-learning/chapter-4-training-evaluating-dl-models/regularization-dropout-weight-decay)

[返回章节目录](README.md) · [返回课程目录](../README.md)

构建和训练模型时，可能会遇到一个常见的问题：模型在训练数据上表现优异，但在面对新的、未见过的数据时，性能会显著下降。这种现象称为过拟合 (overfitting)，它表明模型学习了训练数据的噪声和特定细节，而非数据中普遍的规律。正则化 (regularization)技术旨在对抗过拟合，帮助模型更好地适应新数据。深度学习 (deep learning)中两种常用的正则化方法是Dropout和权重 (weight)衰减（常被称为$L_2$正则化）。

### 理解过拟合 (overfitting)

在讨论解决方案之前，我们先来看看过拟合是什么样子。当模型发生过拟合时，它本质上是在记忆训练集。它的容量过高，或者在相同数据上训练了过长时间，以至于开始学习训练样本中存在的随机波动。这会导致模型过于复杂，无法在未见过的数据上表现良好。

下面的图表描绘了过拟合模型与经过良好正则化 (regularization)的模型在训练和验证损失方面的典型表现。



[交互图表：正则化对模型学习的影响](https://apxml.com/zh/courses/julia-deep-learning/chapter-4-training-evaluating-dl-models/regularization-dropout-weight-decay#plot-12eg8cg)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "正则化对模型学习的影响",
    "xaxis": {
      "title": "训练轮次"
    },
    "yaxis": {
      "title": "损失"
    },
    "height": 400,
    "legend": {
      "orientation": "h",
      "yanchor": "bottom",
      "y": 1.02,
      "xanchor": "right",
      "x": 1
    },
    "plot_bgcolor": "#e9ecef",
    "paper_bgcolor": "#ffffff"
  },
  "data": [
    {
      "name": "过拟合：训练损失",
      "x": [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10
      ],
      "y": [
        1.0,
        0.7,
        0.5,
        0.35,
        0.25,
        0.2,
        0.17,
        0.15,
        0.13,
        0.12
      ],
      "type": "scatter",
      "mode": "lines",
      "line": {
        "color": "#fa5252",
        "dash": "dash",
        "width": 2
      }
    },
    {
      "name": "过拟合：验证损失",
      "x": [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10
      ],
      "y": [
        1.1,
        0.8,
        0.65,
        0.55,
        0.5,
        0.52,
        0.58,
        0.65,
        0.75,
        0.9
      ],
      "type": "scatter",
      "mode": "lines",
      "line": {
        "color": "#fa5252",
        "width": 2
      }
    },
    {
      "name": "正则化：训练损失",
      "x": [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10
      ],
      "y": [
        1.05,
        0.78,
        0.6,
        0.48,
        0.4,
        0.35,
        0.32,
        0.3,
        0.28,
        0.27
      ],
      "type": "scatter",
      "mode": "lines",
      "line": {
        "color": "#40c057",
        "dash": "dash",
        "width": 2
      }
    },
    {
      "name": "正则化：验证损失",
      "x": [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10
      ],
      "y": [
        1.15,
        0.88,
        0.72,
        0.63,
        0.58,
        0.55,
        0.53,
        0.52,
        0.51,
        0.5
      ],
      "type": "scatter",
      "mode": "lines",
      "line": {
        "color": "#40c057",
        "width": 2
      }
    }
  ]
}
```

</details>



> 训练和验证损失曲线的表现。当训练损失持续下降而验证损失开始上升时，就会发生过拟合。正则化的目标是使两种损失都保持较低水平并趋于收敛。

请注意过拟合模型的验证损失在某个点之后开始增加，即使其训练损失持续下降。而正则化模型则倾向于在训练和验证损失之间表现出更好的吻合度，或者至少验证损失会稳定在较低的值。

### Dropout

Dropout是一种由Srivastava等人（2014）提出的简单而有效的正则化 (regularization)技术。在每次训练迭代中，Dropout会随机将层中一部分神经元的输出设置为零。这种神经元的“丢弃”会在每个训练批次中轻微改变网络架构。

**这为什么有帮助？**

1. **防止协同适应：** 神经元学习变得更加独立。它们不能依赖于其他特定神经元的活跃，因为任何神经元都可能被丢弃。这鼓励它们学习单独更有用的特征。
2. **集成效应：** 使用Dropout训练可以看作是训练大量“瘦身”网络（只包含一部分神经元的网络）。在测试时，使用完整网络可以被解释为对这些“瘦身”网络的预测进行平均，这通常会提升性能。

**在Flux.jl中的实现**

Flux.jl提供了`Dropout(p)`层，其中`p`是在训练期间每个神经元输出被设置为零的概率。它通常插入到模型的其他层之间，常用于全连接层中的激活函数 (activation function)之后。

```julia
using Flux

# 定义一个包含Dropout的模型
model = Chain(
  Dense(784, 256, relu),
  Dropout(0.5),  # 以50%的概率应用dropout
  Dense(256, 128, relu),
  Dropout(0.3),  # 以30%的概率应用dropout
  Dense(128, 10)
)
```

在训练期间，Flux会自动启用dropout。当您评估模型时（例如，在训练循环之外使用`model(x)`或显式调用`Flux.testmode!`），dropout会自动禁用，并且层输出会进行适当缩放，以弥补训练期间被丢弃的神经元。这种缩放确保了下一层输入的预期总和在训练和推理 (inference)期间保持一致。具体来说，剩余活跃神经元的输出会按$1/(1-p)$的因子进行放大。

Dropout率`p`是一个您需要调整的超参数 (parameter) (hyperparameter)。常见值范围从0.2到0.5。更高的`p`意味着更强的正则化。

### 权重 (weight)衰减 (L2 正则化 (regularization))

权重衰减，也称为$L_2$正则化，是另一种预防过拟合 (overfitting)的常用方法。它通过向模型的损失函数 (loss function)添加一个惩罚项来起作用。这个惩罚项与模型权重的平方和成比例。

修改后的损失函数变为：


$$
L_{total} = L_{original} + \frac{\lambda}{2} \sum_{i} w_i^2
$$


这里，$L_{original}$是原始损失（例如，交叉熵或均方误差），$w_i$是模型中的单个权重，而$\lambda$（lambda）是正则化强度或权重衰减系数。通常包含$1/2$因子是为了在求导时方便计算。

**这为什么有帮助？**

1. **抑制大权重：** 大权重会使模型对输入的微小变化非常敏感，可能导致决策边界过于尖锐并对噪声过拟合。通过惩罚大权重，$L_2$正则化鼓励模型找到权重更小、分布更均匀的解决方案。
2. **更简单的模型：** 权重较小的模型通常被认为是“更简单”的，并且倾向于更好地泛化。惩罚项将权重推向零，但除非某个权重确实不提供信息，否则不会精确地推到零。

**在Flux.jl中的实现**

在Flux.jl中，权重衰减通常作为优化器的一部分应用。`Optimiser`结构体可以将现有优化器与`WeightDecay`组件进行封装。

```julia
using Flux
using Flux: Optimise

# 原始优化器
opt_rule = Adam(0.001) # 学习率为0.001

# 添加权重衰减
lambda = 0.01 # 权重衰减系数
opt = Optimiser(opt_rule, WeightDecay(lambda))

# 示例模型
model = Dense(10, 5)
ps = Flux.params(model)
gs = gradient(() -> Flux.mse(model(rand(10)), rand(5)), ps)

# 使用带有权重衰减的优化器更新参数
Flux.update!(opt, ps, gs)
```

在此设置中，在参数 (parameter)更新步骤期间，优化器不仅会沿着使$L_{original}$最小化的方向移动参数，还会由于$L_2$惩罚项而隐含地减小权重的幅度。$L_2$惩罚项对权重$w$的梯度是$\lambda w$。因此，更新规则实际上变为$w \leftarrow w - \eta (\frac{\partial L_{original}}{\partial w} + \lambda w)$，这会导致权重趋向于零。

权重衰减系数$\lambda$是一个超参数 (hyperparameter)。典型值很小，例如$10^{-2}$、$10^{-4}$或$10^{-5}$。找到合适的值通常需要实验。

### 选择和使用正则化 (regularization)

Dropout和权重 (weight)衰减都是提升模型泛化能力的有效手段。

- **Dropout** 在大型、深层网络中特别有效，因为这些网络中特征的协同适应可能成为问题。它在全连接层中很常见。
- **权重衰减 ($L_2$)** 是一种更通用的正则化方法，几乎可以应用于任何具有可学习权重的模型。

同时使用这两种技术也是可能的，有时甚至有益。每种技术的最佳强度（dropout概率`p`和权重衰减系数$\lambda$）将取决于您的特定数据集和模型架构。这些都是超参数 (parameter) (hyperparameter)，您通常会使用验证集进行调整，我们将在讨论超参数调整策略时更详细地介绍这个话题。

应用正则化时，监控训练和验证性能都很重要。

- 如果您的模型仍然过拟合 (overfitting)（验证损失远高于训练损失，或验证损失持续增加），您可能需要增加正则化强度（例如，提高dropout的`p`值，增大权重衰减的$\lambda$）。
- 反之，如果您的模型欠拟合 (underfitting)（训练和验证损失都很高且没有改善，或者验证性能不佳），您可能施加了过多的正则化。在这种情况下，尝试降低正则化强度甚至将其移除。

通过谨慎地应用Dropout和权重衰减等正则化技术，您可以在Julia中构建深度学习 (deep learning)模型，这些模型不仅在已见过的数据上表现良好，也能在新的、未见过的数据上表现出色。这是构建可靠且高效机器学习 (machine learning)方案的重要一步。

## 参考资料

- [Dropout: A Simple Way to Prevent Overfitting](http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf) — Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov (2014)
  Journal: Journal of Machine Learning Research (JMLR); Volume: 15; Pages: 1929-1958; DOI: [10.5555/2627435.2670313](https://doi.org/10.5555/2627435.2670313)
  介绍Dropout正则化技术的奠基性论文，阐释其机制和优势。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  一本内容全面的教科书，其中有专门章节讲解正则化技术，包括对L2正则化（权重衰减）和Dropout的详细阐述。
- [Layers: Dropout](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHlEDAl6FpGOylVxRxv3uLhciN2t6SU3lqMzVHBz5AXiLWGv7h3do5dJvehWHC6S4YKBSHA1EEKkRpwby1IgIPWcjmqllV6US_3Lu-ZMGOkSlYp7L4YpsvTaEBv) — The Flux.jl Community (2025)
  Publisher: The Flux.jl Community
  Flux.jl中`Dropout`层的官方文档，详细说明了其在Julia深度学习模型中的用法和参数。
- [Optimisers: WeightDecay](https://fluxml.ai/Optimisers.jl/stable/api/#Optimisers.WeightDecay) — The Flux.jl Community (2024)
  Flux.jl中通过优化器应用`WeightDecay`的官方文档，展示了其在训练过程中的集成方式。

---

[上一节](04-%E5%B8%B8%E7%94%A8%E5%88%86%E7%B1%BB%E5%92%8C%E5%9B%9E%E5%BD%92%E8%AF%84%E4%BC%B0%E6%8C%87%E6%A0%87.md) · [下一节](06-%E8%B6%85%E5%8F%82%E6%95%B0%E8%B0%83%E4%BC%98%E7%AD%96%E7%95%A5.md)
