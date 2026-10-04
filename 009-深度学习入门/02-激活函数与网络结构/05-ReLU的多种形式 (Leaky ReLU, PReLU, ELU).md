# ReLU的多种形式 (Leaky ReLU, PReLU, ELU)

来源：[原文](https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-2-activation-functions-architecture/relu-variants)

[返回章节目录](README.md) · [返回课程目录](../README.md)

虽然标准修正线性单元（ReLU）被广泛采用且通常表现良好，但它对于所有负输入都输出精确零的特点可能引起一个被称为“ReLU死亡”的问题。如果一个神经元的权重 (weight)更新后，其输入持续落入负值区间，它将始终输出零。因此，在反向传播 (backpropagation)过程中流经该神经元的梯度也将为零，有效地阻止了其权重的进一步更新。该神经元变得不活跃。

为处理这个潜在问题并寻求其他非线性函数，已经提出了ReLU的几种变体。这些变体保持了ReLU对正输入的优点，但对负输入引入了一个小斜率，确保即使单元没有被强烈激活时，梯度也能继续流动。

### Leaky ReLU

Leaky ReLU是最简单的修改之一。它不像ReLU那样对负输入输出零，而是输出一个与输入成比例的小负值。用数学形式表示为：


$$
f(x) = \begin{cases}
      x & \text{如果 } x > 0 \\
      \alpha x & \text{如果 } x \le 0
    \end{cases}
$$


这里，$\alpha$ 是一个小的固定超参数 (parameter) (hyperparameter)，通常设定为一个像0.01的值。主要的思路是，该函数现在对于负输入有一个非零梯度 ($\alpha$)。这个小梯度保证了神经元不会完全“死亡”，并能继续学习。

### Parametric ReLU (PReLU)

Parametric ReLU，或PReLU，将Leaky ReLU的思路推进了一步。PReLU不像Leaky ReLU那样对负区域的斜率 $\alpha$ 使用固定值，而是将 $\alpha$ 视为一个*可学习参数 (parameter)*。这表示网络本身可以通过反向传播 (backpropagation)在训练过程中学习负输入的最佳斜率。


$$
f(x) = \begin{cases}
      x & \text{如果 } x > 0 \\
      \alpha x & \text{如果 } x \le 0 \quad (\alpha \text{ 被学习})
    \end{cases}
$$


如果最佳斜率不同于预定义的小常数，PReLU可能比Leaky ReLU提供更好的性能，但它为每个神经元（或卷积层中的每个通道）增加了一个需要学习的额外参数。如果数据有限，这可能会比Leaky ReLU或标准ReLU略微增加过拟合 (overfitting)的风险。

### Exponential Linear Unit (ELU)

指数线性单元（ELU）提供另一种选择，旨在使激活平均值更接近零，这有时可以加快学习速度。对于正输入，它的行为类似于ReLU。对于负输入，它使用一个指数函数，该函数饱和到负值。


$$
f(x) = \begin{cases}
      x & \text{如果 } x > 0 \\
      \alpha (e^x - 1) & \text{如果 } x \le 0
    \end{cases}
$$


超参数 (parameter) (hyperparameter) $\alpha$ 控制ELU对负输入的饱和值（通常设为1.0）。与Leaky ReLU和PReLU不同，ELU可以产生负输出。与ReLU及其泄露变体的尖角相比，它在 $x=0$ 附近有更平滑的过渡（尽管在 $x=0$ 处梯度不连续，除非 $\alpha=1$）。将激活值推向零均值的潜在益处是以指数函数导致的计算开销略微增加为代价的。

### 视觉对比

下图比较了ReLU及其常见变体的形状：



[交互图表：ReLU激活函数变体的对比](https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-2-activation-functions-architecture/relu-variants#plot-1f0w3sw)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "xaxis": {
      "range": [
        -3,
        3
      ],
      "title": "输入 (x)",
      "zeroline": true,
      "zerolinecolor": "#ced4da"
    },
    "yaxis": {
      "range": [
        -1.5,
        3
      ],
      "title": "输出 (f(x))",
      "zeroline": true,
      "zerolinecolor": "#ced4da"
    },
    "title": "ReLU激活函数变体的对比",
    "legend": {
      "x": 0.01,
      "y": 0.99
    },
    "margin": {
      "l": 50,
      "r": 20,
      "t": 40,
      "b": 50
    }
  },
  "data": [
    {
      "x": [
        -3,
        -2,
        -1,
        0,
        1,
        2,
        3
      ],
      "y": [
        0,
        0,
        0,
        0,
        1,
        2,
        3
      ],
      "mode": "lines",
      "name": "ReLU",
      "line": {
        "color": "#228be6",
        "width": 2
      }
    },
    {
      "x": [
        -3,
        -2,
        -1,
        0,
        1,
        2,
        3
      ],
      "y": [
        -0.03,
        -0.02,
        -0.01,
        0,
        1,
        2,
        3
      ],
      "mode": "lines",
      "name": "Leaky ReLU (α=0.01)",
      "line": {
        "color": "#12b886",
        "width": 2,
        "dash": "dash"
      }
    },
    {
      "x": [
        -3,
        -2,
        -1,
        0,
        1,
        2,
        3
      ],
      "y": [
        -0.75,
        -0.5,
        -0.25,
        0,
        1,
        2,
        3
      ],
      "mode": "lines",
      "name": "PReLU (α=0.25 已学习)",
      "line": {
        "color": "#f76707",
        "width": 2,
        "dash": "dot"
      }
    },
    {
      "x": [
        -3,
        -2,
        -1,
        0,
        1,
        2,
        3
      ],
      "y": [
        -0.9502,
        -0.8647,
        -0.6321,
        0,
        1,
        2,
        3
      ],
      "mode": "lines",
      "name": "ELU (α=1.0)",
      "line": {
        "color": "#be4bdb",
        "width": 2,
        "dash": "dashdot"
      }
    }
  ]
}
```

</details>



> ReLU、Leaky ReLU（带有典型小alpha）、PReLU（表示可能学到的较大alpha）和ELU的对比。注意Leaky ReLU和PReLU如何具有线性负斜率，而ELU具有平滑曲线，饱和于-alpha。

### 变体选择

尽管标准ReLU仍然是非常常用且通常有效的选择，特别是作为起点，但这些变体提供了有用的替代方案：

1. **Leaky ReLU:** 处理潜在“ReLU死亡”问题的一个简单方法。如果在使用标准ReLU训练时，你观察到许多不活跃的神经元，则可以考虑使用它。
2. **PReLU:** 如果你认为最佳负斜率可能与0.01这样的小常数显著不同，并且你有足够的数据来学习额外的参数 (parameter)而不会过拟合 (overfitting)，则考虑使用它。
3. **ELU:** 如果希望将激活值推向零均值，它可能是有益的，可能加快收敛速度。它对负输入的平滑曲线和负饱和是其独特的特点。然而，它在计算上比ReLU或Leaky ReLU略微耗费资源。

在实践中，选择通常取决于经验测试。首先尝试标准ReLU，如果性能不佳或怀疑存在“死亡神经元”，则可以尝试Leaky ReLU、PReLU或ELU，看看它们是否能为你的特定任务和架构带来改善。

你可以使用深度学习 (deep learning)库轻松实现这些变体。例如，在PyTorch中：

```python
import torch
import torch.nn as nn

# 示例输入
input_tensor = torch.randn(5) * 2 # 一些正值和负值

# 标准ReLU
relu = nn.ReLU()
output_relu = relu(input_tensor)
print(f"输入: {input_tensor.numpy()}")
print(f"ReLU 输出: {output_relu.numpy()}")

# Leaky ReLU (alpha=0.01 为默认值)
leaky_relu = nn.LeakyReLU(negative_slope=0.01)
output_leaky = leaky_relu(input_tensor)
print(f"Leaky ReLU 输出 (alpha=0.01): {output_leaky.numpy()}")

# PReLU (alpha 可学习，初始值通常为 0.25)
# 注意: num_parameters=1 表示所有元素使用相同的 alpha
prelu = nn.PReLU(num_parameters=1, init=0.25)
output_prelu = prelu(input_tensor)
print(f"PReLU 输出 (初始 alpha=0.25): {output_prelu.detach().numpy()}") # 需要 detach() 因为 alpha 是可学习的

# ELU (alpha=1.0 为默认值)
elu = nn.ELU(alpha=1.0)
output_elu = elu(input_tensor)
print(f"ELU 输出 (alpha=1.0): {output_elu.numpy()}")
```

尝试这些激活函数 (activation function)是设计和改进神经网络 (neural network)架构的常规步骤。

## 参考资料

- [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852) — Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun (2015)
  Journal: Proceedings of the IEEE International Conference on Computer Vision (ICCV); Pages: 1026-1034; DOI: [10.48550/arXiv.1502.01852](https://doi.org/10.48550/arXiv.1502.01852)
  介绍了PReLU，并对ReLU及其变体（包括Leaky ReLU）进行了详尽分析。
- [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)](https://arxiv.org/abs/1511.07289) — Djork-Arné Clevert, Thomas Unterthiner, and Sepp Hochreiter (2016)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1511.07289](https://doi.org/10.48550/arXiv.1511.07289)
  介绍了指数线性单元（ELU）激活函数，讨论了其特性和对深度网络训练的优势。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本基础教材，详细解释了神经网络、激活函数和深度学习架构。
- [Activation Functions](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-feature-space) — PyTorch Team (2024)
  Publisher: PyTorch
  PyTorch官方文档，提供了各种激活函数（包括Leaky ReLU、PReLU和ELU）的详细信息和使用示例。

---

[上一节](04-%E4%BF%AE%E6%AD%A3%E7%BA%BF%E6%80%A7%E5%8D%95%E5%85%83%20%28ReLU%29.md) · [下一节](06-%E9%80%89%E6%8B%A9%E5%90%88%E9%80%82%E7%9A%84%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0.md)
