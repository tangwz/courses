# Sigmoid 激活函数

来源：[原文](https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-2-activation-functions-architecture/sigmoid-activation)

[返回章节目录](README.md) · [返回课程目录](../README.md)

非线性激活函数 (activation function)在深度学习 (deep learning)模型中是不可或缺的，它们能帮助模型学习复杂模式。Sigmoid 函数，也称为逻辑函数，是最早且具有开创性历史地位的激活函数之一。

### 数学定义

Sigmoid 函数的数学定义为：


$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$


$x$ 是函数的输入（通常是神经元的加权输入和偏置 (bias)之和）。该函数可以将任何实数值压缩到 0 到 1 的范围内。

### 特性与形状

Sigmoid 函数的显著特征是其“S”形曲线。



[交互图表：Sigmoid 激活函数](https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-2-activation-functions-architecture/sigmoid-activation#plot-1l5euav)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "xaxis": {
      "title": "输入 (x)",
      "range": [
        -10,
        10
      ]
    },
    "yaxis": {
      "title": "输出 (σ(x))",
      "range": [
        -0.1,
        1.1
      ]
    },
    "title": "Sigmoid 激活函数",
    "template": "plotly_white",
    "legend": {
      "yanchor": "bottom",
      "y": 0.01,
      "xanchor": "right",
      "x": 0.99
    }
  },
  "data": [
    {
      "x": [
        -10,
        -8,
        -6,
        -4,
        -2,
        0,
        2,
        4,
        6,
        8,
        10
      ],
      "y": [
        4.5e-05,
        0.000335,
        0.00247,
        0.01798,
        0.1192,
        0.5,
        0.8807,
        0.982,
        0.9975,
        0.9996,
        0.9999
      ],
      "mode": "lines+markers",
      "type": "scatter",
      "name": "σ(x)",
      "line": {
        "color": "#228be6"
      },
      "marker": {
        "color": "#228be6"
      }
    }
  ]
}
```

</details>



> Sigmoid 函数将输入映射到 (0, 1) 区间。它在 $x=0$ 附近呈现平滑过渡，对于大的负输入饱和趋近于 0，对于大的正输入饱和趋近于 1。

重要特性包括：

1. **输出范围：** 输出始终在 0 和 1 之间（不包含边界）。这一特性使其最初颇受欢迎，特别是在二元分类问题中作为输出层，其输出可以被理解为概率。
2. **非线性：** 它是一个非线性函数，对神经网络 (neural network)学习复杂模式十分必要，线性模型无法处理的模式。
3. **平滑性：** 该函数处处可导，这是梯度优化方法（如反向传播 (backpropagation)）的一个要求。其导数易于计算：$\sigma'(x) = \sigma(x)(1 - \sigma(x))$。

### 优点

- **概率解释：** (0, 1) 的输出范围使得 Sigmoid 神经元的输出可以作为概率处理，这对于二元分类任务来说是符合直觉的（例如，属于类别 1 的概率）。
- **平滑梯度：** 梯度处处有明确定义，防止训练过程中出现突变。

### 缺点

尽管 Sigmoid 函数具有历史地位，但由于一些明显不足，它已不再受青睐用于深层网络的隐藏层：

1. **梯度消失：** 这是最主要的问题。再次观察 Sigmoid 函数的形状。当输入为较大的正值或负值时（即神经元“饱和”时），函数曲线变得非常平坦。平坦的函数意味着导数（梯度）接近于零。

   
   
   [交互图表：Sigmoid 函数及其导数](https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-2-activation-functions-architecture/sigmoid-activation#plot-9qclaq)
   
   <details>
   <summary>图表数据（JSON）</summary>
   
   ```json
   {
     "layout": {
       "xaxis": {
         "title": "输入 (x)",
         "range": [
           -10,
           10
         ]
       },
       "yaxis": {
         "title": "输出",
         "range": [
           -0.05,
           1.05
         ]
       },
       "title": "Sigmoid 函数及其导数",
       "template": "plotly_white",
       "legend": {
         "yanchor": "top",
         "y": 0.99,
         "xanchor": "left",
         "x": 0.01
       }
     },
     "data": [
       {
         "x": [
           -10,
           -8,
           -6,
           -4,
           -2,
           0,
           2,
           4,
           6,
           8,
           10
         ],
         "y": [
           4.5e-05,
           0.000335,
           0.00247,
           0.01798,
           0.1192,
           0.5,
           0.8807,
           0.982,
           0.9975,
           0.9996,
           0.9999
         ],
         "mode": "lines",
         "type": "scatter",
         "name": "Sigmoid σ(x)",
         "line": {
           "color": "#228be6"
         }
       },
       {
         "x": [
           -10,
           -8,
           -6,
           -4,
           -2,
           0,
           2,
           4,
           6,
           8,
           10
         ],
         "y": [
           4.5e-05,
           0.000335,
           0.00245,
           0.01766,
           0.10499,
           0.25,
           0.10499,
           0.01766,
           0.00245,
           0.000335,
           4.5e-05
         ],
         "mode": "lines",
         "type": "scatter",
         "name": "导数 σ'(x)",
         "line": {
           "color": "#fd7e14"
         }
       }
     ]
   }
   ```
   
   </details>
   
   

   > Sigmoid 函数的导数在 $x=0$ 处最大（值为 0.25），随着输入远离 0 迅速趋近于零。

   在反向传播 (backpropagation)过程中，梯度逐层相乘。如果多个层使用 Sigmoid 激活函数 (activation function)，并且它们的神经元在饱和区域运行，反向传播的梯度将重复乘以小数值（导数，小于或等于 0.25）。这会导致到达前面层的梯度变得非常小（“消失”），使得这些层的权重 (weight)难以有效更新。网络实质上停止了在较深层的学习。
2. **输出非零中心：** Sigmoid 函数的输出始终为正（介于 0 和 1 之间）。这可能会带来问题。如果下一层神经元的输入始终为正，反向传播时该神经元权重的梯度将全部具有相同的符号（全部为正或全部为负，取决于损失函数 (loss function)相对于神经元输出的梯度）。这可能导致梯度下降 (gradient descent)过程中更新效率低下，呈锯齿状，与使用零中心输出的激活函数相比，会减缓收敛速度。

### 使用场景

由于梯度消失和输出非零中心的问题，Sigmoid 通常**不建议**用于现代深度学习 (deep learning)模型的隐藏层。ReLU 及其变体等函数（我们将在后面介绍）通常能带来更快、更有效的训练。

然而，Sigmoid 仍在特定情境下有其用处：

- **二元分类的输出层：** 其 (0, 1) 的输出范围使其成为执行二元分类的网络最终层的自然选择，其中输出表示正类的概率。
- **循环网络中的门：** 某些循环神经网络 (neural network) (RNN)架构（如 LSTM 和 GRU，稍后会介绍）在内部使用 Sigmoid 函数作为“门”来控制信息流，凭借其 (0, 1) 范围。

### PyTorch 实现示例

使用 `torch.sigmoid` 或 `nn.Sigmoid` 模块在 PyTorch 中应用 Sigmoid 函数非常简单。

```python
import torch
import torch.nn as nn

# 示例输入张量（例如，线性层的输出）
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

# 使用函数式 API 应用 Sigmoid
sigmoid_output_functional = torch.sigmoid(x)
print("使用 torch.sigmoid 的输出:", sigmoid_output_functional)

# 使用模块 API 应用 Sigmoid
sigmoid_module = nn.Sigmoid()
sigmoid_output_module = sigmoid_module(x)
print("使用 nn.Sigmoid 的输出:", sigmoid_output_module)

# 验证输出范围
print(f"最小输出: {sigmoid_output_module.min()}, 最大输出: {sigmoid_output_module.max()}")

# 示例输出:
# 使用 torch.sigmoid 的输出: tensor([0.1192, 0.2689, 0.5000, 0.7311, 0.8808])
# 使用 nn.Sigmoid 的输出: tensor([0.1192, 0.2689, 0.5000, 0.7311, 0.8808])
# 最小输出: 0.11920292109251022, 最大输出: 0.8807970285415649
```

尽管 Sigmoid 在神经网络 (neural network)的历史上扮演了重要角色，但其局限性，特别是梯度消失问题，促使研究人员寻求替代方案。在接下来的章节中，我们将了解 Tanh 和 ReLU 等其他激活函数 (activation function)，它们解决了其中一些问题。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  全面解释了激活函数，包括 Sigmoid 函数的特性、历史背景以及深度网络中梯度消失问题。
- [Lecture Notes on Neural Networks Part 1: Setting up the Architecture](https://cs231n.github.io/neural-networks-1/) — Andrej Karpathy, Justin Johnson, and Serena Yeung (2023)
  Journal: Stanford University CS231n Course Notes
  提供了神经网络架构的入门介绍，涵盖了 Sigmoid 等激活函数、它们的优点以及梯度消失和非零中心输出等关键缺点。
- [torch.sigmoid](https://pytorch.org/docs/stable/generated/torch.sigmoid.html) — PyTorch Developers (2023)
  Publisher: PyTorch Foundation
  PyTorch 深度学习框架中 Sigmoid 激活函数的官方文档，详细说明了其函数式和基于模块的使用方法，便于实际实现。

---

[上一节](01-%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0%E7%9A%84%E4%BD%9C%E7%94%A8.md) · [下一节](03-%E5%8F%8C%E6%9B%B2%E6%AD%A3%E5%88%87%EF%BC%88Tanh%EF%BC%89%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0.md)
