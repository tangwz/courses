# 修正线性单元 (ReLU)

来源：[原文](https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-2-activation-functions-architecture/relu-activation)

[返回章节目录](README.md) · [返回课程目录](../README.md)

修正线性单元，通常称为ReLU，为Sigmoid和Tanh等其他非线性激活函数 (activation function)提供了一种更简单但效果很好的替代方案。虽然Sigmoid和Tanh等函数引入了非线性，但它们可能存在饱和和梯度消失等潜在弊端。由于其计算效率高以及能够缓解一些梯度问题，ReLU已成为深度学习 (deep learning)中，特别是隐藏层中的常用方法。

ReLU函数在数学上的定义为：


$$
f(x) = \max(0, x)
$$


简单来说，如果输入$x$为正，函数输出$x$本身。如果输入为零或负，函数输出零。这形成了“修正”行为，即将负值截断为零。



[交互图表：ReLU激活函数](https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-2-activation-functions-architecture/relu-activation#plot-12sycvi)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "xaxis": {
      "title": "输入 (x)",
      "range": [
        -5,
        5
      ],
      "zeroline": true,
      "zerolinewidth": 1,
      "zerolinecolor": "#ced4da"
    },
    "yaxis": {
      "title": "输出 (f(x))",
      "range": [
        -1,
        5
      ],
      "zeroline": true,
      "zerolinewidth": 1,
      "zerolinecolor": "#ced4da"
    },
    "title": "ReLU激活函数",
    "template": "plotly_white",
    "height": 350,
    "width": 500
  },
  "data": [
    {
      "x": [
        -5,
        -4,
        -3,
        -2,
        -1,
        0,
        1,
        2,
        3,
        4,
        5
      ],
      "y": [
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        2,
        3,
        4,
        5
      ],
      "type": "scatter",
      "mode": "lines",
      "name": "ReLU",
      "line": {
        "color": "#228be6",
        "width": 3
      }
    }
  ]
}
```

</details>



> ReLU函数$f(x) = \max(0, x)$。它对正输入是线性的，对负输入则为零。

### ReLU的优势

1. **计算简便性：** 与涉及计算量大的指数运算的Sigmoid和Tanh不同，ReLU只需一个简单的比较（是否$x > 0$？）。这使得计算速度明显加快，无论是在前向传播（计算激活值）还是反向传播 (backpropagation)（计算梯度）中。
2. **降低梯度消失的可能性：** 回想一下，Sigmoid和Tanh函数在大的正输入或负输入时会饱和，这意味着它们的梯度趋近于零。在反向传播过程中，将许多小梯度相乘可能导致梯度信号消失，使得较早的层难以有效学习。ReLU对正输入解决了这个问题。ReLU的导数为：
   
   $$
   f'(x) = \begin{cases} 1 & \text{若 } x > 0 \\ 0 & \text{若 } x < 0 \\ \text{未定义} & \text{若 } x = 0 \end{cases}
   $$
   
   对于正输入（$x > 0$），梯度始终为1。这种恒定的梯度防止了Sigmoid/Tanh中出现的收缩效应，使得梯度能够更有效地通过深层网络，并且通常在训练期间导致更快的收敛。（在实际应用中，$x=0$处的梯度通常设置为0或1。）
3. **引入稀疏性：** 因为ReLU对所有负输入输出零，这可能导致网络内部的稀疏激活。这意味着在任何给定时间，某一层中只有一部分神经元可能处于活跃状态（输出非零值）。稀疏性可以提高计算效率，并可能导致数据表示更解耦。

### “死亡ReLU”问题

尽管有其优势，ReLU也并非没有自身问题。最主要的问题是“死亡ReLU”问题。如果一个神经元的输入在训练过程中始终为负，它将始终输出零。因此，流经该神经元的梯度也将始终为零（因为当$x<0$时，$f'(x)=0$）。

当这种情况发生时，与该神经元相关的权重 (weight)将不再通过梯度下降 (gradient descent)进行更新。该神经元基本上变得不活跃，并停止参与学习过程。如果学习率设置过高，或者存在一个大的负偏置 (bias)项将神经元的加权和推入负值范围，就可能发生这种情况。一旦一个ReLU单元“死亡”，它不太可能恢复。

请看这个简单的PyTorch示例，演示如何使用ReLU：

```python
import torch
import torch.nn as nn

# 示例输入张量
input_tensor = torch.randn(1, 5) # 批量大小为1，5个特征
print(f"输入: {input_tensor}")

# 使用torch.nn.ReLU应用ReLU
relu_activation = nn.ReLU()
output_tensor = relu_activation(input_tensor)
print(f"ReLU后的输出: {output_tensor}")

# 使用torch.relu函数式形式应用ReLU
output_functional = torch.relu(input_tensor)
print(f"使用函数式形式的输出: {output_functional}")

# 演示梯度（requires_grad=True）
input_tensor.requires_grad_(True)
output_relu = torch.relu(input_tensor)
# 假设一些上游梯度用于演示
output_relu.backward(torch.ones_like(output_relu))

print(f"输入的梯度: {input_tensor.grad}")
# 注意：当输入>0时，梯度为1；当输入<=0时，梯度为0
```

输出显示了负输入的归零效应，并显示了正输入时梯度为1，负输入时梯度为0。

ReLU的简洁性、速度快以及缓解梯度消失的能力使其在许多深度学习 (deep learning)模型中成为隐藏层的默认选择。然而，“死亡神经元”的可能性意味着仔细的初始化和学习率选择很重要。在下一节中，我们将查看专门设计用于解决这个“死亡”问题的ReLU变体。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 6
  一本权威教材，详细阐述了ReLU、其数学特性、优点以及像ReLU死亡问题这样的挑战。
- [ReLU - PyTorch 2.3 documentation](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html) — PyTorch Team (2024)
  PyTorch中ReLU激活函数实现的官方文档，包含其用法和参数。

---

[上一节](03-%E5%8F%8C%E6%9B%B2%E6%AD%A3%E5%88%87%EF%BC%88Tanh%EF%BC%89%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0.md) · [下一节](05-ReLU%E7%9A%84%E5%A4%9A%E7%A7%8D%E5%BD%A2%E5%BC%8F%20%28Leaky%20ReLU%2C%20PReLU%2C%20ELU%29.md)
