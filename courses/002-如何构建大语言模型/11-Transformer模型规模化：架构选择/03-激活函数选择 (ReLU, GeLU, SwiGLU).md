# 激活函数选择 (ReLU, GeLU, SwiGLU)

来源：[原文](https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-11-scaling-transformers-architectural-choices/choice-activation-functions)

[返回章节目录](README.md) · [返回课程目录](../README.md)

每个Transformer模块中的前馈网络 (FFN) 子层在改变注意力机制 (attention mechanism)学习到的表示方面发挥着重要作用。标准FFN由两个线性变换和一个非线性激活函数 (activation function)组成：

$FFN(x) = max(0, xW_1 + b_1)W_2 + b_2$

其中，$x$ 是注意力子层的输入，$W_1$、$b_1$、$W_2$ 和 $b_2$ 是可学习参数 (parameter)，所示的激活函数是ReLU。这种非线性是必不可少的；没有它，两个线性层将合并为一个线性变换，从而限制模型的表达能力。

随着模型规模扩大，这种激活函数的选择不仅仅是一个小细节。它会影响梯度流动、训练稳定性、计算成本，并最终影响模型的最终表现。下面我们检视大型Transformer中常见的选项：ReLU、GeLU和SwiGLU。

### 修正线性单元 (ReLU)

修正线性单元，即ReLU，定义为 $ReLU(x) = max(0, x)$，是深度学习 (deep learning)的基本激活函数 (activation function)。它的主要优点是简单性和计算效率。它避免了在深度网络中经常与sigmoid或tanh函数出现的梯度消失问题。

```python
import torch
import torch.nn as nn

# 简化版FFN中ReLU的使用示例
d_model = 512
d_ff = 2048 # 典型的内部维度是 4*d_model
relu_ffn = nn.Sequential(
    nn.Linear(d_model, d_ff),
    nn.ReLU(),
    nn.Linear(d_ff, d_model)
)

# 输入示例
x = torch.randn(16, 128, d_model) # 批次，序列长度，维度
output = relu_ffn(x)
print("Output shape:", output.shape)
# 输出： 输出形状： torch.Size([16, 128, 512])
```

然而，ReLU并非没有缺点。主要问题是“死亡ReLU”问题：如果神经元的输入持续低于零，它们可能变得不活跃，导致它们的权重 (weight)停止更新，因为在该区域梯度为零。虽然像仔细初始化和降低学习率等技术可以减轻此问题，但它仍然需要考虑，特别是在非常深的神经网络 (neural network)中。此外，它在 $x=0$ 处的非平滑性质有时会阻碍优化，相较于更平滑的替代品。

### 高斯误差线性单元 (GeLU)

高斯误差线性单元 (GeLU) 作为ReLU的一种更平滑的替代品被提出，并随着BERT和GPT系列模型而获得广泛应用。它根据输入值进行加权，但这种加权是随机的，结合了标准高斯累积分布函数 ($\Phi(x)$)。

$GeLU(x) = x \cdot \Phi(x)$

由于计算精确高斯累积分布函数可能较慢，因此常使用近似方法：

$GeLU(x) \approx 0.5x \left(1 + \tanh\left[\sqrt{2/\pi}(x + 0.044715x^3)\right]\right)$

直观理解是GeLU提供比ReLU更平滑的曲线，可能使得优化更容易，梯度流动更好。经验上，它在Transformer模型中通常表现优于ReLU。

```python
import torch
import torch.nn as nn

# 简化版FFN中GeLU的使用示例
d_model = 512
d_ff = 2048
gelu_ffn = nn.Sequential(
    nn.Linear(d_model, d_ff),
    nn.GELU(), # PyTorch 默认使用近似方法
    nn.Linear(d_ff, d_model)
)

# 输入示例
x = torch.randn(16, 128, d_model)
output = gelu_ffn(x)
print("Output shape:", output.shape)
# 输出： 输出形状： torch.Size([16, 128, 512])
```

GeLU的计算量略大于ReLU，但受到硬件加速器的良好支持。它在许多基本大型语言模型中的成功使其在多年来成为一个标准选择。

### Swish门控线性单元 (SwiGLU)

近期，FFN层中涉及门控机制的变体表现出良好性能。一个流行变体是SwiGLU，它在PaLM论文中提出，并用于Llama等模型。

核心理念是将Swish激活函数 (activation function) ($Swish(x) = x \cdot \sigma(x)$，其中 $\sigma$ 是sigmoid函数) 与门控机制结合。SwiGLU通常不使用单个线性层来扩展维度，而是使用两个线性层，它们的输出进行逐元素相乘。其中一个输出通过Swish函数，作为另一个的门。

$SwiGLU(x, W, V, b, c) = (xW + b) \otimes Swish(xV + c)$

其中，$x$ 是输入，$W$、$V$、$b$ 和 $c$ 是可学习参数 (parameter)，$\otimes$ 表示逐元素相乘。Swish函数定义为：

$Swish(x) = x \cdot \sigma(\beta x)$
通常，$\beta$ 设置为1或设为可学习参数。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUFFN(nn.Module):
    def __init__(self, dim, hidden_dim, bias=True):
        super().__init__()
        # 通常 hidden_dim 会按比例调整，例如 2/3 * 4 * dim，
        # 因为 SwiGLU 会分割中间表示。
        # 这里我们简化处理，假设 hidden_dim 是目标
        # 维度，即门控分割*之前*的维度。

        # 我们需要两个线性层来实现门控机制
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(dim, hidden_dim, bias=bias)
        # 最终的线性层
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x):
        # 应用两个线性层
        hidden1 = self.w1(x)
        hidden2 = self.w2(x)

        # 对第一个输出应用Swish激活并进行逐元素相乘
        gated_hidden = F.silu(hidden1) * hidden2 # F.silu 是 PyTorch 的 Swish

        # 应用最终的线性层
        output = self.w3(gated_hidden)
        return output

# SwiGLU 使用示例
d_model = 512
# SwiGLU 中实际的隐藏维度需要仔细考虑。
# 常见做法是使用隐藏维度，例如 (2/3 * 4 * d_model)
# 这样参数数量与标准 FFN（4 * d_model）相似。
# 为简单起见，这里我们使用较小的 hidden_dim。
d_ff_swiglu = 1024 # 门控的隐藏维度示例

swiglu_ffn = SwiGLUFFN(d_model, d_ff_swiglu)

# 输入示例
x = torch.randn(16, 128, d_model)
output = swiglu_ffn(x)
print("Output shape:", output.shape)
# 输出： 输出形状： torch.Size([16, 128, 512])
```

关于SwiGLU（以及GeGLU等类似门控激活函数）一个细微但重要的点是其对参数数量的影响。为了使SwiGLU实现中使用的 `hidden_dim`（如上文的 `d_ff_swiglu`）与中间维度为 $d_{ff}$ 的标准ReLU/GeLU FFN保持相似的参数数量，该 `hidden_dim` 通常设置为约 $\frac{2}{3} d_{ff}$。这是因为SwiGLU使用两个线性投影 ($W$ 和 $V$) 来达到中间维度，有效地分摊了标准FFN中通常由一个更大矩阵 ($W_1$) 处理的容量。尽管如此，SwiGLU在大型模型中经常被发现能够得到更好的困惑度分数和下游表现，相比GeLU或ReLU，这表明门控机制有好处。

### 比较与选择



[交互图表：常见激活函数](https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-11-scaling-transformers-architectural-choices/choice-activation-functions#plot-xb37qm)

<details>
<summary>图表数据（JSON）</summary>

```json
{
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
        "color": "#228be6"
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
        -0.004,
        -0.045,
        -0.159,
        0,
        0.841,
        1.954,
        2.996
      ],
      "mode": "lines",
      "name": "GeLU (近似)",
      "line": {
        "color": "#12b886"
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
        -0.142,
        -0.238,
        -0.269,
        0,
        0.731,
        1.762,
        2.858
      ],
      "mode": "lines",
      "name": "Swish (β=1)",
      "line": {
        "color": "#fd7e14"
      }
    }
  ],
  "layout": {
    "title": "常见激活函数",
    "xaxis": {
      "title": "x"
    },
    "yaxis": {
      "title": "激活函数值(x)"
    },
    "template": "plotly_white",
    "legend": {
      "yanchor": "bottom",
      "y": 0.01,
      "xanchor": "right",
      "x": 0.99
    }
  }
}
```

</details>



> ReLU、GeLU（近似）和Swish激活函数 (activation function)的比较。请注意从ReLU到GeLU再到Swish，平滑度逐渐增加。

选择合适的激活函数涉及权衡：

- **ReLU：** 计算最快，形式最简单。有神经元“死亡”的风险，非平滑。目前在最先进的大型模型中较少见，但仍然可用。
- **GeLU：** 性能和计算成本之间有良好平衡。比ReLU更平滑，在许多知名大型语言模型中经验上表现出色。一个可靠的默认选择。
- **SwiGLU（及其他门控变体）：** 通常在近期大型模型中获得最佳性能（例如，更低的困惑度）。引入了门控，可能对信息流动提供更好的控制。需要在隐藏维度方面仔细实施，以有效管理参数 (parameter)数量。由于门控乘法，计算成本可能略高于GeLU。

在扩展Transformer模型时，从ReLU转向GeLU或SwiGLU是一种旨在提升表现的常见架构变化。尽管FFN实现较复杂，SwiGLU带来的性能提升促使其被多个近期大型模型采用。与许多架构选择一样，最佳选择可能取决于特定的模型大小、数据集和计算预算，通常需要经验验证。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems 30 (NIPS 2017); DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  引入Transformer架构的开创性论文，包括前馈网络的结构和作用。
- [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415) — Dan Hendrycks, Kevin Gimpel (2016)
  Journal: arXiv preprint; DOI: [10.48550/arXiv.1606.08415](https://doi.org/10.48550/arXiv.1606.08415)
  引入高斯误差线性单元（GeLU）激活函数的原始研究论文，该函数在许多Transformer模型中成为标准选择。

---

[上一节](02-%E6%B7%B1%E5%BA%A6%E4%B8%8E%E5%AE%BD%E5%BA%A6%E5%8F%96%E8%88%8D.md) · [下一节](04-%E8%A7%84%E8%8C%83%E5%8C%96%E5%B1%82%E6%94%BE%E7%BD%AE%E4%BD%8D%E7%BD%AE%EF%BC%88%E5%89%8D%E7%BD%AELN%20vs.%20%E5%90%8E%E7%BD%AELN%EF%BC%89.md)
