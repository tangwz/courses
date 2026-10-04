---
course: "cnns-for-computer-vision"
chapter: "attention-transformers-vision"
lesson: "implementing-attention-blocks-practice"
sourceId: 2597
sourceUrl: "https://apxml.com/zh/courses/cnns-for-computer-vision/chapter-5-attention-transformers-vision/implementing-attention-blocks-practice"
title: "在CNN中实现注意力模块的实践"
description: "动手实践在标准CNN架构中实现注意力机制，如SE模块。"
order: 7
plots: []
sourceHash: "9a2bd9105d7f26ce3123478f7afd756f2fdace08320c49d7891e79c855626e65"
sourceCorrections: []
---

注意力机制 (attention mechanism)，例如Squeeze-and-Excitation (SE) 模块和非局部网络，使卷积神经网络 (neural network) (CNN) 能够动态地调整其特征通道或空间位置的权重 (weight)。这使得CNN能够有效地集中于输入中信息量更大的部分。在实践中实现这些注意力模块对于提升CNN性能是不可或缺的。这里将演示如何在典型的PyTorch CNN架构中实现广泛使用的Squeeze-and-Excitation (SE) 模块。

SE模块执行特征重校准，旨在明确地建立通道间的关联。它包含两个主要操作：

1. \*\*压缩：\*\*从每个通道的空间图中聚合全局信息。这通常通过全局平均池化（GAP）完成，生成一个通道描述向量 (vector) $z \in \mathbb{R}^C$，其中 $C$ 是通道数量。对于通道 $c$，压缩值 $z_c$ 计算如下：

   
   $$
   z_c = \frac{1}{H \times W} \sum_{i=1}^H \sum_{j=1}^W u_c(i, j)
   $$
   

   其中 $u_c$ 是输入 $U$ 的第 $c$ 个特征图，而 $H, W$ 是其空间维度。
2. \*\*激励：\*\*聚合的信息用于学习通道维度的注意力权重。这通常涉及两个全连接（线性）层：一个具有缩减比 $r$ 和激活函数 (activation function)（例如ReLU）的降维层，接着是一个维度增加回 $C$ 个通道并带有门控激活函数（例如Sigmoid）的层。生成的向量 $s \in \mathbb{R}^C$ 包含每个通道介于0和1之间的权重：

   
   $$
   s = \sigma(W_2 \delta(W_1 z))
   $$
   

   此处，$W_1 \in \mathbb{R}^{\frac{C}{r} \times C}$ 和 $W_2 \in \mathbb{R}^{C \times \frac{C}{r}}$ 是线性层的权重，$\delta$ 是ReLU激活函数，而 $\sigma$ 是Sigmoid激活函数。
3. \*\*缩放（或重校准）：\*\*原始输入特征图 $U$ 通过学习到的注意力权重 $s$ 进行缩放。输出特征图 $\tilde{X}$ 通过逐元素相乘得到：

   
   $$
   \tilde{x}_c = s_c \cdot u_c
   $$
   

   其中 $\tilde{x}_c$ 和 $u_c$ 分别是输出 $\tilde{X}$ 和输入 $U$ 的第 $c$ 个通道，$s_c$ 是通道 $c$ 的学习到的标量权重。

### 实现SE模块

让我们将其实现为一个可复用的PyTorch模块。我们定义一个继承自 `torch.nn.Module` 的 `SEBlock` 类。

```python
import torch
import torch.nn as nn

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation 模块。
    为卷积模块添加通道维度注意力。
    """
    def __init__(self, channels, reduction_ratio=16):
        """
        初始化SE模块。
        Args:
            channels (int): 输入通道数。
            reduction_ratio (int): 中间层通道缩减的因子。
                                   默认值: 16。
        """
        super(SEBlock, self).__init__()
        if channels <= reduction_ratio:
             # 避免将通道数降至零或负数
             reduced_channels = channels // 2 if channels > 1 else 1
        else:
            reduced_channels = channels // reduction_ratio

        # 压缩操作：全局平均池化
        self.squeeze = nn.AdaptiveAvgPool2d(1)

        # 激励操作：两个线性层
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        SE模块的前向传播。
        Args:
            x (torch.Tensor): 形状为 (batch, channels, height, width) 的输入张量。
        Returns:
            torch.Tensor: 输出张量，输入通过通道维度注意力权重进行缩放。
        """
        batch_size, num_channels, _, _ = x.size()

        # 压缩：(batch, channels, height, width) -> (batch, channels, 1, 1)
        squeezed = self.squeeze(x)
        # 为线性层重塑：(batch, channels, 1, 1) -> (batch, channels)
        squeezed = squeezed.view(batch_size, num_channels)

        # 激励：(batch, channels) -> (batch, channels)
        channel_weights = self.excitation(squeezed)
        # 为缩放重塑权重：(batch, channels) -> (batch, channels, 1, 1)
        channel_weights = channel_weights.view(batch_size, num_channels, 1, 1)

        # 缩放：将原始输入乘以学习到的通道权重
        scaled_output = x * channel_weights
        return scaled_output
```

此 `SEBlock` 模块现在可以轻松地集成到现有CNN架构中。 `reduction_ratio` 是一个超参数 (parameter) (hyperparameter)，用于控制注意力机制 (attention mechanism)的容量和计算成本。一个常用的值是16。

### 将SE模块集成到ResNet模块中

SE模块通常添加在一个构建块（如ResNet模块）中的主要卷积操作*之后*，但在添加残差连接*之前*。让我们说明如何修改一个基本的ResNet模块以包含一个SE层。

考虑一个简化的ResNet模块结构：

```python
class BasicResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 快捷连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x) # 准备快捷连接

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity # 添加快捷连接
        out = self.relu(out) # 最终ReLU
        return out
```

现在，让我们在残差相加之前添加 `SEBlock`：

```python
class SEResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, reduction_ratio=16):
        super(SEResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 在此处添加SE模块
        self.se_block = SEBlock(out_channels, reduction_ratio)

        # 快捷连接（与之前相同）
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x) # 准备快捷连接

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 将SE模块应用于主路径输出
        out = self.se_block(out)

        out += identity # 添加快捷连接
        out = self.relu(out) # 最终ReLU
        return out
```

下面的图表说明了 `SEResNetBlock` 内的数据流。

> 带有Squeeze-and-Excitation模块的ResNet模块内的数据流。SE模块在与快捷连接结合之前，对主卷积路径的特征图进行重校准。

### 注意事项

- **放置位置：** 尽管将SE模块放置在最终残差相加之前很常见，但也存在其他变体。尝试不同的放置位置（例如，在每个卷积层之后）可能会根据架构和任务产生不同的结果。
- **缩减比 `r`：** 这控制着激励阶段瓶颈层的复杂程度。较小的 `r`（例如8）意味着更复杂的瓶颈，可能更好地捕获通道间的关系，但会增加参数 (parameter)。较大的 `r`（例如32）会减少参数，但可能限制注意力机制 (attention mechanism)的表达能力。默认值16是一个合理的起始点。
- **计算成本：** SE模块会增加少量计算开销，主要来自两个线性层和池化操作。然而，与主要的卷积层相比，这通常可以忽略不计，尤其是在更深的网络中。参数的增加也相对适度。

这个实践例子体现了如何在标准CNN中实现和集成一个基本的通道注意力机制。通过根据全局通道上下文 (context)选择性地增强信息丰富的特征并抑制不那么有用的特征，SE模块通常可以提升各种计算机视觉任务的模型准确度。当集成其他注意力机制（例如空间注意力或非局部模块）时，也适用类似的原理，尽管它们的具体实现会有所不同。

## 参考资料

- [Deep Residual Learning for Image Recognition](https://doi.org/10.1109/CVPR.2016.90) — Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (2016)
  Journal: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR); Publisher: IEEE; Pages: 770-778; DOI: [10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90)
  介绍了残差网络（ResNet）架构，这是深度学习模型中广泛使用的基础网络，常与SE模块等注意力机制结合使用。
