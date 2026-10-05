# 实操：修改带注意力机制的 U-Net

来源：[原文](https://apxml.com/zh/courses/advanced-diffusion-architectures/chapter-2-advanced-unet-architectures/practice-unet-attention-modification)

[返回章节目录](README.md) · [返回课程目录](../README.md)

此处，一个标准的 U-Net 块被修改为包含注意力机制 (attention mechanism)。加入注意力，特别是自注意力 (self-attention)，能让模型捕捉图像特征中的长距离关联，这对于生成有逻辑的结构很有益处。重点在于把一个空间自注意力层加到扩散模型中 U-Net 架构常见的残差块里。

我们假设你对 PyTorch 和 U-Net 的基本组成部分（卷积层、归一化 (normalization)、激活函数 (activation function)、残差连接）有一定了解。我们的目标是增强一个已有的块结构。

### 前提条件：一个基本的残差块

我们从 U-Net 中常用的典型残差块开始。它可能看起来像这样（简化版）：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # 可选的时间嵌入投影
        self.time_mlp = None
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels * 2) # 用于缩放和偏移
            )

        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.activation = nn.SiLU() # Swish 激活函数很常用

        # 跳跃连接匹配
        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_connection = nn.Identity()

    def forward(self, x, t_emb=None):
        residual = x

        h = self.norm1(x)
        h = self.activation(h)
        h = self.conv1(h)

        if self.time_mlp is not None and t_emb is not None:
            time_encoding = self.time_mlp(t_emb)
            time_encoding = time_encoding.view(h.shape[0], h.shape[1] * 2, 1, 1)
            scale, shift = torch.chunk(time_encoding, 2, dim=1)
            h = self.norm2(h) * (1 + scale) + shift # 调节特征
        else:
             h = self.norm2(h) # 如果没有时间嵌入调节，则应用归一化

        h = self.activation(h)
        h = self.conv2(h)

        return h + self.skip_connection(residual)
```

这个 `ResidualBlock` 包含了组归一化 (normalization)、卷积层、一个激活函数 (activation function) (SiLU)，并处理残差连接，以及可选的时间嵌入 (embedding)调节。

### 实现一个多头自注意力 (self-attention)块

现在，我们来定义一个适用于图像特征的自注意力块。由于标准多头自注意力 (MHSA) 作用于序列，我们需要将 2D 空间特征图重塑为序列。

```python
import math

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        assert channels % num_heads == 0, f"Channels ({channels}) must be divisible by num_heads ({num_heads})"
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = 1 / math.sqrt(self.head_dim)

        self.norm = nn.GroupNorm(32, channels)
        # 查询、键、值投影
        self.to_qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        # 输出投影
        self.to_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        residual = x

        qkv = self.to_qkv(self.norm(x))
        # 重塑以进行注意力计算：(b, c, h, w) -> (b, c, h*w) -> (b, num_heads, head_dim, h*w) -> (b*num_heads, head_dim, h*w)
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(b * self.num_heads, self.head_dim, h * w)
        k = k.reshape(b * self.num_heads, self.head_dim, h * w)
        v = v.reshape(b * self.num_heads, self.head_dim, h * w)

        # 带缩放的点积注意力
        # (b*num_heads, head_dim, h*w) @ (b*num_heads, h*w, head_dim) -> (b*num_heads, h*w, h*w)
        attention_scores = torch.bmm(q.transpose(1, 2), k) * self.scale
        attention_probs = F.softmax(attention_scores, dim=-1)

        # (b*num_heads, h*w, h*w) @ (b*num_heads, head_dim, h*w)' -> (b*num_heads, h*w, head_dim)
        out = torch.bmm(attention_probs, v.transpose(1, 2))

        # 重塑回原形：(b*num_heads, h*w, head_dim) -> (b, num_heads, h*w, head_dim) -> (b, c, h, w)
        out = out.transpose(1, 2).reshape(b, c, h, w)

        out = self.to_out(out)

        return out + residual # 添加残差连接
```

`AttentionBlock` 中的要点：

1. 我们在注意力计算前使用 `GroupNorm`，这是常见做法。
2. `to_qkv` 使用 1x1 卷积将输入投影为查询、键和值。
3. 空间维度 `(h, w)` 被展平为序列长度 `h*w`。
4. 进行标准带缩放的点积注意力计算。
5. 输出被重塑回 `(b, c, h, w)`。
6. 最后的 1x1 卷积将特征投影回。
7. 添加了残差连接。

### 将注意力整合到 U-Net 中

注意力块通常插入在残差块之后，尤其是在较低空间分辨率（U-Net 更深层）处，因为这些地方的特征图较小（例如 16x16 或 8x8），计算成本也更易于管理。

以下是你可能修改 U-Net 结构的方式，特别是在编码器或解码器循环内：

```python
# U-Net 编码器循环中的代码片段示例

# ...之前的层...
x = ResidualBlock(in_channels, out_channels, time_emb_dim)(x, t_emb)

# 如果维度合适（例如，较低分辨率），则添加注意力块
if x.shape[-1] <= 16: # 在 16x16 或更低分辨率时应用注意力
    x = AttentionBlock(out_channels, num_heads=8)(x)

# ...可能还有更多残差块或池化...
```

类似地，在解码器中：

```python
# U-Net 解码器循环中的代码片段示例

# ...上采样和连接...
x = ResidualBlock(in_channels, out_channels, time_emb_dim)(x, t_emb)

# 添加与编码器一致的注意力块
if x.shape[-1] <= 16:
    x = AttentionBlock(out_channels, num_heads=8)(x)

# ...更多层...
```

下图展示了注意力块可能在 U-Net 层的一系列操作中放置的位置。

> 该图示了特征流经标准残差块，在为该层生成最终输出之前，可能紧随其后的是一个注意力块。

### 试验和考量

现在你有了这些构建模块，请思考以下几点：

1. **放置位置：** 尝试将注意力块放置在不同的分辨率位置。太早添加（高分辨率）可能导致高昂的计算成本。仅在瓶颈处添加可能限制其效果。常见的选择是在较低分辨率层中的每个残差块之后添加它们。
2. **头数：** `num_heads` 参数 (parameter)控制注意力机制 (attention mechanism)的细致程度。更多的头能够关注不同的特征子空间，但会略微增加计算量。诸如 4、8 或 16 这样的值很常见。确保通道维度可以被 `num_heads` 整除。
3. **交叉注意力：** 对于条件模型（例如，文本到图像），你将实现一个 `CrossAttentionBlock`，作为自注意力 (self-attention)的一种替代或补充。键和值输入将来自条件上下文 (context)（例如，投影的文本嵌入 (embedding)），而查询将来自图像特征 `x`。这通常与自注意力放置在类似的位置。
4. **计算成本：** 注意力机制，特别是自注意力，其计算成本与序列长度（这里是 `h*w`）呈二次方关系。这就是为什么它们通常被限制在较低分辨率的原因。
5. **训练稳定性：** 确保在注意力块周围使用适当的归一化 (normalization)（如这里使用的 `GroupNorm`）和残差连接，以保持训练的稳定性。

通过整合注意力机制，你的 U-Net 能更好地建模复杂的空间关系，并可能更有效地整合条件信息，从而提升你的扩散模型的能力。记得仔细测试这些修改，同时监测样本质量和训练性能。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Volume: 30; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  介绍了Transformer架构和多头自注意力机制，这对于本节中实现的注意力模块至关重要。
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — Jonathan Ho, Ajay Jain, Pieter Abbeel (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Volume: 33; DOI: [10.48550/arXiv.2006.11239](https://doi.org/10.48550/arXiv.2006.11239)
  提出了去噪扩散概率模型及其基于U-Net的架构，其中包含了注意力机制，这对于扩散模型中的时间嵌入和U-Net设计尤为重要。
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer (2022)
  Journal: CVPR 2022; Pages: 10684-10695; DOI: [10.48550/arXiv.2112.10752](https://doi.org/10.48550/arXiv.2112.10752)
  介绍了潜在扩散模型，展示了一种高级的U-Net架构，该架构广泛使用自注意力和交叉注意力进行高分辨率条件图像生成，为生成式AI中带注意力的U-Net提供了实践背景。

---

[上一节](06-%E5%BD%92%E4%B8%80%E5%8C%96%E6%96%B9%E6%B3%95%EF%BC%88GroupNorm%2C%20AdaLN%EF%BC%89.md) · [下一节](../03-%E5%9F%BA%E4%BA%8ETransformer%E7%9A%84%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B/01-%E7%94%9F%E6%88%90%E5%BC%8F%E5%BB%BA%E6%A8%A1%E4%B8%AD%E9%87%87%E7%94%A8Transformer%E6%A8%A1%E5%9E%8B%E7%9A%84%E7%BC%98%E7%94%B1.md)
