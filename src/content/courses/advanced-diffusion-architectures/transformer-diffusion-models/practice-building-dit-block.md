---
course: "advanced-diffusion-architectures"
chapter: "transformer-diffusion-models"
lesson: "practice-building-dit-block"
sourceId: 5419
sourceUrl: "https://apxml.com/zh/courses/advanced-diffusion-architectures/chapter-3-transformer-diffusion-models/practice-building-dit-block"
title: "动手实践：构建一个简单的 DiT 模块"
description: "实现扩散Transformer架构的核心模块。"
order: 7
plots: []
sourceHash: "292a2f4d22480df1bd593f1f418308ad5cd9c831f675234d0ab46c6289a1b254"
sourceCorrections: []
---

构建一个单个的扩散Transformer（DiT）模块是一个实践练习，它体现了其核心组成部分。这个实现说明了调节信息（如时间步嵌入 (embedding)）是如何整合的，以及自注意力 (self-attention)如何在扩散框架内作用于图像块嵌入，从而阐明了DiTs如何调整Transformer原理用于图像生成。

我们将使用PyTorch来构建这个模块，这假定您对PyTorch的`nn.Module`和标准Transformer组件（如层归一化 (normalization)、多头自注意力以及MLP层）有实际的操作知识。

### DiT模块的构成

回顾我们对DiT架构的讨论，它处理图像块嵌入 (embedding)序列以及调节信息。一个标准DiT模块通常包含：

1. **输入：** 令牌嵌入序列 `x`（形状 `[batch_size, num_patches, hidden_dim]`）和一个调节向量 (vector) `c`（形状 `[batch_size, embedding_dim]`，通常源自时间步以及可能的类别标签）。
2. **自适应层归一化 (normalization)（adaLN / adaLN-Zero）：** 其尺度和偏移参数 (parameter)从调节向量 `c` 动态生成的层归一化层。“Zero”变体初始化生成这些参数的投影，使得模块在初始时表现得像一个恒等函数，这有助于提升稳定性。
3. **多头自注意力 (self-attention)（MHSA）：** 允许令牌（图像块）之间相互作用，从而获得空间关系。
4. **MLP模块：** 一个标准的前馈网络（通常是两个带有非线性函数，如GeLU或SiLU的线性层），独立应用于每个令牌。
5. **残差连接：** 将子层（如MHSA或MLP）的输入与其输出相加，有助于梯度流动。

具体的排列方式常遵循Transformer中常见的预归一化风格：

`Input -> adaLN -> MHSA -> Residual Add -> adaLN -> MLP -> Residual Add -> Output`

### 实现adaLN-Zero调制

DiT的一个重要特点是调节信息 `c` 如何调制图像令牌 `x` 的处理过程。这是通过`adaLN-Zero`机制完成的。我们需要一个小型子网络，它接收调节向量 (vector) `c` 并输出用于偏移、缩放和门控归一化 (normalization)激活的参数 (parameter)。

我们来为之定义一个辅助函数或模块。它接收 `c` 并产生 `shift`、`scale` 和 `gate` 参数。由于`adaLN-Zero`在MHSA和MLP层 *之前* 应用，我们需要为它们分别准备参数集。输出维度应为 `6 * hidden_dim`（MHSA的偏移、缩放、门控；MLP的偏移、缩放、门控）。

```python
import torch
import torch.nn as nn

class AdaLNModulation(nn.Module):
    """
    从调节嵌入计算偏移、缩放和门控参数。
    将最终线性层的权重和偏置初始化为零。
    """
    def __init__(self, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        # 将线性层初始化为零，以实现“Zero”特性
        self.linear = nn.Linear(embedding_dim, 6 * hidden_dim, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, c):
        # c 的形状: [batch_size, embedding_dim]
        mod_params = self.linear(self.silu(c))
        # mod_params 的形状: [batch_size, 6 * hidden_dim]
        # 拆分为6部分，分别用于 shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        # 每个部分的形状为 [batch_size, hidden_dim]
        # 添加 unsqueeze(1) 使其能与令牌维度广播：[batch_size, 1, hidden_dim]
        return mod_params.chunk(6, dim=1)
```

### 构建DiT模块

现在我们可以组装`DiTBlock`了。我们将使用标准的PyTorch模块来构建`LayerNorm`、`MultiheadAttention`和MLP。

```python
import torch
import torch.nn as nn

# 假定AdaLNModulation类已在上方定义

class DiTBlock(nn.Module):
    """
    一个带有adaLN-Zero调制的标准DiT模块。
    """
    def __init__(self, hidden_dim: int, embedding_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # 归一化层（使用LayerNorm，不带元素级仿射，
        # 因为adaLN提供了仿射参数）
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)

        # 生成adaLN参数的调制网络
        self.modulation = AdaLNModulation(embedding_dim, hidden_dim)

        # 注意力层
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        # MLP层
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(), # 或 nn.SiLU()
            nn.Linear(mlp_hidden_dim, hidden_dim),
        )

    def forward(self, x, c):
        # x 的形状: [batch_size, num_patches, hidden_dim]
        # c 的形状: [batch_size, embedding_dim]

        # 计算调制参数（MSA和MLP的偏移、缩放、门控）
        # 每个参数的形状：在 unsqueeze 后为 [batch_size, 1, hidden_dim]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [
            p.unsqueeze(1) for p in self.modulation(c)
        ]

        # --- 在MHSA之前应用adaLN-Zero ---
        x_norm1 = self.norm1(x)
        # 调制：缩放，偏移
        x_modulated1 = x_norm1 * (1 + scale_msa) + shift_msa
        # 应用注意力
        attn_output, _ = self.attn(x_modulated1, x_modulated1, x_modulated1)
        # 应用门控并添加残差连接
        x = x + gate_msa * attn_output

        # --- 在MLP之前应用adaLN-Zero ---
        x_norm2 = self.norm2(x)
        # 调制：缩放，偏移
        x_modulated2 = x_norm2 * (1 + scale_mlp) + shift_mlp
        # 应用MLP
        mlp_output = self.mlp(x_modulated2)
        # 应用门控并添加残差连接
        x = x + gate_mlp * mlp_output

        # 输出形状: [batch_size, num_patches, hidden_dim]
        return x
```

### 代码讲解

1. **初始化 (`__init__`)**:

   - 我们初始化了两个`LayerNorm`模块（`norm1`、`norm2`），并将`elementwise_affine`设置为`False`。这一点很重要，因为自适应的缩放和偏移参数 (parameter)将由我们的`modulation`网络提供，而不是在`LayerNorm`层内部静态学习。
   - 创建`AdaLNModulation`实例（`modulation`）是为了从调节向量 (vector) `c` 计算出6个必需的参数（MHSA和MLP路径的偏移、缩放、门控）。
   - 定义了一个标准的`MultiheadAttention`层（`attn`）。`batch_first=True`使得处理形状为`[batch, sequence, feature]`的输入更加方便。
   - 创建了一个标准的MLP模块（`mlp`），它通常会将隐藏维度按一个系数（`mlp_ratio`）扩展，然后投影回去。GELU是这里常用的激活函数 (activation function)。
2. **前向传播 (`forward`)**:

   - 调节向量 `c` 通过`modulation`网络，得到六组参数。我们对它们进行`unsqueeze(1)`操作，使其形状变为`[batch_size, 1, hidden_dim]`，以便在调制过程中能正确地在序列长度（图像块数量）维度上进行广播。
   - **第一个子层（注意力）：**
     - 输入`x`通过`norm1`进行归一化 (normalization)。
     - 归一化后的`x_norm1`使用`scale_msa`和`shift_msa`参数进行调制：`x_modulated1 = x_norm1 * (1 + scale_msa) + shift_msa`。请注意`(1 + scale)`的表示形式，这在自适应归一化方案中很常见。
     - 调制后的张量`x_modulated1`被送入自注意力 (self-attention)层（`attn`）。
     - 注意力层的输出由`gate_msa`进行门控，并加回到原始输入`x`（残差连接）：`x = x + gate_msa * attn_output`。
   - **第二个子层（MLP）：**
     - 注意力子层的输出（`x`）通过`norm2`进行归一化。
     - 这个归一化张量`x_norm2`使用`scale_mlp`和`shift_mlp`进行调制。
     - 调制后的张量`x_modulated2`通过`mlp`。
     - MLP输出由`gate_mlp`进行门控，并加回到该子层的输入（残差连接）：`x = x + gate_mlp * mlp_output`。
   - 返回最终张量`x`。

### 模块的使用

要构建一个完整的DiT，您将堆叠多个此`DiTBlock`的实例。输入`x`最初将是图像块嵌入 (embedding)加上位置嵌入，而`c`将从扩散时间步（以及可能的类别标签或其他调节信息）获取。最终模块的输出通常会投影来预测噪声（或者原始数据`x_0`，这取决于参数 (parameter)化方式）。

```python
# 使用示例（说明性）
batch_size = 4
num_patches = 196  # 例如，对于224x224图像和16x16图像块
hidden_dim = 768
embedding_dim = 256 # 时间步/类别嵌入的维度
num_heads = 12

# 输入张量示例
x_patches = torch.randn(batch_size, num_patches, hidden_dim)
conditioning_vec = torch.randn(batch_size, embedding_dim)

# 实例化模块
dit_block = DiTBlock(hidden_dim, embedding_dim, num_heads)

# 将输入传递给模块
output_tokens = dit_block(x_patches, conditioning_vec)

print(f"Input shape: {x_patches.shape}")
print(f"Conditioning shape: {conditioning_vec.shape}")
print(f"Output shape: {output_tokens.shape}")

# 预期输出：
# Input shape: torch.Size([4, 196, 768])
# Conditioning shape: torch.Size([4, 256])
# Output shape: torch.Size([4, 196, 768])
```

这个动手实现提供了一个具体视角，说明Transformer模块如何在DiT框架内被调整，尤其侧重于`adaLN-Zero`机制如何有效地整合调节信息。在此单个模块的构建基础上，可以搭建扩散模型的整个Transformer主干网络。

## 参考资料

- [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 30; Pages: 5998-6008; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  提出了基础的Transformer模型，详细说明了自注意力、多头注意力、前馈网络和残差连接，这些构成了DiT块的基础。
- [Denoising Diffusion Probabilistic Models](https://papers.nips.cc/paper/2020/file/4c5bcfec8584af0d96d2a1d027e2501a-Paper.pdf) — Jonathan Ho, Ajay Jain, Pieter Abbeel (2020)
  Journal: Advances in Neural Information Processing Systems; Publisher: NeurIPS; Volume: 33; Pages: 6840-6851; DOI: [10.48550/arXiv.2006.11239](https://doi.org/10.48550/arXiv.2006.11239)
  为去噪扩散概率模型提供了理论和实践基础，DiT模型正是为此类生成框架而设计。
