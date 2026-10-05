# 动手实践：搭建U-Net

来源：[原文](https://apxml.com/zh/courses/intro-diffusion-models/chapter-4-model-architecture-training/hands-on-unet-setup)

[返回章节目录](README.md) · [返回课程目录](../README.md)

具体的代码片段将展示使用PyTorch实现U-Net架构和纳入时间步信息。示例说明了如何实现主要组成部分：U-Net结构本身和正弦时间步编码。

### 实现正弦时间步编码

如前所述，网络需要知道它在扩散过程中的*哪一个点*进行操作。正弦编码提供了一种方法，可以将时间步 $t$ 编码成一个固定大小的向量 (vector)，网络可以方便地使用它。

这些编码的公式涉及正弦和余弦函数，它们应用于时间步，并在不同频率上进行缩放：


$$
PE(t, 2i) = \sin(t / 10000^{2i/d})
$$


$$
PE(t, 2i+1) = \cos(t / 10000^{2i/d})
$$


这里，$t$ 是时间步，$i$ 是编码维度索引，$d$ 是总编码维度。该方法使模型能够有效地区分不同的时间步。

让我们在PyTorch中实现一个函数来生成这些编码：

```python
import torch
import torch.nn as nn
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim # 编码的维度

    def forward(self, time):
        """
        为一批时间步生成正弦编码。

        Args:
            time (torch.Tensor): 形状为 (batch_size,) 的张量，包含时间步。

        Returns:
            torch.Tensor: 形状为 (batch_size, dim) 的张量，包含编码。
        """
        device = time.device
        half_dim = self.dim // 2
        # 计算频率（10000 的指数）
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        # 计算正弦和余弦的参数
        embeddings = time[:, None] * embeddings[None, :]
        # 连接正弦和余弦分量
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # 如有必要处理奇数维度（尽管 'dim' 通常是偶数）
        if self.dim % 2 == 1:
             embeddings = torch.cat([embeddings, torch.zeros_like(embeddings[:, :1])], dim=-1)
        return embeddings

# 示例用法:
time_embedding_dim = 128 # 选择一个编码维度
time_embed_module = SinusoidalPosEmb(time_embedding_dim)

# 模拟一批时间步
batch_size = 16
timesteps = torch.randint(0, 1000, (batch_size,)) # 示例：1000个扩散步骤

# 生成编码
time_embeddings = time_embed_module(timesteps)
print(f"时间步形状: {timesteps.shape}")
print(f"生成的编码形状: {time_embeddings.shape}")
# 预期输出:
# 时间步形状: torch.Size([16])
# 生成的编码形状: torch.Size([16, 128])
```

这个 `SinusoidalPosEmb` 模块接收一批时间步并生成相应的编码。这些编码随后将被投影并添加到U-Net层中。

### U-Net基本模块结构

U-Net依靠可重用模块进行下采样和上采样。一个典型模块包含卷积、归一化 (normalization)（例如组归一化，它在批大小变化时表现良好）、激活函数 (activation function)（例如SiLU或GeLU）以及可能的残差连接。时间步编码通常被纳入这些模块中。

这是一个包含时间步编码集成的残差模块的简化示例：

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, groups=8):
        super().__init__()
        # 将时间步编码投影以匹配通道维度
        self.time_mlp = nn.Sequential(
            nn.SiLU(), # Swish 激活
            nn.Linear(time_emb_dim, out_channels)
        )

        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.act1 = nn.SiLU()

        # 第二个卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.act2 = nn.SiLU()

        # 残差连接处理（如果输入/输出通道不同）
        if in_channels != out_channels:
            self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.res_conv = nn.Identity() # 无需更改

    def forward(self, x, t_emb):
        """
        Args:
            x (torch.Tensor): 输入特征图 (批次, 输入通道, 高度, 宽度)。
            t_emb (torch.Tensor): 时间步编码 (批次, 时间编码维度)。

        Returns:
            torch.Tensor: 输出特征图 (批次, 输出通道, 高度, 宽度)。
        """
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act1(h)

        # 纳入时间步编码
        time_bias = self.time_mlp(t_emb)
        # 将 time_bias 重塑为 (批次, 输出通道, 1, 1) 以进行广播
        h = h + time_bias[:, :, None, None]

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act2(h)

        # 添加残差连接
        return h + self.res_conv(x)

# 示例用法（需要示例输入和时间编码）
in_channels = 32
out_channels = 64
time_emb_dim = 128
img_size = 32 # 示例图像维度

# 创建虚拟输入张量和时间编码
x_sample = torch.randn(batch_size, in_channels, img_size, img_size)
t_emb_sample = torch.randn(batch_size, time_emb_dim) # 通常使用预生成的编码

# 实例化并运行模块
block = ResidualBlock(in_channels, out_channels, time_emb_dim)
output = block(x_sample, t_emb_sample)
print(f"输入形状: {x_sample.shape}")
print(f"时间编码形状: {t_emb_sample.shape}")
print(f"输出形状: {output.shape}")
# 预期输出:
# 输入形状: torch.Size([16, 32, 32, 32])
# 时间编码形状: torch.Size([16, 128])
# 输出形状: torch.Size([16, 64, 32, 32])
```

在这个 `ResidualBlock` 中，时间步编码 `t_emb` 首先通过一个小型多层感知机 (`time_mlp`)，它由一个激活函数 (`SiLU`) 和一个线性层组成。这会将编码投影到 `out_channels` 维度。结果在第一次卷积和归一化后作为偏差添加到特征图 `h` 中。`[:, :, None, None]` 部分重塑偏差，使其能够在空间维度（高度和宽度）上进行广播。该模块以第二次卷积、归一化、激活和残差连接 (`res_conv(x)`) 的添加结束。

### 构成U-Net结构

一个完整的U-Net使用这些模块，以及下采样（例如 `nn.MaxPool2d` 或步幅卷积）和上采样（例如 `nn.Upsample` 或 `nn.ConvTranspose2d`）层。跳跃连接将下采样路径中的特征图连接到上采样路径中对应的层。

下面是这些组件如何组合的概略示意图。请注意，这是一个简化表示；实际的U-Net通常在每个分辨率级别有多个模块，并可能包含注意力机制 (attention mechanism)。

> 图解了用于噪声预测的U-Net架构内的流程。输入图像和时间步编码通过下采样模块（编码器）、一个瓶颈和带跳跃连接的上采样模块（解码器）进行处理。时间步信息通常会被注入到多个模块中。

构建一个完整的U-Net涉及堆叠这些 `ResidualBlock` 组件，添加适当的下采样（例如使用 `stride=2` 的 `nn.Conv2d`）和上采样层（例如 `nn.Upsample` 或 `nn.ConvTranspose2d`），并实现跳跃连接（通常是沿通道维度简单的张量连接）。主U-Net类的 `forward` 方法组织此流程，将输入图像 `x` 和生成的时间步编码 `t_emb` 传递通过网络结构。

"请记住，这是一个基础设置。实现中通常会纳入更复杂的元素，例如注意力层（特别是在较低分辨率下），以及对通道数量和模块深度进行细致的调整，这些对于获得高质量的生成结果非常重要。然而，理解这些基本构成模块为处理扩散模型架构提供了一个扎实的起点。"

## 参考资料

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) — Olaf Ronneberger, Philipp Fischer, Thomas Brox (2015)
  Journal: MICCAI 2015; Volume: 9351; Pages: 234-241; DOI: [10.48550/arXiv.1505.04597](https://doi.org/10.48550/arXiv.1505.04597)
  介绍U-Net架构的原始论文，它是图像到图像任务的基本组成部分，也是扩散模型解码器的基础。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Volume: 30; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  介绍了正弦位置嵌入技术，该技术被改编用于扩散模型中编码时间步信息。
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — Jonathan Ho, Ajay Jain, Pieter Abbeel (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Volume: 33; DOI: [10.48550/arXiv.2006.11239](https://doi.org/10.48550/arXiv.2006.11239)
  描述了在扩散模型中，使用带有时间步条件的U-Net进行高质量图像生成的开创性论文。

---

[上一节](05-%E8%AE%AD%E7%BB%83%E7%AE%97%E6%B3%95.md) · [下一节](../05-%E9%87%87%E6%A0%B7%E4%B8%8E%E7%94%9F%E6%88%90%E8%BF%87%E7%A8%8B/01-%E4%BB%8E%E5%99%AA%E5%A3%B0%E7%94%9F%E6%88%90%E6%95%B0%E6%8D%AE.md)
