---
course: "synthetic-data-gans-diffusion"
chapter: "advanced-gan-architectures-techniques"
lesson: "practice-implementing-stylegan"
sourceId: 5664
sourceUrl: "https://apxml.com/zh/courses/synthetic-data-gans-diffusion/chapter-2-advanced-gan-architectures-techniques/practice-implementing-stylegan"
title: "动手实践：实现StyleGAN组件"
description: "StyleGAN核心构成模块的代码实现。"
order: 7
plots: []
sourceHash: "1e3b9a5ccc34f6dcf09c6ffaad84cbc4781f7b0ca526a2a317b48ed92c219c85"
sourceCorrections: []
---

实现StyleGAN生成器架构的基本构建模块是主要目标。这包括映射网络（Mapping Network）、合成网络（Synthesis Network）和自适应实例归一化 (normalization)（Adaptive Instance Normalization, AdaIN）机制的实际应用。理解这些组件对于把握StyleGAN如何实现其高保真结果和可控合成非常重要。

我们假设您熟悉PyTorch（或TensorFlow，尽管示例将使用PyTorch），并且有实现自定义神经网络 (neural network)层和架构（包括卷积网络）的经验。

### 映射网络

StyleGAN生成过程的第一步是将初始潜在代码$z$（通常从标准正态分布$\mathcal{N}(0, I)$中抽取）转换为中间潜在空间$W$。这种转换由映射网络（通常是多层感知机MLP）执行。它的作用是解耦潜在空间，这意味着与Z中的变化相比，W中的变化应更直接地对应于生成图像中不同的语义属性。

一个典型的映射网络由几个全连接层以及LeakyReLU或ReLU等激活函数 (activation function)组成。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MappingNetwork(nn.Module):
    def __init__(self, z_dim, w_dim, num_layers=8):
        """
        初始化映射网络。

        参数:
            z_dim (int): 输入潜在代码z的维度。
            w_dim (int): 输出中间潜在代码w的维度。
            num_layers (int): 网络中的线性层数量。
        """
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.num_layers = num_layers

        layers = []
        # 输入层归一化（可选但常见）
        # layers.append(nn.LayerNorm(z_dim)) # 或 PixelNorm

        # 隐藏层
        current_dim = z_dim
        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, w_dim))
            layers.append(nn.LeakyReLU(0.2))
            current_dim = w_dim # 后续层使用w_dim

        self.network = nn.Sequential(*layers)

    def forward(self, z):
        """
        映射网络的前向传播。

        参数:
            z (torch.Tensor): 输入潜在代码 (batch_size, z_dim)。

        返回:
            torch.Tensor: 输出中间潜在代码w (batch_size, w_dim)。
        """
        # 归一化z（可选，取决于特定的StyleGAN版本）
        # z = F.normalize(z, dim=1) * (self.z_dim ** 0.5)
        w = self.network(z)
        return w

# 使用示例:
z_dim = 512
w_dim = 512
batch_size = 4

mapping_net = MappingNetwork(z_dim, w_dim)
z_input = torch.randn(batch_size, z_dim)
w_output = mapping_net(z_input)

print(f"输入 z 的形状: {z_input.shape}")
print(f"输出 w 的形状: {w_output.shape}")
```

这个实现创建了一个8层MLP。实际中，可以在网络前对$z$应用归一化 (normalization)技术，并且MLP内部也可以使用不同的激活函数或层归一化方案。主要目的是实现从Z到W的非线性转换。

### 自适应实例归一化 (normalization) (AdaIN)

AdaIN是StyleGAN用以将编码在中间潜在向量 (vector)$w$中的风格信息注入合成网络的机制。与批归一化（它对整个批次进行归一化）或实例归一化（它对单个样本的空间维度进行归一化）不同，AdaIN按通道归一化卷积层的激活$x$，然后使用从风格向量$w$获得的参数 (parameter)对其进行缩放和偏移。

AdaIN操作定义为：


$$
\text{AdaIN}(x_i, w) = \gamma_i(w) \frac{x_i - \mu(x_i)}{\sqrt{\sigma^2(x_i) + \epsilon}} + \beta_i(w)
$$


这里，$x_i$表示第$i$个通道的激活。$\mu(x_i)$和$\sigma(x_i)$是针对该通道在空间维度（高和宽）上计算的均值和标准差。$\epsilon$是一个用于数值稳定的小常数。重要之处在于，缩放参数$\gamma_i(w)$和偏置 (bias)参数$\beta_i(w)$是中间潜在代码$w$的学习函数。通常，这些参数是通过对$w$应用单独的学习仿射变换（线性层），为使用AdaIN的每个层的每个通道$i$生成的。

```python
import torch
import torch.nn as nn

class AdaIN(nn.Module):
    def __init__(self, channels, w_dim):
        """
        初始化自适应实例归一化层。

        参数:
            channels (int): 输入激活图中的通道数量。
            w_dim (int): 中间潜在代码w的维度。
        """
        super().__init__()
        self.channels = channels
        self.w_dim = w_dim
        self.instance_norm = nn.InstanceNorm2d(channels, affine=False) # affine=False 因为我们提供自己的缩放/偏置参数

        # 学习的仿射变换，将w映射到缩放因子(gamma)和偏置(beta)
        self.style_scale_transform = nn.Linear(w_dim, channels)
        self.style_bias_transform = nn.Linear(w_dim, channels)

    def forward(self, x, w):
        """
        AdaIN的前向传播。

        参数:
            x (torch.Tensor): 输入激活图 (batch_size, channels, height, width)。
            w (torch.Tensor): 中间潜在代码 (batch_size, w_dim)。

        返回:
            torch.Tensor: 应用AdaIN后的激活图 (batch_size, channels, height, width)。
        """
        # 按通道归一化输入激活
        normalized_x = self.instance_norm(x)

        # 从w生成缩放和偏置参数
        # 形状: (batch_size, channels)
        style_scale = self.style_scale_transform(w)
        style_bias = self.style_bias_transform(w)

        # 重塑缩放和偏置以便广播: (batch_size, channels, 1, 1)
        style_scale = style_scale.view(-1, self.channels, 1, 1)
        style_bias = style_bias.view(-1, self.channels, 1, 1)

        # 应用学习到的缩放和偏置
        transformed_x = style_scale * normalized_x + style_bias
        return transformed_x

# 使用示例:
batch_size = 4
channels = 64
height, width = 32, 32
w_dim = 512

adain_layer = AdaIN(channels, w_dim)
x_input = torch.randn(batch_size, channels, height, width) # 示例激活图
w_vector = torch.randn(batch_size, w_dim) # 示例中间潜在

output_x = adain_layer(x_input, w_vector)

print(f"输入 x 的形状: {x_input.shape}")
print(f"输入 w 的形状: {w_vector.shape}")
print(f"输出 x 的形状: {output_x.shape}")
```

请注意，`InstanceNorm2d`使用`affine=False`，因为仿射参数($\gamma, \beta$)是通过`style_scale_transform`和`style_bias_transform`层从$w$动态生成的。

### 合成网络块

合成网络逐步生成图像，通常从一个学习到的常数张量开始，并逐渐提高分辨率。每个分辨率级别一般包含一个或多个块，内含上采样、卷积、噪声注入、AdaIN和激活函数 (activation function)。

我们来关注实现一个单独的StyleGAN块。该块一般接收来自前一层的激活图和风格向量 (vector)$w$，应用卷积，注入学习到的噪声，应用AdaIN，然后使用激活函数。上采样可能会在块之前或块内部发生，具体取决于特定的StyleGAN版本和分辨率级别。

**噪声注入：** StyleGAN在每次卷积之后但在激活函数之前，独立地向每个通道添加学习到的逐像素噪声。这加入随机细节（如头发位置、雀斑），这些细节不需要由$w$控制。噪声通过学习到的逐通道因子进行缩放。

```python
import torch
import torch.nn as nn

class NoiseInjection(nn.Module):
    def __init__(self, channels):
        """
        初始化噪声注入层。

        参数:
            channels (int): 应用噪声的通道数量。
        """
        super().__init__()
        # 每个通道可学习的缩放因子
        # 初始化为0，以便噪声最初没有影响
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x, noise=None):
        """
        噪声注入的前向传播。

        参数:
            x (torch.Tensor): 输入激活图 (batch_size, channels, height, width)。
            noise (torch.Tensor, optional): 预生成的噪声张量。
                                            如果为None，则生成噪声。默认为None。

        返回:
            torch.Tensor: 添加了缩放噪声的激活图。
        """
        if noise is None:
            batch, _, height, width = x.shape
            # 在正确的设备上生成噪声
            noise = torch.randn(batch, 1, height, width, device=x.device, dtype=x.dtype)

        # 向输入添加缩放噪声
        return x + self.weight * noise

# 组合成合成块
class SynthesisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, w_dim, use_upsample=True):
        """
        初始化一个基本的StyleGAN合成块。

        参数:
            in_channels (int): 输入通道数。
            out_channels (int): 输出通道数。
            w_dim (int): 中间潜在代码w的维度。
            use_upsample (bool): 是否包含最近邻上采样。
        """
        super().__init__()
        self.use_upsample = use_upsample
        if self.use_upsample:
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.noise_injection = NoiseInjection(out_channels)
        self.adain = AdaIN(out_channels, w_dim)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x, w, noise=None):
        """
        合成块的前向传播。

        参数:
            x (torch.Tensor): 输入激活图 (batch_size, in_channels, H, W)。
            w (torch.Tensor): 中间潜在代码 (batch_size, w_dim)。
            noise (torch.Tensor, optional): 此块的可选预生成噪声。

        返回:
            torch.Tensor: 输出激活图 (batch_size, out_channels, H' 或 2H, W' 或 2W)。
        """
        if self.use_upsample:
            x = self.upsample(x)

        x = self.conv(x)
        x = self.noise_injection(x, noise=noise) # 传递可选噪声
        x = self.adain(x, w)
        x = self.activation(x)
        return x

# 使用示例:
batch_size = 4
in_channels = 128
out_channels = 64
height, width = 16, 16
w_dim = 512

# 带上采样的块
synth_block_upsample = SynthesisBlock(in_channels, out_channels, w_dim, use_upsample=True)
x_in_low_res = torch.randn(batch_size, in_channels, height, width)
w_vector = torch.randn(batch_size, w_dim)
output_high_res = synth_block_upsample(x_in_low_res, w_vector)

print(f"输入 x 形状 (低分辨率): {x_in_low_res.shape}")
print(f"输出 x 形状 (高分辨率): {output_high_res.shape}") # 应为 32x32

# 不带上采样的块（例如，第一个块或StyleGAN2的后续版本）
synth_block_no_upsample = SynthesisBlock(out_channels, out_channels, w_dim, use_upsample=False)
output_same_res = synth_block_no_upsample(output_high_res, w_vector)

print(f"输入 x 形状 (高分辨率): {output_high_res.shape}")
print(f"输出 x 形状 (相同分辨率): {output_same_res.shape}") # 应为 32x32
```

### 数据流可视化

下面的图表展示了生成器内部的高级数据流，强调了映射网络和AdaIN在合成块中的作用。

> StyleGAN中的高级数据流。映射网络将$z$转换为$w$。合成网络通过其块内的AdaIN层使用$w$来控制生成特征的风格，而注入的噪声则添加随机细节。

### 进一步考虑

- **完整合成网络：** 一个完整的StyleGAN生成器堆叠这些块，通常从一个学习到的常数张量（例如，`4x4x512`）开始，并逐步提高分辨率（例如，4x4 -> 8x8 -> ... -> 1024x1024）。每个分辨率级别一般有两个块（一个不带上采样，一个带上采样）。
- **风格混合：** 在训练期间，StyleGAN经常使用风格混合，其中两个不同的$w$向量 (vector)（$w_1, w_2$）由两个$z$向量生成。一个向量（$w_1$）控制一部分层（例如，粗糙特征，分辨率从4x4到8x8），另一个向量（$w_2$）控制其余层（例如，精细特征，16x16及以上）。这是一种正则化 (regularization)技术，避免网络假定相邻层之间存在强相关性。
- **W+ 空间：** 为了在推理 (inference)或分析期间实现更细粒度的控制，可以为每个AdaIN层单独输入一个$w$向量，而不是对所有AdaIN层使用单个$w$。这种由每层一个$w$向量组成的扩展空间常被称为$W+$空间。
- **渐进式增长与固定架构：** 虽然最初的StyleGAN使用了渐进式增长（在训练期间逐步添加层），但像StyleGAN2这样的后续版本使用固定架构，并结合了跳跃连接和残差块等技术，在目标分辨率下进行端到端训练。AdaIN和噪声注入的要点保持相似。
- **优化（StyleGAN2/3）：** 后续的StyleGAN版本加入了重要的改进，例如权重 (weight)解调（对AdaIN的改进）、无混叠卷积和不同的网络主干，以提升图像质量并减少伪影。实现这些功能复杂得多，但对于达到先进水平的性能非常重要。

本动手实践部分展示了实现StyleGAN独特能力的核心组件。构建一个完整、优化的StyleGAN需要细致地整合这些部分，以及渐进式增长（或等效的固定架构技术）等训练策略和风格混合等正则化方法。强烈建议参考官方实现和相关论文来完成完整构建。

## 参考资料

- [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948) — Tero Karras, Samuli Laine, Timo Aila (2019)
  Journal: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR); Publisher: IEEE/CVF; Pages: 4401-4410; DOI: [10.48550/arXiv.1812.04948](https://doi.org/10.48550/arXiv.1812.04948)
  介绍StyleGAN的基础论文，详细阐述了其生成器架构，包括映射网络、合成网络、AdaIN和样式混合。
- [Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) — Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, Timo Aila (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Volume: 33; Pages: 8110-8121; DOI: [10.48550/arXiv.1912.04958](https://doi.org/10.48550/arXiv.1912.04958)
  介绍StyleGAN2，通过权重解调、固定架构和跳跃连接等修改，解决了多种伪影并提高了图像质量。
- [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868) — Xun Huang, Serge Belongie (2017)
  Journal: Proceedings of the IEEE International Conference on Computer Vision (ICCV); Pages: 1501-1510; DOI: [10.48550/arXiv.1703.06868](https://doi.org/10.48550/arXiv.1703.06868)
  提出了自适应实例归一化（AdaIN）机制，这是StyleGAN用于样式控制的核心组件。
- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) — Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio (2014)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Volume: 27; Pages: 2672-2680; DOI: [10.48550/arXiv.1406.2661](https://doi.org/10.48550/arXiv.1406.2661)
  介绍了生成对抗网络（GANs）的开创性论文，为整个模型类别提供了基本框架。
