# StyleGAN 组件的动手实现

来源：[原文](https://apxml.com/zh/courses/generative-adversarial-networks-gans/chapter-2-advanced-gan-architectures/implementing-stylegan-practical)

[返回章节目录](README.md) · [返回课程目录](../README.md)

实现 StyleGAN 生成器的主要构成部分，例如其映射网络和基于风格的生成器，使用 PyTorch。实践指导将协助您开发这些组件。在代码层面理解这些组件，可以加深对架构思想的理解，并为您使用或修改高级生成模型做好准备。

我们假定您已扎实掌握 PyTorch、卷积层和神经网络 (neural network)的基本知识。本节我们侧重于 StyleGAN 架构的特点。

### 映射网络

映射网络的主要作用是将初始潜在编码 $z$（通常从标准正态分布 $\mathcal{N}(0, I)$ 中采样）转换为中间潜在空间 $W$。这个中间空间 $W$ 通常关联性更小，从而允许更直接的风格控制。映射网络通常实现为多层感知机（MLP）。

我们来构建一个简化的映射网络。它将由若干个带有 LeakyReLU 激活的全连接层构成。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MappingNetwork(nn.Module):
    def __init__(self, z_dim, w_dim, num_layers=8):
        """
        初始化映射网络。

        参数：
            z_dim (int): 输入潜在编码 z 的维度。
            w_dim (int): 输出中间潜在编码 w 的维度。
            num_layers (int): 映射网络中的线性层数量。
        """
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.num_layers = num_layers

        layers = []
        # 输入层归一化（可选但常用）
        layers.append(nn.BatchNorm1d(z_dim)) # 或 StyleGAN 论文中的 PixelNorm

        in_features = z_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_features, w_dim))
            layers.append(nn.LeakyReLU(0.2))
            in_features = w_dim # 后续层具有 w_dim 特征

        self.network = nn.Sequential(*layers)

    def forward(self, z):
        """
        映射网络的前向传播。

        参数：
            z (torch.Tensor): 输入潜在编码 (批大小, z_dim)。

        返回：
            torch.Tensor: 输出中间潜在编码 w (批大小, w_dim)。
        """
        # 如果需要，对 z 进行归一化（StyleGAN 中常用 PixelNorm）
        # 简单归一化示例：
        # z = z / torch.sqrt(torch.mean(z**2, dim=1, keepdim=True) + 1e-8)
        w = self.network(z)
        return w

# 使用示例：
z_dim = 512
w_dim = 512
mapping_net = MappingNetwork(z_dim, w_dim)
# 生成一批随机潜在编码
z_input = torch.randn(16, z_dim) # 批大小 16
# 获取中间潜在编码
w_output = mapping_net(z_input)

print(f"Input z shape: {z_input.shape}")
print(f"Output w shape: {w_output.shape}")
```

在此实现中：

- 我们以一个可选的 $z$ 归一化 (normalization)层开始。StyleGAN 通常使用 PixelNorm，但此处为简化起见展示了批归一化。
- 一系列线性层将输入 $z$ 的维度转换为 $w$ 的目的维度。
- 层间使用 LeakyReLU 激活函数 (activation function)。
- 最终输出 `w` 代表中间潜在空间 $W$ 中的向量 (vector)。这个 $w$ 将通过 AdaIN 用于合成网络中的风格调控。

### 自适应实例归一化 (normalization) (AdaIN)

自适应实例归一化是 StyleGAN 在每个分辨率级别将风格信息（来自 $w$）注入合成网络的机制。回顾其公式：


$$
\text{AdaIN}(x, y) = y_s \left( \frac{x - \mu(x)}{\sigma(x)} \right) + y_b
$$


这里，$x$ 是卷积层输出的激活图，$\mu(x)$ 和 $\sigma(x)$ 是按*每个通道、每个样本*计算的 $x$ 的均值和标准差（实例归一化）。缩放因子 $y_s$ 和偏置 (bias) $y_b$ 通过学习的仿射变换（通常是线性层）从中间潜在编码 $w$ 得到。

我们来构建 AdaIN 操作。

```python
class AdaIN(nn.Module):
    def __init__(self, num_channels, w_dim):
        """
        初始化 AdaIN 层。

        参数：
            num_channels (int): 输入特征图 x 中的通道数量。
            w_dim (int): 中间潜在编码 w 的维度。
        """
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(num_channels, affine=False) # affine=False 因为我们应用自己的缩放/偏置
        # 学习的仿射变换，用于将 w 映射到风格缩放因子和偏置
        self.style_scale_transform = nn.Linear(w_dim, num_channels)
        self.style_bias_transform = nn.Linear(w_dim, num_channels)

    def forward(self, x, w):
        """
        AdaIN 的前向传播。

        参数：
            x (torch.Tensor): 输入特征图 (批大小, 通道数, 高度, 宽度)。
            w (torch.Tensor): 中间潜在编码 (批大小, w_dim)。

        返回：
            torch.Tensor: 由风格 w 调制的特征图 (批大小, 通道数, 高度, 宽度)。
        """
        # 按通道/样本归一化输入特征图
        normalized_x = self.instance_norm(x)

        # 从 w 计算风格缩放因子和偏置
        # w 的形状: (批大小, w_dim)
        style_scale = self.style_scale_transform(w) # 形状: (批大小, 通道数)
        style_bias = self.style_bias_transform(w)   # 形状: (批大小, 通道数)

        # 重塑缩放因子和偏置，使其与特征图维度匹配以进行广播
        # 目标形状: (批大小, 通道数, 1, 1)
        style_scale = style_scale.unsqueeze(-1).unsqueeze(-1)
        style_bias = style_bias.unsqueeze(-1).unsqueeze(-1)

        # 应用学习的缩放因子和偏置
        transformed_x = style_scale * normalized_x + style_bias
        return transformed_x

# 使用示例：
num_channels = 64
w_dim = 512
height, width = 32, 32
batch_size = 16

adain_layer = AdaIN(num_channels, w_dim)
# 示例特征图和中间潜在编码
feature_map = torch.randn(batch_size, num_channels, height, width)
w_code = torch.randn(batch_size, w_dim) # 通常来自映射网络

# 应用 AdaIN
stylized_feature_map = adain_layer(feature_map, w_code)

print(f"Input feature map shape: {feature_map.shape}")
print(f"Input w shape: {w_code.shape}")
print(f"Output stylized feature map shape: {stylized_feature_map.shape}")
```

关于此实现的要点：

- 带有 `affine=False` 的 `nn.InstanceNorm2d` 执行归一化 $(x - \mu(x)) / \sigma(x)$。
- 两个独立的 `nn.Linear` 层学习将全局风格向量 (vector) $w$ 映射到合成网络中该层特有的每个通道的缩放因子 ($y_s$) 和偏置 ($y_b$) 值。
- 缩放因子和偏置被重塑为 `(批大小, 通道数, 1, 1)`，以便在逐元素乘法和加法期间正确地进行广播。

### 噪声注入

StyleGAN 在合成网络的不同层中加入了明确的噪声输入。这种噪声使得生成器能够模拟随机细节（如头发位置、雀斑），这些细节不易通过全局风格向量 (vector) $w$ 直接控制。噪声通常是高斯噪声，通过学习的每通道权重 (weight)进行缩放，并直接添加到特征图中。

```python
class AddNoise(nn.Module):
    def __init__(self, num_channels):
        """
        初始化噪声注入层。

        参数：
            num_channels (int): 添加噪声的特征图中的通道数量。
        """
        super().__init__()
        # 噪声的可学习缩放因子，每个通道一个
        # 初始化为零，因此在训练开始时噪声没有作用
        self.noise_weight = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x):
        """
        将缩放后的噪声添加到输入特征图。

        参数：
            x (torch.Tensor): 输入特征图 (批大小, 通道数, 高度, 宽度)。

        返回：
            torch.Tensor: 添加了噪声的特征图。
        """
        batch_size, _, height, width = x.shape
        # 在正确设备上生成噪声，匹配输入张量类型
        noise = torch.randn(batch_size, 1, height, width, device=x.device, dtype=x.dtype)
        # 通过学习的权重缩放噪声并添加到特征图
        noisy_x = x + self.noise_weight * noise
        return noisy_x

# 使用示例：
noise_layer = AddNoise(num_channels=64)
# 使用前面示例中的特征图
output_with_noise = noise_layer(feature_map) # 可以在 AdaIN/激活之前或之后应用

print(f"Shape after adding noise: {output_with_noise.shape}")
```

这个简单的模块创建与输入 `x` 具有相同空间分辨率的噪声，使用可学习权重 (`noise_weight`) 对其进行缩放，然后将其添加。

### 简化合成网络块

现在，我们把这些组件组合成 StyleGAN 合成网络的一个示例性块。一个常见的块可能包含：

1. 上采样（用于增加分辨率，第一个块除外）。
2. 卷积层。
3. 噪声注入。
4. 激活函数 (activation function)（例如 LeakyReLU）。
5. AdaIN（使用 $w$）。
6. 另一个卷积层。
7. 噪声注入。
8. 激活函数。
9. AdaIN（使用 $w$）。

下面是一个简化的块结构：

```python
class SynthesisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, w_dim, kernel_size=3, upsample=True):
        """
        初始化一个简化的 StyleGAN 合成块。

        参数：
            in_channels (int): 输入通道数。
            out_channels (int): 输出通道数。
            w_dim (int): 中间潜在编码 w 的维度。
            kernel_size (int): 卷积核大小。
            upsample (bool): 是否在块的开头执行上采样。
        """
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if upsample else None

        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.noise1 = AddNoise(out_channels)
        self.adain1 = AdaIN(out_channels, w_dim)
        self.activation1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.noise2 = AddNoise(out_channels)
        self.adain2 = AdaIN(out_channels, w_dim)
        self.activation2 = nn.LeakyReLU(0.2)

    def forward(self, x, w):
        """
        合成块的前向传播。

        参数：
            x (torch.Tensor): 输入特征图。
            w (torch.Tensor): 中间潜在编码 w。

        返回：
            torch.Tensor: 块的输出特征图。
        """
        if self.upsample:
            x = self.upsample(x)

        # 第一个卷积序列
        x = self.conv1(x)
        x = self.noise1(x)
        x = self.activation1(x)
        x = self.adain1(x, w)

        # 第二个卷积序列
        x = self.conv2(x)
        x = self.noise2(x)
        x = self.activation2(x)
        x = self.adain2(x, w)

        return x

# 使用示例：
# 假设我们有来自前一个块的输出或初始常数输入
# 第一个块以一个学习到的常数输入开始（例如，4x4 分辨率）
initial_input = torch.randn(batch_size, 512, 4, 4) # 示例：4x4 分辨率下 512 个通道
w_code = torch.randn(batch_size, w_dim) # 来自映射网络

# 示例：从 512 通道 (4x4) 到 256 通道 (8x8) 的块
block_4x4_to_8x8 = SynthesisBlock(in_channels=512, out_channels=256, w_dim=w_dim, upsample=True)
output_8x8 = block_4x4_to_8x8(initial_input, w_code)

print(f"Output shape of 8x8 block: {output_8x8.shape}")

# 示例：保持 256 通道（8x8 到 8x8 - 可能第一个块不进行上采样）
# block_8x8_to_8x8 = SynthesisBlock(in_channels=256, out_channels=256, w_dim=w_dim, upsample=False)
# output_8x8_v2 = block_8x8_to_8x8(output_8x8, w_code)
# print(f"Output shape of next 8x8 block: {output_8x8_v2.shape}")
```

这个块结构显示了卷积、噪声注入、激活和 AdaIN 如何交错进行。请注意，块内两个 AdaIN 层都使用相同的 $w$ 向量 (vector)，在这层分辨率上提供了统一的风格调控。

### 数据流

下图显示了单个合成块内的数据流，着重说明了中间潜在编码 $w$ 如何通过 AdaIN 和可能的噪声缩放影响生成过程（尽管我们的 `AddNoise` 为简化起见使用了独立于 $w$ 的可学习权重 (weight)；某些不同版本也可能根据 $w$ 缩放噪声）。

> StyleGAN 单个合成块内的简化数据流。中间潜在编码 $w$ 来自通过映射网络的初始潜在 $z$，通过 AdaIN 层调控特征图 $x$。噪声是独立添加的。

### 进一步的考量

- **权重 (weight)初始化：** 恰当的初始化（例如 He 初始化）对稳定训练很重要。
- **权重解调（StyleGAN2）：** StyleGAN2 用“权重解调”取代了 AdaIN 和实例归一化 (normalization)，以处理归一化伪影。这包括根据 $w$ 直接缩放卷积权重。实现此功能是更高级的步骤。
- **学习率均衡化：** StyleGAN 使用均衡学习率，在运行时缩放权重以归一化其动态范围，这有助于在使用 Adam 等优化器处理不同参数 (parameter)尺度时。
- **渐进式增长 / 分辨率处理：** 一个完整的实现需要一种机制来处理不断增加的分辨率，无论是通过渐进式增长（在训练期间添加层）还是通过设计网络同时输出多个分辨率。
- **常数输入：** 合成网络通常不是从 $z$ 开始，而是从一个学习到的常数张量（例如 4x4xC 张量）开始，该张量随后经过风格化并在块中处理。

本次实践练习着重于映射网络、AdaIN 和噪声注入的运作机制。通过实现这些核心部分，您能更清楚地认识 StyleGAN 如何实现对生成过程的精细控制。构建和训练一个完整的 StyleGAN 模型需要细致地整合这些组件，并应用高级训练稳定技术，这些将在后续章节中介绍。

## 参考资料

- [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948) — Tero Karras, Samuli Laine, Timo Aila (2019)
  Journal: CVPR; Pages: 4401-4410; DOI: [10.48550/arXiv.1812.04948](https://doi.org/10.48550/arXiv.1812.04948)
  介绍StyleGAN架构的开创性论文，涵盖了本节讨论的映射网络、自适应实例归一化（AdaIN）和噪声注入机制。
- [Generative Deep Learning: Teaching Machines to Paint, Write, Compose, and Play](https://books.google.com/books/about/Generative_Deep_Learning.html?id=z1F7EAAAQBAJ) — David Foster (2023)
  Publisher: O'Reilly Media
  一本实用书籍，为实现包括StyleGAN在内的各种生成模型提供实际指导，并清晰解释其组件和实现细节。

---

[上一节](06-%E9%9D%9E%E9%85%8D%E5%AF%B9%E5%9B%BE%E5%83%8F%E5%88%B0%E5%9B%BE%E5%83%8F%E8%BD%AC%E6%8D%A2%20%28CycleGAN%29.md) · [下一节](../03-GAN%E8%AE%AD%E7%BB%83%E7%9A%84%E5%8A%A8%E6%80%81%E4%B8%8E%E7%A8%B3%E5%AE%9A%E6%80%A7/01-%E4%B8%8D%E6%94%B6%E6%95%9B%E7%9A%84%E9%9A%BE%E9%A2%98.md)
