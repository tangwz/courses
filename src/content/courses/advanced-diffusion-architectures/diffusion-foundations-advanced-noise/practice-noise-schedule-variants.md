---
course: "advanced-diffusion-architectures"
chapter: "diffusion-foundations-advanced-noise"
lesson: "practice-noise-schedule-variants"
sourceId: 5379
sourceUrl: "https://apxml.com/zh/courses/advanced-diffusion-architectures/chapter-1-diffusion-foundations-advanced-noise/practice-noise-schedule-variants"
title: "动手实践：实现噪声调度变体"
description: "不同噪声调度的代码实现及其在简单数据集上的效果分析。"
order: 7
plots: ["plots/5379-0.json", "plots/5379-1.json"]
sourceHash: "757031272d55d3a92c68608142b5d3225e091d3b94636058b77e87552a5e65f4"
sourceCorrections: []
---

实践实现将使用 PyTorch 实现多种噪声调度变体。理解如何生成和操控这些调度，对于控制扩散过程并根据特定数据或生成需求进行调整来说，是基本要求。

我们将实现常见的线性调度和余弦调度，同时提供一个简单的自定义例子，并可视化它们的行为。这项实践练习将加深您对随着时间 $t$ 变化，方差 $\beta_t$ 如何变化，以及这如何影响由 $\bar{\alpha}_t$ 表示的累积信号保留的理解。

### 设置和辅助函数

首先，我们导入 PyTorch 并定义一些适用于所有调度类型的辅助函数。我们需要函数来计算 $\alpha_t = 1 - \beta_t$ 的值和累积乘积 $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$。

```python
import torch
import torch.nn.functional as F
import math

# 扩散时间步数量
T = 1000

def get_alphas_and_cumprod(betas):
    """根据 betas 计算 alpha_t 和 alpha_t_cumprod。"""
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    return alphas, alphas_cumprod

# 示例用法占位符（将被具体调度 betas 替换）
# betas_example = torch.linspace(0.0001, 0.02, T)
# alphas_example, alphas_cumprod_example = get_alphas_and_cumprod(betas_example)
# print(f"Example Alphas shape: {alphas_example.shape}")
# print(f"Example Alphas Cumprod shape: {alphas_cumprod_example.shape}")
```

这些辅助函数接收一个 $\beta_t$ 值的张量，并返回对应的 $\alpha_t$ 和 $\bar{\alpha}_t$ 张量，它们是正向和反向扩散过程中的必要组成部分。

### 线性调度

线性调度也许是最直接的。在 $T$ 个时间步内，方差 $\beta_t$ 从起始值 $\beta_{start}$ 线性增加到终止值 $\beta_{end}$。

```python
def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    """生成 beta_t 的线性调度。"""
    return torch.linspace(beta_start, beta_end, timesteps)

# 生成线性调度
betas_linear = linear_beta_schedule(T)
alphas_linear, alphas_cumprod_linear = get_alphas_and_cumprod(betas_linear)

# print(f"Linear Betas (first 5): {betas_linear[:5]}")
# print(f"Linear Alphas Cumprod (first 5): {alphas_cumprod_linear[:5]}")
# print(f"Linear Alphas Cumprod (last 5): {alphas_cumprod_linear[-5:]}")
```

此调度在开始阶段添加噪声相对较慢，并在结束阶段线性加速。

### 余弦调度

余弦调度由 Nichol 和 Dhariwal (2021) 提出，旨在避免信号在过程早期衰减过快，这有益于图像质量。它基于余弦函数定义 $\bar{\alpha}_t$，然后从中推导出 $\beta_t$。

```python
def cosine_beta_schedule(timesteps, s=0.008):
    """
    生成 beta_t 的余弦调度，基于 alpha_t_cumprod。
    提案于：https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((t / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0] # 归一化
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999) # 裁剪以避免数值问题

# 生成余弦调度
betas_cosine = cosine_beta_schedule(T)
alphas_cosine, alphas_cumprod_cosine = get_alphas_and_cumprod(betas_cosine)

# print(f"Cosine Betas (first 5): {betas_cosine[:5]}")
# print(f"Cosine Alphas Cumprod (first 5): {alphas_cumprod_cosine[:5]}")
# print(f"Cosine Alphas Cumprod (last 5): {alphas_cumprod_cosine[-5:]}")
```

注意小的偏移量 `s` 用于防止 $\beta_t$ 在 $t=0$ 附近过小。裁剪操作确保了数值稳定性。观察到，与线性调度相比，$\bar{\alpha}_t$ 值在初始阶段下降得更慢。

### 自定义调度：二次函数例子

为说明其灵活性，我们来实践一个二次调度，其中 $\beta_t$ 呈二次函数式增长。这只是一个例子；您可以根据 sigmoid 函数、指数函数或任何其他单调递增函数来设计调度，具体取决于所需的噪声注入特性。

```python
def quadratic_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    """生成 beta_t 的二次调度。"""
    betas_quad = torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2
    return betas_quad

# 生成二次调度
betas_quadratic = quadratic_beta_schedule(T)
alphas_quadratic, alphas_cumprod_quadratic = get_alphas_and_cumprod(betas_quadratic)

# print(f"Quadratic Betas (first 5): {betas_quadratic[:5]}")
# print(f"Quadratic Alphas Cumprod (first 5): {alphas_cumprod_quadratic[:5]}")
# print(f"Quadratic Alphas Cumprod (last 5): {alphas_cumprod_quadratic[-5:]}")
```

此调度在开始阶段添加噪声比线性调度更慢，但在结束阶段加速更快。

### 调度可视化

可视化比较这些调度有助于理解它们的特点。我们来绘制 $\beta_t$ 值（在时间步 $t$ 添加的方差）和 $\bar{\alpha}_t$ 值（在时间步 $t$ 剩余的累积信号）。



![噪声调度：方差 (beta_t) 随时间步变化](plots/5379-0.json)



> 线性、余弦和二次调度在 T=1000 个时间步上的 $\beta_t$ 值比较。



![噪声调度：累积信号 (alpha_bar_t) 随时间步变化](plots/5379-1.json)



> 线性、余弦和二次调度在 T=1000 个时间步上的 $\bar{\alpha}_t$ 值比较。

这些图表清晰地显示了差异：

- **线性**: $\beta_t$ 恒定增加，$\bar{\alpha}_t$ 稳定下降。
- **余弦**: $\beta_t$ 从非常低的数值开始，缓慢增加，到中间部分加快，导致 $\bar{\alpha}_t$ 初始衰减较慢。
- **二次**: $\beta_t$ 从低数值开始并呈二次函数式增加，导致 $\bar{\alpha}_t$ 曲线初始衰减比线性慢，但后期比余弦快。

### 将调度集成到模型中

在典型的扩散模型实现中（通常是 PyTorch `nn.Module`），您会预先计算这些与调度相关的张量并将其注册为缓冲区。这些缓冲区随后可以在训练（用于正向过程 $q(x_t|x_0)$）和采样（用于反向过程 $p_\theta(x_{t-1}|x_t)$）期间由时间步 $t$ 进行索引。

这是一个简化的示意：

```python
import torch
import torch.nn as nn

class SimpleDiffusionModel(nn.Module):
    def __init__(self, schedule_type='linear', timesteps=1000, beta_start=0.0001, beta_end=0.02):
        super().__init__()
        self.timesteps = timesteps

        if schedule_type == 'linear':
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif schedule_type == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        elif schedule_type == 'quadratic':
             betas = quadratic_beta_schedule(timesteps, beta_start, beta_end)
        else:
            raise ValueError(f"Unknown schedule: {schedule_type}")

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0) # 添加 alpha_cumprod_0 = 1

        # 注册缓冲区
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # 正向/反向过程中使用的其他派生量
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        # ... 可能还有 DDPM/DDIM 采样所需的其他量 ...

    def forward_process(self, x_0, t, noise=None):
        """应用正向扩散过程 q(x_t | x_0)。"""
        if noise is None:
            noise = torch.randn_like(x_0)

        # 提取给定批量时间步 t 的值
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t, None, None, None] # 匹配形状 (B, C, H, W)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t, None, None, None]

        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t

    # ... 用于预测的 __call__ 方法（例如，噪声预测） ...
    # ... 使用缓冲区的采样方法 (DDPM, DDIM) ...

# 示例实例化
# model = SimpleDiffusionModel(schedule_type='cosine')
# print(model.betas.shape)
# print(model.sqrt_alphas_cumprod.shape)
```

这个例子展示了所选调度如何直接填充模型结构中必要的张量。派生出的具体值（如 `sqrt_alphas_cumprod`）直接取决于所选调度函数生成的 $\beta_t$ 值。

### 后续步骤

通过实现和可视化这些调度，您已对它们如何影响扩散过程有了实践认识。请记住，调度的选择是一个设计决定，会影响训练动态和最终样本质量。我们之前讨论过的学习型调度代表了更进一步，其中模型会自行优化这些方差步骤，但理解这些固定、分析性调度提供了必要的根基。随着我们转向更复杂的架构和训练方法，噪声调度仍然是影响模型行为的基本组成部分。

## 参考资料

- [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) — Alex Nichol, Prafulla Dhariwal (2021)
  Journal: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV); DOI: [10.48550/arXiv.2102.09672](https://doi.org/10.48550/arXiv.2102.09672)
  引入余弦噪声调度作为线性调度的改进，在章节内容中明确提及。
