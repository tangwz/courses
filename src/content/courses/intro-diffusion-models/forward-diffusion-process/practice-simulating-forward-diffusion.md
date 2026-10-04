---
course: "intro-diffusion-models"
chapter: "forward-diffusion-process"
lesson: "practice-simulating-forward-diffusion"
sourceId: 5385
sourceUrl: "https://apxml.com/zh/courses/intro-diffusion-models/chapter-2-forward-diffusion-process/practice-simulating-forward-diffusion"
title: "实践：模拟正向扩散"
description: "动手使用代码在简单数据上模拟正向加噪过程。"
order: 6
plots: ["plots/5385-0.json"]
sourceHash: "993e4ee809a1feea50471ecca16441ba1387b17aae1902a5ad04a13ea2bf77f8"
sourceCorrections: []
---

增量添加噪声的数学公式 ($q(x_t | x_{t-1})$) 和能从原始数据 $x_0$ 直接跳到带噪状态 $x_t$ 的便捷闭式方程 ($q(x_t | x_0)$) 是理解正向扩散过程的基础。此过程可以使用代码模拟，呈现数据如何根据预定计划逐渐转变为噪声。

我们将使用 Python 和像 PyTorch 这样的库（或 NumPy/TensorFlow，原理是一样的）来演示这一点。

### 设置扩散参数 (parameter)

首先，我们需要前面部分中定义的核心组成部分：

1. **方差调度 ($\beta_t$)**：这决定了每一步添加多少噪声。一种常用选择是线性调度，从少量开始逐渐增加。
2. **导出项 ($\alpha_t, \bar{\alpha}_t$)**：这些直接从 $\beta_t$ 计算得出，对闭式采样方程必不可少。请记住 $\alpha_t = 1 - \beta_t$ 和 $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$。
3. **时间步数 (T)**：正向过程的总时长。

让我们在 PyTorch 中定义这些：

```python
import torch
import torch.nn.functional as F
import numpy as np

# 定义超参数
T = 1000  # 总时间步数
beta_start = 0.0001
beta_end = 0.02

# 线性方差调度
betas = torch.linspace(beta_start, beta_end, T)

# 计算 alphas
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0) # 这就是 \bar{\alpha}_t

# 辅助函数，用于轻松获取给定 t 的 \bar{\alpha}_t
# 我们在开头为 t=0 添加 1.0
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

print(f"Shape of betas: {betas.shape}")
print(f"Shape of alphas_cumprod: {alphas_cumprod.shape}")
print(f"First value of alphas_cumprod (t=1): {alphas_cumprod[0]:.4f}")
print(f"Last value of alphas_cumprod (t=T): {alphas_cumprod[-1]:.4f}")
```

请注意，当 $t$ 较小时，$\bar{\alpha}_t$ 接近 1；随着 $t$ 接近 $T$，它会趋向 0。这反映了在早期时间步，数据仅受到轻微扰动；而在后期时间步，数据几乎完全是噪声。

### 实现正向采样函数

现在，让我们实现我们推导出的闭式采样方程：


$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon
$$


其中 $\epsilon$ 是从标准正态分布 $\mathcal{N}(0, I)$ 采样的随机噪声，而 $x_0$ 是我们的初始数据点。

我们可以编写一个函数，它接受一个初始数据点 `x_start` ($x_0$)、一个时间步 `t`，并返回相应的带噪样本 `x_t`。

```python
# 根据 x_0 和 t 采样 x_t 的函数
def q_sample(x_start, t, noise=None):
    """
    使用闭式方程采样 x_t：sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise

    参数：
        x_start: 初始数据 (x_0)，任意形状的张量。
        t: 时间步索引（整数张量，从 0 开始，其中 0 对应 t=1）。
        noise: 可选的噪声张量，如果为 None 则从 N(0, I) 采样。

    返回：
        采样的带噪版本 x_t。
    """
    if noise is None:
        noise = torch.randn_like(x_start)

    # 获取给定时间步 t 的累积乘积 alpha_bar
    # alphas_cumprod[0] 是 \bar{\alpha}_1, alphas_cumprod[999] 是 \bar{\alpha}_1000
    sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod[t])
    sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1.0 - alphas_cumprod[t])

    # 应用公式
    # 如果 t 是时间步的批次，确保维度匹配以进行广播
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, *([1]*(len(x_start.shape)-1)))
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, *([1]*(len(x_start.shape)-1)))

    # 计算 x_t
    xt = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    return xt
```

此函数封装了直接从 $x_0$ 采样 $x_t$ 的数学核心。请注意，这里使用 `torch.randn_like(x_start)` 来生成与输入数据形状相同的噪声，并对 $\sqrt{\bar{\alpha}_t}$ 和 $\sqrt{1 - \bar{\alpha}_t}$ 项进行了形状调整，以确保在处理批次数据或时间步时能正确广播。

### 可视化噪声添加

让我们看看实际效果。我们将创建一个简单的 1D 信号（例如正弦波）作为 $x_0$，并观察它在不同时间步 $t$ 如何逐渐变得更嘈杂。

```python
# 创建一个简单的 1D 信号（例如正弦波）
signal_length = 100
x_axis = np.linspace(0, 4 * np.pi, signal_length)
x_start = torch.tensor(np.sin(x_axis)).float().unsqueeze(0) # 添加批次维度

# 选择要可视化的带噪时间步（alphas_cumprod 的索引范围为 0 到 999）
timesteps_to_show = [0, 100, 250, 500, 750, 999]
num_plots = len(timesteps_to_show)

# 为选定时间步生成带噪样本
noisy_samples = []
for t_val in timesteps_to_show:
    t = torch.tensor([t_val]) # 函数需要一个张量
    xt = q_sample(x_start, t)
    noisy_samples.append(xt.squeeze(0).numpy()) # 移除批次维度以便绘图

# 为 Plotly 图表准备数据
chart_data = []

# 原始信号 (x_0)
chart_data.append({
    "type": "scatter",
    "mode": "lines",
    "x": list(range(signal_length)),
    "y": x_start.squeeze(0).tolist(),
    "name": "x_0 (Original)",
    "line": {"color": "#4263eb", "width": 3} # 蓝色
})

# 带噪样本 (x_t)
colors = ["#12b886", "#fab005", "#f76707", "#f03e3e", "#ae3ec9"]
# 我们从第一个带噪样本开始绘图（在数学上对应于 t=1）
for i, t_val in enumerate(timesteps_to_show):
    # 注意：alphas_cumprod[t_val] 对应于数学步骤 t = t_val + 1
    label = f"x_{t_val + 1}"
    chart_data.append({
        "type": "scatter",
        "mode": "lines",
        "x": list(range(signal_length)),
        "y": list(noisy_samples[i]),
        "name": label,
        "line": {"color": colors[i % len(colors)], "width": 1.5},
        "opacity": 0.8
    })
```

现在，让我们使用图表可视化这些样本。



![Forward Diffusion Process on a 1D Signal](plots/5385-0.json)



> 在选定时间步 ($t$) 使用正向扩散过程对 1D 正弦波信号进行渐进式加噪。随着 $t$ 增加，原始信号结构逐渐被高斯噪声模糊。

如图所示：

- 在 $t=0$ 时，我们有原始信号 $x_0$。
- 对于较小的 $t$（例如 $x_1$），信号仅受到第一次噪声增量的轻微扰动。整体形状仍然非常清晰。
- 随着 $t$ 增加（$x_{251}, x_{501}, x_{751}$），噪声的贡献 ($\sqrt{1 - \bar{\alpha}_t}$) 变大，而原始信号的贡献 ($\sqrt{\bar{\alpha}_t}$) 缩小。信号变得越来越失真。
- 到了最后的时间步（$x_{1000}$），样本看起来像随机高斯噪声。$\sqrt{\bar{\alpha}_T}$ 项非常小，这意味着样本几乎完全由缩放后的噪声 $\sqrt{1 - \bar{\alpha}_T} \epsilon$ 决定。原始结构 $x_0$ 实际上已经丢失。

这个模拟演示了正向过程：通过根据固定计划逐渐添加噪声，信息会确定性地降级。这个过程不是学习得来的；它是一个预定义的机制。扩散模型的*巧妙之处*在于学习如何逆转这种降级，从噪声 $x_T$ 开始恢复原始数据 $x_0$ 的估计。理解这个正向模拟是理解这种逆转如何实现的第一步。

## 参考资料

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — Jonathan Ho, Ajay Jain, and Pieter Abbeel (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2006.11239](https://doi.org/10.48550/arXiv.2006.11239)
  介绍去噪扩散概率模型（DDPM）的基础论文，详细阐述了前向扩散过程的数学公式和实现，包括方差调度和闭合形式采样。
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) — PyTorch Developers (2024)
  Publisher: PyTorch Foundation
  PyTorch 官方文档，对于理解张量操作和函数（如 `torch.linspace`、`torch.cumprod` 和 `torch.randn_like`）至关重要，这些是实现前向扩散过程的基础。
