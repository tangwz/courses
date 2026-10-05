# 模型参数化 (epsilon预测 vs. x0预测)

来源：[原文](https://apxml.com/zh/courses/advanced-diffusion-architectures/chapter-4-advanced-diffusion-training/model-parameterization)

[返回章节目录](README.md) · [返回课程目录](../README.md)

训练扩散模型时的一个重要决定是，确定底层神经网络 (neural network)应该预测*什么*。给定含噪声的输入 $x_t$ 和时间步 $t$，模型需要估算与去噪过程相关的某个量。两种最常见的参数 (parameter)化方式是预测被添加的噪声 ($\epsilon$)，或预测原始干净数据 ($x_0$)。这个选择会影响损失函数 (loss function)、训练动态，并可能影响最终的采样质量。

### Epsilon预测 ($\epsilon$-prediction)

这是最初的去噪扩散概率模型（DDPM）论文中提出的标准方法。模型 $f_\theta(x_t, t)$ 被训练来预测噪声 $\epsilon$，该噪声是从标准高斯分布中采样并添加到原始数据 $x_0$ 中以生成 $x_t$，其依据是前向过程方程：


$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon
$$


这里，$\bar{\alpha}_t$ 是噪声调度方差直到时间 $t$ 的累积乘积。目标函数通常是预测噪声 $\hat{\epsilon} = f_\theta(x_t, t)$ 与用于生成 $x_t$ 的实际噪声 $\epsilon$ 之间的简化均方误差（MSE）损失：


$$
L_{\epsilon} = \mathbb{E}_{t, x_0, \epsilon} \left[ || \epsilon - f_\theta(x_t, t) ||^2 \right]
$$


期望是针对随机时间步 $t$、初始数据样本 $x_0$ 和采样噪声 $\epsilon$ 计算的。

**优点：**

1. **稳定性：** 经验上，$\epsilon$预测在各种数据集和架构上通常表现出良好的稳定性。预测一个零均值、单位方差的变量（$\epsilon$）对神经网络 (neural network)来说可以是一个表现良好的回归目标。
2. **与分数函数的直接关联：** 在特定条件下，最小化 $L_{\epsilon}$ 等同于分数匹配，这将其直接与基于分数的生成模型的数学基础联系起来。分数函数是数据对数概率密度关于数据的梯度 $\nabla_{x_t} \log p_t(x_t)$，它与噪声 $\epsilon$ 紧密相关。
3. **常用实践：** 这是在DDPM、Stable Diffusion（v2.0的v-prediction之前）等众多成功模型中最常见的参数 (parameter)化方式，使其成为一个可靠的起点。

在采样（逆向过程）期间，预测噪声 $\hat{\epsilon}$ 用于估算趋向噪声较小状态的方向，通常是通过先估算预测的 $x_0$ (记作 $\hat{x}_0$)，然后将其用于DDPM或DDIM更新步骤。

### x0预测 ($x_0$-prediction)

另一种方法是将模型（我们称之为 $g_\theta(x_t, t)$）参数 (parameter)化，使其能够从含噪声的输入 $x_t$ 和时间步 $t$ 直接预测原始干净数据 $x_0$。相应的MSE损失函数 (loss function)旨在最小化预测的 $\hat{x}_0 = g_\theta(x_t, t)$ 与真实 $x_0$ 之间的差异：


$$
L_{x_0} = \mathbb{E}_{t, x_0, \epsilon} \left[ || x_0 - g_\theta(x_t, t) ||^2 \right]
$$


**优点：**

1. **直观输出：** 模型的输出直接表示估算的干净数据，这可能更易于理解和调试。
2. **图像清晰度潜力：** 一些研究表明，直接预测 $x_0$ 可能在特定情况下生成更清晰的图像，尽管这并非普遍保证。

**参数化之间的关联：**

这两种参数化方式在数学上是相关联的。给定前向过程方程 $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$，我们可以用一种预测表示另一种预测：

- 如果模型预测噪声 $\hat{\epsilon} = f_\theta(x_t, t)$，则对应的 $x_0$ 预测为：

  
  $$
  \hat{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}} (x_t - \sqrt{1 - \bar{\alpha}_t} \hat{\epsilon})
  $$
  
- 如果模型预测干净数据 $\hat{x}_0 = g_\theta(x_t, t)$，则对应的 $\epsilon$ 预测为：

  
  $$
  \hat{\epsilon} = \frac{1}{\sqrt{1 - \bar{\alpha}_t}} (x_t - \sqrt{\bar{\alpha}_t} \hat{x}_0)
  $$
  

这些关系表明，选择一种参数化方式隐式定义了另一种。然而，训练网络以预测其中一个量与预测另一个量会直接影响梯度和表现，从而可能导致不同的训练动态和最终模型表现。

> $\epsilon$预测和$x_0$预测方法的比较。核心模型架构通常是相同的，但损失计算的目标变量不同。

### 比较与注意事项

- **数值稳定性：** 预测 $x_0$ 有时稳定性较差，特别是对于小的 $t$ (此时 $x_t$ 非常接近 $x_0$) 或大的 $t$ (此时 $x_t$ 几乎是纯噪声)。转换公式中的缩放因子 $\sqrt{\bar{\alpha}_t}$ 和 $\sqrt{1 - \bar{\alpha}_t}$ 在时间步范围的两端可能变得非常小或非常大，如果不小心处理 (例如，通过损失加权或梯度裁剪)，可能会导致数值问题。$\epsilon$预测以具有固定方差的变量为目标，通常能避免这些缩放问题。
- **损失加权：** 使用 $L_{x_0}$ 时，与标准 $L_{\epsilon}$ 损失相比，在不同时间步对损失项进行不同的加权可能是有益的。准确预测 $x_0$ 的相对重要性可能随噪声水平而变化。
- **实现：** 在两者之间切换涉及更改损失计算中的目标变量，并确保采样过程正确使用相应的预测。例如，如果您的采样器期望 $\epsilon$ 预测，但您的模型输出 $\hat{x}_0$，您必须首先使用上述公式将 $\hat{x}_0$ 转换为 $\hat{\epsilon}$，然后才能继续采样步骤。

```python
import torch
import torch.nn.functional as F

def get_sqrt_alphas_cumprod(alphas):
  """获取累积乘积的辅助函数"""
  return torch.sqrt(torch.cumprod(alphas, dim=0))

def get_sqrt_one_minus_alphas_cumprod(alphas):
  """获取 sqrt(1 - alpha_bar) 的辅助函数"""
  return torch.sqrt(1.0 - torch.cumprod(alphas, dim=0))

# --- 示例设置 ---
T = 1000
betas = torch.linspace(0.0001, 0.02, T) # 示例线性调度
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

# --- 假设我们每个批次项目有以下变量 ---
# x_start: 原始干净图像 [B, C, H, W]
# noise: 采样高斯噪声 (epsilon) [B, C, H, W]
# t: 采样时间步 [B]
# model: 您的U-Net或Transformer模型

# 提取批次时间步 t 的调度值
sqrt_alpha_bar_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

# 计算含噪声图像 x_t
x_t = sqrt_alpha_bar_t * x_start + sqrt_one_minus_alpha_bar_t * noise

# 获取模型预测
# model_output 形状: [B, C, H, W]
model_output = model(x_t, t)

# --- 损失计算 ---

# 1. Epsilon预测损失
target_eps = noise
loss_eps = F.mse_loss(model_output, target_eps)
# 使用 loss_eps 进行反向传播

# 2. x0预测损失
target_x0 = x_start
loss_x0 = F.mse_loss(model_output, target_x0)
# 使用 loss_x0 进行反向传播

# --- 采样考量 (DDIM步骤示例) ---
# 如果模型预测 epsilon (model_output = predicted_eps):
predicted_x0_from_eps = (x_t - sqrt_one_minus_alpha_bar_t * model_output) / sqrt_alpha_bar_t
# 在DDIM更新公式中使用 predicted_x0_from_eps

# 如果模型预测 x0 (model_output = predicted_x0):
predicted_eps_from_x0 = (x_t - sqrt_alpha_bar_t * model_output) / sqrt_one_minus_alpha_bar_t
# 在DDIM更新公式中使用 predicted_eps_from_x0 (或修改DDIM以直接使用 predicted_x0)
```

> Python代码片段，说明了$\epsilon$预测和$x_0$预测损失的目标计算差异，以及如何在采样时从一种预测推导出另一种预测。

### 选择合适的参数 (parameter)化方式

虽然$\epsilon$预测是完善且通常更稳定的默认选项，$x_0$预测提供了一个有效的替代方案。尝试$x_0$预测可能是有益的，如果：

- 您遇到与预测噪声尺度特别相关的稳定性问题。
- 您假设直接回归到干净图像可能会简化您的特定数据或架构的学习任务。
- 您正在实施专门围绕 $x_0$预测设计的架构或技术。

了解其他参数化方式也很重要，例如下一节将讨论的$v$预测，它旨在结合$\epsilon$和$x_0$预测的优点，特别是改进了模型在不同噪声水平下输出的尺度调整方式。了解预测$\epsilon$与$x_0$的含义，有助于理解这些更高级的技术。

## 参考资料

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — Jonathan Ho, Ajay Jain, Pieter Abbeel (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2006.11239](https://doi.org/10.48550/arXiv.2006.11239)
  介绍了去噪扩散概率模型框架，并确立了标准的$\epsilon$-预测目标。
- [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600) — Yang Song, Stefano Ermon (2019)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.1907.05600](https://doi.org/10.48550/arXiv.1907.05600)
  建立了去噪分数匹配与基于似然的生成模型之间的联系，是理解$\epsilon$-预测理论基础的关键。
- [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364) — Tero Karras, Miika Aittala, Timo Aila, Samuli Laine (2022)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2206.00364](https://doi.org/10.48550/arXiv.2206.00364)
  为扩散模型提供了统一框架，比较了不同参数化（$\epsilon$，$x_0$，$v$）及其对训练稳定性和样本质量的影响。

---

[上一节](04-%E9%AB%98%E7%BA%A7%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%E5%BD%A2%E5%BC%8F%20%28v-%E9%A2%84%E6%B5%8B%2C%20L_simple%29.md) · [下一节](06-%E8%AE%AD%E7%BB%83%E7%A8%B3%E5%AE%9A%E6%80%A7%E7%9A%84%E6%96%B9%E6%B3%95%20%28%E6%A2%AF%E5%BA%A6%E8%A3%81%E5%89%AA%2C%20%E6%8C%87%E6%95%B0%E7%A7%BB%E5%8A%A8%E5%B9%B3%E5%9D%87%29.md)
