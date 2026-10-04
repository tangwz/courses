---
course: "synthetic-data-gans-diffusion"
chapter: "diffusion-models-theory-implementation"
lesson: "practice-implementing-ddpm"
sourceId: 5711
sourceUrl: "https://apxml.com/zh/courses/synthetic-data-gans-diffusion/chapter-4-diffusion-models-theory-implementation/practice-implementing-ddpm"
title: "动手实践：构建基础DDPM"
description: "去噪扩散概率模型的代码实现。"
order: 7
plots: ["plots/5711-0.json"]
sourceHash: "73e9f1b0575c911ff1c32c3c166720594116b3f0cf75a9cacb89b6cf8c27f9be"
sourceCorrections: []
---

构建一个基础去噪扩散概率模型 (DDPM)，侧重于生成图像所需的主要组成部分。我们将使用PyTorch进行演示，并使用MNIST或CIFAR-10等标准数据集，假定你已准备好数据加载器。

我们的目标是实现核心机制：前向加噪过程、用于噪声预测的U-Net模型、简化的训练损失计算以及逆向采样过程。

### 设置扩散时间表

扩散过程依赖于预设的方差时间表 $\beta_t$，其中 $t=1, \dots, T$。常见选择包括线性或余弦时间表。从 $\beta_t$，我们得出 $\alpha_t = 1 - \beta_t$ 和 $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$。这些值控制每个时间步的噪声水平，对前向和逆向过程都很重要。

让我们定义一个函数，用于预先计算在 $T$ 个时间步上的线性时间表的这些值。

```python
import torch
import torch.nn.functional as F

def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    """
    生成beta值的线性时间表。
    """
    return torch.linspace(beta_start, beta_end, timesteps)

# 示例设置
T = 200 # 扩散时间步数
betas = linear_beta_schedule(timesteps=T)

# 预先计算alpha和累积alpha
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
# 为方便索引而移位的累积乘积 (alpha_bar_{t-1})
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

# 预先计算q_sample和后验计算所需的项
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas) # 采样步骤所需

# 后验 q(x_{t-1} | x_t, x_0) 的方差项
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

# 我们常需要为给定批次的时间步t提取正确的值
def extract(a, t, x_shape):
    """
    从'a'中提取与时间步't'对应的值，并将其重塑以广播到图像维度。
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu()) # 获取与时间步t对应的值
    # 重塑为 [batch_size, 1, 1, 1] 以进行广播
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
```

可视化累积乘积 $\bar{\alpha}_t$ 会有所帮助。随着 $t$ 的增加，$\bar{\alpha}_t$ 趋近于零，表示添加的噪声更多。



![累积Alpha时间表 (alpha_bar_t)](plots/5711-0.json)



> 累积乘积 $\bar{\alpha}_t$ 随着时间步 $t$ 的增加而平稳下降，表示根据线性方差时间表 $\beta_t$ 噪声逐渐增加。

### 实现前向过程 (q\_sample)

前向过程 $q(\mathbf{x}_t | \mathbf{x}_0)$ 将高斯噪声添加到图像 $\mathbf{x}_0$ 中，以在给定时间步 $t$ 生成 $\mathbf{x}_t$。这由以下公式定义：


$$
\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}, \quad \text{其中 } \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})
$$


我们可以直接使用预计算的值来实现这一点。

```python
def q_sample(x_start, t, noise=None):
    """
    使用重参数化技巧，根据x_0和t采样x_t。
    x_start: 初始图像 (x_0) [批量大小, 通道数, 高度, 宽度]
    t: 时间步张量 [批量大小]
    noise: 可选的噪声张量；如果为None，则采样标准高斯噪声。
    """
    if noise is None:
        noise = torch.randn_like(x_start)

    # 提取给定时间步的 sqrt(alpha_bar_t) 和 sqrt(1 - alpha_bar_t)
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)

    # 应用公式：x_t = sqrt(alpha_bar_t)*x_0 + sqrt(1-alpha_bar_t)*noise
    noisy_image = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    return noisy_image
```

此函数使我们能够为任何时间步 $t$ 生成输入数据的加噪版本，这是训练噪声预测网络所必需的。

### 模型架构：用于噪声预测的U-Net

DDPM的核心是一个神经网络 (neural network)，通常是U-Net架构，训练用于预测在给定时间步 $t$ 添加到图像中的噪声 $\boldsymbol{\epsilon}$。模型 $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ 将加噪图像 $\mathbf{x}_t$ 和时间步 $t$ 作为输入。

U-Net架构非常适合此任务，因为它结合了下采样（编码器）、上采样（解码器）和跳跃连接。跳跃连接使解码器能够重用编码器中的高分辨率特征，有助于保留图像细节，同时处理多尺度信息。

扩散模型的一个重要调整是纳入时间步信息 $t$。这通常通过将 $t$ 转换为时间嵌入 (embedding)向量 (vector)（常使用正弦位置嵌入，类似于Transformer）并将此嵌入添加到U-Net块内的中间特征图中来实现。

```python
# U-Net定义占位符。
# 假设存在一个 `UNetModel` 类，它接受图像维度和时间嵌入维度。
# 示例签名: model = UNetModel(image_channels=1, model_channels=64, time_embed_dim=256, num_res_blocks=2)
# 模型的正向传播类似于: predicted_noise = model(noisy_image, time_steps)
# 其中 `time_steps` 在内部处理以生成嵌入。
# 实现一个完整的U-Net，存在许多标准实现。
# 有关详细信息，请参考'denoising-diffusion-pytorch'等代码库或论文。

# 使用示例（假设模型已定义并实例化）
# model = UNetModel(...)
# noisy_image = ... # 来自q_sample
# t = ... # 采样的时间步张量
# predicted_noise = model(noisy_image, t)
```

### 训练损失的实现

如前所述，我们常使用简化的训练目标：


$$
L_{simple}(\theta) = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \|^2 \right]
$$


其中 $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$。

实际操作中，期望值通过小批量数据来近似。对于批次中的每张图像 $\mathbf{x}_0$，我们采样一个随机时间步 $t \sim \text{Uniform}(1, \dots, T)$，采样噪声 $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$，使用 `q_sample` 计算 $\mathbf{x}_t$，使用U-Net预测噪声 $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$，并计算真实噪声 $\boldsymbol{\epsilon}$ 与预测噪声 $\boldsymbol{\epsilon}_\theta$ 之间的均方误差 (MSE)。

```python
def p_losses(denoise_model, x_start, t):
    """
    计算用于训练噪声预测模型的损失。
    denoise_model: U-Net模型 (epsilon_theta)。
    x_start: 初始图像 (x_0)。
    t: 批次采样的时间步。
    """
    # 1. 采样噪声 epsilon ~ N(0, I)
    noise = torch.randn_like(x_start)

    # 2. 使用q_sample计算x_t
    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)

    # 3. 使用U-Net模型预测噪声
    predicted_noise = denoise_model(x_noisy, t)

    # 4. 计算真实噪声与预测噪声之间的MSE损失
    loss = F.mse_loss(noise, predicted_noise)

    return loss
```

### 实现采样过程（逆向）

生成新图像涉及逆转扩散过程。我们从纯高斯噪声 $\mathbf{x}_T \sim \mathcal{N}(0, \mathbf{I})$ 开始，并使用训练好的模型 $\boldsymbol{\epsilon}_\theta$ 迭代地对其进行去噪，以从 $\mathbf{x}_t$ 估计 $\mathbf{x}_{t-1}$。基础DDPM采样步骤如下：


$$
\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \right) + \sigma_t \mathbf{z}
$$


其中 $\mathbf{z} \sim \mathcal{N}(0, \mathbf{I})$ 对于 $t > 1$，且 $\mathbf{z} = 0$ 对于 $t=1$。方差 $\sigma_t^2$ 通常设置为 $\beta_t$ 或 $\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \beta_t$。使用 $\tilde{\beta}_t$ 对应于真实后验 $q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0)$ 的方差。

让我们实现逆向过程的一步。

```python
@torch.no_grad() # 重要：采样期间禁用梯度
def p_sample(model, x, t, t_index):
    """
    执行从x_t到x_{t-1}的一次采样步骤（去噪）。
    model: 训练好的U-Net模型。
    x: 当前加噪图像 (x_t)。
    t: 当前时间步（标量张量，例如 torch.tensor([t])）。
    t_index: 对应于时间步t的索引（用于访问预计算值）。
    """
    # 获取模型对噪声的预测 (epsilon_theta(x_t, t))
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # DDPM论文中的公式11：根据模型输出计算均值
    model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)

    if t_index == 0:
        # 如果t=1（索引0），不添加噪声（z=0）
        return model_mean
    else:
        # 计算后验方差 sigma_t^2
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # 添加噪声：sigma_t * z
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def p_sample_loop(model, shape, device):
    """
    从T到0的完整采样循环。
    model: 训练好的U-Net。
    shape: 要生成图像的形状（例如，[批量大小, 通道数, 高度, 宽度]）。
    device: 计算设备（例如，'cuda'）。
    """
    b = shape[0] # 批量大小
    # 从T处的随机噪声 N(0, I) 开始
    img = torch.randn(shape, device=device)
    imgs = [] # 如果需要，存储中间图像

    # 从T-1向后迭代到0
    for i in reversed(range(0, T)):
        timestep = torch.full((b,), i, device=device, dtype=torch.long)
        img = p_sample(model, img, timestep, i)
        # 可选：追加中间结果
        # if i % 50 == 0: # 每50步追加一次
        #     imgs.append(img.cpu())

    return img # 返回最终生成的图像 x_0
```

下图说明了前向 (q) 和逆向 (p) 过程之间的关系。

> 前向过程通过固定步骤 $q$ 将数据 $x_0$ 转换为噪声 $x_T$。逆向过程学习如何逆转此过程，从噪声 $x_T$ 开始，并在每一步 $p_\theta$ 使用模型 $\epsilon_\theta$ 逐步生成数据 $x_0$。

### 训练循环概述

训练过程结合了这些组成部分：

1. 初始化U-Net模型 $\boldsymbol{\epsilon}_\theta$ 和一个优化器（例如Adam）。
2. 加载数据集（例如MNIST）。
3. 循环所需数量的训练轮次：
   - 遍历数据批次 $\mathbf{x}_0$。
   - 对于每个批次：
     - 为批次中的每张图像采样一个随机时间步 $t$: $t \sim \text{Uniform}({1, \dots, T})$。
     - 使用 `p_losses` 函数计算损失: $L = \text{p\_losses}(\boldsymbol{\epsilon}_\theta, \mathbf{x}_0, t)$。
     - 执行反向传播 (backpropagation): `loss.backward()`。
     - 更新模型参数 (parameter): `optimizer.step()`。
     - 梯度清零: `optimizer.zero_grad()`。
   - 定期保存模型检查点并使用 `p_sample_loop` 生成示例图像以监控进展。

```python
# 简化的训练循环草图
# 假设数据加载器、模型、优化器已初始化

# model.train() # 将模型设置为训练模式
# for epoch in range(num_epochs):
#     for step, batch in enumerate(dataloader):
#         optimizer.zero_grad()
#
#         batch = batch.to(device) # 假设批次包含图像 x_0
#         batch_size = batch.shape[0]
#
#         # 为批次采样时间步t
#         t = torch.randint(0, T, (batch_size,), device=device).long()
#
#         # 计算损失
#         loss = p_losses(model, batch, t)
#
#         # 反向传播和更新
#         loss.backward()
#         optimizer.step()
#
#         # 日志记录、检查点、采样...
#         if step % log_interval == 0:
#             print(f"Epoch {epoch} Step {step}: Loss = {loss.item()}")
#         if step % sample_interval == 0:
#             # 生成样本
#             model.eval() # 将模型设置为评估模式进行采样
#             samples = p_sample_loop(model, shape=[num_samples, channels, height, width], device=device)
#             # 保存或显示样本
#             model.train() # 设置回训练模式
```

### 运行实现

有了这些组成部分（扩散时间表设置、`q_sample`、U-Net、`p_losses`、`p_sample`、`p_sample_loop` 和训练循环），你就拥有了一个完整但基础的DDPM实现。训练这些模型需要大量的计算资源和时间，特别是对于更大的图像和更深的网络。从较小的数据集（如MNIST）和适中的时间步数（$T \approx 200-1000$）开始，以便在扩大规模前验证你的实现。

经过充分训练后，`p_sample_loop` 函数应该生成与训练数据分布相似的图像。质量将很大程度上取决于模型容量、数据集复杂度、时间步数 $T$ 和训练时长。

这个动手实现为理解DDPM提供了依据。在此基础上，你可以查看本课程其他部分讨论的改进，例如使用DDIM进行更快的采样、使用分类器或无分类器引导的条件生成，以及应用更复杂的U-Net架构。

## 参考资料

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — Jonathan Ho, Ajay Jain, Pieter Abbeel (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2006.11239](https://doi.org/10.48550/arXiv.2006.11239)
  引入去噪扩散概率模型（DDPM）框架的原始论文，详细说明其理论基础、正向/反向过程和训练目标。
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) — Olaf Ronneberger, Philipp Fischer, Thomas Brox (2015)
  Journal: Medical Image Computing and Computer-Assisted Intervention (MICCAI); DOI: [10.48550/arXiv.1505.04597](https://doi.org/10.48550/arXiv.1505.04597)
  介绍了U-Net架构，因其编解码器结构和跳跃连接，常用于DDPM中预测噪声等图像到图像的任务。
- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) — Jiaming Song, Chenlin Meng, Stefano Ermon (2020)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2010.02502](https://doi.org/10.48550/arXiv.2010.02502)
  引入了一种非马尔可夫扩散过程，在DDPM框架基础上实现更快、更高质量的扩散模型采样。
