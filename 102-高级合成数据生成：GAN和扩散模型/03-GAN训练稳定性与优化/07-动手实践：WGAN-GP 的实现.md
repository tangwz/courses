# 动手实践：WGAN-GP 的实现

来源：[原文](https://apxml.com/zh/courses/synthetic-data-gans-diffusion/chapter-3-gan-training-stability-optimization/practice-implementing-wgan-gp)

[返回章节目录](README.md) · [返回课程目录](../README.md)

Wasserstein GAN (WGAN) 和梯度惩罚 (GP) 技术旨在解决原始 WGAN 权重 (weight)裁剪的局限性，从而稳定训练。WGAN-GP 的实际实现如下。这种方法被普遍认为是 GAN 训练稳定性和样本质量上的显著改进。

本实践指南假定您对在 PyTorch 或 TensorFlow 中实现基础 GAN 已感到熟悉。我们将侧重于 WGAN-GP 所需的具体修改。

### WGAN-GP 的核心组成部分

实现 WGAN-GP 主要涉及对判别器网络、损失函数 (loss function)和训练循环的调整。

1. **判别器架构：** WGAN-GP 中的判别器*不*包含最终的 Sigmoid 激活函数 (activation function)。其作用是输出一个标量分数（根据 Wasserstein 距离近似值代表的“真实度”），而非概率。输出层应为线性层。
2. **损失函数：** 我们将标准 GAN 的对数损失替换为源自 Wasserstein 距离估计和梯度惩罚的损失。
3. **梯度惩罚：** 这是其显著特点。不同于权重 (weight)裁剪，我们向判别器的损失中添加了一个惩罚项，促使判别器对其输入的梯度范数接近 1。这更有效地强制执行 Lipschitz 约束。
4. **训练循环：** 通常，判别器在每个训练迭代中的更新频率高于生成器（例如，每次生成器更新对应 5 次判别器更新）。

### 实现判别器损失

判别器旨在最大化其对真实样本和生成样本的得分差异，同时包含梯度惩罚。判别器 ($D$) 的损失函数 (loss function)为：


$$
L_D = \mathbb{E}_{\tilde{x} \sim P_g}[D(\tilde{x})] - \mathbb{E}_{x \sim P_r}[D(x)] + \lambda \mathbb{E}_{\hat{x} \sim P_{\hat{x}}}[(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2]
$$


其中：

- $P_g$ 是生成器分布（伪造样本 $\tilde{x}$）。
- $P_r$ 是真实数据分布（真实样本 $x$）。
- $P_{\hat{x}}$ 是插值样本 $\hat{x}$ 的分布。
- $\lambda$ 是梯度惩罚系数（通常设为 10）。

让我们分解实现过程，特别是梯度惩罚项。

### 实现梯度惩罚

计算梯度惩罚包含以下几个步骤：

1. **采样插值点：** 对于批次中的每个真实样本 $x$ 和生成样本 $\tilde{x}$，创建一个插值样本 $\hat{x}$。

   
   $$
   \hat{x} = \epsilon x + (1 - \epsilon) \tilde{x}
   $$
   

   这里，$\epsilon$ 是从 $U[0, 1]$ 均匀采样的随机数。这需要对批次进行逐元素操作。
2. **计算插值点的判别器输出：** 将这些插值样本 $\hat{x}$ 输入判别器网络，以获取其得分 $D(\hat{x})$。
3. **计算梯度：** 计算判别器输出 $D(\hat{x})$ 相对于插值输入 $\hat{x}$ 的梯度。这需要使用您的深度学习 (deep learning)框架的自动微分功能（例如 PyTorch 中的 `torch.autograd.grad` 或 TensorFlow 中的 `tf.GradientTape`）。很重要的一点是，要确保为输入计算梯度（PyTorch 中需要 `create_graph=True`，因为梯度惩罚本身就是损失图的一部分）。
4. **计算梯度范数：** 针对每个插值样本，计算这些梯度的 L2 范数（欧几里得范数）。
5. **计算惩罚：** 将每个样本的惩罚计算为 $( \| \nabla_{\hat{x}} D(\hat{x}) \|_2 - 1 )^2$。
6. **平均并缩放：** 对批次中的惩罚进行平均，并乘以系数 $\lambda$。

以下是梯度惩罚函数的一个 PyTorch 实现片段：

```python
import torch
import torch.autograd as autograd

def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    """计算 WGAN GP 的梯度惩罚损失"""
    batch_size = real_samples.size(0)
    # 用于真实样本和伪造样本之间插值的随机权重项
    alpha = torch.rand(batch_size, 1, 1, 1, device=device) # 假设为 4D 张量 (B, C, H, W)
    # 扩展 alpha 以匹配图像维度
    alpha = alpha.expand_as(real_samples)

    # 获取真实样本和伪造样本之间的随机插值
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)

    # 获取插值点的判别器得分
    d_interpolates = critic(interpolates)

    # 使用全一张量作为梯度计算的目标
    fake = torch.ones(batch_size, 1, device=device, requires_grad=False) # 使用匹配判别器输出的尺寸

    # 获取相对于插值点的梯度
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake, # 梯度输出必须匹配 d_interpolates 的形状
        create_graph=True, # 为二阶导数创建计算图 (GP 损失的一部分)
        retain_graph=True, # 保留计算图以进行后续计算 (判别器损失)
        only_inputs=True,
    )[0]

    # 重塑梯度以便于计算每个样本的范数
    gradients = gradients.view(gradients.size(0), -1)
    # 计算 L2 范数和惩罚
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# --- 训练循环内部 ---
# 假设已定义 'critic', 'real_imgs', 'fake_imgs'
# LAMBDA_GP = 10 # 梯度惩罚系数

# gradient_penalty = compute_gradient_penalty(critic, real_imgs.data, fake_imgs.data, device)
# critic_loss = torch.mean(critic_fake) - torch.mean(critic_real) + LAMBDA_GP * gradient_penalty
# critic_loss.backward()
# optimizer_D.step()
```

> **注意：** 确保用于 `alpha`、`fake` 和梯度计算的形状与您的具体数据和判别器输出维度匹配。`interpolates` 上的 `requires_grad_(True)` 以及 `autograd.grad` 中的 `create_graph=True`、`retain_graph=True` 对于正确计算惩罚是必不可少的。

### 实现生成器损失

生成器 ($G$) 旨在生成判别器给予高分的样本（即让判别器认为它们是真实的）。其损失函数 (loss function)更简单：


$$
L_G = - \mathbb{E}_{\tilde{x} \sim P_g}[D(\tilde{x})]
$$


在实践中，这意味着生成一批伪造样本，将它们输入判别器，并最小化所得分数的负平均值。

```python
# --- 训练循环内部，生成器更新阶段 ---
# 生成伪造图像
# z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
# gen_imgs = generator(z)

# 计算生成器损失
# fake_scores = critic(gen_imgs)
# generator_loss = -torch.mean(fake_scores)

# generator_loss.backward()
# optimizer_G.step()
```

### 训练过程

典型的 WGAN-GP 训练循环涉及判别器和生成器之间的交替更新。常见做法是每次生成器更新执行多次判别器更新。

> WGAN-GP 的训练循环结构，侧重于每次生成器更新进行多次判别器更新。

**注意事项：**

- **优化器：** 常用 Adam，通常使用特定的超参数 (parameter) (hyperparameter)，如 $\beta_1=0.0$ 或 $\beta_1=0.5$ 以及 $\beta_2=0.9$。标准的 Adam 设置（$\beta_1=0.9, \beta_2=0.999$）也可以，但可能需要更多调整。为生成器和判别器使用独立的优化器实例。
- **学习率：** 两者相似的学习率（例如 1e-4 或 2e-4）通常是一个不错的起始点，不同于 TTUR 明确使用不同的学习率。
- **判别器更新次数 ($n_{critic}$)：** 像 5 这样的值很常见，但这可以调整。它确保判别器为生成器提供可靠的梯度。
- **梯度惩罚系数 ($\lambda$)：** 通常设为 10，如果训练不稳定或梯度消失/爆炸，可以调整。
- **批归一化 (normalization)：** 在 WGAN-GP 的判别器中通常避免使用批归一化，因为它可能在批次中的样本之间引入依赖关系，干扰梯度惩罚的计算。如果需要归一化，层归一化或实例归一化可能是替代方案。对于生成器，批归一化通常仍在使用。

通过实现这些组成部分，特别是梯度惩罚的计算和调整后的损失函数 (loss function)，您可以运用 WGAN-GP 训练更稳定的 GAN，能够生成更高质量的合成数据，与标准 GAN 公式或原始带有权重 (weight)裁剪的 WGAN 相比。请记住在训练期间监控判别器损失、生成器损失和梯度惩罚的量级，以诊断潜在问题。

## 参考资料

- [Improved Training of Wasserstein GANs](https://proceedings.neurips.cc/paper_files/paper/2017/file/892c3b1c6d66df268f77395e87b642e7-Paper.pdf) — Ishaan Gulrajani, Faruk Ahmed, Martín Arjovsky, Vincent Dumoulin, Aaron Courville (2017)
  Journal: Advances in Neural Information Processing Systems; Publisher: Advances in Neural Information Processing Systems; Volume: 30; Pages: 5767-5777; DOI: [10.5555/3157382.3157605](https://doi.org/10.5555/3157382.3157605)
  介绍了梯度惩罚技术，显著提高了WGAN训练的稳定性和样本质量。
- [Wasserstein GAN](https://proceedings.mlr.press/v70/arjovsky17a/arjovsky17a.pdf) — Martín Arjovsky, Soumith Chintala, Léon Bottou (2017)
  Journal: Proceedings of the 34th International Conference on Machine Learning; Publisher: PMLR (Proceedings of Machine Learning Research); Volume: 70; Pages: 214-223; DOI: [10.5555/3305890.3306071](https://doi.org/10.5555/3305890.3306071)
  提出了Wasserstein GAN的基础论文，通过使用Wasserstein距离解决了原始GAN训练的不稳定性。
- [Automatic differentiation package - torch.autograd](https://pytorch.org/docs/stable/autograd.html) — PyTorch Contributors (2025)
  Publisher: PyTorch Foundation
  PyTorch自动微分的官方文档，对于正确实现梯度惩罚计算至关重要。
- [Generative Deep Learning: Teaching Machines to Paint, Write, Compose, and Play](https://www.oreilly.com/library/view/generative-deep-learning/9781492041931/) — David Foster (2019)
  Publisher: O'Reilly Media
  一本实践指南，其中包含各种生成模型的实现，为WGAN-GP提供了背景和示例。

---

[上一节](06-GAN%E7%9A%84%E8%B6%85%E5%8F%82%E6%95%B0%E8%B0%83%E6%95%B4%E7%AD%96%E7%95%A5.md) · [下一节](../04-%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B%EF%BC%9A%E7%90%86%E8%AE%BA%E4%B8%8E%E8%BF%9B%E9%98%B6%E5%AE%9E%E7%8E%B0/01-%E6%95%B0%E5%AD%A6%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86%EF%BC%9A%E9%9A%8F%E6%9C%BA%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B.md)
