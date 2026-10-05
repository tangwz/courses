# WGAN-GP 的实现：实践

来源：[原文](https://apxml.com/zh/courses/generative-adversarial-networks-gans/chapter-3-gan-training-stabilization/implementing-wgan-gp-practice)

[返回章节目录](README.md) · [返回课程目录](../README.md)

梯度惩罚（WGAN-GP）是一种强制执行 Lipschitz 约束的方法，对于 Wasserstein GAN 的稳定训练很重要。它直接解决了实现稳健 GAN 性能时出现的挑战，例如与权重 (weight)裁剪相关的问题。该方法会直接惩罚判别器（critic），如果其梯度范数在真实数据点和生成数据点之间的路径上明显偏离 1。实现 WGAN-GP 是实现 GAN 稳定训练的常见做法，尤其适用于复杂的架构和数据集。提供了指导和代码示例，帮助您将其应用到项目中。

## 理解梯度惩罚的计算

回想一下，Wasserstein 距离要求判别器是 1-Lipschitz 的。WGAN-GP 通过向判别器的损失函数 (loss function)中添加惩罚项来强制执行此要求。此惩罚项旨在阻止判别器输出相对于其输入的梯度范数远离 1。

该惩罚项专门在均匀采样的点上计算，这些点沿着连接真实数据分布 ($p_{data}$) 和生成器分布 ($p_g$) 中的点对的直线。令 $x$ 为真实样本，$\tilde{x}$ 为生成样本。插值样本 $\hat{x}$ 定义为：


$$
\hat{x} = \epsilon x + (1 - \epsilon) \tilde{x}
$$


其中 $\epsilon$ 是从 $U[0, 1]$ 中均匀采样的随机数。

添加到判别器损失中的梯度惩罚项为：


$$
L_{GP} = \lambda \mathbb{E}_{\hat{x} \sim p_{\hat{x}}}[(\| \nabla_{\hat{x}} D(\hat{x}) \|_2 - 1)^2]
$$


其中：

- $D(\hat{x})$ 是判别器对插值样本 $\hat{x}$ 的输出。
- $\nabla_{\hat{x}} D(\hat{x})$ 是判别器输出对插值输入 $\hat{x}$ 的梯度。
- $\| \cdot \|_2$ 表示 L2 范数（欧几里得范数）。
- $\lambda$ 是惩罚系数，一个控制惩罚强度的超参数 (parameter) (hyperparameter)（通常设为 10）。
- $p_{\hat{x}}$ 表示这些插值样本的分布。

期望 $\mathbb{E}_{\hat{x} \sim p_{\hat{x}}}$ 通过使用当前批次的真实数据和虚假数据生成的插值样本批次来近似得到。

### 实现梯度惩罚函数

我们来列出计算此惩罚在典型的深度学习 (deep learning)框架（如 PyTorch 或 TensorFlow）中的步骤。假设您有一批真实图像 (`real_samples`) 和一批由生成器生成的虚假图像 (`fake_samples`)，两者形状相同（例如，`[batch_size, channels, height, width]`）。

1. **生成 Epsilon：** 创建一个形状为 `[batch_size, 1, 1, 1]`（或与您的图像张量广播的适当形状）的张量 `epsilon`，其中包含在 0 到 1 之间均匀采样的随机数。
2. **创建插值样本：** 使用广播计算 $\hat{x}$：`interpolated_samples = epsilon * real_samples + (1 - epsilon) * fake_samples`。确保这些样本需要梯度以进行后续步骤。
3. **计算插值样本的判别器分数：** 将 `interpolated_samples` 输入判别器网络：`interpolated_scores = critic(interpolated_samples)`。
4. **计算梯度：** 计算 `interpolated_scores` 相对于 `interpolated_samples` 的梯度。大多数框架都提供执行此操作的函数（例如 PyTorch 中的 `torch.autograd.grad`，TensorFlow 中的 `tf.GradientTape` 上下文 (context)）。如果这些梯度将是用于优化判别器的图的一部分（确实如此），则必须设置 `create_graph=True`（PyTorch）或确保梯度计算在磁带（tape）的上下文（TensorFlow）中进行。您需要的是梯度本身，而不仅仅是它们对最终损失的贡献。
5. **计算梯度范数：** 如有必要，重塑梯度使其形状为 `[batch_size, -1]`，并计算每个样本梯度在所有特征上的 L2 范数。在计算范数时，在平方根下添加一个小数（例如 `1e-8`）以保证数值稳定性：`gradient_norms = sqrt(sum(gradients**2, axis=1) + 1e-8)`。
6. **计算惩罚：** 应用 WGAN-GP 公式：`gradient_penalty = lambda * mean((gradient_norms - 1)**2)`。

下图展示了 $\hat{x}$ 的采样过程：

> 插值样本 ($\hat{x}$，绿色) 是沿着连接真实样本 ($x$，蓝色) 和生成样本 ($\tilde{x}$，红色) 的线条选取的。判别器的梯度范数在这些插值点处受到惩罚。

下面是使用 PyTorch 语法的实现片段：

```python
import torch
import torch.autograd as autograd

def compute_gradient_penalty(critic, real_samples, fake_samples, lambda_gp, device):
    """计算 WGAN-GP 的梯度惩罚损失"""
    # 真实样本和虚假样本之间插值的随机权重项
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    # 获取真实样本和虚假样本之间的随机插值
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)

    critic_interpolates = critic(interpolates)

    # 使用 autograd 计算梯度
    gradients = autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(critic_interpolates.size(), device=device), # 确保梯度流向所有输出
        create_graph=True, # 在判别器更新期间为二阶导数创建图
        retain_graph=True, # 为生成器更新保留图
        only_inputs=True,
    )[0] # 获取相对于输入的梯度

    # 重塑梯度并计算范数
    gradients = gradients.view(gradients.size(0), -1)
    # 添加小的 epsilon 以提高数值稳定性
    gradient_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    # 计算惩罚
    gradient_penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()
    return gradient_penalty

# 在训练步骤中的示例用法:
# lambda_gp = 10
# 假设 critic, real_batch, fake_batch, device 已定义
# gp = compute_gradient_penalty(critic, real_batch, fake_batch, lambda_gp, device)
# critic_loss = ... + gp # 加到主判别器损失中
```

## 修改训练循环

集成 WGAN-GP 涉及两个主要修改，与原始 GAN 或带权重 (weight)裁剪的 WGAN 相比：

1. **判别器损失计算：** 判别器的目标函数变为：

   
   $$
   L_{Critic} = \mathbb{E}_{\tilde{x} \sim p_g}[D(\tilde{x})] - \mathbb{E}_{x \sim p_{data}}[D(x)] + \lambda \mathbb{E}_{\hat{x} \sim p_{\hat{x}}}[(\| \nabla_{\hat{x}} D(\hat{x}) \|_2 - 1)^2]
   $$
   

   请注意，我们的目标是 *最小化* 此损失。前两项近似于负 Wasserstein 距离，最后一项是梯度惩罚。如果目标是最大化，某些实现可能会反转前两项的符号。请确保您的优化器步骤与最小化或最大化目标一致。
2. **优化器和架构：**

   - 从判别器优化器中移除权重裁剪。
   - Adam 通常是一个合适的优化器，可以调整超参数 (parameter) (hyperparameter)（例如，某些 WGAN-GP 论文中使用的 `beta1=0.0`、`beta2=0.9`，尽管标准的 `beta1=0.5`、`beta2=0.999` 也有效）。
   - 对于生成器的每次更新，通常会多次更新判别器（例如 5 次），尤其是在训练初期，以确保判别器提供有用的梯度。

典型的 WGAN-GP 训练步骤如下（伪代码）：

```python
# 超参数
lambda_gp = 10
critic_iterations = 5
learning_rate = 0.0001
beta1 = 0.0
beta2 = 0.9

# 优化器 (常使用 Adam)
optimizer_G = Adam(generator.parameters(), lr=learning_rate, betas=(beta1, beta2))
optimizer_C = Adam(critic.parameters(), lr=learning_rate, betas=(beta1, beta2))

for epoch in range(num_epochs):
    for i, real_batch in enumerate(data_loader):
        # ---------------------
        # 训练判别器
        # ---------------------
        optimizer_C.zero_grad()

        real_samples = real_batch.to(device)
        batch_size = real_samples.size(0)

        # 采样噪声并生成虚假样本
        z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        fake_samples = generator(z).detach() # 分离以避免通过 C 训练 G

        # 获取判别器分数
        real_scores = critic(real_samples)
        fake_scores = critic(fake_samples)

        # 计算梯度惩罚
        gradient_penalty = compute_gradient_penalty(
            critic, real_samples.data, fake_samples.data, lambda_gp, device
        )

        # 计算判别器损失：-(Wasserstein 损失) + 梯度惩罚
        # 我们最小化此损失，相当于最大化 (真实分数 - 虚假分数 - 梯度惩罚)
        critic_loss = -torch.mean(real_scores) + torch.mean(fake_scores) + gradient_penalty

        # 反向传播并更新判别器
        critic_loss.backward()
        optimizer_C.step()

        # 每 'critic_iterations' 步才训练生成器
        if i % critic_iterations == 0:
            # -----------------
            # 训练生成器
            # -----------------
            optimizer_G.zero_grad()

            # 生成一批新的虚假样本（带梯度）
            z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            gen_samples = generator(z)

            # 获取生成样本的判别器分数
            gen_scores = critic(gen_samples)

            # 计算生成器损失（最大化虚假样本的分数）
            # 我们最小化负分数
            generator_loss = -torch.mean(gen_scores)

            # 反向传播并更新生成器
            generator_loss.backward()
            optimizer_G.step()

        # 日志记录、模型保存等。
        # ...
```

## 实现注意事项

- **Lambda ($\lambda$) 值：** 选择 $\lambda=10$ 是经验得出的，适用于许多问题。然而，它是一个超参数 (parameter) (hyperparameter)。如果训练不稳定，或者梯度范数持续远离 1，您可以尝试调整 $\lambda$，尽管 10 是一个标准的起始值。
- **数值稳定性：** 在计算 L2 范数时，在取平方根之前添加小的 epsilon（例如 `1e-8` 或 `1e-12`）很重要，以防止梯度向量 (vector)为零时出现 `NaN` 值。
- **框架差异：** 虽然原理相同，但计算相对于中间变量梯度的具体语法在不同框架之间略有不同。确保使用正确的函数（`torch.autograd.grad`、`tf.GradientTape.gradient`) 并适当管理梯度计算上下文 (context)。注意 `create_graph=True` (PyTorch) 或嵌套 `GradientTape` (TensorFlow) 等参数，以允许在判别器优化步骤中梯度通过惩罚计算反向传播 (backpropagation)。
- **调试：** 监控判别器损失的组成部分：Wasserstein 估计值（`mean(fake_scores) - mean(real_scores)`）和梯度惩罚项。同时，监控平均梯度范数（`mean(gradient_norms)`）本身。理想情况下，在稳定训练期间，平均梯度范数应在 1 附近徘徊。如果它持续保持远高于或低于 1 的值，可能表明学习率、网络容量或 $\lambda$ 值存在问题。

通过用梯度惩罚取代权重 (weight)裁剪，您提供了一种更平滑、更可靠的方式来强制执行 Lipschitz 约束，与原始 WGAN 公式相比，通常能明显提高训练稳定性和样本质量。这使得 WGAN-GP 成为您高级 GAN 工具包中的一项有用的技术。

## 参考资料

- [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028) — Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, Aaron Courville (2017)
  Journal: Advances in Neural Information Processing Systems 30 (NIPS 2017); DOI: [10.48550/arXiv.1704.00028](https://doi.org/10.48550/arXiv.1704.00028)
  提出了梯度惩罚方法，用于在Wasserstein GAN中强制执行Lipschitz约束，从而提高训练稳定性和性能。
- [Wasserstein GAN](https://arxiv.org/abs/1701.07875) — Martin Arjovsky, Soumith Chintala, Léon Bottou (2017)
  Journal: Proceedings of the 34th International Conference on Machine Learning (ICML 2017); DOI: [10.48550/arXiv.1701.07875](https://doi.org/10.48550/arXiv.1701.07875)
  介绍了Wasserstein GAN的理论框架，解决了传统GAN的局限性，并为WGAN-GP奠定了基础。
- [Generative Deep Learning: Teaching Machines to Paint, Write, Compose, and Play](https://books.google.com/books/about/Generative_Deep_Learning.html?id=1NlHEAAAQBAJ) — David Foster (2023)
  Publisher: O'Reilly Media; Pages: 453
  一本实践指南，涵盖了各种生成模型，包括WGAN-GP，为实践者提供了代码示例和实现策略。
- [torch.autograd](https://pytorch.org/docs/stable/autograd.html) — PyTorch Core Team (2024)
  Publisher: PyTorch Foundation
  PyTorch自动微分引擎的官方文档，对于理解和实现梯度计算（如梯度惩罚）至关重要。

---

[上一节](08-%E7%9B%B8%E5%AF%B9%E8%AE%BA%E7%94%9F%E6%88%90%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9C.md) · [下一节](../04-%E6%9D%A1%E4%BB%B6%E5%BC%8F%E4%B8%8E%E5%8F%AF%E6%8E%A7%E7%94%9F%E6%88%90/01-%E6%9D%A1%E4%BB%B6%E5%BC%8FGAN%EF%BC%88cGAN%EF%BC%89%E4%BB%8B%E7%BB%8D.md)
