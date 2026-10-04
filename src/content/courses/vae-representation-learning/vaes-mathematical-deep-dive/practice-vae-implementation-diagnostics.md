---
course: "vae-representation-learning"
chapter: "vaes-mathematical-deep-dive"
lesson: "practice-vae-implementation-diagnostics"
sourceId: 6305
sourceUrl: "https://apxml.com/zh/courses/vae-representation-learning/chapter-2-vaes-mathematical-deep-dive/practice-vae-implementation-diagnostics"
title: "VAE 实现与诊断：动手操作"
description: "从零开始实现 VAE，并进行诊断以了解其行为。"
order: 8
plots: ["plots/6305-0.json"]
sourceHash: "9f61958d3d1187db8cff0f2b9bb2f30354be49a8a95763dd01ee225d86c224b6"
sourceCorrections: []
---

实际实现变分自编码器（VAE）。VAEs 的构建基于数学基础，例如证据下界（ELBO）、重参数 (parameter)化技巧以及 KL 散度的作用。本指导将引导完成一个 VAE 的从零开始实现、训练，并进行必要的诊断，以了解其行为和常见问题模式。目标是构建模型，并将其具体输出和训练动态与其基本概念联系起来。

在此实践中，我们将使用 Python 和 PyTorch 等常用深度学习 (deep learning)框架。这些思路只需少量语法修改即可迁移到 TensorFlow 等其他框架。我们将侧重于手写数字的 MNIST 数据集，这是一个经典选择，可以让我们专注于 VAE 的运作方式，而无需纠结于复杂的数据预处理或过于庞大的网络结构。

### 环境搭建

开始之前，请确保您已安装 PyTorch，以及用于 MNIST 数据集的 `torchvision` 和用于可视化的 `matplotlib`。

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
```

我们将预先定义一些超参数 (parameter) (hyperparameter)：

```python
# Hyperparameters
latent_dims = 20  # 潜在空间的维度
image_size = 28 * 28 # MNIST 图像是 28x28
batch_size = 128
learning_rate = 1e-3
num_epochs = 30 # 根据需要调整
```

并载入 MNIST 数据集：

```python
# MNIST Dataset
transform = transforms.Compose([
    transforms.ToTensor(), # 转换为 [0, 1] 范围和 C, H, W 格式
    # 对于伯努利输出的 VAE，我们不进行归一化，
    # 因为像素值被视为概率。
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
```

### 构建 VAE 组件

VAE 主要由两个神经网络 (neural network)组成：编码器和解码器。

> 变分自编码器中的高层数据流。

#### 1. 编码器：$q_\phi(z|x)$

编码器由 $\phi$ 参数 (parameter)化，接收输入数据点 $x$（在本例中为图像），并输出近似后验分布 $q_\phi(z|x)$ 的参数。对于高斯后验，这些参数是均值 $\mu$ 和方差的对数 $\log \sigma^2$ (对数方差)。使用对数方差可以提升数值稳定性，并确保方差 $\sigma^2$ 始终为正。

```python
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(self.fc1(x))
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h) # log_var 用于数值稳定性
        return mu, log_var
```

这里，`input_dim` 是 `image_size`，`latent_dim` 是 `latent_dims`。`hidden_dim` 可以自行选择，例如 400。

#### 2. 重参数化技巧

为了让梯度能够通过采样过程（从 $z \sim q_\phi(z|x)$ 采样）反向传播 (backpropagation)，我们使用重参数化技巧。如果 $z \sim \mathcal{N}(\mu, \sigma^2)$，我们可以写成 $z = \mu + \sigma \cdot \epsilon$，其中 $\epsilon \sim \mathcal{N}(0, I)$。随机性现在已转移到 $\epsilon$ 上。

```python
def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var) # std = exp(log(std)) = exp(0.5 * log(var))
    eps = torch.randn_like(std)    # 从 N(0, I) 采样 epsilon
    return mu + eps * std
```

#### 3. 解码器：$p_\theta(x|z)$

解码器由 $\theta$ 参数化，接收潜在向量 (vector) $z$ 并重构数据点 $x$。对于 MNIST，由于像素值通常在 0 到 1 之间，我们可以将每个像素的输出建模为伯努利分布的参数。解码器的最终层将输出 logits，然后通过 sigmoid 函数得到概率。

```python
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        # 我们将在损失函数或生成时应用 sigmoid，
        # 因为 nn.BCEWithLogitsLoss 更稳定。

    def forward(self, z):
        h = self.relu(self.fc1(z))
        x_reconstructed_logits = self.fc2(h)
        return x_reconstructed_logits
```

这里，`output_dim` 是 `image_size`。

#### 4. VAE 模型

现在，我们将编码器和解码器组合成一个 VAE 模型。

```python
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim) # output_dim 就是 input_dim

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = reparameterize(mu, log_var)
        x_reconstructed_logits = self.decoder(z)
        return x_reconstructed_logits, mu, log_var

# 初始化模型
model = VAE(input_dim=image_size, hidden_dim=400, latent_dim=latent_dims)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

### 定义损失函数 (loss function) (ELBO)

VAE 通过最大化 ELBO 进行训练，这等价于最小化负 ELBO。如您所记得，ELBO 包含两项：

1. **重构损失**：$\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]$。对于伯努利输出（例如 MNIST 像素），这是输入 $x$ 和重构 $\hat{x}$ 之间的二元交叉熵 (BCE)。
2. **KL 散度**：$D_{KL}(q_\phi(z|x) || p(z))$。此项规范潜在空间，促使近似后验 $q_\phi(z|x)$ 接近先验 $p(z)$，先验通常是标准正态分布 $\mathcal{N}(0, I)$。

要最小化的损失函数是：


$$
\mathcal{L}(x, \hat{x}, \mu, \log \sigma^2) = \text{重构损失} + \text{KL散度}
$$


对于高斯 $q_\phi(z|x) = \mathcal{N}(z | \mu, \text{diag}(\sigma^2))$ 和 $p(z) = \mathcal{N}(z | 0, I)$，KL 散度有一个方便的解析形式：


$$
D_{KL}(q_\phi(z|x) || p(z)) = -\frac{1}{2} \sum_{j=1}^{D} (1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2)
$$


其中 $D$ 是潜在空间的维度。

```python
def loss_function(x_reconstructed_logits, x, mu, log_var):
    # 重构损失（使用 BCEWithLogitsLoss 以提升数值稳定性）
    # 它期望 logits 作为输入，原始像素作为目标。
    # reduction='sum' 以对所有像素和批次元素求和
    recon_loss = nn.functional.binary_cross_entropy_with_logits(x_reconstructed_logits, x, reduction='sum')

    # KL 散度
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # 我们对潜在维度求和，然后对批次求平均
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return recon_loss + kld # 这是要最小化的负 ELBO。
```

注意：`binary_cross_entropy_with_logits` 在 `reduction='mean'`（默认）时对像素求平均，在 `reduction='sum'` 时求和。求和后，通常会除以 `batch_size` 以保持批次间损失大小一致。此处，KLD 的 `torch.sum` 对批次中每个项的潜在维度求和。我们将两种损失相加，得到批次的总损失。

### 训练 VAE

训练循环涉及获取一批数据，将其输入 VAE，计算损失，并使用 Adam 等优化器更新模型参数 (parameter)。

```python
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 用于存储损失组件以进行绘图的列表
train_losses = []
recon_losses = []
kld_losses = []

model.train() # 将模型设置为训练模式
for epoch in range(num_epochs):
    epoch_loss = 0
    epoch_recon_loss = 0
    epoch_kld_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(-1, image_size).to(device) # 扁平化图像

        # 前向传播
        x_reconstructed_logits, mu, log_var = model(data)

        # 计算损失
        loss = loss_function(x_reconstructed_logits, data, mu, log_var)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 累加损失（按数据集大小归一化，用于 ELBO 平均估算）
        epoch_loss += loss.item()
        # 对于单独的组件，确保它们在同一尺度上
        # 如果 loss_function 返回求和值，recon_loss_item 将来自：
        # nn.functional.binary_cross_entropy_with_logits(x_reconstructed_logits, data, reduction='sum').item()
        # kld_item 将来自：
        # (-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())).item()

        # 简化：如果您之前将损失除以 len(data)，则存储批次平均值
        # For this example, let's calculate them separately for clarity if needed for plots
        # 或者只将每批次项的总损失相除。
        # 上面的 `loss.item()` 是批次的总和。
        # 要获得每个样本的损失：
        # batch_recon_loss = nn.functional.binary_cross_entropy_with_logits(x_reconstructed_logits, data, reduction='sum')
        # batch_kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        # epoch_recon_loss += batch_recon_loss.item()
        # epoch_kld_loss += batch_kld.item()

    # 计算本轮的平均损失
    avg_epoch_loss = epoch_loss / len(train_loader.dataset)
    # avg_epoch_recon_loss = epoch_recon_loss / len(train_loader.dataset)
    # avg_epoch_kld_loss = epoch_kld_loss / len(train_loader.dataset)

    train_losses.append(avg_epoch_loss)
    # recon_losses.append(avg_epoch_recon_loss) # 如果您单独追踪它们
    # kld_losses.append(avg_epoch_kld_loss)     # 如果您单独追踪它们

    print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_epoch_loss:.4f}')
    # 如果追踪组件：
    # print(f'Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_epoch_loss:.4f}, Avg Recon Loss: {avg_epoch_recon_loss:.4f}, Avg KLD: {avg_epoch_kld_loss:.4f}')
```

**关于损失缩放的重要说明：** 重构损失和 KL 散度的具体值会因您是对像素、潜在维度和批次项求和还是求平均而有显著差异。保持一致性非常重要。ELBO 是一个期望值，因此对批次（和数据集）求平均可得到蒙特卡洛估计。对于 KL 散度，对潜在维度求和然后对批次求平均是标准做法。对于重构，对像素求和然后对批次求平均也很常见。`loss_function` 的代码将两者求和，因此 `loss.item()` 是批次的总和。除以 `len(train_loader.dataset)` 可以对其进行归一化 (normalization)。

### 诊断与解读

训练完成后（或训练过程中），有必要诊断 VAE 的表现。

#### 1. 监控损失组件

绘制总损失（负 ELBO）、重构损失和 KL 散度随训练轮次的曲线，可以提供训练过程的直观信息。



![VAE 训练进展](plots/6305-0.json)



> VAE 训练期间损失组件的示例（值仅作说明）。总损失和重构损失通常应下降。KL 散度在编码器开始学习使用潜在空间时可能初期增加，随后稳定或缓慢下降。

- **重构损失下降**：表明模型在重构输入方面表现更好了。
- **KL 散度行为**：最初，如果编码器输出 $\mu \approx 0, \sigma \approx 1$（匹配先验但未编码信息），KL 项可能很小。随着训练进行，如果编码器学习使用潜在空间，$q(z|x)$ 将偏离 $p(z)$，从而增加 KL 项。理想情况下，它会找到一个平衡点。

#### 2. 可视化重构结果

将原始图像与其重构结果进行对比，是评估 VAE 表现的直接方式。

```python
model.eval() # 将模型设置为评估模式
with torch.no_grad():
    # 获取一批测试数据
    data, _ = next(iter(test_loader))
    data = data.view(-1, image_size).to(device) # 扁平化图像

    x_reconstructed_logits, _, _ = model(data)
    # 应用 sigmoid 获得用于可视化的概率
    x_reconstructed = torch.sigmoid(x_reconstructed_logits)

    # 显示原始图像和重构图像
    n_images = 10
    plt.figure(figsize=(20, 4))
    for i in range(n_images):
        # 原始
        ax = plt.subplot(2, n_images, i + 1)
        plt.imshow(data[i].cpu().numpy().reshape(28, 28), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == 0: ax.set_title("Original")

        # 重构
        ax = plt.subplot(2, n_images, i + 1 + n_images)
        plt.imshow(x_reconstructed[i].cpu().numpy().reshape(28, 28), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == 0: ax.set_title("Reconstructed")
    plt.show()
```

查看清晰度、重要特征的保留情况以及整体保真度。

#### 3. 生成新样本

生成模型的一个标志是它们生成新数据的能力。我们可以通过从先验 $p(z)$（例如 $\mathcal{N}(0, I)$）中采样 $z$，并将其通过解码器来完成此操作。

```python
model.eval()
with torch.no_grad():
    # 从先验 N(0,I) 中采样潜在向量
    num_samples = 10
    z_samples = torch.randn(num_samples, latent_dims).to(device)

    # 解码它们以生成图像
    generated_logits = model.decoder(z_samples)
    generated_images = torch.sigmoid(generated_logits)

    plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        ax = plt.subplot(1, num_samples, i + 1)
        plt.imshow(generated_images[i].cpu().numpy().reshape(28, 28), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.suptitle("从先验生成的样本", fontsize=16)
    plt.show()
```

评估这些生成样本的质量和多样性。它们看起来像 MNIST 数字吗？它们多样吗？

#### 4. 常见问题与调试

- **模糊的重构/生成结果**：
  这是 VAE 的一个常见特点，特别是使用简单解码器和高斯输出假设时。模型可能对多个合理的高频细节进行平均，导致结果平滑。更强的解码器（例如，图像使用转置卷积，或第 3 章讨论的自回归 (autoregressive)解码器）可以有所帮助。重构损失的选择（例如 $L_2$ 与 $L_1$ 或 BCE）也会影响清晰度。
- **后验坍缩 (KL 消失)**：
  当 KL 散度项 $D_{KL}(q_\phi(z|x) || p(z))$ 在训练期间趋近于零时，就会出现这种情况。这意味着近似后验 $q_\phi(z|x)$ 变得非常接近先验 $p(z)$，不论输入 $x$ 是什么。因此，潜在变量 $z$ 几乎不包含关于 $x$ 的信息，解码器基本上学会了忽略 $z$ 并生成一个平均输出。

  - **检测方法**：监控 KL 散度值。如果它持续很低（例如 < 0.1，尽管确切阈值取决于缩放和 `latent_dims`）或迅速降至接近零并保持不变，您可能遇到了后验坍缩。如果解码器足够强大，能够无条件地建模数据分布，重构结果可能看起来仍然可以，但该模型将不利于表示学习或条件生成。
  - **发生原因**：优化过程可能认为满足 KL 约束比学习有意义的表示更容易，特别是当解码器表达能力不足或训练早期 KL 项的权重 (weight)相对于重构项过高时。
  - **缓解方法**：KL 退火（在训练期间逐渐将 KL 项的权重从 0 增加到 1）、使用表达能力更强的解码器，或修改目标（例如“自由位”等后续章节主题）可以缓解此问题。
- **潜在空间中的“空洞”**：
  先验 $p(z)$ 鼓励潜在编码靠近原点。然而，如果学习到的表示的流形 $q(z|x)$ 是稀疏的或存在间隙，那么从 $p(z)$ 中采样的潜在空间中的所有点不一定都能解码成真实的样本。在已知数据点的潜在编码之间进行插值可以帮助查看学习到的流形的平滑性。

#### 5. 查看潜在空间 (进阶)

如果您的 `latent_dims` 为 2，您可以直接看到解码器如何将潜在空间的区域映射到图像。对于更高维度，t-SNE 或 UMAP 等降维技术可以将潜在编码 $z$（通过编码测试数据获得）投影到 2D，然后根据它们的真实标签（例如 MNIST 的数字类别）进行绘图和着色。训练良好的 VAE 通常会在潜在空间中展示相似数据点的聚类。

```python
# 潜在空间可视化示例（如果 latent_dims 合适或使用 t-SNE）
# 此代码段假设您已将 test_data 编码为 test_mu 和 test_labels
# from sklearn.manifold import TSNE
# tsne = TSNE(n_components=2, random_state=0)
# z_tsne = tsne.fit_transform(test_mu.cpu().numpy()) # 假设 test_mu 包含来自编码器的均值

# plt.figure(figsize=(10, 8))
# plt.scatter(z_tsne[:, 0], z_tsne[:, 1], c=test_labels.cpu().numpy(), cmap='tab10', s=5)
# plt.colorbar()
# plt.title('潜在空间的 t-SNE')
# plt.xlabel('t-SNE 维度 1')
# plt.ylabel('t-SNE 维度 2')
# plt.show()
```

此次实践应该能让您对 VAE 是如何构建的以及训练期间需要关注什么有一个实际体会。这里讨论的诊断方法是基本的。当您在后续章节中学习更高级的 VAE 架构和应用时，这些基本检查仍将是您的首要分析手段。请记住，每个组件——编码器、解码器、重参数 (parameter)化和损失项——都直接对应您已学习的数学框架，理解它们的联系对于掌握 VAE 很重要。

## 参考资料

- [Auto-Encoding Variational Bayes](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE3-1kwG3Ojb9lJt944BzimoZxpDlRYqbgTDH-FOlJXUp6ifF7wJuFQ0WRLthmA_tT7AtaTGOcj7uWzArHc59Go8nkla3lJ-ddcpkVq9rksFiOusTmUkSJF1G-nw6GABGt4KCAIm5sk) — Diederik P. Kingma, Max Welling (2013)
  Journal: ICLR; DOI: [10.48550/arXiv.1312.6114](https://doi.org/10.48550/arXiv.1312.6114)
  提出变分自编码器（VAE）模型、证据下界（ELBO）目标和重参数化技巧的原始论文。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 20
  一本权威教材，其中有一章专门介绍深度生成模型，包括对变分自编码器理论和实践方面的全面阐述。
- [Tutorial on Variational Autoencoders](https://doi.org/10.48550/arXiv.1606.05908) — Carl Doersch (2016)
  Journal: arXiv preprint; DOI: [10.48550/arXiv.1606.05908](https://doi.org/10.48550/arXiv.1606.05908)
  一篇被广泛引用的教程，提供了对 VAE 的易懂介绍，详细阐述了其数学基础和实践方面，包括实现中遇到的常见问题。
- [Learn the Basics - PyTorch Tutorials](https://pytorch.org/tutorials/beginner/basics/intro_tutorial.html) — Suraj Subramanian, Seth Juarez, Cassie Breviu, Dmitry Soshnikov, Ari Bornstein (2021)
  Publisher: PyTorch
  学习 PyTorch 的官方入门教程，涵盖了构建、训练和评估神经网络的基本组件和工作流程，与本节中的实现直接相关。
