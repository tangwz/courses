---
course: "applied-autoencoders-feature-extraction"
chapter: "variational-autoencoders-structured-latent-spaces"
lesson: "hands-on-building-vae-inspecting-latent-space"
sourceId: 6488
sourceUrl: "https://apxml.com/zh/courses/applied-autoencoders-feature-extraction/chapter-6-variational-autoencoders-structured-latent-spaces/hands-on-building-vae-inspecting-latent-space"
title: "实作：构建变分自编码器并检验其潜在空间"
description: "关于构建变分自编码器并可视化其学习到的潜在空间的实践环节。"
order: 9
plots: ["plots/6488-0.json"]
sourceHash: "157b45d2846ccada8d9a78b58f4dd8aaf30d13614371d84defea2455a1970785"
sourceCorrections: []
---

我们将着手进行变分自编码器（VAE）的实际实现。VAE 通过学习概率潜在空间，使其区别于标准自编码器，并能够生成新的数据样本。我们将使用 PyTorch 构建一个 VAE，在手写数字的 MNIST 数据集上训练它，然后检查其潜在空间以查看它是如何组织数据的。最后，我们将从这个潜在空间采样以生成新的数字图像。

本次练习将巩固你对以下内容的理解：

- 构建编码器以输出均值 ($z_{mean}$) 和对数方差 ($z_{log\_var}$)。
- 实现重参数 (parameter)化技巧进行采样。
- 定义 VAE 特有的损失函数 (loss function)（重构 + KL 散度）。
- 可视化学习到的潜在空间。
- 通过从潜在空间采样来生成新数据。

让我们开始吧！

### 1. 设置环境和导入库

首先，请确保你已安装 PyTorch。如果尚未安装，通常可以通过 pip 进行安装：
`pip install torch torchvision matplotlib numpy`

现在，让我们导入实现 VAE 所需的库。

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
```

我们导入 `numpy` 用于数值运算，`matplotlib.pyplot` 用于绘图，`scipy.stats.norm` 用于从正态分布生成网格（对可视化生成数字的流形很有用），以及 `torch` 和 `torchvision` 中的各种组件。

### 2. 加载和预处理 MNIST 数据集

MNIST 数据集是机器学习 (machine learning)中的经典数据集，包含 70,000 张手写数字（0-9）的灰度图像，每张为 28x28 像素。它非常适合 VAE，因为学习到的二维潜在空间可以方便地进行可视化。

```python
# 定义一个转换以标准化数据并展平图像
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1)) # 展平图像
])

# 加载数据集
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

# 定义 DataLoader
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义图像尺寸和潜在维度
image_size = 28
original_dim = image_size * image_size # 28 * 28 = 784
latent_dim = 2
```

这里，我们使用 `torchvision.datasets` 加载 MNIST 数据。我们定义了一个 `transform` 来将图像转换为 PyTorch 张量并将其展平为 784 维向量 (vector)。然后我们创建 `DataLoader` 实例，以便在训练期间高效地进行批处理。我们还定义了 `latent_dim = 2`，这意味着我们的 VAE 会将每张图像压缩成一个二维潜在向量。选择这种低维度是为了方便绘制和检查潜在空间的结构。

### 3. 构建 VAE 组件

一个 VAE 有三个主要部分：编码器、采样层（实现重参数 (parameter)化技巧）和解码器。在 PyTorch 中，我们通常将这些定义为 `nn.Module` 类。

#### 编码器网络

编码器接收输入图像，并将其映射到潜在空间中高斯分布的参数（均值和对数方差）。

```python
# 编码器网络
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        z_mean = self.fc_mean(h)
        z_log_var = self.fc_log_var(h)
        return z_mean, z_log_var

encoder = Encoder(original_dim, 256, latent_dim)
print(encoder)
```

我们的编码器是一个简单的前馈神经网络 (neural network)。它接收展平的 784 维图像，通过一个包含 256 个单元和 ReLU 激活的密集层，然后有两个输出层：一个用于 `z_mean`，另一个用于 `z_log_var`。这两个输出层都有 `latent_dim` 个单元，对应于我们选择的潜在空间的维度。

#### 采样层（重参数化技巧）

为了使用反向传播 (backpropagation)训练 VAE，我们需要一种方法在不中断梯度流的情况下从分布 $q(z|x)$（由 $z_{mean}$ 和 $z_{log\_var}$ 定义）中采样。重参数化技巧实现了这一点：$z = z_{mean} + \exp(0.5 \cdot z_{log\_var}) \cdot \epsilon$，其中 $\epsilon$ 从标准正态分布 $\mathcal{N}(0, I)$ 中采样。

```python
# 采样函数（重参数化技巧）
def sampling(z_mean, z_log_var):
    std = torch.exp(0.5 * z_log_var)
    epsilon = torch.randn_like(std) # 从标准正态分布采样
    return z_mean + std * epsilon
```

`sampling` 函数接收 `z_mean` 和 `z_log_var` 张量作为输入。它根据 `z_log_var` 计算标准差，从具有相同形状的标准正态分布生成 `epsilon`，然后应用重参数化公式。

#### 解码器网络

解码器接收从潜在空间采样的点 $z$，并将其映射回原始数据空间，尝试重构输入图像。

```python
# 解码器网络
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        reconstruction = torch.sigmoid(self.fc_out(h)) # Sigmoid 激活函数用于像素值 [0,1]
        return reconstruction

decoder = Decoder(latent_dim, 256, original_dim)
print(decoder)
```

解码器在某种程度上反映了编码器的结构。它接收一个二维潜在向量 (vector)，通过一个包含 256 个单元（ReLU 激活）的密集层，然后输出一个 784 维向量。我们在输出层使用 Sigmoid 激活函数 (activation function)，因为我们希望重构的像素值归一化 (normalization)到 0 和 1 之间。

#### 完整的 VAE 模型

现在，让我们连接这些组件以形成完整的 VAE 模型。

```python
# VAE 模型
class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = sampling(z_mean, z_log_var)
        reconstruction = self.decoder(z)
        return reconstruction, z_mean, z_log_var

vae = VAE(encoder, decoder)
print(vae)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae.to(device)
```

`VAE` 类结合了 `encoder` 和 `decoder`。它的 `forward` 方法接收输入 `x`，通过编码器获取 `z_mean` 和 `z_log_var`，使用 `sampling` 函数采样 `z`，最后通过解码器获取 `reconstruction`。如果 GPU 可用，我们还会将设备设置为 `cuda` 以进行 GPU 加速。

### 4. 定义 VAE 损失函数 (loss function)

VAE 损失函数包含两部分：

1. **重构损失**：这衡量了解码器重构输入图像的效果。对于 MNIST，由于像素值已标准化并可视为概率，因此二元交叉熵 (BCE) 是一个常见选择。
2. **KL 散度损失**：这作为一个正则化 (regularization)项，促使编码器学习到的分布 $q(z|x)$ 接近先验分布 $p(z)$（通常是标准正态分布 $\mathcal{N}(0, I)$）。公式如下：
   $D_{KL}(q(z|x) || p(z)) = -0.5 \cdot \sum_{j=1}^{潜在维度} (1 + z_{log\_var_j} - z_{mean_j}^2 - \exp(z_{log\_var_j}))$

我们将定义一个函数来计算这个组合损失。

```python
# VAE 损失函数
def vae_loss_function(reconstruction, x, z_mean, z_log_var):
    # 重构损失（二元交叉熵）
    # 我们使用 F.binary_cross_entropy，reduction='sum' 以匹配 TensorFlow 的缩放。
    # 默认情况下它在批次上平均，所以我们需要在维度上求和，然后对批次求和。
    reconstruction_loss = F.binary_cross_entropy(reconstruction, x, reduction='sum')

    # KL 散度损失
    kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())

    return reconstruction_loss + kl_loss
```

这里，`reconstruction_loss` 是使用 `F.binary_cross_entropy` 在 `reconstruction` 和原始 `x` 之间计算的。我们使用 `reduction='sum'` 对所有元素求和，从而进行有效缩放。`kl_loss` 是使用上述公式计算的。

### 5. 编译和训练 VAE

定义好模型和损失函数 (loss function)后，我们可以设置优化器并训练 VAE。

```python
# 优化器
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

# 训练循环
epochs = 30 # 你可能需要更多训练轮次以获得更好结果
train_losses = []
val_losses = []

for epoch in range(epochs):
    # 训练
    vae.train()
    total_train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        reconstruction, z_mean, z_log_var = vae(data)
        loss = vae_loss_function(reconstruction, data, z_mean, z_log_var)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_dataset)
    train_losses.append(avg_train_loss)

    # 验证
    vae.eval()
    total_val_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            reconstruction, z_mean, z_log_var = vae(data)
            loss = vae_loss_function(reconstruction, data, z_mean, z_log_var)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(test_dataset)
    val_losses.append(avg_val_loss)

    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

# 绘制训练和验证损失值
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='训练损失')
plt.plot(val_losses, label='验证损失')
plt.title('模型损失')
plt.ylabel('损失')
plt.xlabel('训练轮次')
plt.legend(loc='upper right')
plt.show()
```

我们使用 Adam 优化器进行训练。训练循环遍历训练轮次和批次。对于每个批次，我们执行前向传播，计算 VAE 损失，反向传播 (backpropagation)，并更新模型参数 (parameter)。我们还包括一个验证步骤，以监督模型在未见数据上的表现。监督损失是检验模型是否在学习的好习惯。对于生产模型，你可能需要训练更多轮次。

### 6. 检查潜在空间

VAE 最有价值的方面之一是检查其潜在空间的结构。由于我们选择了 `latent_dim = 2`，我们可以在此空间中创建 MNIST 数字的二维散点图。

```python
def plot_latent_space(encoder_model, data_loader, n_samples=10000):
    encoder_model.eval() # 将编码器设置为评估模式
    z_means = []
    labels = []
    with torch.no_grad():
        for i, (data, label) in enumerate(data_loader):
            if len(z_means) * data.shape[0] >= n_samples:
                break # 限制样本数量以便可视化
            data = data.to(device)
            z_mean, _ = encoder_model(data)
            z_means.append(z_mean.cpu().numpy())
            labels.append(label.cpu().numpy())

    z_means = np.concatenate(z_means, axis=0)[:n_samples]
    labels = np.concatenate(labels, axis=0)[:n_samples]

    plt.figure(figsize=(12, 10))
    plt.scatter(z_means[:, 0], z_means[:, 1], c=labels, cmap='viridis') # Use 'viridis' or another distinct colormap
    plt.colorbar(label='数字标签')
    plt.xlabel("潜在维度 1 ($z_1$)")
    plt.ylabel("潜在维度 2 ($z_2$)")
    plt.title("二维潜在空间中的 MNIST 测试数据（均值）")
    plt.grid(True)
    plt.show()

# 使用训练好的编码器模型部分来获取 z_mean
plot_latent_space(vae.encoder, test_loader)
```

`plot_latent_space` 函数使用我们训练好的 VAE 的 `encoder` 部分来获取测试图像的 $z_{mean}$ 向量 (vector)。然后它创建一个散点图，其中每个点代表一个图像，其位置由其二维潜在表示确定，颜色由其实际数字标签（`labels`）决定。

你应该观察到 VAE 已经学会以某种结构化的方式组织数字。看起来相似的数字（例如 1 和 7，或 3 和 8）可能靠得更近，并且不同数字可能存在清晰的聚类。损失函数 (loss function)中的 KL 散度项促成了这种连续且有组织的结构。

以下是此类可视化 Plotly 图表的示例（此示例 JSON 中数据点较少，以便简洁）。实际上，你会使用 `plot_latent_space` 函数中的完整 `z_mean` 和 `y_data_labels`。



![MNIST 潜在空间示例 (z_mean)](plots/6488-0.json)



> 一个散点图，显示了投影到二维潜在空间中的 MNIST 测试图像样本。每个点代表一个图像，并根据其数字标签着色。这有助于可视化 VAE 如何组织不同的数字。

### 7. 生成新样本（从潜在空间采样）

VAE 作为生成模型的真正魅力在于我们从潜在空间中采样点并使用解码器将其转换为新图像。由于潜在空间被鼓励是连续的，潜在空间中相邻的点应该解码为视觉上相似的图像。

```python
def plot_generated_images_manifold(decoder_model, n=15, figure_size=15, latent_dim_val=2):
    decoder_model.eval() # 将解码器设置为评估模式
    # 显示数字的二维流形
    # 我们将从潜在空间中的高斯网格采样点
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))

    # 对应于标准正态分布百分位数的线性间隔坐标
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    with torch.no_grad():
        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                if latent_dim_val == 2:
                    z_sample = torch.tensor([[xi, yi]], dtype=torch.float32).to(device)
                else: # 对于更高的 latent_dim，只取前两个用于可视化
                    # 或者为其他维度生成随机样本
                    z_sample_base = torch.randn(1, latent_dim_val).to(device)
                    z_sample_base[0,0] = xi
                    z_sample_base[0,1] = yi
                    z_sample = z_sample_base

                x_decoded = decoder_model(z_sample)
                digit = x_decoded[0].cpu().numpy().reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size,
                       j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(figure_size, figure_size))
    plt.imshow(figure, cmap='Greys_r')
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("潜在维度 1 变化")
    ax.set_ylabel("潜在维度 2 变化")
    plt.title("从潜在空间生成的数字流形")
    plt.show()

# 使用训练好的解码器模型部分
plot_generated_images_manifold(vae.decoder, n=20, latent_dim_val=latent_dim)
```

`plot_generated_images_manifold` 函数在二维潜在空间中创建了一个点网格（从高斯先验下可能存在的区域进行采样）。对于每个点 ($z_{sample}$)，它使用 `decoder` 生成一个图像。然后这些图像被排列成一个大网格并显示出来。

当你在潜在空间中移动时，你应该会看到不同类型的数字之间平滑的过渡。例如，一个数字可能从“1”逐渐变形为“7”，或从“4”变为“9”。这展示了 VAE 学习有意义且连续表示的能力。

### 8. 总结与展望

在此实作部分，你已使用 PyTorch 成功构建、训练并检验了变分自编码器。你了解了如何：

- 定义输出分布参数 (parameter)（$z_{mean}, z_{log\_var}$）的编码器。
- 实现重参数化技巧进行采样。
- 创建一个结合重构和 KL 散度项的自定义 VAE 损失函数 (loss function)。
- 训练 VAE 并可视化其学习到的潜在空间，观察它是如何聚类和组织 MNIST 数字等数据的。
- 使用解码器通过遍历潜在空间生成新的数据样本，展示 VAE 的生成能力。

VAE 编码器学习到的特征，特别是 $z_{mean}$ 向量 (vector)，对于后续任务非常有价值。正如本章前面讨论的（“将 VAE 潜在表示作为特征使用”），这些压缩的、结构化的表示通常可以提高分类器或其他机器学习 (machine learning)模型的性能，尤其是在处理高维数据时。

通过以下方式进行更多尝试：

- 尝试编码器和解码器的不同网络架构。
- 调整 `latent_dim`。如果它更大怎么办？或者只有 1？
- 训练更多轮次或在不同数据集上训练。
- 将提取的潜在特征（$z_{mean}$）应用于简单的分类任务（例如，在 MNIST 潜在特征上使用 `sklearn` 的 `LogisticRegression`），并将其性能与在原始像素上训练的分类器进行比较。

这次实践经验为将 VAE 应用于更复杂的问题以及理解它们在特征提取和生成建模中的作用奠定了坚实的基础。

## 参考资料

- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — Diederik P Kingma, Max Welling (2013)
  Journal: arXiv preprint arXiv:1312.6114; DOI: [10.48550/arXiv.1312.6114](https://doi.org/10.48550/arXiv.1312.6114)
  介绍变分自编码器（VAE）的基础论文，详细阐述了概率框架、重参数化技巧和目标函数。
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) — PyTorch Contributors (2024)
  Publisher: PyTorch Foundation
  PyTorch深度学习框架的官方文档，对于理解VAE实现中使用的库和函数至关重要。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 20
  一本权威的深度学习教材，包含关于深度生成模型（如VAE）的详细章节，解释了其理论基础。
- [Deep Learning with PyTorch](https://www.manning.com/books/deep-learning-with-pytorch) — Eli Stevens, Luca Antiga, and Thomas Viehmann (2020)
  Publisher: Manning Publications
  使用PyTorch构建深度学习模型的实用指南，涵盖了基础概念和实际实现，包括自编码器和生成模型。
