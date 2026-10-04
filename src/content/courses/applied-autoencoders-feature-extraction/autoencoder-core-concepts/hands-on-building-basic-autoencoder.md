---
course: "applied-autoencoders-feature-extraction"
chapter: "autoencoder-core-concepts"
lesson: "hands-on-building-basic-autoencoder"
sourceId: 6453
sourceUrl: "https://apxml.com/zh/courses/applied-autoencoders-feature-extraction/chapter-2-autoencoder-core-concepts/hands-on-building-basic-autoencoder"
title: "动手实践：构建一个基础自编码器"
description: "一个关于构建和训练用于图像重构的简单自编码器的实践环节。"
order: 9
plots: []
sourceHash: "1195de762681b10b390f20e508125e59e2f0c22a2e873e7472a1ef4de0384d9b"
sourceCorrections: []
---

这个实践练习将帮助您巩固对编码器、解码器、瓶颈层以及重构损失的理解。我们将使用流行的MNIST数据集，它包含手写数字的灰度图像。我们的目标是训练一个自编码器，将其图像压缩成低维表示，然后重构它们。

### 准备工作：环境和数据

在开始之前，请确保您的深度学习 (deep learning)环境已准备就绪。对于本例，我们将介绍基于PyTorch的设置步骤。您主要需要`torch`和`torchvision`用于数据加载和模型构建，`numpy`用于数值运算，以及`matplotlib`用于可视化结果。

**1. 导入库**
首先，让我们导入必要的库。

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
```

**2. 加载和准备MNIST数据集**
MNIST数据集通过`torchvision.datasets`方便获取。每个图像为28x28像素。对于这个基础自编码器，我们将把这些28x28的图像展平为784像素的向量 (vector)。我们还需要对像素值进行归一化 (normalization)，通常是归一化到0到1的范围，这有助于训练的稳定。PyTorch的`transforms.ToTensor()`会自动处理将像素缩放到[0, 1]。

```python
# 定义一个转换来归一化数据并展平图像
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)), # 归一化到[-1, 1]范围，以便使用tanh时更好地训练（可选，但常见）
    transforms.Lambda(lambda x: x.view(-1)) # 将28x28图像展平为784
])

# 加载MNIST数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 创建数据加载器
batch_size = 256
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 获取一个样本以检查形状（可选）
sample_data, _ = next(iter(train_loader))
print(f"Sample x_train batch shape: {sample_data.shape}")
print(f"Flattened image size: {sample_data.shape[1]}")
```

您会注意到我们只加载了图像数据并忽略了标签（`_`）。这是因为自编码器以无监督方式训练；它们的目的是重构输入，而不是预测标签。展平后的图像大小应为784。

### 设计我们的基础自编码器

自编码器由两个主要部分组成：编码器和解码器。编码器将输入数据映射到低维表示（瓶颈层），而解码器则尝试从该表示中重构原始输入。

**1. 定义架构**
让我们使用PyTorch的`nn.Module`定义一个简单架构。我们将使用`nn.Linear`层。

- **输入层**：这将与我们展平的MNIST图像的形状（784个特征）匹配。
- **编码器**：一系列`nn.Linear`层，逐步降低维度。例如，784 -> 128 -> 64。
- **瓶颈层**：这是我们网络中最小的层，表示压缩的潜在空间。在本例中，我们选择维度为32。这远小于输入784，使其成为一个欠完备自编码器。
- **解码器**：一系列`nn.Linear`层，逐步增加维度，与编码器镜像对应。例如，32 -> 64 -> 128 -> 784。
- **输出层**：该层应具有与输入相同的单元数量（784），并使用适用于重构归一化 (normalization)像素值（例如，若归一化到[0,1]则使用`nn.Sigmoid`，若归一化到[-1,1]则使用`nn.Tanh`）的激活函数 (activation function)。

我们可以这样定义它：

```python
class Autoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.ReLU() # 瓶颈层
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Tanh() # 如果输入被归一化到[-1, 1]，则使用Tanh，否则对[0, 1]使用Sigmoid
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

latent_dim = 32
autoencoder = Autoencoder(latent_dim)

# 如果有GPU可用，将模型移至GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autoencoder.to(device)

print(autoencoder)
```

下面是一张图，展示了我们自编码器的一般结构：

> 数据通过自编码器的流程，从输入，经过编码器和瓶颈层进行压缩，再由解码器进行重构。

**2. 定义损失函数 (loss function)和优化器**
在训练之前，我们需要定义损失函数和优化器。
正如本章所讨论的，均方误差（MSE）是处理像我们归一化后的像素值这类连续数据时，重构损失的常用选择。

$MSE = \frac{1}{N} \sum_{i=1}^{N} (x_i - \hat{x}_i)^2$

这里，$x_i$是原始输入，$\hat{x}_i$是重构输出。我们将使用Adam优化器，这是许多深度学习 (deep learning)任务中流行且有效的方法。

```python
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)
```

### 训练自编码器

现在，我们来训练自编码器。特别之处在于，输入数据同时作为输入*和*目标输出。网络学习重构它所接收到的内容。

```python
num_epochs = 50
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # 训练阶段
    autoencoder.train()
    running_train_loss = 0.0
    for data, _ in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        outputs = autoencoder(data)
        loss = criterion(outputs, data)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item() * data.size(0)

    epoch_train_loss = running_train_loss / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)

    # 验证阶段
    autoencoder.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            outputs = autoencoder(data)
            loss = criterion(outputs, data)
            running_val_loss += loss.item() * data.size(0)

    epoch_val_loss = running_val_loss / len(test_loader.dataset)
    val_losses.append(epoch_val_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {epoch_train_loss:.4f}, '
          f'Validation Loss: {epoch_val_loss:.4f}')
```

我们可以绘制训练损失和验证损失图，以观察模型掌握的如何：

```python
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='训练损失')
plt.plot(val_losses, label='验证损失')
plt.title('模型训练期间的损失')
plt.xlabel('周期')
plt.ylabel('损失 (MSE)')
plt.legend()
plt.grid(True)
plt.show()
```

### 可视化重构结果

检验自编码器能力的关键是它重构输入图像的程度。让我们使用训练好的`autoencoder`从测试集中预测（重构）图像，并将其中一些与原始图像并排显示。

```python
# 从测试集重构图像
autoencoder.eval() # 将模型设置为评估模式
with torch.no_grad():
    data_iter = iter(test_loader)
    data, _ = next(data_iter) # 获取一批测试数据
    data = data.to(device)
    decoded_imgs = autoencoder(data).cpu().numpy() # 获取重构结果并移至CPU

# 显示原始图像和重构图像
n = 10  # 要显示的数字数量
plt.figure(figsize=(20, 4))
for i in range(n):
    # 显示原始图像
    ax = plt.subplot(2, n, i + 1)
    # 为显示目的撤销归一化：数据被归一化到[-1, 1]，因此将其缩放回[0, 1]
    original_img = (data[i].cpu().numpy().reshape(28, 28) + 1) / 2
    plt.imshow(original_img, cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if i == 0:
        ax.set_title("原始图像")

    # 显示重构图像
    ax = plt.subplot(2, n, i + 1 + n)
    # 为显示目的撤销归一化：decoded_imgs在[-1, 1]范围内，将其缩放回[0, 1]
    reconstructed_img = (decoded_imgs[i].reshape(28, 28) + 1) / 2
    plt.imshow(reconstructed_img, cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if i == 0:
        ax.set_title("重构图像")
plt.show()
```

您会看到，重构后的数字虽然可能比原始数字略显模糊或不清晰，但总体上是可识别的。这表明我们的自编码器在其32维瓶颈层中学到了有效的压缩表示，并能使用该表示来产生对原始784维输入的合理近似。

### 我们已完成的工作

在这次动手实践中，您成功地使用PyTorch构建并训练了一个基础自编码器。您已了解：

- **编码器**将高维输入映射到低维的**瓶颈层**。
- **解码器**尝试从这种压缩表示中重构原始输入。
- 网络通过最小化**重构损失**（在本例中为MSE）进行训练，其中输入本身就是目标。

这个简单的自编码器演示了基本原理。瓶颈层学习到的潜在表示是特征提取的根本，我们将在后续章节中更详细地介绍。例如，您可以通过将输入数据传入`autoencoder.encoder(data)`来获得潜在表示。我们很快就会看到不同种类的自编码器和更精巧的架构如何能学习到更强且有价值的特征。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  提供了深度学习的全面理论基础，包括专门介绍自编码器以及重建损失和瓶颈等相关概念的章节。
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) — PyTorch Core Team (2024)
  Publisher: PyTorch Foundation
  用于本实践中PyTorch库和模块（如torch.nn、torch.optim和torchvision）的官方文档。
- [Gradient-based learning applied to document recognition](https://ieeexplore.ieee.org/document/726791) — Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner (1998)
  Journal: Proceedings of the IEEE; Publisher: IEEE; Volume: 86; Pages: 2278-2324; DOI: [10.1109/5.726791](https://doi.org/10.1109/5.726791)
  介绍了MNIST数据集的基础论文，该数据集是自编码器实践示例的核心。
- [Deep Learning with PyTorch](https://www.manning.com/books/deep-learning-with-pytorch) — Eli Stevens, Luca Antiga, and Thomas Viehmann (2020)
  Publisher: Manning Publications
  一本专门使用PyTorch框架实现深度学习模型（包括自编码器）的实用指南。
