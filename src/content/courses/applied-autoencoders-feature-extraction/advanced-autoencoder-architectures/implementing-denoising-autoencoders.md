---
course: "applied-autoencoders-feature-extraction"
chapter: "advanced-autoencoder-architectures"
lesson: "implementing-denoising-autoencoders"
sourceId: 6467
sourceUrl: "https://apxml.com/zh/courses/applied-autoencoders-feature-extraction/chapter-4-advanced-autoencoder-architectures/implementing-denoising-autoencoders"
title: "去噪自编码器的实现"
description: "实现去噪自编码器的实际步骤，包括向输入数据添加噪声的方法。"
order: 4
plots: []
sourceHash: "c1b972bf1ec03ceadeb751f721d6b5bf44e70c2ad7c1200e3f190ded7c5a5b8c"
sourceCorrections: []
---

去噪自编码器（DAE）被训练来重建原始的、干净的输入，即使输入的是其损坏版本。这种简单而有效的方法促使自编码器学习到更有用的特征，因为它无法只学习一个简单的恒等映射。

实现去噪自编码器涉及几个不同的步骤，主要围绕您如何准备数据以及您要求模型预测什么。

### 创建损坏的输入数据

去噪自编码器的显著特点是其在噪声输入上进行训练。因此，第一步是人为地损坏您的干净数据集。设原始干净输入为 $x$。我们通过添加某种形式的噪声或遮蔽 $x$ 的某些部分来创建损坏版本 $\tilde{x}$。自编码器随后将学习把 $\tilde{x}$ 映射回 $x$。

有几种常见的引入这种损坏的方式：

1. **加性高斯噪声**：这涉及向您的输入数据添加从高斯（正态）分布中采样的随机数。
   如果您的数据已归一化 (normalization)（例如，像素值在0到1之间），您通常会添加均值为0和选定标准差（噪声因子）的噪声。添加噪声后，您可能需要裁剪这些值，以确保它们保持在有效范围（例如，归一化图像的`[0, 1]`）内。

   使用PyTorch在Python中的典型操作可能如下所示：

   ```python
   import torch

   noise_factor = 0.5 # 调整此超参数
   # 假设 x_train_clean 是一个PyTorch张量
   x_train_noisy = x_train_clean + noise_factor * torch.randn_like(x_train_clean)
   x_train_noisy = torch.clamp(x_train_noisy, 0., 1.) # 确保数据保持在有效范围
   ```

   `noise_factor` 控制噪声量。因子越大意味着损坏程度越高。
2. **遮蔽噪声（椒盐噪声）**：这涉及随机地将输入特征（例如，图像中的像素，数据向量 (vector)中的值）的一小部分设置为最小值、最大值，或直接将其归零。
   例如，您可以随机将图像中一定百分比的像素设置为0。

   使用PyTorch，这可以这样实现：

   ```python
   # 用于将随机元素设置为0
   noise_factor = 0.5 # 要归零的元素比例（例如，0.5表示50%被丢弃）
   # 创建二进制掩码：以 (1 - noise_factor) 的概率为1，否则为0
   mask = (torch.rand_like(x_train_clean) > noise_factor).float()
   x_train_noisy = x_train_clean * mask
   ```

   此处，`noise_factor` 表示被“丢弃”或遮蔽的输入特征的比例。

噪声类型及其强度（噪声因子）的选择是重要的超参数 (parameter) (hyperparameter)。您应选择一种能反映数据集中预期干扰类型，或反映您希望模型学习的不变性的噪声类型。噪声量应足够大，以防止自编码器学习简单的恒等函数，但又不能太大，以至于模型无法恢复原始信号。

### 模型结构

去噪自编码器（编码器和解码器网络）的结构可以与标准自编码器非常相似。您可以将全连接层用于表格数据，或将卷积层和转置卷积层用于图像数据，就像您对非去噪的对应模型所做的那样。

- **编码器**：接收损坏的输入 $\tilde{x}$ 并将其映射到低维潜在表示 $z$。
  $z = f(\tilde{x})$
- **解码器**：接收潜在表示 $z$ 并尝试重建*原始、干净*的输入 $x$。
  $x' = g(z)$
  $x'$ 即为重建输出。

主要区别不在于层本身，而在于网络在训练期间看到的内容以及它试图生成的内容。

### 训练过程与损失函数 (loss function)

这是去噪自编码器与标准自编码器区别明显的地方。在训练期间：

- **模型输入**：损坏的数据，$\tilde{x}$。
- **模型目标**：原始的、干净的数据，$x$。

损失函数衡量解码器的输出 $x'$ 与原始干净数据 $x$ 之间的差异。常见的损失函数包括：

- **均方误差 (MSE)**：适用于连续输入数据，如归一化 (normalization)像素强度。在PyTorch中，这是 `nn.MSELoss()`。
  $L(x, x') = \frac{1}{N} \sum_{i=1}^{N} (x_i - x'_i)^2$
- **二元交叉熵 (BCE)**：如果输入数据是二元的或表示概率（例如，黑白图像，或像素值缩放到[0,1]并被视为概率的图像），则常用此函数。在PyTorch中，这是 `nn.BCELoss()`（用于原始概率）或 `nn.BCEWithLogitsLoss()`（用于logits）。
  $L(x, x') = - \frac{1}{N} \sum_{i=1}^{N} [x_i \log(x'_i) + (1-x_i) \log(1-x'_i)]$

让我们以图示方式呈现数据流和训练目标：

> 此图表呈现了去噪自编码器的训练流程。干净数据 $x$ 首先被损坏以生成 $\tilde{x}$。自编码器将 $\tilde{x}$ 作为输入，其解码器尝试重建原始干净数据 $x$。损失函数通过比较重建结果 $x'$ 与原始 $x$ 来计算。

使用PyTorch时，训练循环设置将如下所示：

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 假设您的自编码器类已像之前一样定义（包含编码器和解码器部分）
# 示例：
# class Autoencoder(nn.Module):
#     def __init__(self, input_dim, latent_dim):
#         super().__init__()
#         self.encoder = nn.Sequential(...)
#         self.decoder = nn.Sequential(...)
#     def forward(self, x):
#         return self.decoder(self.encoder(x))

# autoencoder_model = Autoencoder(input_dim, latent_dim).to(device)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(autoencoder_model.parameters(), lr=0.001)

# 假设 x_train_clean_tensor 和 x_test_clean_tensor 是您的原始干净数据张量
# 假设您的数据加载器 train_loader 和 test_loader 是使用 TensorDataset 创建的

epochs = 50

for epoch in range(epochs):
    autoencoder_model.train()
    for clean_data_batch, _ in train_loader: # _ 只是一个占位符，因为 clean_data_batch 既是输入也是目标
        clean_data_batch = clean_data_batch.to(device)

        # 应用噪声以创建损坏的输入
        noise_factor = 0.2
        noisy_data_batch = clean_data_batch + noise_factor * torch.randn_like(clean_data_batch)
        noisy_data_batch = torch.clamp(noisy_data_batch, 0., 1.) # 裁剪到有效范围

        optimizer.zero_grad()
        reconstructed_output = autoencoder_model(noisy_data_batch) # 输入：噪声数据
        loss = criterion(reconstructed_output, clean_data_batch) # 目标：干净数据
        loss.backward()
        optimizer.step()

    # （可选）添加与之前示例类似的验证循环
    # autoencoder_model.eval()
    # with torch.no_grad():
    #     for clean_data_batch_val, _ in test_loader:
    #         clean_data_batch_val = clean_data_batch_val.to(device)
    #         noisy_data_batch_val = clean_data_batch_val + noise_factor * torch.randn_like(clean_data_batch_val)
    #         noisy_data_batch_val = torch.clamp(noisy_data_batch_val, 0., 1.)
    #         val_reconstruction = autoencoder_model(noisy_data_batch_val)
    #         val_loss = criterion(val_reconstruction, clean_data_batch_val)
    # ... 打印验证损失

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
```

请仔细注意，`noisy_data_batch` 作为输入传入，而 `clean_data_batch` 作为目标传入。这迫使网络学习去除噪声。

### 为何这适用于特征学习

通过训练自编码器进行去噪，您实际上是在迫使它学习数据的底层流形或结构。它无法简单地学习一个恒等函数，因为输入和目标不同。为了成功地从噪声版本中重建干净数据，编码器必须捕捉到基本特征并舍弃噪声。因此，得到的潜在表示 $z$ 往往更有效，并且包含更多关于数据的重要信息。

一旦去噪自编码器训练完成，您通常会使用其编码器部分。您会将新的、*干净*的数据通过这个训练好的编码器，以获取特征表示。这些特征随后可用于分类或聚类等后续任务。

例如，要从一个训练好的PyTorch DAE中获取编码器：

```python
# 假设 'autoencoder_model' 是您训练好的 DAE 实例
# 并且 'x_new_clean_data' 是您新的、干净的数据（作为 PyTorch 张量）
autoencoder_model.eval() # 将模型设置为评估模式
with torch.no_grad(): # 禁用梯度计算
    clean_features = autoencoder_model.encoder(x_new_clean_data.to(device))
# clean_features 将是设备上的 PyTorch 张量；如果需要，可以转换为 NumPy：
# clean_features_np = clean_features.cpu().numpy()
```

实现去噪自编码器是标准自编码器的直接扩展，但这种训练目标的变化，即从噪声中重建干净数据，对学习到的特征的质量和稳定性有着明显影响。本章后面的实践练习将指导您从头开始构建和训练一个。

## 参考资料

- [Extracting and Composing Robust Features with Denoising Autoencoders](https://dl.acm.org/doi/10.1145/1390156.1390294) — Pascal Vincent, Hugo Larochelle, Yoshua Bengio, and Pierre-Antoine Manzagol (2008)
  Journal: Proceedings of the 25th International Conference on Machine Learning (ICML); Publisher: Association for Computing Machinery; Pages: 1096-1103; DOI: [10.1145/1390156.1390294](https://doi.org/10.1145/1390156.1390294)
  这篇论文介绍了去噪自动编码器，详细阐述了其架构、训练目标以及通过从损坏输入重建干净输入来学习鲁棒特征表示的动机。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  这本综合性教材深入地解释了自动编码器，包括去噪自动编码器，涵盖了其原理、不同噪声类型以及对表示学习的益处。
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) — PyTorch Development Team (2024)
  Publisher: PyTorch Foundation
  PyTorch官方文档是实现深度学习模型的实用指南。它涵盖了核心模块和函数，如`torch.randn_like`、`torch.clamp`、`nn.MSELoss`、`nn.BCELoss`和`nn.Module`，这些都在去噪自动编码器的实现示例中使用。
