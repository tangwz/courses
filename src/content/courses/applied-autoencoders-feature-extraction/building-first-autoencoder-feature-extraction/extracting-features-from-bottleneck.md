---
course: "applied-autoencoders-feature-extraction"
chapter: "building-first-autoencoder-feature-extraction"
lesson: "extracting-features-from-bottleneck"
sourceId: 6461
sourceUrl: "https://apxml.com/zh/courses/applied-autoencoders-feature-extraction/chapter-3-building-first-autoencoder-feature-extraction/extracting-features-from-bottleneck"
title: "从瓶颈层提取特征的方法"
description: "从自编码器瓶颈层提取学习到的特征表示的实用方法。"
order: 8
plots: []
sourceHash: "30a48e9d0927b2a847b61234ba07a5de58a7b8c5599ca4b799930f4144340a8a"
sourceCorrections: []
---

您的自编码器成功训练后，它的编码器部分已学习到一种映射，将输入数据映射到一个紧凑的、低维度的表示，位于瓶颈层。这种学习到的表示就是我们所求的。下一步是使用这个训练好的编码器将您现有和新的数据转换为这些有意义的特征。这个过程有效地将您的编码器变成了一个专门的**特征提取**工具。

### 分离编码器

完整的自编码器模型，包含编码器和解码器，对于训练过程是必需的。解码器的作用是从瓶颈层的表示重建输入，从而促使编码器学习有用的特征。然而，对于特征提取，我们只需要编码器。

在PyTorch等深度学习 (deep learning)框架中，自编码器可以方便地定义，其**编码器组件是模型中一个独立可调用的部分**。自编码器训练完成后，特征提取过程变得简单直接。

设想您的自编码器架构如下所示。对于特征提取，我们实质上只使用编码器组件。

> 已训练的自编码器（上方）学习将数据压缩为瓶颈层表示。对于特征提取（下方），仅使用编码器部分将输入数据转换为这些特征。

### 创建和使用特征提取器

在PyTorch中，处理此事的常用方法是将**编码器定义为主`Autoencoder`类中一个独立的`nn.Sequential`块或`nn.Module`**。然后，在训练完完整的`Autoencoder`模型后，您可以直接访问其`encoder`属性来执行特征提取。

假设您有一个已训练的`autoencoder`对象，基于前面示例中的`Autoencoder`类。

```python
import torch
import torch.nn as nn
import numpy as np

# 重新定义Autoencoder类，以确保编码器是一个清晰的模块
class Autoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim

        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.ReLU()
        )

        # 解码器部分
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Tanh() # 或Sigmoid，取决于您的归一化方式
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# 创建一个自编码器实例（假设 latent_dim=32）
autoencoder = Autoencoder(latent_dim=32)

# --- 重要：在此处加载已训练的权重 ---
# 这很重要。没有训练好的权重，编码器将无法提取有意义的特征。
# 为了演示，假设您已保存了训练好的模型：
# torch.save(autoencoder.state_dict(), 'autoencoder_mnist.pth')
# autoencoder.load_state_dict(torch.load('autoencoder_mnist.pth'))
# ---------------------------------------------

# 将模型移至设备（CPU/GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autoencoder.to(device)
autoencoder.eval() # 将模型设置为评估模式

print("自编码器架构：")
print(autoencoder)

# 示例：准备一些模拟新数据（请替换为您的实际数据）
# 这些模拟数据应模仿您原始训练数据的形状和预处理方式。
# 对于MNIST，那是展平的784维向量，归一化到 [-1, 1]。
num_new_samples = 5
dummy_new_data = torch.randn(num_new_samples, 784).to(device) # 随机数据，请替换为实际的预处理数据

# 通过编码器生成特征
with torch.no_grad(): # 禁用梯度计算，因为我们不进行训练
    extracted_features = autoencoder.encoder(dummy_new_data)

# 将特征移回CPU并转换为NumPy数组以供进一步使用
extracted_features_np = extracted_features.cpu().numpy()

print(f"\n原始（模拟）数据的形状：{dummy_new_data.shape}")
print(f"提取特征的形状：{extracted_features_np.shape}")
```

`extracted_features_np` 数组将有 `N` 行（其中 `N` 是输入样本的数量）和 `D_latent` 列（其中 `D_latent` 是瓶颈层的维度，在本例中为 32）。每一行都是一个密集向量 (vector) (dense vector)，表示相应输入样本的压缩形式。

---

### 提取特征的特点

您刚刚提取的特征有几个显著特点：

- **较低维度**：它们通常比原始输入数据的维度低得多，实现了降维的主要目标。
- **学习到的表示**：这些不是人工设计的特征。相反，它们是由自编码器学习的，旨在捕捉数据中重建所需的显著变动。
- **密集**：与某些情况下可能寻求的稀疏表示（例如，稀疏自编码器）不同，基本自编码器瓶颈层中的特征通常是密集的数值向量 (vector)。
- **可能非线性**：由于自编码器使用非线性激活函数 (activation function)，它们可以捕获数据中复杂的非线性关系，通常会产生比PCA等线性方法更具表现力的特征。

---

### 重要：数据预处理的一致性

一个无论如何强调都不过分的注意事项：任何您想提取特征的新数据**必须**经过完全相同的预处理步骤（例如，归一化 (normalization)、缩放、重塑），这些步骤适用于训练自编码器的数据。

如果您的训练数据被缩放至 [0, 1] 范围，新数据也必须使用*相同参数 (parameter)*（例如，训练集中的最小值/最大值）缩放至 [0, 1] 范围。如果数据经过标准化（零均值，单位方差），新数据必须使用*训练集中的均值和标准差*进行标准化。未能做到这一点将导致编码器接收到与训练时不同分布的数据，从而导致提取的特征可能不是最佳的，甚至没有意义。

---

### 如何处理这些特征？

现在您有了这些紧凑的、学习到的特征向量 (vector)，接下来该怎么做？
这些特征用途广泛：

1. **作为下游模型的输入**：它们可以作为其他机器学习 (machine learning)模型的输入，例如分类器（例如，逻辑回归、支持向量机、随机森林）或回归模型。通常，使用这些压缩特征可以加快训练速度，减少过拟合 (overfitting)，有时还能为下游任务带来更好的性能。
2. **数据可视化**：如果潜在空间的维度是2或3，您可以直接将数据点可视化。对于更高维的潜在空间，您可以使用t-SNE或UMAP等进一步的降维技术对提取的特征进行处理，以便在2D或3D中进行可视化并查看数据簇或流形。
3. **相似性搜索和异常检测**：特征可用于查找相似项或检测异常。
4. **数据压缩**：虽然自编码器可用于有损压缩，但提取的特征本身就是压缩表示。

从瓶颈层提取特征的过程是运用自编码器进行特征工程中的基本步骤。凭借这些技术，您现在可以转换原始数据成为更强大的表示形式，适用于各种机器学习应用。在即将到来的实操部分，您将这些步骤应用于使用表格数据的实际示例。

## 参考资料

- [Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion](http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf) — Pascal Vincent, Hugo Larochelle, Yoshua Bengio, Pierre-Antoine Manzagol (2010)
  Journal: Journal of Machine Learning Research (JMLR); Publisher: Journal of Machine Learning Research; Volume: 11; Pages: 3371-3408; DOI: [10.5555/1953048.2078174](https://doi.org/10.5555/1953048.2078174)
  本文介绍了堆叠去噪自编码器，这些方法能有效从数据中学习到强大且有意义的特征。
- [Deep Learning](http://www.deeplearningbook.org) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  一本关于深度学习的综合资源，其中有一章专门介绍自编码器及其在表征学习中的作用。
- [PyTorch torch.nn.Module Documentation](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) — PyTorch Contributors (2024)
  Publisher: PyTorch.org
  这份官方文档提供了定义神经网络层和模型的基础信息，包括 `nn.Module` 基类，这对于构建自编码器及其编码器等子组件至关重要。
