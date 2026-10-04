---
course: "applied-autoencoders-feature-extraction"
chapter: "building-first-autoencoder-feature-extraction"
lesson: "hands-on-feature-extraction-tabular-data"
sourceId: 6463
sourceUrl: "https://apxml.com/zh/courses/applied-autoencoders-feature-extraction/chapter-3-building-first-autoencoder-feature-extraction/hands-on-feature-extraction-tabular-data"
title: "实践：从表格数据中获取特征"
description: "关于构建自编码器以从表格数据集中获取特征的实践环节。"
order: 10
plots: ["plots/6463-0.json"]
sourceHash: "bd290a463045689e34276f9a1bb168d7c4077304ca4b3330a4beeb410bee14f0"
sourceCorrections: []
---

好的，我们来构建一个自编码器以从表格数据中获取特征。在前面几节中，我们讨论了相关原理：数据准备、网络设计、损失函数 (loss function)和优化器。现在，我们将所有这些付诸实践。我们的目标是接收一个含有特定数量特征的数据集，训练一个自编码器来重构它，然后将自编码器的压缩内部表示（瓶颈层）用作新的、低维的特征集。

我们将逐步进行以下步骤：

1. 设置我们的环境并生成一个合成数据集。
2. 准备数据：缩放和拆分。
3. 设计和构建自编码器模型。
4. 训练自编码器。
5. 从瓶颈层中获取学到的特征。
6. 可视化潜在空间。
7. 在一个简单的分类任务中，简要评估这些已获取特征的实用性。

使用 PyTorch 构建 autoencoder，用于特征提取。其原理可以直接应用于其他框架。

### 设置和生成数据

首先，请确保您已安装必要的库。我们将主要使用 `torch`、`numpy`、`pandas` 和 `scikit-learn`。

```python
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
```

对于本次练习，我们将使用 `scikit-learn` 生成一个合成数据集。这使我们能够控制数据的特性，并能够专注于自编码器的机制。我们将创建一个包含 20 个特征的数据集。我们的自编码器将尝试学习这些特征的压缩表示。

```python
from sklearn.datasets import make_classification

# 生成一个合成数据集
X_orig, y_orig = make_classification(
    n_samples=2000,
    n_features=20,        # 原始特征数量
    n_informative=15,    # 有用特征数量
    n_redundant=3,        # 冗余特征数量
    n_repeated=0,        # 重复特征数量
    n_classes=2,        # 目标变量 y 的类别数量
    n_clusters_per_class=2,
    flip_y=0.01,
    random_state=42
)

print(f"原始数据形状: {X_orig.shape}")
# 原始数据形状: (2000, 20)
```

这里，`X_orig` 包含我们的特征，而 `y_orig` 是一个二元类别标签。我们将使用 `X_orig` 以无监督方式训练自编码器（在训练期间它不会看到 `y_orig`）。稍后，`y_orig` 将对于可视化潜在空间和可选的分类任务很有用。

### 数据准备

神经网络 (neural network)，包括自编码器，当输入特征处于相似的尺度上时通常表现更好。我们将使用 `scikit-learn` 中的 `MinMaxScaler` 将特征缩放到 0 到 1 之间。

```python
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_orig)

# 将数据拆分为训练集和测试集
# 自编码器学习重构其输入，因此 X 既是输入也是目标。
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    X_scaled, y_orig, test_size=0.2, random_state=42, stratify=y_orig
)

# 将 NumPy 数组转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_np, dtype=torch.long)
y_test_tensor = torch.tensor(y_test_np, dtype=torch.long)

# 创建 TensorDatasets 和 DataLoaders
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, X_train_tensor) # 对于自编码器，输入和目标是相同的
test_dataset = TensorDataset(X_test_tensor, X_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"训练数据形状 (PyTorch 张量): {X_train_tensor.shape}")
print(f"测试数据形状 (PyTorch 张量): {X_test_tensor.shape}")
# 训练数据形状 (PyTorch 张量): torch.Size([1600, 20])
# 测试数据形状 (PyTorch 张量): torch.Size([400, 20])
```

缩放确保所有特征更均匀地贡献于学习过程，并帮助优化器更快收敛。解码器输出层中的 `Sigmoid` 激活函数 (activation function)很适合缩放到 [0, 1] 范围的数据。

### 设计自编码器

我们将构建一个简单的全连接自编码器。该架构将包含一个将输入压缩到一个小潜在空间的编码器，以及一个从这个潜在表示中重构输入的解码器。

我们来定义瓶颈层的维度。这是一个重要的超参数 (parameter) (hyperparameter)。更小的瓶颈层会强制进行更强的压缩，并可能会学到更抽象的特征。对于这个示例，我们的目标是生成一个二维潜在空间，这将易于可视化。

```python
input_dim = X_train_tensor.shape[1]  # 特征数量，即 20
latent_dim = 2                      # 瓶颈层的维度

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
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
            nn.Linear(128, input_dim),
            nn.Sigmoid() # Sigmoid 适用于 [0,1] 输出
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

autoencoder_model = Autoencoder(input_dim, latent_dim)

# 如果 GPU 可用，则将模型移至 GPU
device = torch.device("cuda" if torch.cuda_is_available() else "cpu")
autoencoder_model.to(device)

print(autoencoder_model)
```

`autoencoder_model` 的打印输出显示了其结构：`encoder` 是一个 `Sequential` 模块，将 20 个维度压缩到 2 个；`decoder` 是另一个 `Sequential` 模块，将其扩展回 20 个维度。`ReLU` 激活函数 (activation function)用于隐藏层，而 `Sigmoid` 用于输出层，因为我们的输入数据缩放到 0 和 1 之间。如果我们使用 `StandardScaler`，线性激活函数可能更适合输出。

现在，我们来定义损失函数 (loss function)和优化器。我们将使用 Adam 优化器和均方误差 (`MSELoss`) 作为损失函数。`MSE` 适合处理连续（或缩放后的连续）数据的重构任务。

```python
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder_model.parameters(), lr=0.001)
```

### 训练自编码器

我们通过要求自编码器重构训练数据来对其进行训练。输入和目标输出是相同的。

```python
epochs = 100
train_losses = []
val_losses = []

for epoch in range(epochs):
    # 训练
    autoencoder_model.train()
    running_train_loss = 0.0
    for data, _ in train_loader: # _ 是目标，与数据相同
        data = data.to(device)
        optimizer.zero_grad()
        outputs = autoencoder_model(data)
        loss = criterion(outputs, data)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item() * data.size(0)

    epoch_train_loss = running_train_loss / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)

    # 验证
    autoencoder_model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for data, _ in test_loader: # _ 是目标，与数据相同
            data = data.to(device)
            outputs = autoencoder_model(data)
            loss = criterion(outputs, data)
            running_val_loss += loss.item() * data.size(0)

    epoch_val_loss = running_val_loss / len(test_loader.dataset)
    val_losses.append(epoch_val_loss)

    print(f'周期 [{epoch+1}/{epochs}]，训练损失: {epoch_train_loss:.4f}，验证损失: {epoch_val_loss:.4f}')
```

训练完成后，我们可以可视化训练损失和验证损失，以检查是否过拟合 (overfitting)并查看模型是否收敛。



![自编码器训练和验证损失](plots/6463-0.json)



> 自编码器训练进度，显示均方误差损失在训练集和验证集上随周期下降。理想情况下，两种损失都会下降并收敛。

一个好的自编码器在验证集上会有较低的重构损失。

### 从瓶颈层获取特征

一旦自编码器训练完成，编码器部分（`autoencoder_model.encoder`）就可以用作特征获取器。我们将原始（缩放过的）数据输入到编码器中，以获取压缩后的潜在表示。

```python
autoencoder_model.eval() # 将模型设置为评估模式
with torch.no_grad(): # 禁用梯度计算
    X_train_encoded = autoencoder_model.encoder(X_train_tensor.to(device)).cpu().numpy()
    X_test_encoded = autoencoder_model.encoder(X_test_tensor.to(device)).cpu().numpy()

print(f"原始训练数据形状: {X_train_np.shape}")
print(f"编码训练数据形状: {X_train_encoded.shape}")
# 原始训练数据形状: (1600, 20)
# 编码训练数据形状: (1600, 2)

print(f"原始测试数据形状: {X_test_np.shape}")
print(f"编码测试数据形状: {X_test_encoded.shape}")
# 原始测试数据形状: (400, 20)
# 编码测试数据形状: (400, 2)
```

如您所见，我们已成功将数据的维度从 20 个特征降低到 2 个特征。这 2 个特征是由自编码器学习所得，它们捕捉了重构原始数据所需的最显著信息。

### 可视化潜在空间

由于我们选择了 `latent_dim = 2`，我们可以轻松地在二维散点图中可视化已获取的特征。我们可以使用原始类别标签（`y_train_np` 或 `y_test_np`）为点着色，以查看自编码器是否学习到了一个能够区分类别的表示，即使它没有明确地为分类任务进行训练。

```python
plt.figure(figsize=(8, 6))
plt.scatter(X_test_encoded[y_test_np == 0, 0], X_test_encoded[y_test_np == 0, 1],
            label='类别 0', alpha=0.7, c='#4263eb')
plt.scatter(X_test_encoded[y_test_np == 1, 0], X_test_encoded[y_test_np == 1, 1],
            label='类别 1', alpha=0.7, c='#f06595')
plt.title('二维潜在空间可视化')
plt.xlabel('潜在维度 1')
plt.ylabel('潜在维度 2')
plt.legend(title='原始类别')
plt.grid(True)
plt.show()
```

> 来自测试集的二维潜在空间表示的散点图，按其原始类别标签着色。簇或分离可能表示有效的特征学习。(注意：上述数据点仅为示意占位符。)

如果图表显示出一些基于原始类别的分离或聚类，则表明自编码器已学到与数据底层结构相关的特征。

### 评估特征实用性（快速检查）

已获取特征的最终检验是它们在下游任务上的表现。让我们快速地训练一个简单的逻辑回归分类器，基于：

1. 原始（缩放过的）20 个特征。
2. 自编码器获取的 2 个特征。

我们将比较它们在测试集上的准确率。

```python
# 1. 基于原始特征的分类器
lr_original = LogisticRegression(solver='liblinear', random_state=42)
lr_original.fit(X_train_np, y_train_np)
y_pred_original = lr_original.predict(X_test_np)
accuracy_original = accuracy_score(y_test_np, y_pred_original)
print(f"使用原始 {X_train_np.shape[1]} 个特征的准确率: {accuracy_original:.4f}")

# 2. 基于自编码器获取特征的分类器
lr_encoded = LogisticRegression(solver='liblinear', random_state=42)
lr_encoded.fit(X_train_encoded, y_train_np) # 使用二维编码特征
y_pred_encoded = lr_encoded.predict(X_test_encoded)
accuracy_encoded = accuracy_score(y_test_np, y_pred_encoded)
print(f"使用自编码器 {X_train_encoded.shape[1]} 个特征的准确率: {accuracy_encoded:.4f}")

# 示例输出:
# 使用原始 20 个特征的准确率: 0.8850
# 使用自编码器 2 个特征的准确率: 0.8725
```

在这个输出中，使用 2 个特征的准确率略低于使用 20 个特征。这并不意外，因为压缩过程中可能会丢失一些信息。然而，在特征数量显著减少（2 个 vs. 20 个）的情况下获得可比的性能，展现了自编码器在降维方面的能力。在某些情况下，特别是对于嘈杂或高度冗余的原始特征，自编码器获取的特征甚至能通过充当去噪或正则化 (regularization)机制而带来性能提升。降维和性能之间的权衡是一个常见议题。

### 本次实践会话总结

在本次实践章节中，我们成功地：

- 生成了一个合成表格数据集。
- 使用缩放和拆分对数据进行了预处理。
- 在 **PyTorch** 中设计、构建并训练了一个带有二维瓶颈的简单自编码器。
- 使用训练好的编码器从数据中获取低维特征。
- 在潜在空间中可视化了这些特征。
- 通过将它们输入到一个简单的下游分类模型中，简要评估了它们的实用性。

这个实践练习提供了一个使用自编码器进行特征获取的基础工作流程。您可以将这些步骤调整到您自己的表格数据集上。尝试调整层数、每层神经元数量、潜在空间维度以及其他超参数 (parameter) (hyperparameter)（如本章前面所述，并将在第 7 章中更详细地介绍）对于为您的特定问题获得最佳结果很重要。

后面章节将介绍更高级的自编码器架构，如去噪自编码器、用于图像数据的卷积自编码器和变分自编码器，进一步扩充您进行高效特征获取的工具包。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本全面的教材，涵盖自编码器理论、网络设计原则、损失函数和优化策略，对于理解所呈现的概念基础尤为重要。
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) — PyTorch Core Team (2024)
  PyTorch深度学习框架的官方文档，对于神经网络模块（`torch.nn`）、优化器（`torch.optim`）和数据工具（`DataLoader`、`TensorDataset`）的实际实现细节至关重要。
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html) — scikit-learn developers (2024)
  详细介绍了数据预处理技术（如`MinMaxScaler`）、`make_classification`数据集生成以及表格数据模型评估工具（如`train_test_split`和`accuracy_score`）的官方指南。
- [Reducing the Dimensionality of Data with Neural Networks](https://doi.org/10.1126/science.1127647) — Geoffrey E. Hinton, Ruslan R. Salakhutdinov (2006)
  Journal: Science; Publisher: American Association for the Advancement of Science; Volume: 313; Pages: 504-507; DOI: [10.1126/science.1127647](https://doi.org/10.1126/science.1127647)
  一篇奠基性论文，展示了深度自编码器在无监督降维和从高维数据中学习有用特征方面的有效性。
