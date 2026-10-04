---
course: "vae-representation-learning"
chapter: "advanced-vae-architectures"
lesson: "practice-advanced-vae-architectures"
sourceId: 6332
sourceUrl: "https://apxml.com/zh/courses/vae-representation-learning/chapter-3-advanced-vae-architectures/practice-advanced-vae-architectures"
title: "动手实践：实现高级VAE架构"
description: "实现并比较CVAE和VQ-VAE等多种高级VAE架构。"
order: 8
plots: []
sourceHash: "9901623ea331ecd532a770a7b4da2f3ade11c1f0c862b1a276d20fded9d63a1c"
sourceCorrections: []
---

提供了两种主要的高级VAE架构（条件VAE (CVAE) 和矢量量化 (quantization)VAE (VQ-VAE)）的实现细节。目的是提供构建、训练和评估这些模型所需的基础代码结构和知识，为处理更复杂的生成任务做好准备。

我们将重点说明实现这些高级变体所需的具体修改。在本次环节中，我们假设您使用MNIST或Fashion-MNIST等数据集，因为这些数据集能够清晰演示条件生成以及离散隐变量对样本质量的影响。您应该熟悉Python以及PyTorch或TensorFlow等深度学习 (deep learning)框架。

## 本次实践的前期准备

在开始之前，请确保您已具备：

1. 可运行的Python环境 (例如，Python 3.8+)。
2. 已安装深度学习 (deep learning)库：PyTorch (推荐版本 1.10+) 或 TensorFlow (推荐版本 2.5+)。
3. 熟悉使用您选择的框架构建和训练标准VAE。
4. 标准数据科学库：NumPy, Matplotlib (用于可视化)。

我们将提供高级别的代码结构和逻辑。您将根据您的具体框架和数据集进行调整。

## 实现条件VAE (CVAE)

条件VAE通过将条件信息（表示为 $c$）整合到生成和推理 (inference)过程中来扩展VAE框架。这使我们能够引导VAE生成具有由 $c$ 定义的特定属性的数据样本。例如，对于MNIST， $c$ 可以是数字标签（0-9），从而使我们能够请求CVAE生成特定数字的图像。

### CVAE架构

核心思想是使编码器 $q_\phi(z|x,c)$ 和解码器 $p_\theta(x|z,c)$ 都依赖于条件 $c$。

1. **条件表示**: 条件 $c$ (例如，类别标签) 通常在输入网络前被转换为数值格式，通常是one-hot编码。假设 $c_{embed}$ 是这种数值表示。
2. **编码器 $q_\phi(z|x,c)$**:

   - 输入：原始数据 $x$ 和条件 $c_{embed}$。
   - 修改：将 $c_{embed}$ 与 $x$ (如果 $x$ 被展平) 或与编码器内 $x$ 的中间特征表示进行拼接。
   - 输出：近似后验 $q_\phi(z|x,c)$ 的参数 (parameter) $(\mu, \log \sigma^2)$，该后验通常是高斯分布。
3. **解码器 $p_\theta(x|z,c)$**:

   - 输入：隐变量样本 $z$ (训练期间从 $q_\phi(z|x,c)$ 中采样，生成期间从 $p(z|c)$ 中采样) 和条件 $c_{embed}$。
   - 修改：在通过解码器网络之前，将 $c_{embed}$ 与 $z$ 进行拼接。
   - 输出：重构数据 $\hat{x}$ 的分布参数 (例如，图像的像素值)。

A diagram illustrating the CVAE structure:

> 条件变分自动编码器中的数据流。条件 $c$ 被整合到编码器和解码器中。

### CVAE目标函数

CVAE的目标函数是ELBO的条件版本：


$$
L_{CVAE}(\phi, \theta; x, c) = \mathbb{E}_{q_\phi(z|x,c)}[\log p_\theta(x|z,c)] - D_{KL}(q_\phi(z|x,c) || p(z|c))
$$


训练期间，我们最大化此 $L_{CVAE}$。

- 第一项是条件重构似然。
- 第二项是近似后验 $q_\phi(z|x,c)$ 与先验 $p(z|c)$ 之间的KL散度。
  通常，先验 $p(z|c)$ 被简化为标准正态分布 $p(z) = \mathcal{N}(0, I)$，特别是当条件 $c$ 主要影响解码器时。如果使用 $p(z|c)$，它可能是一个也依赖于 $c$ 的学习先验。

### 实现草图 (PyTorch伪代码示例)

```python
# 条件: 标签 (例如，MNIST的0到9的整数)
# 将标签转换为one-hot编码: c_embed

# 编码器
class CVAEEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, condition_dim, hidden_dim):
        super().__init__()
        # 定义层 (例如，nn.Linear, nn.Conv2d)
        # 示例: self.fc_x = nn.Linear(input_dim, hidden_dim)
        # 示例: self.fc_c = nn.Linear(condition_dim, hidden_dim)
        # 示例: self.fc_combined = nn.Linear(hidden_dim * 2, hidden_dim)
        # self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        # self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, c_embed):
        # h_x = F.relu(self.fc_x(x.view(x.size(0), -1))) # 如果是图像，展平 x
        # h_c = F.relu(self.fc_c(c_embed))
        # combined = torch.cat([h_x, h_c], dim=1)
        # h_combined = F.relu(self.fc_combined(combined))
        # mu = self.fc_mu(h_combined)
        # logvar = self.fc_logvar(h_combined)
        return mu, logvar

# 解码器
class CVAEDecoder(nn.Module):
    def __init__(self, latent_dim, condition_dim, hidden_dim, output_dim):
        super().__init__()
        # 定义层
        # 示例: self.fc_z = nn.Linear(latent_dim, hidden_dim)
        # 示例: self.fc_c = nn.Linear(condition_dim, hidden_dim)
        # 示例: self.fc_combined = nn.Linear(hidden_dim * 2, hidden_dim)
        # self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z, c_embed):
        # h_z = F.relu(self.fc_z(z))
        # h_c = F.relu(self.fc_c(c_embed))
        # combined = torch.cat([h_z, h_c], dim=1)
        # h_combined = F.relu(self.fc_combined(combined))
        # reconstruction = torch.sigmoid(self.fc_out(h_combined)) # 假设像素值使用sigmoid
        # return reconstruction.view(-1, num_channels, height, width) # 重塑为图像
        pass # 实际实现取决于网络设计

# 训练循环:
# 对于每个批次 (x_batch, c_batch_labels):
#   c_batch_embed = one_hot_encode(c_batch_labels)
#   mu, logvar = encoder(x_batch, c_batch_embed)
#   z_sampled = reparameterize(mu, logvar)
#   x_reconstructed = decoder(z_sampled, c_batch_embed)
#
#   reconstruction_loss = F.binary_cross_entropy(x_reconstructed, x_batch, reduction='sum')
#   kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#   loss = reconstruction_loss + kl_divergence
#   optimizer.zero_grad()
#   loss.backward()
#   optimizer.step()
```

关键在于编码器和解码器的适当位置拼接条件嵌入 (embedding) `c_embed`。

### 评估与生成

- **条件重构**: 给定输入图像 $x$ 的真实标签 $c$，对其进行重构。
- **条件生成**: 从 $p(z)$ (例如，$\mathcal{N}(0,I)$) 中采样 $z$，选择所需的条件 $c_{target}$，并生成 $\hat{x} = \text{decoder}(z, c_{target\_embed})$。您应该看到与 $c_{target}$ 对应的样本。对于MNIST，您可以按需生成特定数字的图像。

## 实现矢量量化 (quantization)VAE (VQ-VAE)

VQ-VAE通过将编码器的输出量化到学习到的码本（或嵌入 (embedding)空间）中最接近的向量 (vector)来引入离散隐空间。这常常生成比具有连续隐变量的标准VAE更清晰的样本，因为解码器学习将有限的表示集合映射到输出。

### VQ-VAE架构

1. **编码器 $E$**: 将输入 $x$ 映射到连续表示 $z_e(x)$。此输出通常是一个向量 (vector)张量，例如，如果 $x$ 是图像，则为 $H' \times W' \times D$ 的特征图。
2. **矢量量化 (quantization) (VQ) 层**:
   - **码本**: 一个可学习的嵌入 (embedding)空间 $E = \{e_1, e_2, \dots, e_K\}$，其中每个 $e_i \in \mathbb{R}^D$ 是一个嵌入向量。$K$ 是码本的大小。
   - **量化**: 对于编码器输出特征图中的每个向量 $z_{e,j}(x)$，找到最近的码本嵌入 $e_k$：
     
     $$
     k_j = \arg\min_i ||z_{e,j}(x) - e_i||_2
     $$
     
     $z_{e,j}(x)$ 的量化表示是 $z_{q,j}(x) = e_{k_j}$。
   - **直通估计器 (STE)**: 在反向传播 (backpropagation)期间，来自解码器的梯度 $\nabla_{z_q} L$ 被直接复制到编码器输出 $z_e(x)$，即 $\nabla_{z_e} L = \nabla_{z_q} L$。这使得梯度能够通过不可微分的 $\arg\min$ 操作进行传递。
3. **解码器 $D$**: 将量化的隐向量 $z_q(x)$ 映射回数据空间以生成 $\hat{x}$。

A diagram illustrating the VQ-VAE structure:

> 矢量量化变分自动编码器中的数据流。编码器输出 $z_e(x)$ 使用可学习的码本进行量化。直通估计器 (STE) 用于梯度传播。

### VQ-VAE目标函数

VQ-VAE通过最小化组合损失进行训练：


$$
L_{VQVAE} = L_{reconstruction} + L_{codebook} + \beta \cdot L_{commitment}
$$


其中：

1. **重构损失 $L_{reconstruction}$**: 衡量解码器从量化 (quantization)隐变量 $z_q(x)$ 重构输入 $x$ 的效果。对于图像，这通常是均方误差 (MSE)：
   
   $$
   L_{reconstruction} = ||x - D(z_q(x))||^2_2
   $$
   
2. **码本损失 $L_{codebook}$**: 旨在使码本向量 (vector) $e_i$ 更接近它们所映射的编码器输出 $z_e(x)$。它使用停止梯度 (`sg`) 操作来防止编码器输出任意增大：
   
   $$
   L_{codebook} = ||\text{sg}[z_e(x)] - e_k||^2_2
   $$
   
   这里，$e_k$ 是最接近 $z_e(x)$ 的码本向量。梯度只更新 $e_k$。
3. **承诺损失 $L_{commitment}$**: 旨在确保编码器承诺使用一个嵌入 (embedding)，并且其输出不会增大。它鼓励 $z_e(x)$ 接近其选择的码本向量 $e_k$：
   
   $$
   L_{commitment} = ||z_e(x) - \text{sg}[e_k]||^2_2
   $$
   
   超参数 (parameter) (hyperparameter) $\beta$ (通常在0.1到2.0之间，常取0.25) 控制此项的强度。梯度只更新 $z_e(x)$。

基本VQ-VAE中没有对先验的显式KL散度项。隐空间的离散性本身就是一种正则化 (regularization)形式。在VQ-VAE训练后，可以学习离散隐代码的先验（例如，使用PixelCNN）用于生成。

### 实现草图 (PyTorch伪代码示例)

```python
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # 初始化码本 (嵌入)
        # self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        # self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, inputs):
        # inputs: (批次, 通道, 高度, 宽度) -> (B*H*W, C)
        # flat_input = inputs.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)

        # 计算输入与所有码本向量的距离
        # distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
        #             + torch.sum(self.embedding.weight**2, dim=1)
        #             - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

        # 找到最近的编码 (索引)
        # encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        # quantized = self.embedding(encoding_indices).view(inputs.shape) # 获取量化向量

        # 计算损失
        # codebook_loss = F.mse_loss(quantized.detach(), inputs) # 原始论文中的 sg[inputs], 用于EMA更新。或 sg[quantized] 用于码本向量
        # commitment_loss = F.mse_loss(inputs, quantized.detach())
        # loss = codebook_loss + self.commitment_cost * commitment_loss

        # 直通估计器:
        # quantized = inputs + (quantized - inputs).detach() # STE

        # 将量化后的结果重新塑形回 (批次, 通道, 高度, 宽度)
        # return quantized, loss, encoding_indices.squeeze()
        pass # 实际实现需要仔细处理维度和STE

class VQVAE(nn.Module):
    def __init__(self, encoder, decoder, vq_layer):
        super().__init__()
        # self.encoder = encoder
        # self.vq_layer = vq_layer
        # self.decoder = decoder

    def forward(self, x):
        # z_e = self.encoder(x)
        # quantized_latents, vq_loss, _ = self.vq_layer(z_e)
        # x_reconstructed = self.decoder(quantized_latents)
        # return x_reconstructed, vq_loss
        pass

# 训练循环:
# 对于每个批次 (x_batch, _): # 除非评估需要，否则不需要标签
#   x_reconstructed, vq_loss = vq_vae_model(x_batch)
#
#   reconstruction_loss = F.mse_loss(x_reconstructed, x_batch)
#   total_loss = reconstruction_loss + vq_loss # vq_loss 已经包含码本损失和承诺损失
#   optimizer.zero_grad()
#   total_loss.backward()
#   optimizer.step()
```

`VectorQuantizer` 模块是其中最复杂的部分。请确保停止梯度（PyTorch中的`.detach()`）已正确应用于码本损失和承诺损失，并且在前向传播中实现了STE，以便梯度能够通过量化 (quantization)步骤流回编码器。

### 评估与生成

- **重构质量**: 由于离散隐变量瓶颈，VQ-VAE通常比标准VAE生成更清晰的重构。
- **生成**: 要生成新样本，您首先需要学习离散隐代码 $k$ 的先验 $p(k)$。这通常通过在从训练数据获得的 `encoding_indices` 序列上训练自回归 (autoregressive)模型（如PixelCNN或Transformer）来完成。一旦学习了 $p(k)$，就可以从中采样索引，获取相应的码本向量 (vector) $e_k$，并将其传递给解码器。

## CVAE与VQ-VAE的比较

CVAE和VQ-VAE相比基础VAE架构都有显著改进，但它们解决不同的问题，并具有独特的特点。

| 特征 | 条件VAE (CVAE) | 矢量量化 (quantization)VAE (VQ-VAE) |
| --- | --- | --- |
| **主要目标** | 基于属性的受控生成 | 改进样本保真度，离散隐变量表示 |
| **隐变量空间** | 连续，受 $c$ 条件约束 | 离散，有限集合的学习码本向量 (vector) |
| **控制机制** | 显式输入条件 $c$ | 通过学习到的码本结构隐式实现 |
| **样本质量** | 内容受 $c$ 控制；仍可能出现模糊 | 通常更清晰，更少模糊的样本；生成需要代码的先验 |
| **训练目标** | 条件ELBO ($L_{reconstruction} + D_{KL}$) | $L_{reconstruction} + L_{codebook} + \beta \cdot L_{commitment}$ |
| **梯度流** | 标准重参数 (parameter)化技巧 | 量化步骤使用直通估计器 |
| **常见挑战** | 确保条件 $c$ 被有效利用，后验坍塌 | 码本坍塌（未使用代码），选择 $K$ 和 $\beta$ |
| **生成过程** | 采样 $z \sim p(z)$，提供 $c$，解码 \$p\_\theta(x | z,c)\$ |

### 何时选择哪种？

- **CVAE**在以下情况更适用：
  - 您需要生成具有特定、可控属性的数据。
  - 希望隐变量空间对条件具有可解释性。
  - 应用于基于属性的风格迁移或数据增强等。
- **VQ-VAE**在以下情况是一个有力的选择：
  - 主要目标是高保真、清晰的样本生成。
  - 数据的离散表示有益（例如，用于下游任务或学习先验）。
  - 处理复杂数据时，连续隐变量可能导致过度平滑或模糊的输出（例如，高分辨率图像、音频）。

## 进一步实践

在CVAE和VQ-VAE的实现基础上，您可以延伸您的实践工作：

1. 在MNIST或Fashion-MNIST等数据集上实现这两种模型。
2. 定性比较两种模型的重构和生成样本。对于CVAE，测试其生成特定类别的能力。对于VQ-VAE，观察样本的清晰度。
3. 调整超参数 (parameter) (hyperparameter)进行实验：
   - 对于CVAE：隐变量维度、网络深度、条件如何整合。
   - 对于VQ-VAE：码本大小 ($K$)、嵌入 (embedding)维度 ($D$)、承诺成本 ($\beta$)。观察 $K$ 对样本多样性和重构质量的影响。
4. 可视化隐变量空间：对于CVAE，尝试可视化不同条件如何映射到隐变量空间的不同区域（例如，使用t-SNE对根据条件 $c$ 染色过的 $z$ 样本进行可视化）。对于VQ-VAE，分析码本使用情况。
5. 组合架构：思考如何结合这些思想，例如，一个条件VQ-VAE (C-VQ-VAE)，其中码本或其使用方式可以被条件化。
6. 研究其他架构：回顾本章讨论的其他模型，例如分层VAE或 $\beta$-VAE，并思考实现它们所需的具体架构或损失函数 (loss function)变化。修改编码器、解码器、隐变量空间或损失函数的原理是所有这些高级VAE的根本所在。

本次动手实践对于更透彻地理解架构选择如何影响变分自动编码器的行为和能力非常有帮助。通过构建和实验，您将能更好地选择、设计和调整VAE模型，以应对您具体的表示学习和生成建模任务。

## 参考资料

- [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114.pdf) — Diederik P. Kingma, Max Welling (2013)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1312.6114](https://doi.org/10.48550/arXiv.1312.6114)
  介绍了变分自编码器（VAE），提供了所有讨论的先进VAE架构的基础理论和重参数化技巧。
- [Neural Discrete Representation Learning](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFtckXJ4reCLP4NaZZTPvnC-N5xqAvwORf2p1-7WJlhlcESnBg4saAEGjdFZwDofhz6I5G-t8_gRZ2_kP8G7FRRCRiujZ3pKnJ_cR51-9k_sBibmL8uIey-T1pd9GYwRrsq2OCTDaPGzHXp7HgWLJoEKTHkhT5YFZ5CxZpYLzhgpRpjBMDelen6sebPFMzQiQ==) — Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu (2017)
  Journal: Advances in Neural Information Processing Systems; Volume: 30; Pages: 6306–6315; DOI: [10.5591/978-1-57753-855-5.1](https://doi.org/10.5591/978-1-57753-855-5.1)
  介绍了向量量化变分自编码器（VQ-VAE），阐释了其新颖的离散潜在空间、码本学习以及用于梯度流的直通估计器。
