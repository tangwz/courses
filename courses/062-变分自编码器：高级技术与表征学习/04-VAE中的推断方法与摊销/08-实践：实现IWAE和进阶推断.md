# 实践：实现IWAE和进阶推断

来源：[原文](https://apxml.com/zh/courses/vae-representation-learning/chapter-4-vae-inference-techniques/practice-iwaes-advanced-inference)

[返回章节目录](README.md) · [返回课程目录](../README.md)

增强变分自编码器（VAEs）的推断能力有多种方法。近似后验分布 $q_\phi(z|x)$ 的质量对 VAE 性能有很大作用。虽然摊销推断提供了效率，但像重要性加权自编码器（IWAEs）这样的方法提供了一种实现更紧密证据下界（ELBO）和可能更准确的后验近似的方式。一个 IWAE 的实现及其作用分析将被呈现。其他进阶推断策略的考虑因素也将被讨论。

### 理解 IWAE 目标函数

请回顾，标准 VAE 最大化 ELBO：


$$
\mathcal{L}_{ELBO}(x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) || p(z))
$$


Burda 等人（2015）提出的 IWAE，通过为每个数据点 $x$ 使用来自近似后验分布 $q_\phi(z|x)$ 的多个样本，为对数边缘似然 $\log p_\theta(x)$ 提供了一个更紧密的下界。IWAE 目标函数，通常记作 $\mathcal{L}_K$，为：


$$
\mathcal{L}_K(x) = \mathbb{E}_{z_1, ..., z_K \sim q_\phi(z|x)} \left[ \log \left( \frac{1}{K} \sum_{k=1}^K \frac{p_\theta(x, z_k)}{q_\phi(z_k|x)} \right) \right]
$$


此处 $z_k$ 是从 $q_\phi(z|x)$ 中抽取的 $K$ 个独立样本。这可被重写为更便于实现的形式：


$$
\mathcal{L}_K(x) = \mathbb{E}_{z_1, ..., z_K \sim q_\phi(z|x)} \left[ \log \left( \frac{1}{K} \sum_{k=1}^K \exp(\log p_\theta(x|z_k) + \log p(z_k) - \log q_\phi(z_k|x)) \right) \right]
$$


请注意，当 $K=1$ 时，$\mathcal{L}_1$ 会恢复为标准 ELBO（这与实际中期望的处理方式存在细微的差别，但对于优化来说，它们非常相似）。随着 $K \to \infty$，$\mathcal{L}_K \to \log p_\theta(x)$。

### 实现 IWAE

我们来概述将标准 VAE 实现（你可能在第2章的实践练习中已有此实现）修改为 IWAE 的步骤。我们假设你有一个编码器，它为 $q_\phi(z|x)$ 输出参数 (parameter)（例如，均值 $\mu_\phi(x)$ 和对数方差 $\log \sigma^2_\phi(x)$），以及一个解码器，它对 $p_\theta(x|z)$ 进行建模。

#### 1. 采样多个潜在变量

对于小批量中的每个输入 $x$，你不需要只从 $q_\phi(z|x)$ 中抽取一个样本 $z$，而是需要抽取 $K$ 个样本。如果你的编码器为高斯 $q_\phi(z|x)$ 输出 $\mu_\phi(x)$ 和 $\sigma_\phi(x)$，则重参数化技巧将被应用 $K$ 次：
$z_k = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon_k$，这里 $\epsilon_k \sim \mathcal{N}(0, I)$，对于 $k=1, \dots, K$。

在张量操作方面（例如，在 PyTorch 或 TensorFlow 中）：

- 如果你的编码器输出形状为 `(batch_size, latent_dim)` 的 `mu` 和 `logvar`。
- 你需要将它们扩展到 `(batch_size, K, latent_dim)` 的形状，或者以允许每个输入 $K$ 个样本的方式处理它们。
- 生成形状为 `(batch_size, K, latent_dim)` 的 `eps`。
- 计算形状为 `(batch_size, K, latent_dim)` 的 `z_samples`。

```python
# 伪代码/PyTorch 风格的示例
# mu, logvar 的形状: (batch_size, latent_dim)
# K: 重要性样本的数量

# 为 K 个样本扩展 mu 和 logvar
mu_expanded = mu.unsqueeze(1).expand(-1, K, -1)  # (batch_size, K, latent_dim)
logvar_expanded = logvar.unsqueeze(1).expand(-1, K, -1) # (batch_size, K, latent_dim)
std_expanded = torch.exp(0.5 * logvar_expanded)

# 采样 epsilon
epsilon = torch.randn_like(std_expanded) # (batch_size, K, latent_dim)

# 为每个输入生成 K 个潜在样本
z_samples = mu_expanded + std_expanded * epsilon # (batch_size, K, latent_dim)
```

#### 2. 计算每个样本的对数权重 (weight)

对于每个样本 $z_k$，我们需要计算其未归一化 (normalization)的对数重要性权重 $w'_k$：


$$
\log w'_k = \log p_\theta(x|z_k) + \log p(z_k) - \log q_\phi(z_k|x)
$$


- $\log p_\theta(x|z_k)$: 重构对数似然。这涉及到将每个 $z_k$ 通过解码器，以获取 $p_\theta(x|z_k)$ 的参数，然后计算 $x$ 的对数概率。如果 $x$ 为解码器进行了形状调整，请确保它与 $K$ 个样本对齐 (alignment)。例如，如果 $x$ 的形状为 `(batch_size, data_dim)`，则在计算每个样本的重构损失时，可能需要将其扩展为 `(batch_size, K, data_dim)` 以匹配 `z_samples`。
- $\log p(z_k)$: 先验对数概率。通常，$p(z) = \mathcal{N}(0, I)$，因此这很容易计算。
- $\log q_\phi(z_k|x)$: 在近似后验 $q_\phi(z|x)$ 下的对数概率。

这些分量将得到一个形状为 `(batch_size, K)` 的对数权重张量，例如 `log_w_prime`。

#### 3. 使用 Log-Sum-Exp 进行平均

IWAE 目标函数涉及 $\log (\frac{1}{K} \sum_k \exp(\log w'_k))$。直接对指数项求和可能导致数值下溢或上溢。`log-sum-exp` (LSE) 技巧在这里非常重要：


$$
\log \left( \sum_{k=1}^K \exp(a_k) \right) = \alpha + \log \left( \sum_{k=1}^K \exp(a_k - \alpha) \right)
$$


这里 $\alpha = \max_k a_k$。
单个数据点 $x$ 的 IWAE 损失为：


$$
\mathcal{L}_K(x) = \text{LSE}(\log w'_1, \dots, \log w'_K) - \log K
$$


小批量的最终损失是批次中所有 $x$ 的 $\mathcal{L}_K(x)$ 的平均值。

```python
# 伪代码/PyTorch 风格的 IWAE 损失项示例
# log_p_x_given_z: (batch_size, K)，每个样本的重构对数似然
# log_p_z: (batch_size, K)，每个样本的先验对数概率
# log_q_z_given_x: (batch_size, K)，每个样本的近似后验对数概率

log_w_prime = log_p_x_given_z + log_p_z - log_q_z_given_x # (batch_size, K)

# 为了数值稳定性进行 Log-sum-exp
log_sum_exp_w = torch.logsumexp(log_w_prime, dim=1) # (batch_size,)

# 每个数据点的 IWAE 目标函数
iwae_elbo_per_sample = log_sum_exp_w - torch.log(torch.tensor(K, dtype=torch.float32))

# 在批次上求平均
batch_iwae_elbo = torch.mean(iwae_elbo_per_sample)

# 需要最小化的损失是 -batch_iwae_elbo
loss = -batch_iwae_elbo
```

> **关于维度：** 仔细管理张量维度。当你将 $z_k$（形状为 `(batch_size, K, latent_dim)`）传递给解码器时，它可能会将其处理为 `(batch_size * K, latent_dim)`。然后，输出的重构结果将是 `(batch_size * K, data_dim)`。你需要将 $x$ 的形状调整以匹配这一点，从而计算 $\log p_\theta(x|z_k)$，然后将得到的对数似然调整回 `(batch_size, K)`。

### 实验与分析

一旦你实现了 IWAE，是时候进行实验了：

1. **改变 $K$ 值**：使用不同的 $K$ 值（例如，$K=1, 5, 10, 50$）训练 IWAE 模型。

   - **标准 VAE 作为基准**：$K=1$ 的情况有效地模拟了你的标准 VAE（尽管如前所述，估计器在理论上存在细微的差异，但实践中它常被用作 VAE 的基准）。
   - **监控边界**：在验证集上绘制报告的 $\mathcal{L}_K$ 值。你会发现 $\mathcal{L}_K$ 通常随 $K$ 增加，这表明边界变得更紧密。
   - **重构质量**：直观检查重构结果。随着 $K$ 增加，它们是否更清晰？如果合适，用 MSE 进行量化 (quantization)，但请注意 IWAE 优化的是 $\log p(x)$ 的下界，而不是直接的重构误差。
   - **样本质量**：从先验 $p(z)$ 中生成样本并进行解码。生成样本的质量是否随 $K$ 的增加而提高？
   - **训练时间**：注意随着 $K$ 增加，每个 epoch 的训练时间也会增加。这是一个直接的权衡。

   
   
   [交互图表：IWAE 性能对比 K](https://apxml.com/zh/courses/vae-representation-learning/chapter-4-vae-inference-techniques/practice-iwaes-advanced-inference#plot-1vvxsxy)
   
   <details>
   <summary>图表数据（JSON）</summary>
   
   ```json
   {
     "data": [
       {
         "x": [
           1,
           5,
           10,
           25,
           50
         ],
         "y": [
           -120,
           -110,
           -105,
           -102,
           -100
         ],
         "type": "scatter",
         "mode": "lines+markers",
         "name": "验证集 L_K"
       },
       {
         "x": [
           1,
           5,
           10,
           25,
           50
         ],
         "y": [
           30,
           140,
           280,
           650,
           1300
         ],
         "type": "scatter",
         "mode": "lines+markers",
         "name": "训练时间 (秒/epoch)",
         "yaxis": "y2"
       }
     ],
     "layout": {
       "title": "IWAE 性能对比 K",
       "xaxis": {
         "title": "重要性样本数量 (K)"
       },
       "yaxis": {
         "title": "验证集下界 (L_K)",
         "color": "#228be6"
       },
       "yaxis2": {
         "title": "训练时间 (秒/epoch)",
         "overlaying": "y",
         "side": "right",
         "color": "#f06595"
       },
       "legend": {
         "x": 0.05,
         "y": 0.95
       }
     }
   }
   ```
   
   </details>
   
   

   > 上图说明了一个典型趋势：随着 $K$ 的增加，IWAE 下界（$\mathcal{L}_K$）得到提升（变得不那么负），但每个 epoch 的计算成本也随之增加。
2. **活跃单元**：如果你对解耦或表示学习（在第5章中会讲到）也感兴趣，请考察使用 IWAE（当 $K > 1$ 时）是否会影响潜在空间中“活跃单元”的数量，与标准 VAE 相比。有时，更紧密的边界可以阻止过早的 KL 消失。
3. **后验坍缩**：对于容易出现后验坍缩（即 $q_\phi(z|x)$ 变得与 $p(z)$ 非常相似，导致潜在变量不提供信息）的模型，使用 $K > 1$ 的 IWAE 是否有助于缓解这个问题？IWAE 提供的更准确的梯度可能提供更好的优化路径。

### 其他进阶推断方法的考虑

IWAE 侧重于通过多样本改进边界，而其他方法则修改 $q_\phi(z|x)$ 的结构或推断过程本身：

- **结构化变分推断**：

  - **归一化 (normalization)流**：在编码器中实现带有归一化流的 VAE 涉及将简单的 Gaussian 样本转换为更复杂的分布。你会在从 $N(\mu_\phi(x), \sigma^2_\phi(x))$ 进行初始采样后插入流层（例如，平面流、径向流，或更进阶的自回归 (autoregressive)流如 MAF 或 IAF）。主要挑战是计算这些变换的雅可比行列式的对数，这需要加到 $\log q_\phi(z_0|x)$ 中以获得 $\log q_\phi(z_K|x)$（这里 $z_0$ 是基础样本，$z_K$ 是变换后的样本）。
  - **自回归后验**：不同于对角协方差 Gaussian，$q_\phi(z|x) = \prod_i q_\phi(z_i | z_{<i}, x)$。这要求编码器使用自回归网络（例如 RNN 或掩码自编码器）。
- **辅助变量（例如，半摊销 VI，分层 VAEs）**：

  - 引入辅助变量 $u$ 来扩展潜在空间，例如 $q_\phi(z, u | x) = q_\phi(u|x)q_\phi(z|x,u)$。这通常会产生表达能力更强的后验。实现上涉及为这些辅助变量设计推断网络，并修改 ELBO 以考虑它们。
  - 对于半摊销 VI，你可能需要从一个摊销提议开始，执行几步优化（例如 SGD）来精炼每个数据点的 $z$。这计算量更大，但可以得到非常准确的后验。
- **对抗变分贝叶斯 (AVB)**：

  - AVB 用一个对抗鉴别器替换了 ELBO 中的 KL 散度项，该鉴别器试图区分来自 $q_\phi(z|x)$ 的样本和来自真实先验 $p(z)$ 的样本。编码器随后被训练来欺骗这个鉴别器。这需要在你的 VAE 框架内设置一个 GAN 类似的训练循环。

实现这些进阶方法通常涉及对概率建模的更细致研究和仔细的网络架构设计。共同点是转向捕捉更复杂的后验结构，或获取模型证据的更好估计。

### 后续展望

这项关于 IWAE 的实践练习应能为你理解如何改进 VAE 推断提供扎实的基础。通过使得 ELBO 更紧密，IWAE 可以带来更好的生成模型和更忠实的潜在表示。增加的计算成本是一种权衡，但通常，一个适中的 $K$ 值（例如 5-10）可以提供一个良好的平衡。

随着你的学习深入，请思考 IWAE 的原理（通过多个样本获得更好估计）或其他进阶方法的结构改进，如何可以结合或适用于你的特定 VAE 应用。批判性评估和改进推断机制的能力是进阶 VAE 开发的一个标志。

## 参考资料

- [Importance Weighted Autoencoders](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHCpM7b8sUp8A8HnJkRCnm8w2SWe9jC1CpR6VriRE9qA06kO7xFsKR_NKNhnKP6raTw9D2RynYLFISgtrrtJ61fiK-fKYMhM7Rl1A1xRDxZx9sR-vx9vGik3fk=) — Yuri Burda, Roger Grosse, Ruslan Salakhutdinov (2016)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1502.03388](https://doi.org/10.48550/arXiv.1502.03388)
  介绍了重要性加权自编码器 (IWAE)，并展示了如何使用多个重要性样本来提供比标准 VAE ELBO 更紧密的边际似然下界，从而提高模型性能。
- [Auto-Encoding Variational Bayes](https://openreview.net/forum?id=bU3Vq5b1m) — Diederik P. Kingma, Max Welling (2014)
  Journal: ICLR; DOI: [10.48550/arXiv.1312.6114](https://doi.org/10.48550/arXiv.1312.6114)
  这篇基础性论文介绍了变分自编码器 (VAE)，确立了证据下界 (ELBO) 和重参数化技巧，用于高效训练深度生成模型。
- [Variational Inference with Normalizing Flows](https://arxiv.org/abs/1505.05770) — Danilo Jimenez Rezende, Shakir Mohamed (2015)
  Journal: International Conference on Machine Learning (ICML); Publisher: JMLR.org; Pages: 1530–1538; DOI: [10.48550/arXiv.1505.05770](https://doi.org/10.48550/arXiv.1505.05770)
  提出了归一化流 (Normalizing Flows) 作为一种在变分推断中构建更具表达力的近似后验的方法，增强了推断模型超越简单高斯分布的灵活性。
- [Adversarial Variational Bayes: Unifying Variational Autoencoders and Generative Adversarial Networks](https://proceedings.mlr.press/v70/mescheder17a.html) — Lars Mescheder, Sebastian Nowozin, Andreas Geiger (2017)
  Journal: International Conference on Machine Learning (ICML); Publisher: Proceedings of Machine Learning Research; Pages: 2391-2400; DOI: [10.48550/arXiv.1701.04722](https://doi.org/10.48550/arXiv.1701.04722)
  引入了对抗变分贝叶斯 (AVB)，这种方法将 VAE 目标函数中的 KL 散度项替换为对抗性目标，以使近似后验与先验相匹配。

---

[上一节](07-%E5%AF%B9%E6%8A%97%E5%8F%98%E5%88%86%E8%B4%9D%E5%8F%B6%E6%96%AF%20%28AVB%29.md) · [下一节](../05-%E8%A7%A3%E8%80%A6%E8%A1%A8%E7%A4%BA%E5%AD%A6%E4%B9%A0%E4%B8%8E%E5%8F%98%E5%88%86%E8%87%AA%E7%BC%96%E7%A0%81%E5%99%A8%20%28VAEs%29/01-%E5%AE%9A%E4%B9%89%E8%A7%A3%E8%80%A6%EF%BC%9A%E5%85%AC%E5%BC%8F%E5%8C%96%E4%B8%8E%E9%9A%BE%E9%A2%98.md)
