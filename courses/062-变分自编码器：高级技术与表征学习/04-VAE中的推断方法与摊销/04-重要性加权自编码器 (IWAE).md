# 重要性加权自编码器 (IWAE)

来源：[原文](https://apxml.com/zh/courses/vae-representation-learning/chapter-4-vae-inference-techniques/importance-weighted-autoencoders-iwaes)

[返回章节目录](README.md) · [返回课程目录](../README.md)

标准证据下界（ELBO）是训练变分自编码器（VAE）的核心。然而，根据定义，ELBO是对数边缘似然 $\log p_\theta(x)$ 的一个*下界*。ELBO与真实对数似然之间的差距正是KL散度 $KL(q_\phi(z|x) || p_\theta(z|x))$。一个更简单的近似后验 $q_\phi(z|x)$ 通常会导致更大的差距，这意味着下界更宽松。虽然使 $q_\phi(z|x)$ 更具表达能力（例如，使用即将介绍的归一化 (normalization)流）是收紧此下界的一种方式，但重要性加权自编码器（IWAE）提供了一种替代方法：通过从*相同*的 $q_\phi(z|x)$ 中使用多个样本来获得更紧密的下界。

### IWAE目标：使用多样本获得更紧密的下界

IWAE的核心思想（由Burda、Grosse和Salakhutdinov提出）是使用近似后验 $q_\phi(z|x)$ 中的多个样本来形成对 $p_\theta(x)$ 更好的蒙特卡洛估计。回想一下，边缘似然可以写成 $p_\theta(x) = \mathbb{E}_{z \sim q_\phi(z|x)} \left[ \frac{p_\theta(x,z)}{q_\phi(z|x)} \right]$。
标准ELBO是 $\mathbb{E}_{z \sim q_\phi(z|x)} \left[ \log \frac{p_\theta(x,z)}{q_\phi(z|x)} \right]$。Jensen不等式告诉我们 $\log \mathbb{E}[Y] \ge \mathbb{E}[\log Y]$，这就是ELBO是一个下界的原因。

IWAE目标，表示为 $L_K(x)$，对于给定的输入 $x$，使用从 $q_\phi(z|x)$ 中独立抽取的 $K$ 个样本 $z_1, \dots, z_K$：


$$
L_K(x) = \mathbb{E}_{z_1, \dots, z_K \sim q_\phi(z|x)} \left[ \log \left( \frac{1}{K} \sum_{k=1}^K \frac{p_\theta(x, z_k)}{q_\phi(z_k|x)} \right) \right]
$$


每个项 $w_k = \frac{p_\theta(x, z_k)}{q_\phi(z_k|x)}$ 是一个重要性权重 (weight)。IWAE目标在取对数*之前*对这些权重进行平均。这与标准ELBO存在细微但重要的差异。

$L_K$ 目标具有几个良好性质：

1. **单调改进的下界**：对于任何 $K \ge 1$，我们有：
   
   $$
   \log p_\theta(x) \ge L_K(x) \ge L_{K-1}(x)
   $$
   
   这意味着，随着样本数量 $K$ 的增加，IWAE目标为真实对数边缘似然提供了一个逐步收紧的下界。
2. **与ELBO的关联**：当 $K=1$ 时，IWAE目标 $L_1(x)$ 正好是标准ELBO：
   
   $$
   L_1(x) = \mathbb{E}_{z_1 \sim q_\phi(z|x)} \left[ \log \left( \frac{p_\theta(x, z_1)}{q_\phi(z_1|x)} \right) \right]
   $$
   
3. **收敛到真实对数似然**：当 $K \to \infty$ 时，IWAE目标 $L_K(x)$ 收敛到真实对数边缘似然 $\log p_\theta(x)$。这是因为 $\frac{1}{K} \sum_{k=1}^K w_k$ 成为 $p_\theta(x)$ 的一个更好的估计。

下图呈现了使用 $K$ 个样本计算IWAE目标的一般过程。

> 从推断网络 $q_\phi(z|x)$ 中抽取多个样本 $z_1, \dots, z_K$。每个样本都贡献一个重要性权重 $w_k = p_\theta(x|z_k)p(z_k)/q_\phi(z_k|x)$。这些权重被平均，并且这个平均值的对数构成了IWAE目标 $L_K$。

### 直觉：为何先平均权重 (weight)很重要

标准ELBO ($L_1$) 可以被认为是 $\log(w_k)$ 的平均。IWAE目标 ($L_K$) 是先平均 $w_k$，*然后*取对数：$\log(\text{avg}(w_k))$。由于Jensen不等式 ($\log \mathbb{E}[Y] \ge \mathbb{E}[\log Y]$)，这种改变直接导致了更紧密的下界。

直观上看，$q_\phi(z|x)$ 可能不是真实后验 $p_\theta(z|x)$ 的良好近似。它可能会在 $p_\theta(z|x)$ 值高的区域分配较低的概率密度。如果我们只取一个样本 $z_1$（如标准ELBO计算中），并且它落在一个估计不佳的区域，那么我们就会得到该数据点的一个糟糕估计。
使用IWAE，即使 $K$ 个样本中的一个 $z_k$ 恰好落入真实后验概率较高的区域（从而获得较大的重要性权重 $w_k$），它也能显著提高平均值 $\frac{1}{K}\sum w_j$。这使得整体估计对来自 $q_\phi(z|x)$ 的任何单个“不幸”样本的敏感度降低。推断网络 $q_\phi(z|x)$ 仍然“简单”，但下界的多样本估计更加稳定。

### 使用IWAE的益处

通过最大化 $L_K$（对于 $K>1$）而不是标准ELBO ($L_1$) 来训练VAE，可带来多项优势：

1. **更紧密的对数似然下界**：这是最直接的益处。与使用ELBO训练的VAE相比，IWAE通常在测试数据上报告更好的（即更高的）对数似然估计，尤其是在评估时使用相同的 $K$ 值。
2. **模型学习的改进**：优化更紧密的下界可以为生成模型 $p_\theta(x|z)$（解码器）和推断网络 $q_\phi(z|x)$（编码器）带来更好的参数 (parameter)更新。已经观察到，使用IWAE训练的模型可以学习到更丰富、信息量更大的潜在表示，并生成更高质量的样本。编码器 $q_\phi(z|x)$ 可能会学习提出更具多样性或能更好覆盖真实后验模式的样本 $z_k$，因为这些样本将更有效地促成高 $L_K$ 值。
3. **模型架构不变**：IWAE实现这些益处无需更复杂的推断网络架构（如结构化VI或归一化 (normalization)流所需）。$q_\phi(z|x)$ 的表达能力保持不变；改变的是目标函数。

### 计算成本与考量

IWAE的主要缺点是其在训练和评估期间增加了计算成本。
对于每个数据点和每个梯度步骤：

- 你需要从 $q_\phi(z|x)$ 中采样 $K$ 个潜在变量 $z_k$。
- 你需要对每个 $z_k$ 执行 $K$ 次解码器前向传播以计算 $p_\theta(x|z_k)$。
- 你需要对每个 $z_k$ 计算 $q_\phi(z_k|x)$。

这意味着计算成本随 $K$ 线性增长。如果 $K=50$，一个训练周期将比标准VAE长大约50倍，前提是解码器是瓶颈。

**$K$ 的选择**：
样本数量 $K$ 是一个超参数 (parameter) (hyperparameter)。

- 较小的 $K$（例如，5到10）：在计算量可控的情况下，对下界提供了适度的改进。
- 较大的 $K$（例如，50、100，甚至评估时可达5000）：提供了更紧密的下界，接近真实对数似然，但计算成本显著。
  在实践中，训练通常使用中等 $K$ 值（例如，$K=5$ 或 $K=50$），而评估可能会使用更大的 $K$ 值以获得更准确的对数似然估计。

**梯度方差**：
虽然 $L_K$ 是一个更紧密的下界，但其梯度估计器的方差有时可能成为问题，特别是当 $K$ 非常大或重要性权重 (weight)高度偏斜时。然而，更紧密下界带来的益处通常超过了这一担忧，并且像对 $z_k$ 使用重参数化技巧这样的技术对于保持梯度方差可控非常重要。

### 使用IWAE目标进行训练

使用IWAE目标训练VAE与训练标准VAE相似，主要区别在于损失计算：

1. **前向传播**：
   - 对于输入 $x$，获取 $q_\phi(z|x)$ 的参数 (parameter)（例如，如果 $q$ 是高斯分布，则为 $\mu(x), \sigma(x)$）。
   - 使用重参数化技巧从 $q_\phi(z|x)$ 中抽取 $K$ 个样本 $z_k$。对于每个 $z_k$：
     - 计算 $\log q_\phi(z_k|x)$。
     - 将 $z_k$ 通过解码器以获得 $p_\theta(x|z_k)$。
     - 计算 $\log p(z_k)$（来自先验）。
     - 计算未归一化 (normalization)的对数权重 (weight)：$\log \tilde{w}_k = \log p_\theta(x|z_k) + \log p(z_k) - \log q_\phi(z_k|x)$。
2. **目标计算**：
   - 计算IWAE目标 $L_K$。为了避免非常小的权重引起的数值问题，通常使用对数和指数技巧（log-sum-exp trick）：

   
   $$
   L_K(x) = \text{LogSumExp}(\log \tilde{w}_1, \dots, \log \tilde{w}_K) - \log K
   $$
   
   其中 $\text{LogSumExp}(a_1, \dots, a_K) = \log \sum_{j=1}^K \exp(a_j)$。
3. **反向传播 (backpropagation)**：计算 $-L_K(x)$ 相对于 $\theta$ 和 $\phi$ 的梯度，并更新参数。

VAE的其他部分，包括编码器和解码器的网络架构，基本保持不变。

### 总结与何时使用IWAE

重要性加权自编码器通过使用多个重要性样本，提供了一种有原则的方法来获得VAE对数似然的更紧密下界。这通常意味着模型性能的提升，例如更高质量的生成样本和更有意义的潜在表示，而无需明确增加推断网络 $q_\phi(z|x)$ 的复杂性。

在以下情况下可考虑使用IWAE：

- 当你需要更准确的数据对数似然估计时。
- 当你怀疑标准ELBO的宽松性是模型性能的限制因素时。
- 当你有足够的计算预算来承担每个样本计算量 $K$ 倍的增加时。
- 当你希望提升VAE性能，但不想立即转向更复杂的推断网络架构时。

IWAE代表了从基本VAE目标向前迈出的重要一步。它们表明，重新思考目标本身的估计，而不仅仅是模型组件，可以为生成建模带来很大的改进。接下来，我们将考察直接增强近似后验 $q_\phi(z|x)$ 表达能力的技术。

## 参考资料

- [Importance Weighted Autoencoders](https://arxiv.org/abs/1509.00519) — Yuri Burda, Roger Grosse, Ruslan Salakhutdinov (2016)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1509.00519](https://doi.org/10.48550/arXiv.1509.00519)
  介绍了用于VAE中更紧密似然下界的IWAE目标的原论文。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 20
  一本内容全面的教科书，提供了变分自编码器、ELBO和相关生成模型的背景知识。

---

[上一节](03-VAE%20%E4%B8%AD%E7%9A%84%E7%BB%93%E6%9E%84%E5%8C%96%E5%8F%98%E5%88%86%E6%8E%A8%E6%96%AD.md) · [下一节](05-%E8%BE%85%E5%8A%A9%E5%8F%98%E9%87%8F%E4%B8%8E%E5%8D%8A%E5%88%86%E6%91%8A%E5%8F%98%E5%88%86%E6%8E%A8%E6%96%AD.md)
