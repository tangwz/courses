# FactorVAE 和总相关性 VAE (TCVAE)

来源：[原文](https://apxml.com/zh/courses/vae-representation-learning/chapter-3-advanced-vae-architectures/factorvaes-tcvaes)

[返回章节目录](README.md) · [返回课程目录](../README.md)

Beta-VAE 提供了一种机制，通过增加 ELBO 中 $KL(q_\phi(z|x) || p(z))$ 项的权重 (weight) (eta > 1) 来促成分离。然而，它们是通过统一惩罚此 KL 散度的所有方面来做到这一点的。这有时可能导致不理想的权衡，可能过度惩罚对学习有意义表示很重要的项，或过度简化了后验分布。FactorVAE 和总相关性 VAE (TCVAE) 提供更具针对性的方法。这些方法旨在直接处理与分离表示相关的一个主要统计属性：潜在因子之间的独立性。这是通过关注一个称为总相关性（Total Correlation）的量来实现的。

### 理解总相关性 (TC)

FactorVAE 和 TCVAE 都围绕着 **总相关性 (TC)** 的理念。对于一组随机变量 $z = (z_1, z_2, ..., z_D)$，其联合分布为 $q(z)$，边缘分布为 $q(z_j)$，总相关性被定义为联合分布与其边缘分布乘积之间的 Kullback-Leibler 散度：


$$
TC(z) = KL(q(z) || \prod_{j=1}^D q(z_j))
$$


在 VAE 的背景下，$q(z)$ 通常指的是**聚合后验分布**，$q(z) = \int q_\phi(z|x) p_{data}(x) dx$。此分布表示编码器处理整个数据集时生成的潜在编码的整体分布。

本质上，TC 量化 (quantization)了变量 $z_j$ 之间的统计依赖程度。

- 如果潜在变量 $z_j$ 完全独立，那么 $q(z) = \prod_j q(z_j)$，且 $TC(z) = 0$。
- 如果 $z_j$ 之间存在依赖关系，那么 $q(z)$ 将与 $\prod_j q(z_j)$ 不同，且 $TC(z) > 0$。

分离的目标是让每个潜在维度 $z_j$ 对应数据中单一且独立的变异因子。最小化 $TC(z)$ 直接促使这些潜在维度在统计上变得独立，这是分离的一个有力替代指标。

> 总相关性 (TC) 量化了潜在空间 $z$ 维度之间统计依赖的程度。较低的 TC（右侧）表明潜在因子 $z_j$ 更加独立，这是分离表示的一种理想属性。例如，形状、颜色和大小可能被学习为独立的因子。

### FactorVAE：使用判别器惩罚总相关性

FactorVAE 将总相关性的直接惩罚引入 VAE 目标函数。FactorVAE 的目标是：


$$
L_{FactorVAE} = E_{p_{data}(x), q_\phi(z|x)}[\log p_\theta(x|z)] - KL(q_\phi(z|x) || p(z)) - \lambda \cdot TC(q(z))
$$


这里，$KL(q_\phi(z|x) || p(z))$ 是标准的 VAE KL 散度项，用于正则化 (regularization)每个样本的后验分布。新的项 $\lambda \cdot TC(q(z))$ 明确惩罚了聚合后验分布 $q(z)$ 的总相关性，其中 $\lambda$ 是一个超参数 (parameter) (hyperparameter)，控制此惩罚的强度。

一个主要难点在于 $TC(q(z))$ 无法直接计算，因为 $q(z)$ 本身难以处理（它是数据分布上的积分）。FactorVAE 提出了一个巧妙的解决方案：使用判别器网络 $D(z)$ 来估计 $TC(q(z))$。

该过程包括：

1. **判别器采样**：
   - 从 $q(z)$ 中采样：这些样本的获取方式是：首先从 $p_{data}(x)$ 中采样一个数据点 $x$（即从您的训练批次中），然后从 $q_\phi(z|x)$ 中采样 $z$。
   - 从 $\prod_j q(z_j)$ 中采样：这些样本的生成方式是：首先如上所述从 $q(z)$ 中采样一批 $z$。然后，对于每个维度 $j$，将批次中样本的 $z_j$ 值随机排列。这打破了维度之间的依赖关系，同时保留了边缘分布 $q(z_j)$。
2. **训练判别器**：判别器 $D(z)$ 经过训练，以区分从 $q(z)$ 中抽取的样本（标记 (token)为“真实”或类别 1）和从 $\prod_j q(z_j)$ 中抽取的样本（标记为“伪造”或类别 0）。判别器的目标通常是二元交叉熵损失：
   
   $$
   L_D = - E_{z \sim q(z)}[\log D(z)] - E_{z' \sim \prod_j q(z_j)}[\log(1-D(z'))]
   $$
   
3. **估计 TC**：判别器训练完成后，可以使用其输出近似 TC 项。一种常见的近似方法是：
   
   $$
   TC(q(z)) \approx E_{z \sim q(z)}[\log D(z) - \log(1-D(z))]
   $$
   
   然后将此项（或其变体）添加到 VAE 的损失函数 (loss function)中（带系数 $\lambda$），并与其他 VAE 组件（重建误差和每样本 KL 散度）一同最小化。

通过交替训练 VAE 编码器/解码器和 TC 判别器，FactorVAE 促使编码器生成维度独立的潜在编码，因为这会增加判别器区分 $q(z)$ 和 $\prod_j q(z_j)$ 的难度。

### 总相关性 VAE ($\beta$-TCVAE)：分离纠缠因子

Chen 等人（2018）在“VAEs 中分离源”一文中提出的 $\beta$-TCVAE，首先分解标准 VAE 目标函数中的平均 KL 散度项 $E_{p_{data}(x)}[KL(q_\phi(z|x) || p(z))$]，从而采取了不同的路径。假设先验分布 $p(z) = \prod_j p(z_j)$ 是因子化的（例如，各向同性高斯分布 $\mathcal{N}(0, I)$），这一项可以分解为三个有意义的组成部分：

1. **索引-编码互信息 $I(x;z)$**：这是 $E_{p_{data}(x)}[KL(q_\phi(z|x) || q_\phi(z))]$。它衡量输入数据 $x$ 和潜在编码 $z$ 之间的互信息。值越高意味着 $z$ 保留了关于 $x$ 的更多信息。
2. **总相关性 $TC(z)$**：这是 $KL(q_\phi(z) || \prod_j q_\phi(z_j))$，与之前讨论的 TC 项相同，衡量聚合后验分布中的依赖关系。对于分离而言，值越低越好。
3. **维度 KL $\sum_j KL(q_\phi(z_j) || p(z_j))$**：这一项促使每个潜在维度 $q_\phi(z_j)$ 的边缘分布（来自聚合后验分布）与先验分布 $p(z_j)$ 的相应边缘分布匹配。

因此，平均 KL 散度可以写为：


$$
E_{p_{data}(x)}[KL(q_\phi(z|x) || p(z))] = I(x;z) + TC(z) + \sum_j KL(q_\phi(z_j) || p(z_j))
$$


标准的 $\beta$-VAE 会以因子 $\beta$ 相等地惩罚这三项。$\beta$-TCVAE 背后的想法是，为了实现更好的分离，我们可能希望专门加大对 $TC(z)$ 的惩罚权重 (weight)，而不必过多地增加对 $I(x;z)$（这可能会损害重建）或维度 KL 项的惩罚。

$\beta$-TCVAE 目标函数修改了 ELBO，以允许对这些组件使用不同的权重：


$$
L_{\beta-TCVAE} = E_{p_{data}(x), q_\phi(z|x)}[\log p_\theta(x|z)] - w_I \cdot I(x;z) - w_{TC} \cdot TC(z) - w_{DKL} \cdot \sum_j KL(q_\phi(z_j) || p(z_j))
$$


通常，$w_I$（互信息的权重）保持为 1。主要关注点在于 $w_{TC}$（在此背景下常表示为 $\beta$，因此称为 $\beta$-TCVAE），其值设为大于 1，以强调最小化总相关性。$w_{DKL}$ 也可能需要调整。

**eta-TCVAE 中的项估计**：
与 FactorVAE 不同，$\beta$-TCVAE 通常使用小批量蒙特卡洛方法估计这些项，包括 $TC(z)$，而无需辅助判别器。对于给定的小批量数据点 $\{x_1, ..., x_M\}$ 及其对应的潜在样本 $\{z_1, ..., z_M\}$，其中 $z_i \sim q_\phi(z|x_i)$：

- $q_\phi(z)$ 通过小批量中样本 $z_i$ 的经验分布近似。
- $q_\phi(z_j)$（边缘分布）从这些样本中近似。
- TC 公式 $E_{q_\phi(z)}[\log \frac{q_\phi(z)}{\prod_j q_\phi(z_j)}]$ 中的密度，如 $q_\phi(z_i)$ 和 $q_\phi(z_{ij})$（对于 $z_i$ 的第 j 个分量）被估计。例如，$\log q_\phi(z_i)$ 可以使用核密度估计器近似，或者在实践中更常见的是，通过利用 $q_\phi(z|x_k)$ 的高斯形式并进行平均来近似：$\log q_\phi(z_i) \approx \log \frac{1}{M} \sum_{k=1}^M q_\phi(z_i | x_k)$。（注意：$q_\phi(z_i|x_k)$ 表示在由 $x_k$ 参数 (parameter)化的后验分布下评估 $z_i$ 的密度。）
  精确的估计器可能有些复杂，但它们基于对当前小批量样本的操作。

### 方法比较与实际考量

| 特性 | $\beta$-VAE | FactorVAE | $\beta$-TCVAE |
| --- | --- | --- | --- |
| **TC 控制** | 间接，通过整体 KL 惩罚 | 直接，通过明确的 TC 项 | 直接，通过分解的 KL 和 TC 项加权 |
| **TC 估计** | 未明确估计 | 基于判别器 | 小批量蒙特卡洛估计 |
| **复杂度** | 简单（一个超参数 (parameter) (hyperparameter) $\beta$） | 较高（训练 VAE + 判别器） | 中等（估计器可能复杂） |
| **超参数** | $\beta$ | $\lambda$，判别器架构/训练 | $w_I, w_{TC}, w_{DKL}$ |
| **稳定性** | 通常稳定 | 由于类似 GAN 的训练，稳定性可能较差 | 估计噪声可能影响稳定性 |

**考量因素：**

- **TC 惩罚的有效性**：FactorVAE 和 $\beta$-TCVAE 在相似的重建质量水平下，通常显示出比 $\beta$-VAE 更强的分离结果，正因为它们更直接地针对 TC。$\beta$-VAE 对 $KL(q(z|x)||p(z))$ 的更强惩罚可能导致潜在空间的“过度修剪”，可能丢弃对重建有用的信息（过度降低 $I(x;z)$），或过度强迫 $q(z|x)$ 过于接近 $p(z)$。
- **估计难题**：
  - 对于 FactorVAE，有效训练判别器很重要。TC 估计的质量取决于判别器近似密度比的程度。
  - 对于 $\beta$-TCVAE，基于小批量的 TC 和其他项的估计可能存在噪声，尤其是在批量大小较小时。这些估计的准确性会影响学习的动态过程。
- **超参数调整**：FactorVAE ($\lambda$) 和 $\beta$-TCVAE ($w_{TC}$ 等) 都引入了需要仔细调整的新超参数。最优值可能与数据集有关。
- **计算成本**：FactorVAE 增加了训练和运行判别器的成本。$\beta$-TCVAE 增加了每批次估计分解 KL 项的计算开销。

FactorVAE 和 TCVAE 代表了通过超越简单的 KL 加权来实现更分离表示的重要进展。通过将总相关性确定为一个需要控制的重要属性，它们提供了一个更扎实的框架。尽管它们在估计和超参数调整方面带来了一系列复杂性，但它们分离和惩罚纠缠来源的能力通常会带来更具解释性和实用性的潜在空间，这是表示学习中的一个主要目标。当您实现和试用这些模型时，请密切注意 TC 估计过程以及各个超参数对分离指标和重建质量的影响。

## 参考资料

- [Disentangling by Factorising](https://proceedings.mlr.press/v80/kim18b.html) — Hyunjik Kim, Andriy Mnih (2018)
  Journal: Proceedings of the 35th International Conference on Machine Learning; Publisher: PMLR; Volume: 80; Pages: 2649-2658; DOI: [10.48550/arXiv.1802.05983](https://doi.org/10.48550/arXiv.1802.05983)
  介绍了FactorVAE，该方法利用判别器对总相关性进行正则化以实现解缠。
- [beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2sy-5xx) — Irina Higgins, Loic Matthey, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, Shakir Mohamed, Alexander Lerchner (2017)
  Journal: International Conference on Learning Representations
  该论文介绍了$eta$-VAE，一个利用加权KL散度项来促进潜在空间解缠的框架。它是后续方法的基础。

---

[上一节](06-%E7%94%A8%E4%BA%8E%E8%A7%A3%E8%80%A6%E8%A1%A8%E7%A4%BA%E7%9A%84Beta-VAE.md) · [下一节](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%9E%E7%8E%B0%E9%AB%98%E7%BA%A7VAE%E6%9E%B6%E6%9E%84.md)
