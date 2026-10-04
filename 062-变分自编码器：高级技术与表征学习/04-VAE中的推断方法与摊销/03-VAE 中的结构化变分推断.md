# VAE 中的结构化变分推断

来源：[原文](https://apxml.com/zh/courses/vae-representation-learning/chapter-4-vae-inference-techniques/structured-variational-inference)

[返回章节目录](README.md) · [返回课程目录](../README.md)

均值场近似，$q_\phi(z|x) = \prod_i q_\phi(z_i|x)$，通过假设在给定输入 $x$ 的情况下，潜在变量 $z_i$ 之间相互独立，大幅简化了 VAE 的训练。然而，这个假设往往限制性过强。真实后验 $p_\theta(z|x)$ 可能在潜在维度之间显示出复杂的依赖关系，而强制 $q_\phi(z|x)$ 因子化会使其无法准确地建模这些关联。这种差异，通常是“摊销差距”的一部分，可能会限制 VAE 学习丰富表示和生成高保真数据的能力。结构化变分推断提供了一种方法，通过明确地建模近似后验中的依赖关系来处理此限制。

### 在近似后验中建模依赖关系

结构化变分推断旨在通过允许潜在变量 $z_1, \dots, z_D$ 之间存在关联来丰富分布族 $q_\phi(z|x)$。并非完全因子化的形式，$q_\phi(z|x)$ 旨在捕获一些统计结构。这使得 $q_\phi(z|x)$ 可以更准确地近似真实后验 $p_\theta(z|x)$，从而可能带来更紧密的证据下界（ELBO）和更好的模型表现。

核心思想是使用能够表示依赖关系的模型来定义 $q_\phi(z|x)$。常见方法包含自回归 (autoregressive)模型和归一化 (normalization)流，两者都允许灵活且富有表现力的后验分布。

> 均值场和结构化（自回归）近似后验的比较。在均值场情况中，潜在变量 $z_i$ 在给定 $x$ 的情况下是条件独立的。在结构化自回归情况中，$z_i$ 依赖于前面的 $z_j$ (对于 $j<i$) 和 $x$。

### 用于 $q_\phi(z|x)$ 的自回归 (autoregressive)模型

引入结构的一种有效方法是以自回归方式建模 $q_\phi(z|x)$。这意味着潜在向量 (vector) $z = (z_1, \dots, z_D)$ 上的分布被分解为条件分布的乘积：


$$
q_\phi(z|x) = q_\phi(z_1|x) \prod_{j=2}^D q_\phi(z_j | z_{<j}, x)
$$


此处，$z_{<j}$ 表示 $(z_1, \dots, z_{j-1})$。每个条件分布 $q_\phi(z_j | z_{<j}, x)$ 可以由一个神经网络 (neural network)参数 (parameter)化，该网络以 $x$ 和先前采样的潜在变量 $z_1, \dots, z_{j-1}$ 作为输入。例如，如果每个 $q_\phi(z_j | z_{<j}, x)$ 是高斯分布，它的均值 $\mu_j$ 和标准差 $\sigma_j$ 将是 $x$ 和 $z_{<j}$ 的函数：


$$
\mu_j, \log \sigma_j = f_j(x, z_{<j}; \phi_j)
$$


这种结构允许 $q_\phi(z|x)$ 捕获任意依赖关系，只要条件神经网络 $f_j$ 具有足够的表达能力。从这样的模型中采样是顺序的：首先采样 $z_1 \sim q_\phi(z_1|x)$，然后 $z_2 \sim q_\phi(z_2|z_1, x)$，依此类推。虽然计算密度 $q_\phi(z|x)$ 很直接（$D$ 个项的乘积），但如果 $D$ 很大，顺序采样可能会很慢。

诸如逆自回归流（IAF）等技术（您可能在第 3 章中回顾过），提供了一种实现此类富有表现力的自回归模型的方法，使得采样能够并行化，从而大幅加快此过程。在 IAF 中，$z$ 通过使用自回归变换，从噪声向量 $\epsilon$（其中 $\epsilon_j$ 相互独立）变换而获得：$z_j = g_j(\epsilon_j; h_j(x, \epsilon_{<j}))$。

### 使用归一化 (normalization)流实现富有表现力的后验

归一化流，也在第 3 章中讨论过（第 3.5 节“用于灵活先验和后验的归一化流”），提供了另一种通用且强大的框架，用于构建复杂的后验分布。归一化流通过一系列可逆变换 $f_1, \dots, f_K$ 变换一个简单的基础分布 $q_0(u)$（例如，一个标准多元高斯分布）：


$$
z = f_K \circ \dots \circ f_1(u), \quad u \sim q_0(u)
$$


$z$ 的密度可以使用变量变换公式计算：


$$
q_\phi(z|x) = q_0(u) \left| \det \left( \frac{\partial (f_K \circ \dots \circ f_1)}{\partial u} \right) \right|^{-1}
$$


这些变换 $f_k$ 的参数 (parameter)（以及潜在的基础分布 $q_0$）作为 $\phi$ 的一部分进行学习，并且通常以 $x$ 为条件。这使得 $q_\phi(z|x)$ 能够学习高度灵活的分布。重点是，变换 $f_k$ 的设计使得其雅可比矩阵（以及行列式）计算上是可处理的。例子包含平面流、径向流以及更复杂的流架构，如 RealNVP、MAF 和 IAF。

为 $q_\phi(z|x)$ 使用归一化流可以显著提高推断网络的表达能力，使其能更好地匹配真实后验，从而收紧 ELBO。

### 影响与权衡

采用结构化变分推断有几个重要影响：

1. **ELBO 和模型质量的改进**：更灵活的 $q_\phi(z|x)$ 可以为真实对数似然 $\log p_\theta(x)$ 提供更紧密的下界。这通常意味着更好的生成表现，例如更清晰的生成样本和测试数据上更高的似然分数。潜在空间中学习到的表示也可能变得更有意义，因为推断网络更好地捕获了底层数据流形。
2. **计算复杂度的增加**：主要的权衡是计算成本。

   - **训练**：参数 (parameter)化和优化结构化的 $q_\phi(z|x)$ 涉及更多参数，并且通常每次迭代的计算更复杂（例如，计算流的雅可比行列式或自回归 (autoregressive)模型的顺序条件）。
   - **推断**：从某些结构化后验（如标准自回归模型）中采样，如果无法并行化，可能会更慢。
3. **模型设计选择**：您现在在 $q_\phi(z|x)$ 的架构方面有更多选择。对于自回归模型，这包含潜在变量的排序以及条件网络的架构。对于归一化 (normalization)流，它涉及选择流层的类型和数量。这些选择会影响表现和计算负载。
4. **KL 散度项**：ELBO 中的 KL 散度 $D_{KL}(q_\phi(z|x) || p(z))$ 可能会变得更具挑战性。如果 $p(z)$ 是标准高斯分布，并且 $q_\phi(z|x)$ 是复杂分布（例如，来自归一化流），KL 散度可能不再有解析解。在这种情况下，通常需要进行估计，例如，通过采样 $z \sim q_\phi(z|x)$ 并计算 $\mathbb{E}_{q_\phi(z|x)}[\log q_\phi(z|x) - \log p(z)]$。

### 何时考虑结构化推断

结构化变分推断在以下情况中特别有益：

- 您怀疑您的数据中真实潜在变异因子之间存在强关联或依赖关系，而均值场 VAE 无法捕获这些关系。
- 标准 VAE 产生不理想的结果，例如过度平滑或模糊的生成样本，或低似然，并且您假设推断网络是瓶颈。
- 应用要求高准确度的后验推断，例如在潜在变量本身是直接关注对象的场景中。

虽然引入结构会增加复杂性，但模型表达能力和表现方面的潜在提升通常能弥补额外开销，特别是对于具有挑战性的数据集或目标是先进结果时。此处讨论的技术，例如用于 $q_\phi(z|x)$ 的自回归 (autoregressive)模型和归一化 (normalization)流，对于构建更复杂、更强大的 VAE 来说是根本。随着我们的前进，您会看到这些改进的推断机制如何与其他高级 VAE 组件结合。

## 参考资料

- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — Diederik P Kingma, Max Welling (2013)
  Journal: arXiv; DOI: [10.48550/arXiv.1312.6114](https://doi.org/10.48550/arXiv.1312.6114)
  这篇基础论文介绍了变分自编码器（VAE）框架，为使用潜在变量的生成模型和均场近似奠定了基础。
- [Variational Inference with Normalizing Flows](https://arxiv.org/abs/1505.05770) — Danilo Jimenez Rezende and Shakir Mohamed (2015)
  Journal: International Conference on Machine Learning (ICML); Pages: 1530-1538; DOI: [10.48550/arXiv.1505.05770](https://doi.org/10.48550/arXiv.1505.05770)
  引入了正态流作为在变分推断中构建更具表达力和灵活性的近似后验分布的方法，增强了模型捕捉复杂依赖关系的能力。
- [Improving Variational Inference with Inverse Autoregressive Flow](https://arxiv.org/abs/1606.04934) — Diederik P. Kingma, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya Sutskever, and Max Welling (2016)
  Journal: Advances in Neural Information Processing Systems (NIPS); Volume: 29; DOI: [10.48550/arXiv.1606.04934](https://doi.org/10.48550/arXiv.1606.04934)
  介绍了逆自回归流（IAF），这是一种特定类型的正态流，能够在推断过程中通过高效的并行采样构建强大的自回归近似后验。
- [Masked Autoregressive Flow for Density Estimation](https://arxiv.org/abs/1705.07057) — George Papamakarios, Theo Pavlakou, Iain Murray (2017)
  Journal: Advances in Neural Information Processing Systems (NIPS); Volume: 30; DOI: [10.48550/arXiv.1705.07057](https://doi.org/10.48550/arXiv.1705.07057)
  介绍了蒙版自回归流（MAF），这是一种自回归正态流架构，广泛用于高质量密度估计和 VAE 中富有表达力的后验建模。

---

[上一节](02-%E5%B9%B3%E5%9D%87%E5%9C%BA%E8%BF%91%E4%BC%BC%E7%9A%84%E5%B1%80%E9%99%90%E6%80%A7.md) · [下一节](04-%E9%87%8D%E8%A6%81%E6%80%A7%E5%8A%A0%E6%9D%83%E8%87%AA%E7%BC%96%E7%A0%81%E5%99%A8%20%28IWAE%29.md)
