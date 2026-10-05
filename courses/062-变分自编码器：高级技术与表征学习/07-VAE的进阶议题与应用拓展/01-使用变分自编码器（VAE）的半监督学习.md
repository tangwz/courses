# 使用变分自编码器（VAE）的半监督学习

来源：[原文](https://apxml.com/zh/courses/vae-representation-learning/chapter-7-advanced-vae-topics-extensions/semi-supervised-learning-vaes)

[返回章节目录](README.md) · [返回课程目录](../README.md)

半监督学习 (supervised learning) (semi-supervised learning)（SSL）应对机器学习 (machine learning)中的一种常见情况：即拥有大量无标签数据，同时只有少量、通常获取成本较高的有标签数据。其目标是运用这两种数据类型，构建性能超越仅使用有标签数据训练的模型。变分自编码器（VAE）凭借其从无标签输入中习得丰富数据表示的固有能力，为半监督学习提供了一个有用的方法体系。

### VAE在半监督分类中的应用

VAE在半监督学习 (supervised learning) (semi-supervised learning)中使用的核心思想是：VAE从所有可用数据（有标签的 $X_L$ 和无标签的 $X_U$）学到的潜在空间 $z$，相比原始输入数据 $x$，能为后续分类任务提供更具结构性且信息量更多的表示。在有标签数据稀少时，这一点尤其有利。

一种普遍且有效的方法包括一个VAE，它学习从 $x$ 到 $z$ 再返回 $x$ 的映射，并结合一个分类器，该分类器在此习得的潜在表示上进行操作。

1. **VAE用于表示习得**: 变分自编码器（VAE），由编码器 $q_\phi(z|x)$ 和解码器 $p_\theta(x|z)$ 构成，在整个数据集（$X_L \cup X_U$）上进行训练。其目标是最大化证据下界（ELBO）：

   
   $$
   \mathcal{L}_{VAE}(x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) || p(z))
   $$
   

   此过程促使编码器在潜在空间 $z$ 中捕获输入数据的显著特征。先验 $p(z)$ 通常是标准高斯分布 $N(0,I)$。
2. **潜在空间上的分类器**: 一个独立的分类网络 $f_\psi(y|z)$ 被训练用来使用从数据有标签部分 $X_L$ 获得的潜在编码 $z_L$ 预测标签 $y$。分类损失 $\mathcal{L}_{C}$ 通常是真实标签 $y_L$ 和预测标签 $\hat{y}_L = f_\psi(z_L)$ 之间的交叉熵。

   
   $$
   \mathcal{L}_{C}(y_L, \hat{y}_L) = -\sum_{i} y_{L,i} \log \hat{y}_{L,i}
   $$
   

这两个组成部分通常联合训练。总目标函数结合了VAE的ELBO（应用于所有数据）和分类损失（仅应用于有标签数据），并通过超参数 (parameter) (hyperparameter) $\gamma$ 进行加权：


$$
\mathcal{J}_{SSL} = \sum_{x \in X_L \cup X_U} \left( \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) || p(z)) \right) + \gamma \sum_{(x_l, y_l) \in D_L} \mathcal{L}_{C}(y_l, f_\psi(q_\phi(z|x_l)))
$$


超参数 $\gamma$ 平衡了生成（VAE）任务和判别（分类）任务的贡献。以下是说明这种常见架构的图表。

> VAE半监督学习的一种常见架构。VAE处理有标签和无标签数据以习得表示。分类器随后使用来自有标签数据的这些表示进行监督训练。

### 半监督学习 (supervised learning) (semi-supervised learning)的集成生成模型

更紧密集成的方法，例如Kingma等人（2014）提出的M1和M2模型，将 $y$ 视为生成模型本身的随机变量。例如，M2模型假设的生成过程是 $p(y) \rightarrow p(z|y) \rightarrow p(x|z)$。标签 $y$ 影响潜在变量 $z$ 的生成，而 $z$ 又生成 $x$。

此类模型的损失函数 (loss function)变得更为复杂。对于有标签数据 $(x_l, y_l)$，ELBO的制定是为了反映 $\log p(x_l, y_l)$。对于无标签数据 $x_u$，模型必须对未知标签进行边缘化。这通常通过引入一个辅助推断网络 $q_\psi(y|x_u)$ 来完成，该网络预测无标签样本的标签分布。然后，无标签数据的ELBO涉及对这些预测标签分布的期望：


$$
\mathcal{L}_{unsup}(x_u) = \sum_k q_\psi(y=k|x_u) \left( \mathbb{E}_{q_\phi(z|x_u)}[\log p_\theta(x_u|z) + \log p(z|y=k) + \log p(y=k)] - D_{KL}(q_\phi(z|x_u)||p(z|y=k)) \right) + H(q_\psi(y|x_u))
$$


项 $p(z|y=k)$ 是潜在变量的类别条件先验，而 $H(q_\psi(y|x_u))$ 是一个熵项，它在适当时鼓励 $q_\psi(y|x_u)$ 给出确定性预测。分类器 $q_\psi(y|x)$ 也使用有标签数据进行判别性训练，通常通过总损失函数中的一个额外项，例如：


$$
\mathcal{L}_{classifier\_sup} = \sum_{(x_l, y_l) \in D_L} -\log q_\psi(y_l|x_l)
$$


这些集成模型理论上能更好地捕捉潜在的联合分布 $p(x,y)$，从而可能带来更好的生成和分类表现。然而，它们也引入了更大的模型复杂度和更精细的训练动态。

### 架构与训练考量

- **网络组成**: 编码器 $q_\phi(z|x)$、解码器 $p_\theta(x|z)$（或在某些模型中的 $p_\theta(x|z,y)$）以及分类器 $f_\psi(y|z)$（或 $q_\psi(y|x)$）通常是神经网络 (neural network)。它们的架构（例如，用于图像的CNN，用于序列的RNN）应根据数据模态选择。
- **参数 (parameter)共享**: 编码器 $q_\phi(z|x)$ 共享用于处理有标签和无标签数据。分类器 $f_\psi(y|z)$ 在此共享编码器生成的潜在编码上操作。
- **训练批次**: 训练期间，小批量数据通常通过从有标签池 $D_L$ 和无标签池 $D_U$ 中采样来构建。这确保了VAE重构/正则化 (regularization)项和监督分类项都对梯度更新有所贡献。
- **损失平衡**: 权重 (weight)因子 $\gamma$（以及在更复杂目标中可能存在的其他各项权重）很重要。它的值决定了习得良好表示与在有标签集上实现高分类准确率之间的相对重要性。为了获得最佳性能，通常需要适当调整这些权重。

### 基于VAE的半监督学习 (supervised learning) (semi-supervised learning)的优势

- **无标签数据的有效运用**: VAE自然地运用大量无标签数据来习得数据的潜在结构和变化。
- **少量标签下的分类改进**: 习得的表示能够显著提升分类器的性能，尤其当有标签样本数量较少时。
- **更优的泛化能力**: 通过从更广的数据分布（包括无标签样本）中学习，模型可能更好地泛化到未见数据。
- **类别条件生成**: 明确将 $y$ 纳入生成过程的模型（如M2的一些变体）可以用于生成以特定类别标签为条件的新样本，例如 $p(x|y,z)$ 或 $p(x|z)$，并且 $z \sim p(z|y)$。

### 挑战与局限

- **后验坍缩**: 与标准VAE类似，如果解码器过于强大或KL散度项权重 (weight)过高，潜在变量 $z$ 可能会被解码器忽略，导致潜在空间信息不足。这会阻碍分类器 $f_\psi(y|z)$ 的性能。
- **有标签与无标签数据不匹配**: 如果无标签数据的分布与有标签数据显著不同（领域漂移），从无标签数据习得的表示可能对有标签数据的分类无益，甚至有害。
- **复杂性与调优**: 集成半监督VAE模型相比更简单的流程，在实现和训练上可能更复杂。调整超参数 (parameter) (hyperparameter)如 $\gamma$ 以及不同模型组件的学习率需要仔细实验。
- **优化难度**: 平衡多个目标（重构、KL正则化 (regularization)、分类准确率）有时可能导致具有挑战性的优化局面。

尽管存在这些挑战，VAE仍为半监督学习 (supervised learning) (semi-supervised learning)提供了一种多功能且有原则的方法体系。通过有效结合VAE进行无监督表示学习的能力与来自有限有标签数据的监督信号，这些模型能在有标签数据成为瓶颈的任务上获得良好表现。选择更简单的特征提取方法还是更集成的生成模型，通常取决于具体问题、可用数据量和计算资源。

## 参考资料

- [Semi-Supervised Learning with Deep Generative Models](https://arxiv.org/abs/1406.5298) — Diederik P. Kingma, Danilo J. Rezende, Shakir Mohamed, Max Welling (2014)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Publisher: Advances in Neural Information Processing Systems; Volume: 27; Pages: 3581-3589; DOI: [10.48550/arXiv.1406.5298](https://doi.org/10.48550/arXiv.1406.5298)
  介绍了M1和M2模型，是基于VAE的集成生成式半监督学习的奠基性工作。
- [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114.pdf) — Diederik P. Kingma and Max Welling (2013)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1312.6114](https://doi.org/10.48550/arXiv.1312.6114)
  提出了原始的变分自编码器框架和证据下界（ELBO）目标函数，是VAE的基础。
- [Deep Semi-Supervised Learning: A Survey](https://arxiv.org/pdf/2008.06023.pdf) — Yahya Ouali, Jaehyun Shin, and Jean-François Lee (2020)
  Journal: arXiv preprint arXiv:2008.06023; DOI: [10.48550/arXiv.2008.06023](https://doi.org/10.48550/arXiv.2008.06023)
  对深度半监督学习技术进行了全面概述，包含生成模型相关章节。
- [Importance Weighted Autoencoders](https://arxiv.org/pdf/1509.00519.pdf) — Yuri Burda, Roger Grosse, and Ruslan Salakhutdinov (2015)
  Journal: International Conference on Learning Representations (ICLR); Publisher: PMLR (Proceedings of Machine Learning Research); DOI: [10.48550/arXiv.1509.00519](https://doi.org/10.48550/arXiv.1509.00519)
  提出了重要性加权自编码器（IWAE），以提供更紧密的对数似然下界，提升了VAE的性能并减轻了后验坍缩。

---

[上一节](../06-VAEs%E5%9C%A8%E5%A4%84%E7%90%86%E5%BA%8F%E5%88%97%E5%92%8C%E7%BB%93%E6%9E%84%E5%8C%96%E6%95%B0%E6%8D%AE%E4%B8%AD%E7%9A%84%E5%BA%94%E7%94%A8/07-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%B8%BA%E5%BA%8F%E5%88%97%E6%95%B0%E6%8D%AE%E5%AE%9E%E7%8E%B0VAE.md) · [下一节](02-VAEs%E5%9C%A8%E5%BC%82%E5%B8%B8%E5%92%8C%E5%88%86%E5%B8%83%E5%A4%96%E6%95%B0%E6%8D%AE%E6%A3%80%E6%B5%8B%E4%B8%AD%E7%9A%84%E5%BA%94%E7%94%A8.md)
