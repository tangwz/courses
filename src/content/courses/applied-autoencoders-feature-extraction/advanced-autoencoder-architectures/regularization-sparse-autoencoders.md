---
course: "applied-autoencoders-feature-extraction"
chapter: "advanced-autoencoder-architectures"
lesson: "regularization-sparse-autoencoders"
sourceId: 6465
sourceUrl: "https://apxml.com/zh/courses/applied-autoencoders-feature-extraction/chapter-4-advanced-autoencoder-architectures/regularization-sparse-autoencoders"
title: "稀疏自编码器的正则化方法"
description: "研究 L1 正则化和 KL 散度惩罚，以实现在自编码器激活中的稀疏性。"
order: 2
plots: []
sourceHash: "fcb2b7d38f43dfd481cd0720d998d11b8381ac7d992ef713e512c6c888110e1f"
sourceCorrections: []
---

稀疏自编码器旨在生成表示，其中在任意给定时刻只有少量隐藏单元处于激活状态。这通常不能仅通过架构设计实现，而是通过在自编码器的损失函数 (loss function)中加入正则项。这种惩罚机制阻止网络激活过多神经元，引导它学习更具选择性和更有效的特征。实现这种稀疏性有两种常用方法：$L_1$ 正则化 (regularization)和 KL 散度。

### 稀疏性的 $L_1$ 正则化 (regularization)

一种直接促进稀疏性的方法是直接惩罚隐藏层的激活值。$L_1$ 正则化在损失函数 (loss function)中增加一个项，该项与隐藏单元激活值的绝对值之和成比例。

如果 $h_i$ 是给定输入的第 $i$ 个隐藏单元的激活值，$L_1$ 惩罚项是：


$$
R_{L1} = \sum_i |h_i|
$$


该惩罚项被加到标准的重构损失（例如，均方误差，MSE）中。总损失函数变为：


$$
L(x, \hat{x}) = \text{MSE}(x, \hat{x}) + \lambda \sum_i |h_i|
$$


这里，$x$ 是输入，$\hat{x}$ 是重构输出，而 $\lambda$（lambda）是一个超参数 (parameter) (hyperparameter)，用于控制稀疏惩罚的强度。

为什么这会产生稀疏性？$L_1$ 范数（绝对值之和）的特点是将许多不那么重要的激活值推向精确的零，从而使这些神经元在给定输入下不激活。较大的 $\lambda$ 值将强制实现更强的稀疏性，意味着更少的神经元将处于激活状态，但将其设置过高可能会妨碍自编码器有效重构输入的能力。找到 $\lambda$ 的适当平衡通常需要多次实验。

### KL 散度正则化 (regularization)

另一种常用的诱导稀疏性的方法是使用 Kullback-Leibler (KL) 散度。这种方法不是直接惩罚单个激活值，而是尝试使每个隐藏神经元在训练样本批次中的平均激活值与一个预设的低活动水平匹配。

我们将 $\rho$ (rho) 定义为隐藏神经元的期望平均激活值，通常称为稀疏参数 (parameter)。$\rho$ 的典型值可能很小，例如 0.05，这表示我们希望每个神经元平均只在批次中 5% 的训练样本上激活。

现在，让 $\hat{\rho}_j$（rho-hat sub j）表示第 $j$ 个隐藏单元的实际平均激活值，在一个训练小批次中计算得出：


$$
\hat{\rho}_j = \frac{1}{m} \sum_{k=1}^{m} h_j(x^{(k)})
$$


其中 $m$ 是小批次中的样本数量，$h_j(x^{(k)})$ 是第 $k$ 个训练样本的第 $j$ 个隐藏单元的激活值。

KL 散度项衡量期望分布（均值为 $\rho$ 的伯努利分布）与观察分布（均值为 $\hat{\rho}_j$ 的伯努利分布）之间的差异。对于每个隐藏单元 $j$，KL 散度是：


$$
KL(\rho || \hat{\rho}_j) = \rho \log \frac{\rho}{\hat{\rho}_j} + (1-\rho) \log \frac{1-\rho}{1-\hat{\rho}_j}
$$


如果 $\hat{\rho}_j$ 接近 $\rho$，此项将很小；而当 $\hat{\rho}_j$ 偏离 $\rho$ 时，此项会增加。自编码器的总损失函数 (loss function)随后包含这些 KL 散度项在所有隐藏单元上的总和，并由另一个超参数 (hyperparameter) $\beta$（beta）加权：


$$
L(x, \hat{x}) = \text{MSE}(x, \hat{x}) + \beta \sum_j KL(\rho || \hat{\rho}_j)
$$


超参数 $\beta$ 控制此稀疏惩罚的权重 (weight)。较高的 $\beta$ 值将更强制地使每个隐藏单元的平均激活值 $\hat{\rho}_j$ 与目标稀疏度 $\rho$ 匹配。

KL 散度方法不像 $L_1$ 正则化那样直接将单个激活值强制归零。相反，它鼓励每个神经元的*平均表现*稀疏。这表示一个神经元可能在少数特定输入模式下强烈激活，但在大多数其他模式下保持不激活，从而实现目标平均值 $\rho$。

> 图表显示了稀疏性正则化如何融入自编码器的损失计算。在使用 $L_1$ 或 KL 散度正则化等方法时，密集激活会比稀疏激活产生更高的惩罚。

### 选择和调整正则化 (regularization)

$L_1$ 和 KL 散度正则化都能有效诱导稀疏性，但它们实现的方式略有不同：

- **$L_1$ 正则化**：直接促使单个隐藏单元激活值为零。如果您的框架提供针对层或激活值的 $L_1$ 正则器，它通常更容易实现。
- **KL 散度正则化**：鼓励每个神经元（在一个批次中）的*平均*激活值与目标稀疏度 $\rho$ 匹配。这可以产生更专业化的神经元，它们会强烈激活但频率较低。

无论采用哪种方法，正则化的强度（由 $L_1$ 的 $\lambda$ 或 KL 散度的 $\beta$ 控制）是一个重要的超参数 (parameter) (hyperparameter)。

- 如果惩罚太弱，自编码器可能无法实现有意义的稀疏性。
- 如果惩罚太强，它可能会过度限制模型，导致重构质量不佳，因为自编码器难以用极少数活跃单元表示数据。

调整这些超参数通常涉及使用不同值训练自编码器，并观察重构损失和隐藏层激活的稀疏性。您还可以评估下游任务中提取特征的质量，以找到最佳正则化强度。在训练期间监控隐藏层中的平均激活水平也可以提供帮助，尤其是在使用 KL 散度时。

通过整合这些正则化技术，稀疏自编码器可以从数据中学习到更精细且通常更易于解释的特征，从简单的压缩转向能够捕捉输入更不同方面的表示。

## 参考资料

- [UFLDL Tutorial: Sparse Autoencoders](http://ufldl.stanford.edu/wiki/index.php/UFLDL_Tutorial) — Andrew Ng, Jiquan Ngiam, Chuan-Chang Wang, Christopher D. Manning, Erhan Baydogan, Pradnya Kuchibhotla, Vivek Ramavajjala, Rahul Gaur, David F. Miller (2011)
  Publisher: Stanford University
  详细解释了稀疏自编码器，包括L1和KL散度正则化方法，常被深度学习课程引用。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  一本涵盖深度学习基础概念的综合性教材，包括自编码器、正则化技术和稀疏表示。
- [Representation Learning: A Review and New Perspectives](https://ieeexplore.ieee.org/document/6474136) — Yoshua Bengio, Aaron Courville, Pascal Vincent (2013)
  Journal: IEEE Transactions on Pattern Analysis and Machine Intelligence; Publisher: IEEE; Volume: 35; Pages: 1798-1828; DOI: [10.1109/TPAMI.2013.42](https://doi.org/10.1109/TPAMI.2013.42)
  一篇讨论表征学习的开创性综述文章，稀疏自编码器在学习解耦和高效特征方面发挥着重要作用。
