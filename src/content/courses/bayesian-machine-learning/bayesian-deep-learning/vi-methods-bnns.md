---
course: "bayesian-machine-learning"
chapter: "bayesian-deep-learning"
lesson: "vi-methods-bnns"
sourceId: 3627
sourceUrl: "https://apxml.com/zh/courses/bayesian-machine-learning/chapter-6-bayesian-deep-learning/vi-methods-bnns"
title: "BNN的变分推断（例如：反向传播贝叶斯）"
description: "实现VI方法，例如反向传播贝叶斯，用于可扩展的BNN训练。"
order: 5
plots: []
sourceHash: "07c2e415b0f211128c661cb2d9887aad93484749fe923500c1c91ad4b57a7260"
sourceCorrections: []
---

虽然MCMC方法能够从贝叶斯神经网络 (neural network)权重 (weight)$w$的后验分布$p(w | \mathcal{D})$中提供渐近精确的样本，但它们在深度学习 (deep learning)中常见的高维度和大数据集上往往表现不佳。每次迭代的计算成本以及收敛所需的迭代次数可能变得过高。这就是变分推断（VI）提供有吸引力替代方案的地方。VI不进行采样，而是将贝叶斯推断重新构建为优化问题，在易于处理的族中寻求一个最接近真实后验$p(w | \mathcal{D})$的近似分布$q_\phi(w)$，这通常通过最小化库尔巴克-莱布勒（KL）散度$KL[q_\phi(w) || p(w | \mathcal{D})]$来衡量。

正如我们在第三章中看到的那样，最小化此KL散度等同于最大化证据下界（ELBO）：


$$
\mathcal{L}(\phi) = \mathbb{E}_{q_\phi(w)}[\log p(\mathcal{D}|w)] - KL[q_\phi(w) || p(w)]
$$


这里，$p(\mathcal{D}|w)$是给定权重的数据的似然，$p(w)$是权重的先验分布，而$q_\phi(w)$是由$\phi$参数 (parameter)化的变分近似。第一项促使近似后验良好地解释数据，而第二项作为正则项，使近似保持与先验接近。

然而，由于非线性激活函数 (activation function)引入的权重与数据之间复杂的非共轭关系以及参数的庞大数量，将坐标上升变分推断（CAVI）等传统VI方法直接应用于BNN是困难的。

### 反向传播 (backpropagation)贝叶斯：使用梯度优化ELBO

Blundell等人（2015）提出了一种将VI应用于BNN的突破性技术，即**神经网络 (neural network)中的权重 (weight)不确定性**，通常称为**反向传播贝叶斯（BBB）**。其核心思想是使用随机梯度上升法优化ELBO，以变分参数 (parameter)$\phi$为目标，这与使用反向传播训练标准深度学习 (deep learning)模型非常相似。

主要挑战在于计算ELBO的梯度，特别是期望项$\mathbb{E}_{q_\phi(w)}[\log p(\mathcal{D}|w)]$，因为期望是针对$q_\phi(w)$取的，并且我们要微分的参数$\phi$位于此分布内部。计算期望的梯度很棘手。

#### 重参数化技巧

反向传播贝叶斯巧妙地绕过这个问题，使用了**重参数化技巧**。我们不直接从$q_\phi(w)$中采样权重$w$，而是引入一个具有固定分布的辅助噪声变量$\epsilon$（例如：标准高斯，$\epsilon \sim \mathcal{N}(0, I)$），并定义一个确定性变换$w = g(\phi, \epsilon)$，使得得到的$w$具有$q_\phi(w)$的分布。

对于将$q_\phi(w)$选择为对角高斯分布的常见情况，其中$\phi = \{\mu, \rho\}$表示均值向量 (vector)$\mu$以及与标准差向量$\sigma$相关的参数$\rho$（通常$\sigma = \log(1 + \exp(\rho))$以保证正数性），重参数化很简单：


$$
w = \mu + \sigma \odot \epsilon = \mu + \log(1 + \exp(\rho)) \odot \epsilon, \quad \text{此处 } \epsilon \sim \mathcal{N}(0, I)
$$


现在，ELBO中的期望可以根据$\epsilon$的固定分布进行重写：


$$
\mathbb{E}_{q_\phi(w)}[\log p(\mathcal{D}|w)] = \mathbb{E}_{\epsilon \sim p(\epsilon)}[\log p(\mathcal{D}|g(\phi, \epsilon))]
$$


现在可以使用链式法则将梯度$\nabla_\phi$推入期望内部，因为给定$\phi$和$\epsilon$，$g(\phi, \epsilon)$是确定性的：


$$
\nabla_\phi \mathbb{E}_{q_\phi(w)}[\log p(\mathcal{D}|w)] = \mathbb{E}_{\epsilon \sim p(\epsilon)}[\nabla_\phi \log p(\mathcal{D}|g(\phi, \epsilon))]
$$


这个期望通常使用蒙特卡洛采样来近似。对于小批量数据$\mathcal{D}_i$，我们采样一个（或几个）$\epsilon$值，计算相应的$w = g(\phi, \epsilon)$，计算对数似然$\log p(\mathcal{D}_i | w)$，然后通过确定性变换$g$和网络本身使用标准反向传播计算梯度$\nabla_\phi \log p(\mathcal{D}_i | w)$。

#### 完整的目标函数和梯度

结合似然项和KL散度项，对于单个数据点$(x, y)$要最大化的目标函数（ELBO）变为：


$$
\mathcal{L}(\phi) \approx \log p(y | x, w) - KL[q_\phi(w) || p(w)]
$$


其中$w = g(\phi, \epsilon)$是使用重参数化技巧采样的。如果$q_\phi(w)$和$p(w)$选择得当（例如，都是高斯分布），KL散度项$KL[q_\phi(w) || p(w)]$通常可以解析计算。

训练中使用的损失函数 (loss function)通常是负ELBO，为整个数据集或小批量数据进行了缩放：


$$
\text{损失} = - \sum_{i=1}^N \left( \mathbb{E}_{q_\phi(w)}[\log p(y_i|x_i, w)] - \frac{1}{N} KL[q_\phi(w) || p(w)] \right)
$$


在实践中，对于大小为$M$的小批量数据，损失近似为：


$$
\text{损失}_{\text{小批量}} \approx -\sum_{j=1}^M \log p(y_j | x_j, w_j) + \frac{1}{B} KL[q_\phi(w) || p(w)]
$$


这里，$w_j = g(\phi, \epsilon_j)$是为小批量数据中的每个数据点独立采样的（或者对整个批次使用一个单独的样本$w$），$B$是数据集中小批量的数量（适当缩放KL项）。该损失函数对$\phi$（均值$\mu$和方差参数$\rho$）的梯度通过反向传播计算，并用于使用Adam或RMSprop等优化器更新$\phi$。

> 反向传播贝叶斯的训练过程。变分参数$\phi$（均值$\mu$和方差参数$\rho$）定义了近似后验$q_\phi(w)$。在每一步中，通过重参数化采样噪声$\epsilon$以生成网络权重$w$。前向传播计算小批量数据的预测输出。损失（负ELBO）结合了数据的对数似然和近似后验$q_\phi(w)$与先验$p(w)$之间的KL散度。损失对$\phi$的梯度通过反向传播计算，并使用优化器更新$\phi$。

### 实践考量

- **先验的选择$p(w)$：** 标准高斯先验（$\mathcal{N}(0, \sigma_p^2 I)$）是常见的选择，当$q_\phi(w)$也是高斯分布时，它能简化KL散度的计算。可以使用更复杂的先验，但可能需要对KL项进行数值估计。
- **初始化：** 将均值$\mu$初始化为与标准网络权重 (weight)类似，并保持初始方差（由$\rho$控制）较小，通常有助于稳定性。
- **方差参数 (parameter)化：** 使用$\rho$使得$\sigma = \log(1 + \exp(\rho))$能够确保$\sigma$始终为正。
- **梯度方差：** 梯度估计的随机性（由于采样$\epsilon$）可能导致训练过程中出现高方差。例如在每一步中使用更多蒙特卡洛样本或方差减少方法（如局部重参数化技巧，它将重参数化应用于激活而不是权重）的技术有所帮助，尽管这可能会增加计算成本。
- **计算成本：** 尽管比MCMC更具可扩展性，但BNN的VI计算成本仍然高于训练标准确定性神经网络 (neural network)。每次前向/反向传播 (backpropagation)都涉及权重采样和KL项计算，大致使参数数量翻倍（每个权重的$\mu$和$\rho$），并增加了计算图的复杂性。

### VI应用于BNN的优点与局限性

**优点：**

- **可扩展性：** 使用小批量随机梯度优化，使其适用于MCMC不可行的大型数据集和复杂模型。
- **兼容性：** 与现有深度学习 (deep learning)框架和硬件加速（GPU/TPU）结合得相对较好。
- **单次优化：** 在一次优化运行后提供变分参数 (parameter)$\phi$的点估计，这与MCMC需要多个链和收敛检查不同。

**局限性：**

- **近似质量：** 后验近似的准确性受到所选变分族$q_\phi(w)$的限制。像平均场高斯这样简单的族可能无法捕捉真实后验中的复杂依赖关系。
- **优化困难：** 优化ELBO可能具有挑战性，因为存在噪声梯度和潜在的局部最优。
- **方差低估：** VI有时会低估后验分布的方差，可能导致过度自信的预测。

与蒙特卡洛Dropout（在后续章节讨论）相比，反向传播 (backpropagation)贝叶斯代表了一种更具原则性且更灵活的VI方法，它明确定义并优化权重 (weight)的近似后验分布的参数。尽管通常比MC Dropout计算密集，但它允许对先验和变分近似有更多控制，如果优化成功且变分族足够，可能带来更准确的不确定性估计。

## 参考资料

- [Weight Uncertainty in Neural Networks](https://arxiv.org/abs/1505.05424) — Charles Blundell, Julien Cornebise, Koray Kavukcuoglu, Daan Wierstra (2015)
  Journal: Proceedings of the 32nd International Conference on Machine Learning (ICML); Volume: 37; Pages: 163-172; DOI: [10.48550/arXiv.1505.05424](https://doi.org/10.48550/arXiv.1505.05424)
  介绍贝叶斯反向传播算法的原始论文，是深度学习中变分推断的核心方法。
- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — Diederik P Kingma, Max Welling (2014)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1312.6114](https://doi.org/10.48550/arXiv.1312.6114)
  介绍了重参数化技巧，这是变分推断中基于梯度的优化（尤其对VAE和BNN等模型）的核心方法。
- [Variational Inference: A Review for Statisticians](https://doi.org/10.1080/01621459.2017.1285773) — David M. Blei, Alp Kucukelbir, Jon D. McAuliffe (2017)
  Journal: Journal of the American Statistical Association; Volume: 112; Pages: 859-877; DOI: [10.1080/01621459.2017.1285773](https://doi.org/10.1080/01621459.2017.1285773)
  对变分推断方法的全面综述，提供了该领域的广泛概述及其理论基础。
- [Probabilistic Machine Learning: Advanced Topics](https://probml.github.io/pml-book/book2.html) — Kevin Patrick Murphy (2023)
  Publisher: MIT Press; Pages: Chapter 21: Variational Inference
  一本高级教材，包含关于变分推断及其在贝叶斯神经网络中应用的详细章节。
