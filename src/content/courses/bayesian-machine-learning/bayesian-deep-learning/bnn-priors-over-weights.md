---
course: "bayesian-machine-learning"
chapter: "bayesian-deep-learning"
lesson: "bnn-priors-over-weights"
sourceId: 3624
sourceUrl: "https://apxml.com/zh/courses/bayesian-machine-learning/chapter-6-bayesian-deep-learning/bnn-priors-over-weights"
title: "贝叶斯神经网络（BNNs）：权重的先验分布"
description: "通过对网络权重和偏置设定先验分布来定义贝叶斯神经网络。"
order: 2
plots: []
sourceHash: "59625f466d38ae1e9b9ac22be61ed7323597b81b0737495a2336d40666a3720b"
sourceCorrections: []
---

标准神经网络 (neural network)学习一组单一的最佳参数 (parameter)，通常是权重 (weight)和偏置 (bias)，这些参数通常通过使用梯度下降 (gradient descent)等方法最小化损失函数 (loss function)来得到。这些代表了参数的庞大集合中的点估计。虽然在预测方面有效，但这种方法本身不能捕捉与这些参数值或所得预测相关的不确定性。如果训练数据有限或有噪声，多组不同的权重可能以几乎同样好的方式解释数据。标准网络只选择其中一组，可能导致其预测过于自信。

贝叶斯神经网络（BNNs）通过采纳贝叶斯思想来解决这个问题：将网络参数（权重 $w$ 和偏置，统称为 $w$）视为随机变量，而不是要优化的固定值。在BNN中，我们的目标不是找到一组单一的最佳权重 $\hat{w}$，而是根据观测数据 $\mathcal{D}$ 推断权重的完整*后验分布*：$p(w | \mathcal{D})$。这个后验分布反映了我们看到数据后对合理参数值的信念，本质上量化 (quantization)了与之相关的不确定性。

## 定义先验：$p(w)$

任何贝叶斯模型的基础部分是先验分布，$p(w)$。这个分布编码了我们*在*观测任何数据*之前*对参数 (parameter)的信念。在BNN的背景下，先验分布设定在网络中的所有权重 (weight)和偏置 (bias)之上。

对权重有信念意味着什么？通常，我们没有关于特定权重值的强烈先验信息。一种常见且实用的选择是使用一个简单、数学上方便且能反映一般假设的先验。一个非常常见的选择是为每个权重 $w_i$ 使用一个以零为中心的独立高斯先验：


$$
w_i \sim \mathcal{N}(0, \sigma_p^2)
$$


这意味着整个权重向量 (vector) $w$ 的先验是：


$$
p(w) = \prod_i \mathcal{N}(w_i | 0, \sigma_p^2)
$$


这里，$\sigma_p^2$ 是先验方差，一个超参数 (hyperparameter)。这个先验表示一种信念，即权重可能较小并集中在零附近。$\sigma_p^2$ 的较大值对应于“较弱”的先验，允许权重偏离零更远，而较小值则施加更强的约束，将权重推向零。

选择零均值高斯先验与标准神经网络 (neural network)训练中使用的L2正则化 (regularization)（权重衰减）有直接关系。回想一下，L2正则化会在损失函数 (loss function)中添加一个惩罚项 $\lambda \sum_i w_i^2$。在高斯先验下最大化后验概率（MAP估计）等同于最小化带有L2惩罚项的损失函数，其中正则化强度 $\lambda$ 与先验方差 $\sigma_p^2$ 有关。然而，完整的贝叶斯方法更进一步；我们的目标是描述整个后验分布 $p(w|\mathcal{D})$，而不仅仅是找到它的众数。先验影响着这个完整分布的形状。

其他先验也是可行的：

- **拉普拉斯先验：** $p(w_i) \propto \exp(-|w_i|/\beta)$。这鼓励稀疏性（许多权重精确为零），类似于L1正则化。
- **分层先验：** 先验超参数（如 $\sigma_p^2$）本身可以被赋予先验分布，允许模型从数据中学习合适的正则化程度。
- **信息性先验：** 如果领域知识表明某些权重结构或值更合理，这可以编码在更具体的先验中。

目前，由于其简单性以及与权重衰减的关联，我们通常会假设一个简单的高斯先验。

## 似然：$p(\mathcal{D}|w)$

似然函数 $p(\mathcal{D}|w)$ 描述了在给定网络参数 (parameter) $w$ 的*特定*设置下观测到数据集 $\mathcal{D = \{ (x_n, y_n) \}_{n=1}^N}$ 的概率。这与标准神经网络 (neural network)中的情况相同，由网络架构和假定的输出分布决定。

- **回归：** 如果我们假设目标变量 $y$ 遵循以网络输出 $f(x; w)$ 为中心、噪声方差为 $\sigma^2$ 的高斯分布，那么单个数据点 $(x_n, y_n)$ 的似然是 $\mathcal{N}(y_n | f(x_n; w), \sigma^2)$。整个数据集的似然（假设独立性）是乘积：

  
  $$
  p(\mathcal{D}|w) = \prod_{n=1}^N \mathcal{N}(y_n | f(x_n; w), \sigma^2)
  $$
  

  在这种情况下，最小化负对数似然 $-\log p(\mathcal{D}|w)$ 对应于最小化均方误差（MSE）损失，再加上一个与噪声方差 $\sigma^2$ 相关的项。
- **分类：** 对于分类任务，网络输出通常代表概率（例如，通过softmax层）。如果 $y_n$ 是独热编码标签，单个数据点的似然通常使用分类分布建模，其中概率由网络输出 $f(x_n; w)$ 给出。

  
  $$
  p(y_n | x_n, w) = \text{分类}(y_n | f(x_n; w))
  $$
  

  在这里，最小化负对数似然 $-\log p(\mathcal{D}|w)$ 对应于最小化交叉熵损失。

似然的选择将抽象的网络参数 $w$ 与实际数据 $\mathcal{D}$ 联系起来。

## 后验：$p(w|\mathcal{D})$

在定义了先验 $p(w)$ 和似然 $p(\mathcal{D}|w)$ 后，我们可以使用贝叶斯定理将它们结合起来，得到权重 (weight)的后验分布：


$$
p(w | \mathcal{D}) = \frac{p(\mathcal{D} | w) p(w)}{p(\mathcal{D})}
$$


- $p(w | \mathcal{D})$ 是**后验分布**：我们观测数据后对权重的更新信念。这是我们在BNN中想要计算或近似的核心对象。
- $p(\mathcal{D} | w)$ 是**似然**：在给定特定权重下数据的概率（与标准损失函数 (loss function)相关）。
- $p(w)$ 是**先验分布**：我们看到数据之前对权重的初始信念（起到正则化 (regularization)器的作用）。
- $p(\mathcal{D})$ 是**边际似然**或**证据**：$p(\mathcal{D}) = \int p(\mathcal{D} | w) p(w) dw$。此项作为归一化 (normalization)常数，确保后验分布积分为1。

下图对比了标准方法和贝叶斯方法：

> 标准神经网络 (neural network)学习权重的点估计与贝叶斯神经网络学习权重的后验分布的比较。

BNN的主要难点在于后验 $p(w | \mathcal{D})$ 的计算。对于拥有数百万参数 (parameter)的深度神经网络，边际似然 $p(\mathcal{D})$ 所需的积分是难以计算的。此外，后验分布本身是高维且复杂的，使得直接计算变得不可能。

因此，我们必须依赖近似方法，而不是计算精确的后验。本章后续部分将介绍解决此挑战的两种主要技术类别：马尔可夫链蒙特卡洛（MCMC）方法和变分推断（VI）。这些方法使我们能够从后验分布中抽取样本或找到其近似值，从而能够在深度学习 (deep learning)任务中使用贝叶斯框架，特别是用于量化 (quantization)不确定性。

## 参考资料

- [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/) — Christopher M. Bishop (2006)
  Publisher: Springer
  一本广泛使用的教科书，提供了贝叶斯方法的全面背景，包括先验分布、似然函数原理及其在各种机器学习模型中的应用。
- [Bayesian Learning for Neural Networks](https://www.cs.toronto.edu/~radford/bayes-book.html) — Radford M. Neal (1996)
  Publisher: Springer Science & Business Media; Volume: 118
  一本经典著作，奠定了贝叶斯神经网络的理论基础，讨论了权重先验分布的规定和后验推断的统计机制。
- [Weight Uncertainty in Neural Networks](http://proceedings.mlr.press/v37/blundell15.pdf) — Charles Blundell, Julien Cornebise, Koray Kavukcuoglu, and Daan Wierstra (2015)
  Journal: Proceedings of the 32nd International Conference on Machine Learning (ICML); Publisher: PMLR (Proceedings of Machine Learning Research); Volume: 37; Pages: 1631-1640; DOI: [10.5555/2969239.2969374](https://doi.org/10.5555/2969239.2969374)
  这篇论文介绍了“反向传播贝叶斯”，一种实用且可扩展的变分推断方法，用于训练贝叶斯神经网络，展示了如何近似权重的后验分布。
