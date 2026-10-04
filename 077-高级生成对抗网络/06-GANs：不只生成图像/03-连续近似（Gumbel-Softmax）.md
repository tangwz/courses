# 连续近似（Gumbel-Softmax）

来源：[原文](https://apxml.com/zh/courses/generative-adversarial-networks-gans/chapter-6-gans-beyond-image-generation/gumbel-softmax-text-gans)

[返回章节目录](README.md) · [返回课程目录](../README.md)

将标准GAN框架直接用于文本序列等离散数据会遇到一个主要难题。生成器通常会输出下一个词元 (token)在词汇表 (vocabulary)上的概率。为生成序列，我们需要从此分布中进行采样。`argmax`操作或从多项式分布中采样本质上是不可微分的。这会中断从判别器到生成器的梯度流动，导致无法使用标准反向传播 (backpropagation)进行有效训练。

解决此问题的一种有效方法是使用离散随机变量的连续松弛。Gumbel-Softmax技巧（也称为Concrete分布）提供了一种方法，能用可微分函数近似从分类分布中采样，从而实现端到端训练。

## Gumbel-Max技巧：从分类分布中采样

在了解Gumbel-Softmax之前，我们先看看Gumbel-Max技巧。它是从具有类别概率 $\pi_1, \pi_2, ..., \pi_k$ 的分类分布中抽取样本 $z$ 的一种方法。步骤如下：

1. 从标准Gumbel分布（Gumbel(0, 1)）中抽取 $k$ 个独立的样本 $g_1, ..., g_k$。标准Gumbel分布的概率密度函数是 $f(x) = e^{-(x + e^{-x})}$。样本可以通过逆变换采样生成： $g_i = -\log(-\log(u_i))$，其中 $u_i \sim \text{Uniform}(0, 1)$。
2. 计算 `argmax`：
   
   $$
   z = \text{独热编码} \left( \underset{i \in \{1, ..., k\}}{\text{最大值索引}} (\log(\pi_i) + g_i) \right)
   $$
   

这个过程可以正确地从目标分类分布中采样。然而，`argmax`函数与直接采样一样，是不可微分的。"softmax"部分的作用就在于此。

## 用Softmax对Argmax进行松弛

Gumbel-Softmax的核心思想是用其连续、可微分的近似函数——`softmax`函数来替代不可微分的`argmax`操作。

假设生成器为每个类别 $i$ 生成了logits（未归一化 (normalization)的对数概率） $\alpha_i = \log(\pi_i)$，以及独立的Gumbel噪声样本 $g_i$，我们可以按如下方式计算松弛样本向量 (vector) $y$ 的分量：


$$
y_i = \frac{\exp((\alpha_i + g_i) / \tau)}{\sum_{j=1}^k \exp((\alpha_j + g_j) / \tau)}
$$


在此，$y = (y_1, ..., y_k)$ 是一个位于单纯形上的向量（即 $y_i \ge 0$ 且 $\sum y_i = 1$），类似于概率分布。

### 温度参数 (parameter)（$\tau$）的作用

参数 $\tau > 0$ 是温度。它控制着Gumbel-Softmax分布与实际分类分布的近似程度：

- **当 $\tau \to 0$ 时：** `softmax`函数越来越接近`argmax`。生成的向量 (vector) $y$ 会趋近于通过Gumbel-Max技巧采样的类别的独热编码。分布将集中在概率单纯形的顶点上。
- **当 $\tau \to \infty$ 时：** logits $\alpha_i$ 和噪声 $g_i$ 的影响减小，`softmax`的输出趋近于均匀分布 $(1/k, ..., 1/k)$。
- **中间温度 $\tau$：** 提供平滑、可微分的近似。较高的温度会产生“更柔和”（更均匀）的分布，而较低的温度会产生“更硬”（更接近独热）的分布。

### 退火计划

实践中，一种常见做法是使用温度退火。训练开始时使用相对较高的温度 $\tau$。这有助于初期阶段的生成，并提供更平滑的梯度。随着训练进行，$\tau$ 会逐渐降低（退火）到一个较小的正值（例如，0.1或0.01）。这使得样本逐渐“更硬”，促使生成器产生更接近实际离散词元 (token)的输出。

## 将Gumbel-Softmax整合到文本GAN中

在文本生成GAN中：

1. 生成器（通常是RNN或Transformer）根据先前的状态或词元 (token)，为序列中的下一个词元生成logits $\alpha = (\alpha_1, ..., \alpha_k)$。
2. 并非直接应用`argmax`或采样，而是使用这些logits和附加的Gumbel噪声，并给定特定温度 $\tau$，来应用Gumbel-Softmax函数。
   
   $$
   y = \text{Gumbel-Softmax}(\alpha, \tau)
   $$
   
3. 生成的向量 (vector) $y$ 相对于生成器的参数 (parameter)（通过logits $\alpha$）是可微分的。
4. 这种“软”词元表示 $y$ 可以被馈送给：
   - 作为生成器下一个时间步的输入（如果使用循环架构）。
   - 作为完整生成序列的一部分输入到判别器。
5. 判别器处理包含这些软表示的序列（或者有时，在“直通”变体中，是软表示和从它们派生出的硬独热向量的混合）。
6. 重要地，判别器计算的梯度现在可以通过Gumbel-Softmax操作回传，以更新生成器的权重 (weight)。

> 流图显示了GAN中的Gumbel-Softmax机制。生成器输出logits，这些logits与Gumbel噪声结合，并通过带温度参数的Gumbel-Softmax函数处理。这会产生一个可微分的“软”样本，该样本可以传递给判别器，从而使梯度回传到生成器。

## 优点与缺点

**优点：**

- **实现梯度流：** 提供了一条梯度路径，可以在不依赖强化学习 (reinforcement learning)的情况下，训练生成器处理离散数据。
- **训练动态更简化：** 通常比基于强化学习的方法（如SeqGAN，其梯度常有较高方差）带来更平稳的训练动态。
- **端到端：** 允许使用标准深度学习 (deep learning)优化器联合训练整个系统。

**缺点：**

- **近似性：** 训练期间生成的样本并非真正的离散，尤其是在较高温度下。这会引入偏差。
- **温度敏感性：** 性能很大程度上依赖于温度 $\tau$ 的选择和退火计划，这些都成为需要调整的重要超参数 (parameter) (hyperparameter)。
- **潜在的模式崩溃：** 尽管在离散数据上通常比基础GAN更稳定，但它不能完全消除模式崩溃等问题。
- **近似质量：** 梯度信号的质量取决于Gumbel-Softmax分布在当前温度下近似真实分类分布的程度。

Gumbel-Softmax技巧是使GAN适应离散数据生成的重要进展。尽管它并非完美方案，但它提供了一种实用且广泛使用的实现基于梯度训练的方法，适用于文本生成等方面，为强化学习方法的复杂性和潜在不稳定性提供了替代方案。理解其机制和权衡对处理图像等连续数据的GAN使用者来说很重要。

## 参考资料

- [Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/pdf/1611.01144.pdf) — Eric Jang, Shixiang Gu, Ben Poole (2016)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1611.01144](https://doi.org/10.48550/arXiv.1611.01144)
  介绍Gumbel-Softmax技巧以实现从分类分布中可微分采样的两篇原始论文之一。
- [The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables](https://arxiv.org/pdf/1611.00712.pdf) — Chris J. Maddison, Andriy Mnih, Yee Whye Teh (2016)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1611.00712](https://doi.org/10.48550/arXiv.1611.00712)
  同期发表的另一项具有影响力的工作，提出了与Gumbel-Softmax数学上等价的Concrete分布。
- [Text Generation with Gumbel-Softmax](https://arxiv.org/pdf/1704.05814.pdf) — Matt J. Kusner, José Miguel Hernández-Lobato (2017)
  Journal: Proceedings of the 34th International Conference on Machine Learning (ICML); Publisher: PMLR (Proceedings of Machine Learning Research); Volume: 70; Pages: 1980-1988; DOI: [10.48550/arXiv.1704.05814](https://doi.org/10.48550/arXiv.1704.05814)
  展示了Gumbel-Softmax重参数化技巧在文本生成中的实际应用，包括在生成对抗网络（GANs）和变分自编码器（VAEs）中的使用。

---

[上一节](02-%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95%20%28SeqGAN%2C%20RankGAN%29.md) · [下一节](04-%E5%9F%BA%E4%BA%8EGAN%E7%9A%84%E9%9F%B3%E9%A2%91%E5%90%88%E6%88%90%20%28WaveGAN%2C%20SpecGAN%29.md)
