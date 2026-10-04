---
course: "generative-adversarial-networks-gans"
chapter: "conditional-controllable-generation"
lesson: "infogan-disentangled-representations"
sourceId: 2715
sourceUrl: "https://apxml.com/zh/courses/generative-adversarial-networks-gans/chapter-4-conditional-controllable-generation/infogan-disentangled-representations"
title: "信息最大化GAN (InfoGAN)"
description: "使用互信息，以无监督方式学习解耦潜在变量。"
order: 3
plots: []
sourceHash: "f1fe743efe8873bc14fe54ebd9ccaa8c1a00d03cbfd0d5431c251cc3729a1bde"
sourceCorrections: []
---

尽管条件GAN（cGAN）提供了一种使用明确标签（$y$）来指导生成过程的有效方法，但它们基本依赖于此类带标签数据的可用性。如果我们想在*不*依赖预设标签的情况下，找出并操控数据的有意义属性，该怎么办？这就是信息最大化GAN，即InfoGAN发挥作用之处。InfoGAN旨在以完全无监督的方式学习潜在空间中的*解耦表征*。其目标是确定输入噪声向量 (vector)中与生成数据显著且可解释特征相对应的特定维度。

设想在MNIST手写数字数据集上训练一个GAN。一个标准GAN可能会学习到一个复杂、纠缠的潜在空间，其中改变单个潜在变量会同时影响生成数字的多个方面（例如，其身份和书写风格）。InfoGAN则尝试构建潜在空间，使其某些部分控制不同的因素，如数字类型（0-9）、旋转或笔画粗细，即便训练数据没有提供这些因素的标签。

### 核心思想：互信息最大化

InfoGAN通过修改标准GAN框架实现此目的。我们将输入到生成器$G$的内容不再仅仅是随机噪声向量 (vector)$z$，而是分为两部分：

1. 传统的不可压缩噪声向量$z$。
2. 一组新的潜在变量$c = (c_1, c_2, ..., c_L)$，我们称之为*潜在编码*。

生成器的任务现在是产生输出$x = G(z, c)$。InfoGAN的核心思路是鼓励潜在编码$c$与生成的样本$G(z, c)$之间建立强关联。这种关联使用*互信息*进行量化 (quantization)，记作$I(X; Y)$。直观地说，互信息衡量的是给定变量Y的知识后，变量X不确定性的减少量。在我们的案例中，我们希望最大化互信息$I(c; G(z, c))$。如果$I(c; G(z, c))$很高，则表示潜在编码$c$包含关于生成输出$G(z, c)$中存在特征的大量信息。

### InfoGAN目标函数

为了实现此目标，InfoGAN在标准GAN目标函数中添加了一个正则化 (regularization)项。总体目标变为：


$$
\min_G \max_D V_{InfoGAN}(D, G) = V(D, G) - \lambda I(c; G(z, c))
$$


在此：

- $V(D, G)$是原始GAN价值函数（例如，极小极大目标或 Wasserstein 距离）。
- $I(c; G(z, c))$是潜在编码$c$与生成器输出$G(z, c)$之间的互信息。
- $\lambda$是一个非负超参数 (parameter) (hyperparameter)，用于控制互信息正则化的强度。典型值为$\lambda = 1$。

生成器$G$现在不仅旨在欺骗判别器$D$，而且旨在最大化其潜在编码$c$与其输出之间的互信息。判别器$D$仍尝试区分真实样本和伪造样本。

### 互信息估计

直接最大化$I(c; G(z, c))$在计算上是困难的，因为它涉及后验概率$P(c|x)$，其中$x = G(z, c)$，这通常难以处理。InfoGAN巧妙地避开了此问题，通过最大化互信息的*变分下界*来实现。

我们引入了一个由神经网络 (neural network)参数 (parameter)化的辅助分布$Q(c|x)$，它作为真实后验$P(c|x)$的近似。可以证明，互信息$I(c; G(z, c))$有一个下界：


$$
I(c; G(z, c)) \ge E_{c \sim P(c), x \sim G(z, c)} [\log Q(c|x)] + H(c)
$$


在此，$H(c)$是潜在编码采样的先验分布$P(c)$的熵。由于$H(c)$对于固定的先验分布$P(c)$（例如，均匀分类或标准高斯）是常数，最大化此下界实际归结为最大化该项$E_{c \sim P(c), x \sim G(z, c)} [\log Q(c|x)]$。

此期望可以通过采样高效估计：从$c$的先验分布$P(c)$中采样$c$，从其噪声分布中采样$z$，生成$x = G(z, c)$，然后使用辅助网络$Q$计算$\log Q(c|x)$。

### InfoGAN架构

InfoGAN架构修改了标准GAN的设置：

1. **生成器（$G$）**：将噪声向量 (vector)$z$和潜在编码$c$作为输入，生成$G(z, c)$。
2. **判别器（$D$）**：接收样本$x$（真实或生成）并输出其为真实的概率。
3. **辅助网络（$Q$）**：接收生成的样本$x = G(z, c)$并输出分布$Q(c|x)$的参数 (parameter)，旨在预测用于生成$x$的潜在编码$c$。

通常，判别器$D$和辅助网络$Q$共享大部分卷积层，仅在最终层分化，以产生各自的输出（$D$的真实/伪造概率和$Q$的$Q(c|x)$参数）。

> InfoGAN的简化架构概览。生成器使用噪声$z$和潜在编码$c$。判别器网络具有共享层，分别连接到不同的头部：一个用于真实/伪造分类（$D$），另一个用于预测潜在编码（$Q$）。总损失同时引导对抗性博弈和$c$与$G(z, c)$之间互信息的最大化。

### 实现考量

- **潜在编码先验（$P(c)$）**：你需要定义潜在编码$c$的结构和先验分布。常见选择包括：
  - 离散特征（例如，数字身份）的分类分布。
  - 旋转或缩放等特征的均匀连续分布（例如，$U[-1, 1]$）。
  - 其他连续因素的高斯分布。
- **辅助损失（$Q$）**：用于训练$Q$的损失函数 (loss function)取决于为$c$选择的先验。
  - 如果$c_i$是分类的，$Q(c_i|x)$会为每个类别输出概率，损失通常是采样$c_i$与预测概率之间的交叉熵。
  - 如果$c_i$是连续的（例如，高斯），$Q(c_i|x)$可能会输出近似高斯后验的均值和方差。损失可以是采样$c_i$在此预测高斯分布下的对数似然。
- **训练**：生成器$G$、判别器$D$和辅助网络$Q$联合训练。来自标准GAN损失的梯度更新$G$和$D$。来自互信息下界（使用$Q$计算）的梯度同时更新$G$和$Q$。

### 找出可解释因素

成功训练后，InfoGAN通常会找出与数据中有意义变化相对应的潜在编码$c$。例如，在MNIST上，一个分类编码可能会学习表示数字类别（0-9），而连续编码可能会捕捉旋转和笔画宽度，所有这些都在未曾见过这些属性的明确标签的情况下完成。通过固定噪声$z$并改变$c$的特定维度，您可以直接操控生成输出中的这些学到因素。

### 优点与局限性

**优点：**

- 使得无监督找出数据中可解释、解耦的因素。
- 提供了一种在无需带标签数据的情况下，操控生成样本特定属性的方法。
- 直接基于标准GAN框架，架构改动相对较小。

**局限性：**

- 找出的因素取决于所选择的$c$结构和数据集固有的变化。如果数据分布中不显著，它可能无法找出人类觉得直观的因素。
- 解耦通常不完美；编码可能仍会在某种程度上影响多个属性。
- 训练稳定性仍可能令人担忧，可能需要调整超参数 (parameter) (hyperparameter)$\lambda$。
- 需要基于先验$P(c)$精心设计辅助网络$Q$和相应的损失函数 (loss function)。

InfoGAN代表着在构建更可控、更易理解的生成模型方面的一个重要进展。通过结合信息理论的原则，它提供了一个直接从原始、无标签数据中学习结构化潜在表征的框架，为生成过程的细粒度操控开辟了可能性。

## 参考资料

- [InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets](https://arxiv.org/abs/1606.03657) — Xi Chen, Yan Duan, Rein Houthooft, John Schulman, Ilya Sutskever, and Pieter Abbeel (2016)
  Journal: Advances in Neural Information Processing Systems (NIPS 29); Pages: 2172-2180; DOI: [10.48550/arXiv.1606.03657](https://doi.org/10.48550/arXiv.1606.03657)
  提出信息最大化生成对抗网络（InfoGAN）的开创性论文，介绍了一种通过最大化互信息来无监督学习可分离和可解释表示的方法。
- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) — Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio (2014)
  Journal: Advances in Neural Information Processing Systems (NIPS 27); Pages: 2672-2680; DOI: [10.48550/arXiv.1406.2661](https://doi.org/10.48550/arXiv.1406.2661)
  介绍生成对抗网络（GANs）的原始论文，建立了InfoGAN及其他高级GAN架构所基于的基本框架。
- [Generative Deep Learning: Teaching Machines to Paint, Write, Compose, and Play](https://www.oreilly.com/library/view/generative-deep-learning/9781098134171/) — David Foster (2023)
  Publisher: O'Reilly Media; Pages: 453
  一本关于生成模型的综合指南，包含对各种GAN架构（如InfoGAN）的详细解释和实践实现。(第三版)
