# VAE编码器：输出分布参数

来源：[原文](https://apxml.com/zh/courses/applied-autoencoders-feature-extraction/chapter-6-variational-autoencoders-structured-latent-spaces/vae-encoder-distribution-parameters)

[返回章节目录](README.md) · [返回课程目录](../README.md)

自编码器通常接收输入$x$，并确定性地将其映射到潜在空间中的一个点$z$。这个点$z$是$x$的压缩表示。变分自编码器则采用不同的方式。VAE编码器不是将输入编码为单个点，而是为每个输入在潜在空间中描述一个*概率分布*。

### 为何是分布？

你可能会疑惑，为何是分布？通过为每个输入学习一个分布，VAE能够形成一个更连续且有结构的潜在空间。这意味着潜在空间中彼此靠近的点，在解码时将对应相似的数据。这种特性对于生成新的数据样本非常有用，我们将在本章后面看到。编码器的职责是根据输入数据学习此分布的参数 (parameter)。

### 输出分布参数 (parameter)

为了定义这种概率分布，VAE编码器会输出一组参数。为求简化和符合一般做法，VAE通常假设为每个输入$x$学习到的分布是多元高斯分布（也称作正态分布）。一个高斯分布由两个主要参数来描述：

- 一个**均值向量 (vector)** $\mu$。这个向量定义了潜在空间中分布的中心。
- 一个**方差向量** $\sigma^2$（或等价地，一个标准差向量 $\sigma$）。这个向量定义了分布在潜在空间每个维度上的扩展程度或不确定性。

因此，对于给定的输入$x$，编码器网络不仅仅输出一个单一的潜在向量$z$。相反，它输出两个独立的向量：

1. 均值向量 $\mu(x)$。
2. 对数方差向量 $\log(\sigma^2(x))$。

这些向量的维度将与我们期望的潜在空间大小相等。例如，如果我们的目标是32维潜在空间，编码器将输出一个32维的 $\mu(x)$ 向量和一个32维的 $\log(\sigma^2(x))$ 向量。

### 对数方差的缘由

你可能会问：“为何输出 $\log(\sigma^2)$ 而不是直接输出 $\sigma^2$？” 这是一个很好的疑问，做出这种选择有几个重要的原因：

1. **数值稳定性：** 对数转换有助于保持训练过程更稳定。神经网络 (neural network)的输出若不加限制，有时可能导致方差出现非常大或非常小的正值。对数变换将方差的整个正实数轴 $(0, \infty)$ 映射到对数方差的整个实数轴 $(-\infty, \infty)$，这更便于网络学习和处理。
2. **确保方差为正：** 方差根据其数学定义，必须是非负的（如果存在任何不确定性，则为严格正值）。标准神经网络层的输出可以是任意实数。通过让网络预测 $\log(\sigma^2)$，我们就可以将方差计算为 $\sigma^2 = \exp(\log(\sigma^2))$。由于指数函数 $e^x$ 对任意实数 $x$ 始终为正，这保证了我们的方差 $\sigma^2$ 将始终为正。

类似地，标准差 $\sigma$ 可以从对数方差计算得到，如 $\sigma = \exp(\frac{1}{2} \log(\sigma^2))$ 或 $\sigma = \sqrt{\exp(\log(\sigma^2))}$。

### 用于参数 (parameter)预测的编码器网络结构

这一要求意味着，与标准自编码器相比，VAE的编码器部分在其输出阶段将拥有略微不同的结构。通常，编码器网络的主体（可能包含全连接层、卷积层等）处理输入$x$，并将其转换为某种中间的高层表示。这个共享的中间表示随后被送入两个独立的输出层（常被称为“头部”）:

- 一个层（例如，一个全连接层）输出均值向量 (vector) $\mu(x)$。
- 另一个独立的层（例如，另一个全连接层）输出对数方差向量 $\log(\sigma^2(x))$。

这些用于 $\mu$ 和 $\log(\sigma^2)$ 的最终层通常不对其输出应用激活函数 (activation function)（或者使用线性激活，这等同于无激活）。这是因为均值 $\mu$ 可以取任意实数值，对数方差 $\log(\sigma^2)$ 也可以取任意实数值。

以下是一个说明这种分支结构的图示：

> VAE编码器通过共享层处理输入，随后分成两个头部。一个头部预测与输入$x$相关的潜在分布的均值向量 $\mu(x)$，另一个预测对数方差向量 $\log(\sigma^2(x))$。

### 理解编码器的输出

对于输入到我们VAE的每个数据点$x$，编码器不仅仅提供一个压缩表示。它实际上告诉我们：“对于这个输入$x$，其在潜在空间中对应的表示应该从一个高斯分布中采样，该分布以 $\mu(x)$ 为中心，并在每个潜在维度上具有 $\sigma^2(x)$ 的方差。”

更正式地说，编码器学习将每个输入$x$映射到一个特定的条件概率分布 $q(z|x)$。该分布被假定为高斯分布：
$q(z|x) = \mathcal{N}(z; \mu(x), \text{diag}(\sigma^2(x)))$
这里，$\mathcal{N}(z; \mu, \Sigma)$ 表示 $z$ 上的多元高斯分布，均值为 $\mu$，协方差矩阵为 $\Sigma$。术语 $\text{diag}(\sigma^2(x))$ 表明我们通常处理的是对角协方差矩阵。这意味着在给定输入$x$的情况下，潜在空间的各个维度被假定为条件独立的，这简化了模型。该矩阵的每个对角线元素都是方差向量 (vector) $\sigma^2(x)$ 中的一个 $\sigma_i^2(x)$ 值。

### 后续步骤：从学得的分布中采样

一旦编码器生成了这些参数 (parameter) $\mu(x)$ 和 $\log(\sigma^2(x))$，VAE前向传播的下一个合乎逻辑的步骤就是从这个定义的分布 $\mathcal{N}(\mu(x), \sigma^2(x))$ 中*采样*一个潜在向量 (vector)$z$。这个采样得到的$z$随后会传递给VAE的解码器，解码器将尝试从$z$重建原始输入（或生成一个新的样本）。

然而，采样本身是一个随机过程。一个重大挑战随之出现：在训练期间，我们如何通过这个随机采样步骤反向传播 (backpropagation)梯度？这就是巧妙的数学技巧——“重参数化技巧”发挥作用的地方。我们将在紧接着的下一节讨论VAE的这个重要组成部分，因为它使得这些概率编码器的端到端训练成为可能。

## 参考资料

- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — Diederik P Kingma, Max Welling (2013)
  Journal: International Conference on Learning Representations (ICLR 2014); DOI: [10.48550/arXiv.1312.6114](https://doi.org/10.48550/arXiv.1312.6114)
  这篇基础论文介绍了变分自编码器，解释了概率编码器、其输出分布参数（均值和方差）以及重参数化技巧。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 20
  第20章详细解释了变分自编码器的理论和实践，包括编码器在输出分布参数中的作用。
- [Variational Autoencoders (from "Deep Generative Models" course notes)](https://deepgenerativemodels.github.io/notes/vae/) — Stefano Ermon and course staff (2023)
  Journal: Stanford University CS236: Deep Generative Models Course
  这些课程笔记提供了对变分自编码器的结构化学术解释，涵盖了编码器的概率输出和特定的参数化（均值和对数方差）。

---

[上一节](02-%E5%8F%98%E5%88%86%E8%87%AA%E7%BC%96%E7%A0%81%E5%99%A8%E6%9E%84%E6%88%90.md) · [下一节](04-%E9%87%8D%E5%8F%82%E6%95%B0%E5%8C%96%E6%8A%80%E5%B7%A7%E8%A7%A3%E9%87%8A.md)
