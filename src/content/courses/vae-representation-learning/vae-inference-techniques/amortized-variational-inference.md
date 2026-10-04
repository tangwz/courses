---
course: "vae-representation-learning"
chapter: "vae-inference-techniques"
lesson: "amortized-variational-inference"
sourceId: 6335
sourceUrl: "https://apxml.com/zh/courses/vae-representation-learning/chapter-4-vae-inference-techniques/amortized-variational-inference"
title: "摊销变分推断：优点与局限"
description: "详细分析摊销变分推断、其效率以及固有的局限性。"
order: 1
plots: []
sourceHash: "b33b312c86cc840655150fa620dd4432b17b55e43731a7cb829232a922183036"
sourceCorrections: []
---

变分自编码器的效果与近似后验 $q_\phi(z|x)$ 与真实的、通常难以处理的后验分布 $p_\theta(z|x)$ 匹配的程度紧密相关。变分推断是 VAE 中的一项基础技术，提供了一种处理这种近似的实用方法。其运作机制、优势以及固有的不足之处将被分析。

在经典变分推断中，您可能会为每个数据点 $x_i$ 优化一组独立的变分参数 (parameter)，以定义其特定的 $q_i(z|x_i)$。对于大型数据集来说，这会导致计算成本过高。相比之下，摊销变分推断学习一个单一的参数化函数，通常是一个神经网络 (neural network)，称为推断网络或编码器，表示为 $q_\phi(z|x)$。该网络接收数据点 $x$ 作为输入，并输出给定 $x$ 时隐变量 $z$ 的近似后验分布的参数。例如，如果 $q_\phi(z|x)$ 是高斯分布，推断网络可能会输出其均值 $\mu_\phi(x)$ 和协方差 $\Sigma_\phi(x)$。这个推断网络的参数 $\phi$ 与生成模型的参数 $\theta$ 通过最大化证据下界（ELBO）联合学习。“摊销”一词表示推断的成本通过共享参数 $\phi$ 分摊到所有数据点上。

> 在摊销推断中，一个单一的推断网络 $q_\phi(z|x)$ 学习将任意输入数据点 $x_i$ 映射到其对应的近似后验分布 $q_\phi(z|x_i)$ 的参数。

### 摊销变分推断的优点

摊销变分推断已成为 VAE 的事实标准，凭借其多项显著优势：

1. **推断时的效率高：** 这是一项重要的实用优势。一旦推断网络 $q_\phi(z|x)$ 训练完成，获取新（未见过）数据点 $x_{new}$ 的近似后验分布速度极快。它只要求一次前向传播通过推断网络。这与非摊销方法形成鲜明对比，后者对每个新数据点都需要单独的、通常是迭代的优化过程来查找其变分参数 (parameter)。
2. **对大型数据集的适用性好：** 使用摊销推断训练 VAE 对大型数据集适用性好。推断网络的参数 $\phi$ 在所有数据点之间共享。这有助于使用随机梯度下降 (gradient descent)（SGD）或其变体在小批量数据上进行高效训练。每个小批量都有助于改进共享的 $\phi$ 和 $\theta$。
3. **与深度学习 (deep learning)架构的良好配合：** 摊销推断与深度学习框架非常契合。我们能够使用功能强大且灵活的神经网络 (neural network)架构，例如用于图像数据的卷积神经网络 (CNN)（CNNs）或用于序列数据的循环神经网络 (RNN)（RNNs），作为推断网络 $q_\phi(z|x)$ 的主要构成。这使得从观测数据 $x$ 到近似后验参数的高度非线性、复杂映射得以学习。
4. **编码器与解码器的联合优化：** 推断网络（编码器）的参数 $\phi$ 和生成模型（解码器）的参数 $\theta$ 同时优化以最大化 ELBO。这种联合优化允许编码器和解码器相互适应并学习互补的作用。编码器学习生成对解码器有用的隐表示，而解码器学习从这些表示中重建数据。

### 摊销变分推断的不足与局限

尽管有其优点，摊销变分推断并非没有缺点，主要源于所做的近似处理：

1. **近似后验的表达能力受限：** 推断网络 $q_\phi(z|x)$ 负责生成近似后验的参数 (parameter)，该近似后验理想情况下应接近所有可能输入 $x$ 的真实后验 $p_\theta(z|x)$。如果真实后验表现出复杂的特性，例如多峰性、隐变量之间存在复杂关联，或者对于不同的 $x$ 其形态发生剧烈变化，那么固定形式的 $q_\phi(z|x)$（例如对角高斯分布）——其参数仅仅由神经网络 (neural network)输出——可能会限制性过强。这种不匹配会引入“摊销鸿沟”，它是选用摊销 $q_\phi(z|x)$ 所能达到的 ELBO 与在没有摊销约束下为每个 $x_i$ 优化单个 $q(z|x_i)$ 所能达到的 ELBO 之间的差异。较大的摊销鸿沟意味着 ELBO 对真实对数似然的约束更为宽松，可能降低学习表示和生成样本的质量。
2. **平均场假设：** 许多 VAE 实现中常见的简化假设是近似后验 $q_\phi(z|x)$ 在隐变量维度上因子化。即，

   
   $$
   q_\phi(z|x) = \prod_{j=1}^{D} q_\phi(z_j|x)
   $$
   

   其中 $D$ 是隐空间 $z$ 的维度。这被称为平均场近似。这意味着，给定 $x$ 时，隐变量 $z_j$ 在*近似*后验中是条件独立的。然而，真实后验 $p_\theta(z|x)$ 通常在隐变量之间具有丰富的依赖结构。强制使用因子化的 $q_\phi(z|x)$ 可能会阻碍模型捕捉这些相关性，导致后验近似的准确性降低，进而使得 ELBO 更加宽松。我们将在“平均场近似的局限”部分更详细地分析这一假设的影响。
3. **优化难题及次优解的可能性：** 推断网络参数 $\phi$ 和生成模型参数 $\theta$ 的联合优化是一个复杂的非凸优化问题。训练过程有时会收敛到次优的局部最小值。例如，推断网络可能会学习生成过度简化的近似后验（例如，始终非常接近先验 $p(z)$，以积极最小化 ELBO 中的 KL 散度项）。这会以牺牲重建质量为代价，这种现象通常被称为“后验坍塌”，尤其是在强大的解码器中。当 $q_\phi(z|x)$ 坍塌到先验时，隐变量 $z$ 几乎不包含关于输入 $x$ 的信息，使它们在表示学习中失去作用。
4. **准确建模后验不确定性的困难：** 尽管 $q_\phi(z|x)$ 本身是一个分布，但其参数（例如高斯分布的均值和方差）通常是给定 $x$ 时推断网络的确定性输出。这种固定映射可能不总是有足够的灵活性来捕捉 $p_\theta(z|x)$ 中固有的不确定性的真实范围和形态，尤其当真实后验的形式随 $x$ 变化显著时。
5. **训练时的计算负担：** 尽管测试时推断速度快，但与可能没有明确推断组件或使用更简单的非摊销推断方案的生成模型相比，训练推断网络本身会引入额外的参数和计算步骤（尽管后者对于大型 VAE 通常难以处理）。然而，考虑到测试时推断的快速性和对大型数据集的适用性，这种训练成本通常是可以接受的。

理解这些优点和不足很重要，有助于有效地使用 VAE 并了解本章后续讨论的高级推断技术背后的动因。许多这些高级方法旨在减轻标准摊销变分推断的一项或多项局限，例如，通过提出更具表达能力的 $q_\phi(z|x)$ 族或改进 ELBO 的估计。

## 参考资料

- [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114) — Diederik P. Kingma, Max Welling (2014)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1312.6114](https://doi.org/10.48550/arXiv.1312.6114)
  介绍了变分自编码器（VAE）和使用推断网络进行摊销变分推断的概念，奠定了VAE的基础。
- [Variational Inference: A Review for Statisticians](https://www.tandfonline.com/doi/full/10.1080/01621459.2017.1305993) — David M. Blei, Alp Kucukelbir, Jon D. McAuliffe (2017)
  Journal: Journal of the American Statistical Association; Publisher: Taylor & Francis Ltd.; Volume: 112; Pages: 859-877; DOI: [10.1080/01621459.2017.1305993](https://doi.org/10.1080/01621459.2017.1305993)
  全面回顾了变分推断这一通用技术，为VAE中应用的变分推断提供了更广泛的理论背景。
- [Importance Weighted Autoencoders](https://arxiv.org/pdf/1509.00519) — Yuri Burda, Roger Grosse, Ruslan Salakhutdinov (2016)
  Journal: International Conference on Learning Representations (ICLR); Publisher: OpenReview.net; DOI: [10.48550/arXiv.1509.00519](https://doi.org/10.48550/arXiv.1509.00519)
  提出了重要性加权自编码器（IWAE），以获得更紧密的对数似然下界，从而间接解决了“摊销差距”问题并改进了后验近似。
- [Variational Inference with Normalizing Flows](https://doi.org/10.48550/arXiv.1505.05770) — Danilo Jimenez Rezende, Shakir Mohamed (2015)
  Journal: Proceedings of the International Conference on Machine Learning; Publisher: PMLR; Volume: 37; Pages: 1083-1092; DOI: [10.48550/arXiv.1505.05770](https://doi.org/10.48550/arXiv.1505.05770)
  引入了归一化流来增强近似后验分布的表达能力，直接解决了标准摊销VAE的一个关键局限。
