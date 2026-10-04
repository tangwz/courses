---
course: "vae-representation-learning"
chapter: "advanced-vae-topics-extensions"
lesson: "denoising-vaes-robustness"
sourceId: 6385
sourceUrl: "https://apxml.com/zh/courses/vae-representation-learning/chapter-7-advanced-vae-topics-extensions/denoising-vaes-robustness"
title: "去噪VAE与输入扰动鲁棒性"
description: "通过去噪目标和输入扰动增强VAE鲁棒性和特征学习。"
order: 6
plots: []
sourceHash: "3692a913890bcb33dbf13d680c43a2edd3d43ec2657584f743017a88db0cc25d"
sourceCorrections: []
---

去噪变分自编码器（DVAE）是一种显著提升VAE应用中模型能力和所学表征质量的技术。数据很少是完全纯净的；它常被噪声、缺失值或其他扰动所污染。仅在纯净、理想化数据上表现良好的模型，其实用价值有限。DVAE通过训练VAE从受损版本中重建纯净数据来解决此问题，从而学会忽略噪声，并关注数据的内在结构。

这种方法不仅使VAE对噪声输入更具韧性，还作为一种有效的正则化 (regularization)器，常促成发现更有意义且解耦的潜在特征。通过迫使模型将信号与噪声分离，我们鼓励它捕获数据中变动的本质因素。

### 变分自编码器中的去噪原理

去噪的核心思想并非新生；它借鉴自去噪自编码器（DAE），其中标准自编码器经过训练，从其随机损坏版本$\tilde{x}$重建原始输入$x$。在VAE的背景下，此原理巧妙地融入概率框架。

去噪VAE（DVAE）对标准VAE设置进行如下修改：

1. 从数据集中取一个纯净输入数据点 $x$。
2. 通过对 $x$ 施加噪声过程生成一个损坏版本 $\tilde{x}$。
3. 编码器 $q_\phi(z|\tilde{x})$ 将此损坏输入 $\tilde{x}$ 映射到潜在空间中的一个分布。
4. 从 $q_\phi(z|\tilde{x})$ 中采样一个潜在向量 (vector) $z$。
5. 解码器 $p_\theta(x|z)$ 随后尝试从 $z$ 重建*原始的、纯净的*输入 $x$。

目标函数，即证据下界（ELBO），经过调整以体现此点。如果标准VAE的ELBO为：


$$
L(x; \theta, \phi) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - KL(q_\phi(z|x) || p(z))
$$


DVAE的ELBO变为：


$$
L_{DVAE}(x, \tilde{x}; \theta, \phi) = \mathbb{E}_{q_\phi(z|\tilde{x})}[\log p_\theta(x|z)] - KL(q_\phi(z|\tilde{x}) || p(z))
$$


注意主要区别：重建项 $\log p_\theta(x|z)$ 的期望是根据 $q_\phi(z|\tilde{x})$ 取的，这意味着潜在表征 $z$ 是从*损坏的*输入 $\tilde{x}$ 推断而来，但解码器任务是重建*纯净的*输入 $x$。KL散度项也以 $\tilde{x}$ 为条件，基于噪声观测对后验近似进行正则化 (regularization)。

### 去噪的机制与益处

用去噪目标训练VAE促使模型学习几项重要特性。

**1. 对扰动的鲁棒性：**
通过在训练期间使编码器接触各种形式的噪声，模型在测试时学会对这类扰动不那么敏感。它实质上学会了“看穿”噪声，并提取底层信号。这在输入数据可能降级的应用中尤其有价值，例如来自低质量传感器的图像或带有拼写错误的文本。

**2. 增强特征学习：**
将真实数据结构与噪声分离的要求迫使VAE学习更显著和不变的特征。潜在空间 $z$ 倾向于捕获数据更根本的方面，因为表面化的、噪声引起的变动被阻止编码。这常会带来：

- **更平滑的潜在流形：** 所学数据在潜在空间中的流形变得更“平滑”，或者对与噪声相关的微小输入变化不那么敏感。
- **改进解耦（间接）：** 尽管这不是其主要目标，但对本质特征的关注有时能有助于更好的解耦，因为噪声是DVAE学会忽略的混淆因素。

下面的图表说明了DVAE过程：

> 去噪VAE过程：纯净输入 $x$ 被损坏为 $\tilde{x}$。编码器将 $\tilde{x}$ 映射到潜在表征 $z$。解码器随后尝试从 $z$ 重建原始纯净输入 $x$。训练目标优化 $x$ 的精确重建，并对潜在空间进行正则化 (regularization)。

**3. 正则化：**
去噪任务本身就是一种正则化形式，防止模型仅仅学习一个恒等函数（如果潜在空间足够大且不存在噪声，这对于自编码器来说是微不足道的）。它促使模型学习一个压缩表征，只保留重建纯净数据所需的本质信息。

### 输入扰动的种类

噪声或损坏过程 $C(x)$ 的选择灵活，可根据数据模态和预期的噪声类型进行调整。常见选择包括：

- **加性高斯噪声：** $\tilde{x} = x + \epsilon$，其中 $\epsilon \sim \mathcal{N}(0, \sigma^2I)$。这是一种通用型噪声。
- **椒盐噪声：** 对于图像，随机将一部分像素设为最小值或最大值。
- **遮蔽噪声：** 随机将一部分输入特征（例如，图像中的像素，句子中的单词）设为零或特殊遮蔽标记 (token)。这迫使模型学习从上下文 (context)中推断缺失信息。
- **Dropout作为噪声：** 对输入层应用dropout也可视为一种遮蔽噪声形式。

噪声的强度和类型是超参数 (parameter) (hyperparameter)。噪声过少可能无法提供足够的正则化 (regularization)效果，而噪声过多会使重建任务变得难以处理，从而阻碍学习。

### 对潜在空间和数据流形的影响

DVAE的一种主要解释是从流形学习的角度。假设纯净数据点 $x$ 位于或接近嵌入 (embedding)在高维输入空间中的低维流形。噪声过程 $C(x)$ 将这些点 $\tilde{x}$ 推离此流形。
DVAE学习到：

- 一个编码器 $q_\phi(z|\tilde{x})$，能够有效地将这些偏离流形的点 $\tilde{x}$ 投射回潜在空间中对应于纯净数据流形的区域。
- 一个解码器 $p_\theta(x|z)$，将这些潜在表征映射回纯净数据流形上（或非常接近）的点 $x'$。

本质上，去噪目标鼓励VAE学习数据流形的底层结构，并对偏离此流形的情况变得更有适应力。模型学会“拉回”损坏数据点至此流形，有效地平滑从输入空间到潜在空间的映射，在真实数据分布附近。

### 实际实现考量

在实现DVAE时，请考量以下几点：

- **噪声水平 $(\sigma)$：** 噪声的大小（例如，高斯噪声的标准差、遮蔽概率）是一个重要的超参数 (parameter) (hyperparameter)。它常需要调优，可能通过交叉验证。
- **噪声退火：** 一些策略涉及从低噪声水平开始，并随着训练进行逐渐增加（或反之）。这有时能帮助模型首先学习大体结构，然后改进其去噪能力。
- **随机损坏：** 噪声通常在训练期间对每个数据批次随机地即时施加。这意味着模型在不同时期会看到相同输入的不同损坏版本，这有助于更好的泛化。
- **噪声类型：** 噪声的选择理想情况下应模仿应用领域中预期的扰动类型，或者作为一种有意义的数据增强技术。例如，对于文本，随机丢弃单词或交换相邻单词可能比高斯噪声更合适。

### 去噪VAE与对抗鲁棒性

尽管DVAE增强了对常见、常为随机的噪声模式的一般鲁棒性，但它们与为*对抗鲁棒性*设计的方法不同。对抗性攻击涉及精心制作小的、最差情况的扰动，专门设计来欺骗模型。DVAE并非天生对这类目标攻击免疫，尽管它们带来的改进特征学习和流形平滑可能提供一些有限的、间接的益处。真正的对抗鲁棒性通常需要专门的训练程序，例如对抗训练，我们曾在讨论对抗变分贝叶斯（AVB）时简要提及，并在考察GANs时将与VAE进行比较。

"去噪VAE提供一种直接而有效的机制，以提高变分自编码器的可靠性和特征学习能力。通过训练模型从损坏输入中重建纯净信号，我们不仅使它们对数据缺陷更具韧性，也引导它们学习更根本和有用的表征。这项技术是对VAE工具集有价值的补充，特别是在处理噪声数据集或需要鲁棒特征提取时。"

## 参考资料

- [Extracting and Composing Robust Features with Denoising Autoencoders](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG0Q8JM1TkBK-JEUZqRW5IvyU6Vlt9Uc66FzR3r1pqoWAituUbiwv8iGfNDlJuSvNwbX6mY70qjXVD3aqLaACYSstXAxz2PhGqdpSnrDDlEVg8vA5Ws-gRB6yymH4xaYkWzKYvNkEWF3Bw=) — Pascal Vincent, Hugo Larochelle, Isabelle Lajoie, Yoshua Bengio, Pierre-Antoine Manzagol (2010)
  Journal: Journal of Machine Learning Research; Volume: 11; Pages: 3371-3408
  这篇基础性论文介绍了去噪自编码器（DAEs），通过训练模型从损坏的输入中重建干净数据，为去噪变分自编码器提供了直接的启发。
- [Deep Learning](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE94t8jWg_TXjR9RqvEGcP0T7otfWtC7f12Qjqli5pIqj9ONRdQG-zJPWYSAUaoPvQ-mQ1v5DdOdghBwGQQwCKw7tMV3zpVp9WkieMcsfzn2TFANozycbNK69g/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  这本权威教科书全面概述了深度学习，包括对自编码器、变分自编码器及其正则化技术的详细解释，为DVAE原理提供了更广泛的背景知识。
