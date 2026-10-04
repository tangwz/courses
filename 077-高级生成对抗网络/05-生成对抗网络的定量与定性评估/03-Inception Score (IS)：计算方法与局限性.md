# Inception Score (IS)：计算方法与局限性

来源：[原文](https://apxml.com/zh/courses/generative-adversarial-networks-gans/chapter-5-evaluation-of-gans/inception-score-formulation-limitations)

[返回章节目录](README.md) · [返回课程目录](../README.md)

尽管视觉检查能提供有价值的定性观察，但它具有主观性且难以扩展。我们需要自动化的定量指标来全面评估GAN性能，比较不同模型，并追踪训练进展。Inception Score (IS) 是最早且最普遍采用的指标之一。它旨在体现生成图像的两个理想属性：

1. **图像质量：** 生成图像应看起来是特定的且可识别的某种东西。当输入到预训练 (pre-training)的图像分类器（如在ImageNet上训练的Inception-v3）时，分类器应为每个图像分配高概率给单个类别。
2. **图像多样性：** 生成器应生成多种多样的图像，涵盖真实数据中存在的不同类型或类别。在大量生成的图像中，预测类别的分布应是多样化或相对均匀的。

### Inception Score的计算方法

Inception Score使用预训练 (pre-training)的Inception-v3网络，因其在ImageNet数据集上的出色性能而被选择。核心思路是利用分类器对生成样本的预测结果来衡量上述属性。

令$x$为生成器$G$生成的图像，即$x \sim p_g$。我们将$x$通过Inception-v3模型，以获得条件概率分布$p(y|x)$，其中$y$代表类别标签（来自1000个ImageNet类别）。

1. **质量衡量：** 如果图像$x$质量高且清晰描绘了ImageNet类别中的一个对象，则分布$p(y|x)$应具有低熵。这意味着模型应能很确定$x$属于哪个类别。低熵表示概率质量集中在少数几个类别上（理想情况下是一个）。
2. **多样性衡量：** 如果生成器生成了涵盖许多类别的多样化图像，则所有生成图像上的边缘分布$p(y)$应具有高熵。这个边缘分布是通过对所有生成样本的条件分布求平均得到的：

   
   $$
   p(y) = \int_x p(y|x) p_g(x) dx
   $$
   

   实际中，$p(y)$通过对大量生成样本$\{x_i\}_{i=1}^N$的$p(y|x)$求平均来估算：

   
   $$
   \hat{p}(y) \approx \frac{1}{N} \sum_{i=1}^N p(y|x_i)
   $$
   

   $\hat{p}(y)$的高熵表明，经Inception-v3分类的生成图像相对均匀地涵盖了广泛的类别。

Inception Score使用Kullback-Leibler (KL) 散度结合了这两个思路。具体来说，它衡量了每个图像的条件分布$p(y|x)$与边缘分布$p(y)$之间的散度。我们希望$p(y|x)$是集中（低熵）的，而$p(y)$是均匀（高熵）的。KL散度$D_{KL}(p(y|x) || p(y))$量化 (quantization)了$p(y|x)$与$p(y)$的差异程度。在这里，大的KL散度是期望的，表明单个图像强烈对应特定类别，而整体类别使用是多样的。

最终的Inception Score通过对所有生成样本的KL散度求平均并对结果进行指数运算来计算：


$$
IS(G) = \exp\left( \mathbb{E}_{x \sim p_g} [ D_{KL}(p(y|x) || p(y)) ] \right)
$$


实际中，这通过使用大量样本集来近似：


$$
IS(G) \approx \exp\left( \frac{1}{N} \sum_{i=1}^N D_{KL}(p(y|x_i) || \hat{p}(y)) \right)
$$


较高的Inception Score通常被解读为更好的性能，表明生成器生成的图像既高质量（易于分类）又多样（涵盖许多类别）。

### Inception Score的局限性

尽管具有直观吸引力并被广泛使用，Inception Score仍存在几个显著局限，尤其在使用高级GAN时值得了解：

1. **对预训练 (pre-training)模型的依赖：** IS本质上与在ImageNet上训练的Inception-v3模型相关联。它衡量的是该特定分类器认为对区分ImageNet类别重要的特征。如果您的目标数据集与ImageNet显著不同（例如，医疗扫描、抽象艺术、特定人脸数据集），这些特征可能无法完全吻合人类对图像质量的感知或您目标数据集的特性。
2. **未与真实数据比较：** 该分数*仅*根据生成图像计算。它不直接比较生成图像的分布 ($p_g$) 与真实图像的分布 ($p_{data}$)。理论上，生成器可以通过生成多样化、清晰可分类的图像来获得高IS，即使这些图像与实际训练数据毫无相似之处。例如，生成完美的狗和猫的图像可能会得到一个好的IS，即使训练数据只包含汽车。
3. **对ImageNet类别的敏感性：** 该分数本质上奖励生成类似于ImageNet中1000个类别的图像的生成器。如果您的GAN是在具有不同对象类别的数据集上训练的，那么IS可能不是一个有意义的性能衡量标准。例如，一个训练用于生成MNIST数字的GAN，很可能会得到一个非常低的IS，因为数字不强烈映射到像“狗”或“汽车”这样的ImageNet类别。
4. **发现模式崩溃的能力有限：** 尽管严重的模式崩溃（仅生成一种或极少数不同图像类型）应导致低熵的边缘分布$p(y)$，从而降低IS，但此指标并非万无一失。生成器可能会崩溃到只完美生成少数ImageNet类别。如果这少数类别*在它们之间*是多样的，边缘熵可能仍然相当高，从而掩盖了相对于完整数据集的多样性不足。
5. **平均性质：** 该分数对样本的KL散度进行平均。生成器可能会生成许多好的样本和少数糟糕的样本；平均分数可能仍然看起来可以接受，从而隐藏潜在的失败模式。
6. **计算成本：** 计算IS需要生成大量样本（通常是数万个），并对每个样本使用相对较大的Inception-v3模型执行推断，这可能计算量大。

> Inception Score提供一个单一数字，从预训练分类器的视角总结质量和多样性。然而，它不比较生成样本与真实样本，并且偏向ImageNet特征。

由于这些局限性，尽管IS是GAN评估方面的一个重要进步，但它经常被更新的指标（如Fr\u00e9chet Inception 距离 (FID)）补充或取代。FID使用来自相同Inception网络的特征直接比较生成样本与真实样本的统计数据。理解IS为理解GAN评估技术的发展提供了有价值的背景。

## 参考资料

- [Improved Techniques for Training GANs](https://arxiv.org/pdf/1606.03498.pdf) — Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, Xi Chen (2016)
  Journal: Advances in Neural Information Processing Systems; Publisher: Advances in Neural Information Processing Systems; Volume: 29; Pages: 2234-2242; DOI: [10.48550/arXiv.1606.03498](https://doi.org/10.48550/arXiv.1606.03498)
  这篇基础性论文介绍了Inception Score (IS) 作为衡量GAN生成图像质量和多样性的定量指标。
- [Rethinking the Inception Architecture for Computer Vision](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf) — Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, Zbigniew Wojna (2016)
  Journal: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR); Publisher: IEEE; Pages: 2818-2826; DOI: [10.1109/CVPR.2016.36](https://doi.org/10.1109/CVPR.2016.36)
  本文介绍了Inception-v3神经网络架构，它是用于计算Inception Score的预训练分类器。
- [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://proceedings.neurips.cc/paper/2017/file/8a1d69470766624e5b974b7c1528b6f7-Paper.pdf) — Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, Sepp Hochreiter (2017)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 30; Pages: 6626-6637; DOI: [10.48550/arXiv.1706.08500](https://doi.org/10.48550/arXiv.1706.08500)
  本文介绍了Fréchet Inception Distance (FID)，这是一个广泛使用的度量标准，通过比较生成数据和真实数据分布，解决了Inception Score的一些局限性。
- [Are GANs Created Equal? A Large-Scale Study](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGlcUCrEn11ICzkZ4Himf1mrs9_tRzxZkZiMBnf3NPqeVPPGjsSWxzVR94GoUtPyFSDrVcmpehYT74G6BrN37Xorve2ify062mdMpnWf735CbBHy5G8VOR5rNtv5xKXMQas2FcZP9xxKeWpwT8Vk5Ygs9_pyo_rh5n44SeW7_-h4sgIL5jDla_0NXozLw==) — Mario Lucic, Karol Kurach, Marcin Michalski, Sylvain Gelly, Olivier Bousquet (2018)
  Journal: Advances in Neural Information Processing Systems; Volume: 31; Pages: 520-529; DOI: [10.5591/978-1-57766-081-6.1030](https://doi.org/10.5591/978-1-57766-081-6.1030)
  这项研究全面比较了包括Inception Score在内的多种GAN评估指标，提供了对其经验表现和实际考虑的见解。

---

[上一节](02-%E5%AE%9A%E6%80%A7%E8%AF%84%E4%BC%B0%EF%BC%9A%E8%A7%86%E8%A7%89%E5%9B%BE%E7%81%B5%E6%B5%8B%E8%AF%95.md) · [下一节](04-Fr%C3%A9chet%20Inception%20%E8%B7%9D%E7%A6%BB%20%28FID%29-%20%E5%85%AC%E5%BC%8F.md)
