---
course: "evaluating-synthetic-data-quality"
chapter: "specialized-model-specific-metrics"
lesson: "metrics-gan-evaluation"
sourceId: 5709
sourceUrl: "https://apxml.com/zh/courses/evaluating-synthetic-data-quality/chapter-5-specialized-model-specific-metrics/metrics-gan-evaluation"
title: "生成对抗网络（GAN）评估指标"
description: "讨论生成对抗网络背景下常用的一些指标。"
order: 4
plots: ["plots/5709-0.json"]
sourceHash: "5b7af8f20df6474f57bbd920c647d2fa9c435b878a16c8246e70fe57c46056d0"
sourceCorrections: []
---

通用指标能对合成数据质量进行全面评估，而生成对抗网络（GAN）因其对抗训练过程而具备独特之处。生成器（$G$）和判别器（$D$）之间的动态作用在评估时需特别关注。尽管Fréchet Inception Distance (FID) 或 Inception Score (IS) 等指标是评估图像GAN的标准方法，且精确率/召回率能衡量保真度和多样性，但某些评估方法更直接地关注GAN的内部运行状态或潜在故障模式。

### 使用判别器

判别器的作用是区分真实样本和生成样本。其在训练过程乃至训练完成后的表现，能为生成器的能力提供诊断信息。

#### 判别器损失作为诊断工具

训练期间，生成器（$L_G$）和判别器（$L_D$）的损失会持续监控。理想情况下，在运行良好的GAN中，这些损失应达到某种平衡状态。

- **判别器损失高（$L_D$ 高）：** 表明判别器难以区分真实样本和合成样本。这可能意味着生成器正在生成逼真的样本。但它也可能意味着判别器未能有效学习。
- **判别器损失低（$L_D$ 接近零）：** 表明判别器容易区分真实样本和虚假样本。这通常表示生成器表现不佳，生成了容易识别的合成样本。模式崩溃也可能以这种方式出现，即生成器仅生成少量不同且可识别的输出。
- **生成器损失（$L_G$）：** 低 $L_G$ 通常表明生成器成功欺骗了判别器。然而，解释此值需要结合 $L_D$ 的上下文 (context)。如果 $L_D$ 也高，这可能表明生成成功。如果 $L_D$ 低，则低的 $L_G$ 可能意义不大，因为生成器正在欺骗一个无效的判别器。

**注意：** 原始损失值通常非常嘈杂，且高度依赖于具体的GAN架构、损失函数 (loss function)（例如，minimax、Wasserstein）和超参数 (parameter) (hyperparameter)设置。它们本身很少是绝对样本质量或多样性的可靠指标。绘制训练周期中的损失曲线，能更好地诊断稳定性及潜在问题，例如不收敛或振荡。它们最适合作为训练期间的*相对*指标或用于比较不同的训练过程，而非独立的质量分数。



![GAN训练期间损失曲线](plots/5709-0.json)



> 示例图显示判别器损失下降而生成器损失上升，这可能表明生成器正在学习欺骗判别器，同时判别器随时间略有改善，达到某种稳定状态。

#### 事后判别器评估

一种方法是，在GAN训练完成后，使用训练好的判别器（或训练一个新的分类器）来区分保留的真实样本和新生成的合成样本。该分类器的准确率可作为衡量指标。高准确率表明合成数据容易与真实数据区分，这意味着较低的质量或保真度。这类似于倾向得分评估（在第2章中讨论过），但它使用GAN自己的（或类似的）判别器架构。

### 评估收敛性和稳定性

GAN训练不以传统意义上的方式收敛，即最小化单个损失函数 (loss function)。它在零和博弈中寻求平衡。评估是否有效达到此平衡，或训练是否不稳定，这很重要。

- **指标稳定性：** 监控在整个训练过程中定期计算的标准质量指标（例如图像的FID）。如果指标改善后稳定或趋于平稳，这可能表明朝向根据该指标定义的最佳生成质量收敛。在最初改善后，指标出现不规则波动或恶化可能表示GAN组件内部的不稳定性或过拟合 (overfitting)。
- **模式崩溃检测：** 尽管难以通过仅与GAN机制相关的单一指标直接量化 (quantization)，但严重的模式崩溃（生成器产生非常有限的多样性）通常表现为：
  - 判别器损失非常低（如前所述）。
  - 在多样性指标（如召回率）上得分不佳（在专业指标章节中有所介绍）。
  - 视觉检查显示重复输出。

### 生成器固有属性

不常用，但有时相关的是基于生成器内部结构的评估：

- **潜在空间插值：** 对于具有行为良好的潜在空间（如StyleGAN）的GAN，在两个潜在向量 (vector) $z_1$ 和 $z_2$ 之间插值应在生成的输出空间中产生平滑、逼真的过渡。沿着插值路径的“平滑度”或“逼真度”的视觉检查或定量测量，能提供对生成器数据流形理解的信息。锯齿状过渡或不逼真的中间样本可能表明存在问题。

### 与通用及特定领域指标的关系

认识到GAN评估高度依赖于本课程其他地方讨论过的通用及特定领域指标，这很重要。

- **FID, IS, KID（图像）：** 这些是评估GAN生成图像质量和多样性的标准方法。它们比较预训练 (pre-training)网络（如Inception V3）提取的特征分布。
- **精确率和召回率（通用）：** 这些适用于分布的指标，对GAN诊断有价值。高精确率表明生成的样本逼真（落在真实数据分布内）。高召回率表明生成器覆盖了真实数据分布的大部分模式，有助于抵抗模式崩溃。
- **特定领域指标（文本、时间序列）：** 如果GAN生成文本或时间序列数据，困惑度、BLEU分数或自相关比较等指标（在各自章节中有所介绍）是不可或缺的。

### 实际考量

- **没有单一的最佳指标：** 有效评估GAN需要一套指标。仅依赖一个指标，例如判别器损失或甚至仅FID，都可能产生误导。
- **计算成本：** FID等指标的计算成本可能较高，因为它们需要生成大量样本并使其通过大型预训练 (pre-training)网络。
- **指标相关性：** 了解不同指标之间的关系。例如，FID的改善可能与更好的视觉质量相关，但不一定与多样性（召回率）的改善相关。
- **视觉检查：** 绝不能低估查看生成样本的作用，尤其是在开发和调试期间。定性评估经常能发现定量指标可能遗漏的问题。

总之，尽管许多强大的评估指标适用于不同的生成模型，但了解GAN的特定运行状态，能使用判别器表现和稳定性检查进行有针对性的诊断。这些应始终结合更广泛的统计保真度、效用和特定领域评估使用，以对GAN生成的合成数据形成全面评估。

## 参考资料

- [Generative Adversarial Networks](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf) — Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio (2014)
  Journal: Advances in Neural Information Processing Systems; Pages: 2672-2680; DOI: [10.48550/arXiv.1406.2661](https://doi.org/10.48550/arXiv.1406.2661)
  这篇开创性论文介绍了生成对抗网络及其对抗训练框架，对于理解GAN评估的内在机制至关重要。
- [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/pdf/1706.08500.pdf) — Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, Sepp Hochreiter (2017)
  Journal: Advances in Neural Information Processing Systems; Publisher: Neural Information Processing Systems Foundation, Inc. (NeurIPS); Volume: 30; Pages: 6626-6637; DOI: [10.5555/3295222.3295293](https://doi.org/10.5555/3295222.3295293)
  本文提出了Fréchet Inception Distance (FID)，这是一个广泛使用且重要的指标，用于评估GAN生成的图像质量和多样性。
- [Improved Precision and Recall Metric for Assessing Generative Models](https://arxiv.org/pdf/1904.06991.pdf) — Tuomas Kynkäänniemi, Tero Karras, Samuli Laine, Jaakko Lehtinen, Timo Aila (2019)
  Journal: Advances in Neural Information Processing Systems; Publisher: Neural Information Processing Systems Foundation; Volume: 32; Pages: 3929-3938; DOI: [10.48550/arXiv.1904.06991](https://doi.org/10.48550/arXiv.1904.06991)
  该论文提出了一种改进的Precision和Recall指标，用于评估生成模型，为判别数据保真度和多样性以及诊断模式崩溃等问题提供了特定见解。
- [Wasserstein GAN](http://proceedings.mlr.press/v70/arjovsky17a/arjovsky17a.pdf) — Martin Arjovsky, Soumith Chintala, Léon Bottou (2017)
  Journal: Proceedings of the 34th International Conference on Machine Learning (ICML); Volume: 70; Pages: 214-223; DOI: [10.5555/3305890.3305953](https://doi.org/10.5555/3305890.3305953)
  本文介绍了Wasserstein GAN，它解决了GAN训练中的稳定性问题和模式崩溃，为理解和诊断GAN收敛及故障模式提供了重要背景。
