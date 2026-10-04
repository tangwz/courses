---
course: "deep-learning-regularization-optimization"
chapter: "optimization-refinements-tuning"
lesson: "batch-size-learning-rate"
sourceId: 5005
sourceUrl: "https://apxml.com/zh/courses/deep-learning-regularization-optimization/chapter-7-optimization-refinements-tuning/batch-size-learning-rate"
title: "批处理大小与学习率的关系"
description: "考察批处理大小与合适学习率的关系。"
order: 8
plots: ["plots/5005-0.json"]
sourceHash: "caa575f3a42af4518156a3d6079884b0224c79b0931e84585ac3129dcd8a3d54"
sourceCorrections: []
---

当我们调整超参数 (parameter) (hyperparameter)以优化模型训练时，**批处理大小**和**学习率**是两个常相伴的设置。改变其中一个通常需要调整另一个以获得良好表现。让我们分析一下这两者为何联系紧密。

小批量梯度下降 (gradient descent)在每一步使用训练数据的一个小部分（一个小批量）来计算梯度估计。小批量的大小，$B$，直接影响梯度估计的质量和特点。

### 梯度噪声与批处理大小

随机梯度下降 (gradient descent)（SGD）及其小批量变体的一个基本特点是，仅使用数据的一个子集来估计梯度会引入噪声。

- **小批处理大小：** 使用小批处理大小（例如，16、32、64）会导致梯度估计具有高方差。由于样本量小，每个批处理可能提供明显不同的梯度方向。这种噪声有时会有益，可能帮助优化器避开不良局部最小值或鞍点。但是，它也会使收敛路径不稳定。
- **大批处理大小：** 反之，使用大批处理大小（例如，256、512、1024或更大）会导致梯度估计具有较低方差。从大批处理计算出的梯度是对整个数据集真实梯度的更好近似。这会带来更平滑的收敛路径，但每一步可能需要更多计算，并且可能收敛到更尖锐的最小值，有时泛化效果较差。

小批量梯度估计$\hat{g}_B$的方差大致与批处理大小$B$成反比：


$$
Var(\hat{g}_B) \approx \frac{1}{B} Var(g_i)
$$


其中$g_i$表示单个数据样本的梯度。较大的$B$会降低方差（噪声）。

### 学习率与梯度噪声的关联

学习率$\alpha$决定了沿着（估计的）负梯度方向迈进的步长。合适的学习率与该梯度估计的可靠性（噪声水平）联系紧密。

- **有噪声的梯度（小批处理）：** 如果梯度方向在批处理之间剧烈波动，采取大步长（高学习率）可能导致优化器越过最小值甚至发散。更新可能不稳定地跳动。通常需要较小的学习率来平均噪声，并稳定地向最小值前进。
- **更平滑的梯度（大批处理）：** 当梯度估计更稳定并准确反映真实梯度方向时，可以更自信地迈出更大的步长。较高的学习率可以导致更快的收敛，而没有噪声梯度所带来的发散风险。

### 线性缩放规则：一种常用经验法则

基于批处理大小、梯度噪声和合适步长之间的关系，出现了一个流行的经验法则，通常被称为**线性缩放规则**：

> 如果将批处理大小乘以因子$k$，则将学习率也乘以相同的因子$k$。

**直观理解：** 如果您使用的批处理大小增大$k$倍，您的梯度估计将基于$k$倍的数据量，因此（大致）噪声减少$k$倍；或者从另一个角度看，如果单个梯度相似，则较大批处理中的梯度之和在幅度上会增大$k$倍。为了保持相似的更新幅度或利用更可靠的梯度方向，您可以按比例增加学习率。例如，如果您将批处理大小从64更改为256（增大$4 \times$），此规则建议将学习率增大$4 \times$。

**注意事项：** 线性缩放规则是一种经验法则，而非严格的定律。

- 它通常在带有动量的SGD上表现最佳。像Adam这样的自适应优化器可能不那么敏感。
- 对于非常大的批处理大小，它往往会失效，此时泛化性能可能会下降。
- 通常建议将此缩放与**学习率预热**期（前面已讨论）结合使用，尤其是在使用大学习率时，以防止训练早期出现不稳定。

### 权衡与实际考量

选择批处理大小和学习率涉及多种权衡：

1. **计算成本：** 较大的批处理可以更有效地利用并行处理能力（如GPU），可能导致每个周期的实际运行时间更快。但是，它们需要更多内存。较小的批处理需要更少内存，但如果硬件未得到充分利用，或由于学习率较小而需要更多周期，则可能导致整体训练变慢。
2. **泛化性能：** 这是一个活跃的研究方向。一些发现表明，较小的批处理大小会引入有益的噪声，帮助优化器找到更平坦的最小值，这通常比大批处理训练有时找到的更尖锐的最小值具有更好的泛化能力。但是，大批处理训练有时也可以通过调整来达到可比的结果。
3. **收敛速度：** 较大的批处理与适当缩放的大学习率结合，可以在所需周期数方面加快收敛。



![批处理大小/学习率关联](plots/5005-0.json)



> 不同批处理大小和学习率组合的损失曲线图示。大批处理/大学习率通常最初收敛最快，而小批处理/大学习率可能变得不稳定。

### 调优策略

调优这些超参数 (parameter) (hyperparameter)时：

- **从常用批处理大小开始：** 选择适合您硬件内存的批处理大小（例如，32、64、128、256），并为其找到一个好的学习率。
- **考虑缩放：** 如果您需要更改批处理大小（例如，出于内存原因或为了泛化表现而试验），请尝试使用线性缩放规则作为起点来调整学习率。
- **微调 (fine-tuning)：** 线性缩放规则提供了一个起点，不一定是最终的合适值。更改批处理大小后，您可能需要对学习率进行一些微调。
- **监测：** 始终监测您的训练和验证损失曲线。不稳定（损失增加或剧烈波动）通常表明所选批处理大小的学习率过高。收敛非常慢可能意味着学习率过低。

理解批处理大小和学习率之间的关系对高效的超参数调优非常重要。虽然线性缩放规则等经验法则提供了指导，但仍需要通过经验测试和仔细监测来找到最适合您的特定模型和数据集的组合。

## 参考资料

- [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677) — Priya Goyal, Piotr Dollár, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, Kaiming He (2017)
  Journal: arXiv preprint; DOI: [10.48550/arXiv.1706.02677](https://doi.org/10.48550/arXiv.1706.02677)
  介绍了学习率随大批量大小线性缩放的规则，并讨论了学习率预热的重要性。
- [On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima](https://arxiv.org/abs/1609.04836) — Nitish Shirish Keskar, Dheevatsa Mudigere, Jorge Nocedal, Mikhail Smelyanskiy, Ping Tak Peter Tang (2017)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1609.04836](https://doi.org/10.48550/arXiv.1609.04836)
  研究了使用大批量训练时出现的泛化差距，将其归因于收敛到更尖锐的最小值。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  一本基础教材，涵盖了深度学习中的各种优化算法及其特性。
- [Don't Decay The Learning Rate, Increase The Batch Size](https://arxiv.org/abs/1711.00489) — Samuel L. Smith, Pieter-Jan Kindermans, Chris Ying, Quoc V. Le (2017)
  Journal: International Conference on Learning Representations (ICLR) 2018; DOI: [10.48550/arXiv.1711.00489](https://doi.org/10.48550/arXiv.1711.00489)
  探讨了一种替代策略，即通过增加批量大小而非衰减学习率来实现高效训练。
