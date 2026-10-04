---
course: "foundations-transformers-architecture"
chapter: "advanced-architectural-variants-analysis"
lesson: "scaling-laws-transformers"
sourceId: 2052
sourceUrl: "https://apxml.com/zh/courses/foundations-transformers-architecture/chapter-6-advanced-architectural-variants-analysis/scaling-laws-transformers"
title: "神经网络语言模型的缩放法则"
description: "审视模型性能如何随数据、模型大小和计算量的变化而缩放的经验发现。"
order: 9
plots: ["plots/2052-0.json"]
sourceHash: "e82854e4cd140a966eef9e75d98bd4c83cacc2024f7bab593e3493018b6e7773"
sourceCorrections: []
---

理解性能如何随着资源的增加——特别是计算能力、数据和模型大小——而变化，对提高Transformer架构的效率具有重要作用。这种关系对于设计实验、分配预算以及预测未来大型模型的能力非常重要。经验研究发现了一些令人惊讶的可预测规律，这些规律常被称为缩放法则。

### 核心思想：可预测的性能缩放

开创性的工作，特别是Kaplan等人（2020年）的研究表明，语言模型的性能（通常通过在未见过数据上的交叉熵损失$L$衡量）随着三个主要因素的变化而可预测地提高：

1. **模型大小 ($N$)**: 模型中非嵌入 (embedding)参数 (parameter)的数量。
2. **数据集大小 ($D$)**: 训练期间处理的token数量。
3. **计算量 ($C$)**: 训练的总计算成本，以浮点运算（FLOPs）衡量，对于Transformer模型通常近似为$C \approx 6ND$。

主要发现是，损失与这些因素之间的关系通常遵循幂律，至少在几个数量级范围内。这意味着随着规模的增加，损失会平滑且可预测地降低：


$$
L(N, D, C) \approx L_\infty + \left( \left( \frac{N_c}{N} \right)^{\alpha_N} + \left( \frac{D_c}{D} \right)^{\alpha_D} + \left( \frac{C_c}{C} \right)^{\alpha_C} \right)
$$


或者，更简单地说，当一个因素是瓶颈时：

- 损失与模型大小：$L(N) \propto N^{-\alpha_N}$
- 损失与数据集大小：$L(D) \propto D^{-\alpha_D}$
- 损失与计算量：$L(C) \propto C^{-\alpha_C}$

这里，$N_c, D_c, C_c$ 是表示特征尺度的常数，而 $\alpha_N, \alpha_D, \alpha_C$ 是正指数，表示性能随每个因素的缩放程度。$L_\infty$ 表示一个不可减少的损失分量。这种平滑的幂律行为表明，性能提升并非源于特定规模下的突然突破，而是来自对资源的持续投入。



![缩放法则：损失与参数](plots/2052-0.json)



> 一张双对数图表，说明了测试损失通常如何随着模型大小（参数数量）的增加而降低，遵循可预测的幂律趋势。

### 优化资源分配：计算预算限制

Kaplan等人的一项核心发现与在固定计算预算 ($C$) 下优化资源分配有关。由于 $C \approx 6ND$，如果计算预算固定，增加模型大小 ($N$) 可能意味着你需要减少数据集大小 ($D$)。他们的分析表明，为了获得最佳性能（在给定 $C$ 下实现最低损失），通常最好优先更快速地增加模型大小 ($N$) 而非数据集大小 ($D$)。他们发现性能与模型大小的缩放关系强于与数据集大小的缩放关系（$\alpha_N$ 略大于 $\alpha_D$）。

这意味着，在相对较小的数据集上训练较少步骤的大型模型，可能比在更大型数据集上训练更长时间的小型模型更具计算效率。

### 修正的缩放法则：Chinchilla的发现

然而，Hoffmann等人（2022年）的后续研究，即“Chinchilla”论文，通过更广泛的实验重新审视了这些缩放法则。他们的发现对优化分配提出了不同的看法。

Chinchilla研究得出结论，对于计算优化训练，模型大小 ($N$) 和数据集大小 ($D$) 应大致*等比例*缩放。具体来说，每当模型大小翻倍时，训练token的数量也应翻倍，以期为投入的计算量获得最佳性能。

这表明，在此发现之前开发的许多大型语言模型（如GPT-3、Gopher）相对于其尺寸而言，都严重*训练不足*。它们相对于所训练的数据量而言，模型过大，未达优化。根据Chinchilla的缩放法则，一个在大量更多数据上训练的较小模型可以用相同的计算预算达到相同的性能，或者，以相同的预算获得更好的性能。

例如，Chinchilla模型（700亿参数 (parameter)）在1.4万亿token上进行训练，在众多基准测试中，其性能优于大得多的Gopher模型（2800亿参数，在3000亿token上训练），尽管两者使用了相似的训练计算量。

### 实际意义

这些缩放法则为实践者提供了有益的指导：

- **性能预测**: 它们允许合理准确地预测通过扩展计算、模型大小或数据可实现的性能提升。
- **资源分配**: 它们为如何分配固定预算（例如GPU小时）提供了参考。Chinchilla的发现表明，模型参数 (parameter)和训练token的平衡增长通常是最佳选择。
- **识别瓶颈**: 如果性能显著偏离预测的缩放曲线，则可能表明训练设置、数据质量或架构选择存在问题。
- **训练策略**: 相较于Kaplan最初研究的含义，修正后的法则更倾向于在更大的数据集上训练模型更长时间。

### 局限性与注意事项

认识到这些法则的局限性很重要：

- **经验性**: 它们是源自特定模型家族（主要是密集型Transformer）和数据集（大多是网络文本）的经验观察。它们可能无法完美推广到所有架构或数据模态。
- **关注预训练 (pre-training)损失**: 缩放法则主要预测预训练损失（如交叉熵）。虽然这与下游任务性能强相关，但这种关系并非总是完美或线性的。特定任务的性能提升可能以不同的方式达到饱和。
- **不可约损失**: 性能不能无限期地提升。存在一个与语言本身固有不可预测性相关的理论最小损失 ($L_\infty$)。随着模型变得非常大，当它们接近此极限时，收益会减少。
- **常数因素**: 精确指数 ($\alpha_N, \alpha_D, \alpha_C$) 和常数 ($N_c, D_c, C_c$) 可能因模型架构细节、数据质量和分词 (tokenization)方法等因素而异。

尽管有这些考量，缩放法则代表了在理解如何有效构建和训练大型语言模型方面的重大进步。它们提供了一个量化 (quantization)框架，用于推理 (inference)扩展AI系统所涉及的权衡。当你设计或使用高级Transformer架构时，理解这些经验关系对于对模型大小、数据需求和计算资源做出明智的决策是基本的。

## 参考资料

- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei (2020)
  Journal: International Conference on Learning Representations (ICLR 2020); DOI: [10.48550/arXiv.2001.08361](https://doi.org/10.48550/arXiv.2001.08361)
  这篇论文介绍了语言模型性能的经验性缩放定律。
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) — Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals, Laurent Sifre (2022)
  Journal: arXiv preprint arXiv:2203.15556; DOI: [10.48550/arXiv.2203.15556](https://doi.org/10.48550/arXiv.2203.15556)
  此论文提出了Chinchilla研究成果，修订了大型语言模型计算资源最优分配的策略，建议模型和数据集规模应更均衡地扩展。
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS 2020); DOI: [10.48550/arXiv.2005.14165](https://doi.org/10.48550/arXiv.2005.14165)
  这篇论文介绍了GPT-3，展示了大型模型中涌现的少样本学习能力和缩放原则的实际应用。
