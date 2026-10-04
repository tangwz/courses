---
course: "generative-adversarial-networks-gans"
chapter: "gan-implementation-optimization"
lesson: "hyperparameter-tuning-gans"
sourceId: 2771
sourceUrl: "https://apxml.com/zh/courses/generative-adversarial-networks-gans/chapter-7-gan-implementation-optimization/hyperparameter-tuning-gans"
title: "超参数调整策略"
description: "调整学习率、批大小和架构参数的系统性方法。"
order: 3
plots: []
sourceHash: "62d41309a0edf775d413c403311db240dbe29f912c221c4893448f3430dc1e04"
sourceCorrections: []
---

调整生成对抗网络（GAN）的超参数 (parameter) (hyperparameter)，相比于标准监督学习 (supervised learning)模型，面临独特难题。GAN训练的动态双人游戏特性意味着损失持续变化，收敛性无法保证。一组初期表现良好的超参数，在训练后期可能导致不稳定或模式崩溃。此外，标准损失值（生成器损失、判别器损失）通常不能很好地反映样本质量，这使得直接优化它们变得困难。因此，针对超参数调整，采取系统且有耐心的办法，并以恰当的评估指标为指引，是成功训练高级GAN模型的必要条件。

### 在GAN中识别超参数 (parameter) (hyperparameter)

尽管具体超参数取决于所选架构（例如StyleGAN、BigGAN、WGAN-GP）和优化器，但有几个参数始终需要仔细调整：

1. **学习率：** 这通常是最敏感的超参数。生成器（$lr_G$）和判别器（$lr_D$）使用单独的学习率很常见。第三章讨论的双时间尺度更新规则（TTUR）等技术明确建议使用不同的学习率（通常$lr_D > lr_G$）。值通常在$1e^{-5}$到$5e^{-4}$之间。
2. **优化器参数：** 对于Adam或AdamW优化器，动量参数$\beta_1$和$\beta_2$影响训练过程。尽管默认值（$\beta_1=0.9, \beta_2=0.999$）是常见的起点，但GAN文献常建议使用较低的$\beta_1$值（例如0.0或0.5），以减少动量并可能提升稳定性，尤其是对于生成器而言。
3. **批大小：** 较大的批大小通常能提供更稳定的梯度估计，但会增加内存需求，有时可能导致更尖锐的局部最小值，潜在地损害泛化能力。BigGAN展示了使用超大批大小的成功，但这通常需要仔细调整其他参数，例如学习率和稳定化技术。批大小与归一化 (normalization)层（如批量归一化）和正则化 (regularization)技术有明显影响。
4. **损失函数 (loss function)权重 (weight)：** 许多高级GAN在其损失函数中加入额外项。例如：
   - 在WGAN-GP中，梯度惩罚系数（$\lambda_{GP}$）平衡了 Wasserstein 距离估计与 Lipschitz 约束的施加。常见值约为10。
   - 在CycleGAN中，权重控制着对抗损失与循环一致性损失的相对重要性。
   - 在InfoGAN中，权重调节着标准GAN损失与互信息项之间的平衡。
     这些权重直接影响训练优先级，需要仔细调整。
5. **架构参数：** 尽管StyleGAN等核心架构已有既定结构，但网络深度、层宽度（通道数）、归一化类型（批量归一化、实例归一化、层归一化、谱归一化）以及激活函数 (activation function)的变化，在将模型应用于新数据集或任务时，可以被视为超参数。
6. **正则化：** 权重衰减或Dropout等技术，如果使用，有相关的强度参数需要调整。谱归一化，主要是一种稳定化技术，它影响网络的容量和动态表现。

### 系统性调整策略

纯粹依靠直觉或手动调整通常效率低下，并且难以找到复杂GAN的最佳设置。建议采用更结构化的方法：

#### 网格搜索

网格搜索涉及为每个超参数 (parameter) (hyperparameter)定义一组离散值，并评估每种可能组合的模型表现。例如，你可能测试学习率$[1e^{-5}, 5e^{-5}, 1e^{-4}]$和$\beta_1$值$[0.0, 0.5]$。

- **优点：** 易于理解和实现。在定义的网格内是详尽的。
- **缺点：** 受“维度诅咒”影响。组合数量随超参数数量呈指数增长。它会浪费计算资源在没有前景的区域上，并可能遗漏网格点之间的最佳值。对于超过少数几个超参数的情况，通常在计算上不可行，尤其是在GAN训练时间较长时。

#### 随机搜索

随机搜索，由Bergstra和Bengio（2012）提出，从指定范围或分布中随机采样超参数组合。例如，从$10^{-5}$到$10^{-3}$均匀采样$lr$，从$0.0$到$0.9$均匀采样$\beta_1$。

- **优点：** 经验表明，在相同的计算预算下，它比网格搜索更高效，尤其当某些超参数远比其他超参数更重要时。更易于并行化。不太可能完全遗漏良好区域。
- **缺点：** 不如网格搜索那样系统化。可能需要更多样本才能密集覆盖最有前景的区域。表现取决于所选的采样分布。

> 2D超参数空间中评估点的比较。网格搜索在固定网格上评估点，而随机搜索在空间内随机采样点，通常能更有效地探查重要维度。

#### 贝叶斯优化

贝叶斯优化是一种序列化、基于模型的方法，特别适用于优化代价高昂的黑盒 (black box)函数，例如训练好的GAN的性能指标（如FID分数）。

1. **代理模型：** 它根据先前评估过的点，构建超参数与目标指标之间关系的概率模型（通常是高斯过程）。
2. **采集函数：** 它使用采集函数（例如，预期改进、上置信界）来确定下一组要评估的超参数。该函数平衡了对参数空间不确定区域的勘察与对已知表现良好区域的利用。
3. **评估：** 使用选定的超参数训练GAN，并计算目标指标。
4. **更新：** 结果用于更新代理模型。

- **优点：** 比网格搜索或随机搜索更样本高效，寻找良好超参数所需的GAN训练次数更少。对高维空间 (high-dimensional space)和昂贵评估有效。
- **缺点：** 实现更复杂。序列性质限制了与随机搜索相比的并行化（尽管存在并行变体）。表现取决于代理模型和采集函数的选择。

> 贝叶斯优化在超参数调整中的流程。它迭代地构建目标函数的模型，并用它来智能选择下一组要评估的超参数。

#### 基于种群的训练（PBT）

PBT并行训练一组模型。定期地，表现较差的模型会采用表现较好模型的权重 (weight)和超参数，可能还会对超参数施加随机扰动。这使得在训练过程中可以同时优化权重和超参数。

- **优点：** 能够发现复杂的超参数调度。对大规模实验有效。集成了优化和调整。
- **缺点：** 训练一个种群需要大量计算资源。需要更复杂的设施。

### GAN特有调整考量

- **目标指标：** 不要仅仅依靠生成器或判别器损失进行调整。使用定量指标，如FID、IS、PPL，或任务特定指标（例如条件GAN的分类准确率），在训练期间或最终生成样本上定期评估。选择与期望结果相符的指标（例如，FID用于图像保真度和多样性）。
- **稳定性与质量：** 通常存在权衡。那些能最大化稳定性的超参数 (parameter) (hyperparameter)（例如，非常高的梯度惩罚权重 (weight)）可能会轻微损害可达到的最高质量。调整在于寻找一个适合你目标的平衡点。
- **早期停止与检查点：** 在整个训练过程中，在验证集上监测所选指标（例如FID）。定期保存模型检查点，尤其当指标改善时。GAN训练可能突然发散，因此拥有良好状态的检查点很重要。
- **计算预算：** 要切合实际。没有大量计算资源，完整的贝叶斯优化或PBT可能不可行。从有限次数的随机搜索开始，或根据以往工作集中手动调整最敏感的参数（如学习率）。
- **实验追踪：** 使用MLflow、Weights & Biases或TensorBoard HParams等工具，记录每次运行的超参数、代码版本、评估指标以及定性结果（生成的样本）。这种组织方式对于比较实验和复现结果必不可少。

为高级GAN找到合适的超参数是一个迭代过程，它结合了算法知识、系统搜索策略、细致评估和实验管理。尽管计算密集，但结构化的方法显著增加了训练出成功且高质量生成模型的可能性。

## 参考资料

- [Random Search for Hyper-Parameter Optimization](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf) — James Bergstra and Yoshua Bengio (2012)
  Journal: Journal of Machine Learning Research; Publisher: Microtome Publishing; Volume: 13; Pages: 281-305; DOI: [10.1622/jmlr.v13.12-140](https://doi.org/10.1622/jmlr.v13.12-140)
  这篇原创论文介绍了随机搜索，作为网格搜索在超参数优化方面更高效的替代方法。
- [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://papers.nips.cc/paper_files/paper/2017/file/8a1d6947076a0d3d526c9656461a34ba-Paper.pdf) — Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS) 30; Publisher: Curran Associates, Inc.; Pages: 6626-6637
  介绍了用于稳定GAN训练的双时间尺度更新规则（TTUR）以及用于评估GAN样本质量的Fréchet Inception距离（FID）。
- [Population Based Training of Neural Networks](https://arxiv.org/abs/1711.09846) — Max Jaderberg, Valentin Dalibard, Simon Osindero, Wojciech M. Czarnecki, Jeff Donahue, Ali Razavi, Oriol Vinyals, Tim Green, Iain Dunning, Karen Simonyan, Chrisantha Fernando, Koray Kavukcuoglu (2017)
  Journal: arXiv preprint arXiv:1711.09846; DOI: [10.48550/arXiv.1711.09846](https://doi.org/10.48550/arXiv.1711.09846)
  提出了基于种群的训练（PBT），这是一种通过并行训练一组模型来结合超参数优化和模型训练的方法。
- [Improved Training of Wasserstein GANs](https://papers.nips.cc/paper_files/paper/2017/hash/892c3b1c6dccd5296996d51e7a2b02ad-Paper.pdf) — Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, Aaron Courville (2017)
  Journal: Advances in Neural Information Processing Systems; Publisher: NeurIPS; Volume: 30; Pages: 5767-5777; DOI: [10.5555/3295222.3295327](https://doi.org/10.5555/3295222.3295327)
  介绍了WGAN-GP公式，该公式使用梯度惩罚来强制执行Lipschitz约束，显著提高了GAN训练的稳定性。
- [Taking the human out of the loop: A review of Bayesian optimization](https://ieeexplore.ieee.org/document/7352306) — Babak Shahriari, Kevin Swersky, Ziyu Wang, Ryan P. Adams, and Nando de Freitas (2016)
  Journal: Proceedings of the IEEE; Publisher: IEEE; Volume: 104; Pages: 148-175; DOI: [10.1109/JPROC.2015.2494218](https://doi.org/10.1109/JPROC.2015.2494218)
  贝叶斯优化的全面综述，解释了其原理、组成部分以及在优化昂贵黑盒函数方面的应用。
