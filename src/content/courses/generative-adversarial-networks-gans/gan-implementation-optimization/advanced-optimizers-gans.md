---
course: "generative-adversarial-networks-gans"
chapter: "gan-implementation-optimization"
lesson: "advanced-optimizers-gans"
sourceId: 2769
sourceUrl: "https://apxml.com/zh/courses/generative-adversarial-networks-gans/chapter-7-gan-implementation-optimization/advanced-optimizers-gans"
title: "高级优化器 (AdamW, Lookahead)"
description: "检查超越标准Adam并有益于GAN训练的优化器。"
order: 2
plots: []
sourceHash: "00d2c3f8d8c61580c16d0538b5f0a46313ba7ffc60b6e449fec153f09b279436"
sourceCorrections: []
---

标准的Adam优化器是深度学习 (deep learning)中的主力，但GAN训练的特定动态（表现为极小极大博弈而非简单的损失最小化）通常需要更专门的优化策略。模式崩溃或收敛振荡等不稳定性有时可归因于优化器的行为。两种先进优化器——AdamW和Lookahead——提供了改进的机制，可以为复杂的GAN带来更稳定的训练和可能更好的最终模型性能。

### AdamW：解耦权重 (weight)衰减

标准的L2正则化 (regularization)（与优化器一起使用时常被称为权重衰减）是防止过拟合 (overfitting)的常用技术。在Adam等优化器中，L2正则化通常通过在自适应矩估计和更新步骤之前，将正则项（$\lambda \theta$，其中$\lambda$是衰减因子，$\theta$是权重）直接添加到梯度中来实现。

数学上，带有传统L2正则化的Adam更新简化视图如下：

1. 计算梯度：$g_t = \nabla_\theta L(\theta_{t-1})$
2. 将权重衰减添加到梯度：$g'_t = g_t + \lambda \theta_{t-1}$
3. 更新有偏矩：$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g'_t$， $v_t = \beta_2 v_{t-1} + (1 - \beta_2) (g'_t)^2$
4. 计算偏差校正后的矩：$\hat{m}_t = m_t / (1 - \beta_1^t)$， $\hat{v}_t = v_t / (1 - \beta_2^t)$
5. 更新参数 (parameter)：$\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$

然而，Loshchilov & Hutter（《解耦权重衰减正则化》，2019）的研究指出，Adam中L2正则化与自适应学习率的这种耦合可能存在问题。因为Adam根据每个参数的历史梯度（通过$m_t$和$v_t$）调整其学习率，应用于参数的有效权重衰减也变得自适应。具有较大历史梯度的参数（$v_t$较大）会收到较小的更新，因此，经历的权重衰减少于预期。这种相互作用会使正则化效果不如与SGD一起使用的标准权重衰减。

AdamW通过将权重衰减与梯度更新解耦来解决此问题。它不将衰减项添加到梯度中，而是首先仅使用损失梯度执行梯度更新，然后单独应用权重衰减步骤，类似于标准SGD中权重衰减的实现方式。

AdamW的更新流程是：

1. 计算梯度：$g_t = \nabla_\theta L(\theta_{t-1})$
2. *仅*使用$g_t$更新有偏矩：$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$， $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$
3. 计算偏差校正后的矩：$\hat{m}_t = m_t / (1 - \beta_1^t)$， $\hat{v}_t = v_t / (1 - \beta_2^t)$
4. 计算梯度更新：$\Delta \theta_t = \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$
5. 应用权重衰减并更新：$\theta_t = \theta_{t-1} - \Delta \theta_t - \eta \lambda \theta_{t-1}$ （注意：此处衰减按学习率$\eta$缩放，有时使用固定策略。）

**对GAN的益处：**

- **更有效的正则化：** 确保权重衰减更像其预期的L2正则化效果，可能提高生成器和判别器的泛化能力。
- **提高稳定性：** 通过单独处理衰减，它可能防止与自适应学习率之间产生一些不良相互作用，这些相互作用可能导致训练不稳定。
- **更好的最终性能：** 经常观察到能使模型泛化更好，相比于带有耦合L2正则化的Adam。

AdamW在现代深度学习 (deep learning)库中易于获得（`torch.optim.AdamW`，`tf.keras.optimizers.experimental.AdamW`），通常是一个不错的默认选项，尤其在需要正则化时。请记住与学习率一同调整`weight_decay`参数。

### Lookahead优化器

Lookahead并非独立的优化器，而是一种包装技术，可以与Adam、AdamW或SGD等现有优化器结合使用。由Zhang等人（《Lookahead优化器：前向k步，后退1步》，2019）提出，它旨在通过鼓励优化器更谨慎地检查参数 (parameter)空间来提高稳定性和收敛性。

核心思想是维护两组权重 (weight)：

- **快速权重（$\phi$）：** 这些权重由内部优化器（例如AdamW）多次（$k$步）更新。这允许进行快速试探。
- **慢速权重（$\theta$）：** 这些权重代表一个更稳定、平均的位置。它们每$k$个快速步骤只更新一次，部分地朝着快速权重试探的方向移动。

**机制：**

该过程循环重复：

1. **同步**：快速权重等于慢速权重开始：$\phi_0 = \theta_t$。
2. **内部循环（试探）**：使用选定的内部优化器（例如AdamW），根据每个快速步骤计算的梯度，更新快速权重$\phi$共$k$步。令$k$步后的最终快速权重为$\phi_k$。
   $\phi_{i+1} = \phi_i - \text{内部优化器更新}(L, \phi_i) \quad \text{对于 } i = 0 \dots k-1$
3. **外部循环（更新慢速权重）**：通过在当前慢速权重$\theta_t$和最终快速权重$\phi_k$之间插值来更新慢速权重$\theta$。步长$\alpha$控制慢速权重向快速权重最终位置移动的程度。
   $\theta_{t+1} = \theta_t + \alpha (\phi_k - \theta_t)$
4. 增加$t$，并从步骤1重复。

**对GAN的益处：**

- **降低方差：** 对$k$步的试探方向进行平均可以降低参数更新的方差，从而使收敛更平滑。这在GAN通常嘈杂的优化中特别有帮助。
- **提高稳定性：** $\theta$权重的较慢移动可以防止仅使用快速权重时可能发生的快速振荡或发散。
- **更好的泛化能力：** 经验结果常表明Lookahead能带来具有更好泛化性能的解决方案。

**实现与参数：**

Lookahead的实现通常包装现有优化器实例。主要超参数 (hyperparameter)包括：

- `k`（或`la_steps`）：在更新慢速权重之前，内部优化器的步数。常用值为5或10。
- `alpha`（或`la_alpha`）：用于更新慢速权重的插值因子。常用值为0.5。

内部优化器（例如AdamW）仍然有其自身的超参数（学习率、beta值、权重衰减）需要调整。

**开销：** Lookahead会带来一些开销：它需要存储两份模型参数的副本（快速权重和慢速权重），从而增加内存使用。计算开销最小，主要涉及每$k$次迭代中的插值步骤。

### 选择和使用高级优化器

无论是AdamW还是Lookahead，都无法保证在每种GAN场景中都胜过标准Adam。然而，它们在您的优化工具包中是很有价值的：

- **从AdamW开始：** 考虑到其理论基础和经验成功，在使用权重 (weight)衰减时，AdamW通常是比标准Adam更好的起点。仔细调整其学习率和权重衰减。
- **考虑Lookahead以提高稳定性：** 如果您在使用Adam或AdamW时遇到显著的训练不稳定性（振荡、模式崩溃），用Lookahead（使用适度的$k$和$\alpha$）包装它可能提高收敛平滑性。
- **超参数 (parameter) (hyperparameter)调整仍然重要：** 这些优化器并未消除仔细调整超参数的需要。内部优化器的学习率仍然很重要，权重衰减（对于AdamW）和Lookahead参数（$k, \alpha$）也同样重要。

试用这些优化器，尤其在训练StyleGAN或BigGAN等大型复杂GAN架构时，或在面临顽固的稳定性问题时，可能是一项值得的投入。它们对优化过程提供了更精细的控制，可能带来更快的收敛、更稳定的训练动态，并最终产生更高质量的生成模型。

## 参考资料

- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) — Ilya Loshchilov and Frank Hutter (2019)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1711.05101](https://doi.org/10.48550/arXiv.1711.05101)
  演示了分离权重衰减如何改进 Adam 等自适应优化器的性能和泛化能力，从而引入了 AdamW。
- [Lookahead Optimizer: k steps forward, 1 step back](https://arxiv.org/abs/1907.08610) — Michael R. Zhang, James Lucas, Geoffrey Hinton, and Jimmy Ba (2019)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.1907.08610](https://doi.org/10.48550/arXiv.1907.08610)
  提出了 Lookahead 优化器，一种通过快慢权重插值来提高稳定性和收敛性的包装器。
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — Diederik P. Kingma and Jimmy Ba (2015)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980)
  介绍 Adam 优化器的基础论文，该优化器使用梯度的一阶和二阶矩的自适应估计。
