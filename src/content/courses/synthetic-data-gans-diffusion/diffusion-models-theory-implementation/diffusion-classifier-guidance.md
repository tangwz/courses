---
course: "synthetic-data-gans-diffusion"
chapter: "diffusion-models-theory-implementation"
lesson: "diffusion-classifier-guidance"
sourceId: 5704
sourceUrl: "https://apxml.com/zh/courses/synthetic-data-gans-diffusion/chapter-4-diffusion-models-theory-implementation/diffusion-classifier-guidance"
title: "分类器引导与无分类器引导"
description: "利用分类器或联合训练控制扩散模型输出的方法。"
order: 5
plots: []
sourceHash: "f8000d8c75f354b87bb06618045982f2ad9b00bfb355d6cb21f4005ed02b714d"
sourceCorrections: []
---

扩散模型擅长学习训练数据的基本分布并生成多样化的样本。然而，我们通常需要对生成过程有更好的控制。例如，生成特定物体类别（如“猫”或“狗”）的图像，或合成具有特定属性的数据，这属于条件生成的范围。在扩散模型中实现这种控制的两种重要方法是分类器引导和无分类器引导（CFG）。

## 分类器引导

分类器引导使用一个独立的、预训练 (pre-training)的分类器模型来引导扩散采样过程，使其偏向于展示所需特征的样本，这些特征通常由类别标签 $y$ 定义。主要思想是修改采样步骤，不仅对图像进行去噪，还要根据分类器使其更容易识别为类别 $y$。

回顾一下，逆向扩散过程旨在近似得分函数 $\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t)$，该函数引导从噪声到数据的采样。为了包含对类别 $y$ 的条件信息，我们希望从条件分布 $p(\mathbf{x}_t | y)$ 中采样。使用贝叶斯定理，我们可以将条件得分与无条件得分以及分类器的预测关联起来：


$$
\log p(\mathbf{x}_t | y) = \log p(y | \mathbf{x}_t) + \log p(\mathbf{x}_t) - \log p(y)
$$


对 $\mathbf{x}_t$ 取梯度得到：


$$
\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t | y) = \nabla_{\mathbf{x}_t} \log p(y | \mathbf{x}_t) + \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t)
$$


这里，$\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t)$ 是由无条件扩散模型估计的得分，而 $\nabla_{\mathbf{x}_t} \log p(y | \mathbf{x}_t)$ 是由分类器 $p_\phi(y | \mathbf{x}_t)$ 提供的对数似然梯度，该分类器经过训练可以从噪声输入 $\mathbf{x}_t$ 中预测类别 $y$。

在实际应用中，对于通过噪声预测 $\boldsymbol{\epsilon}_\theta$ 参数 (parameter)化的模型，采样期间的更新方向会进行调整。标准噪声预测 $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ 被修改以引入分类器的梯度。引导噪声预测 $\hat{\boldsymbol{\epsilon}}_\theta$ 的常用公式为：


$$
\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t, y) = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) - s \cdot \sqrt{1-\bar{\alpha}_t} \nabla_{\mathbf{x}_t} \log p_\phi(y | \mathbf{x}_t)
$$


这里，$s$ 是引导尺度，一个控制条件强度超参数 (hyperparameter)。$s$ 值越高，生成过程越强烈地偏向于分类器 $p_\phi$ 识别为属于类别 $y$ 的样本。

**机制：** 在逆向扩散过程的每一步，分类器检查当前噪声样本 $\mathbf{x}_t$，并计算对 $\mathbf{x}_t$ 的改变将如何提高目标类别 $y$ 的概率。然后，此梯度信息用于轻微调 (fine-tuning)整去噪步骤，从而有效地使生成偏向所需类别。

**优点：**

- 可以潜在地使用强大的、独立训练的分类器。
- 直接：使用外部“专家”来引导生成。

**缺点：**

- 需要一个分类器 $p_\phi(y | \mathbf{x}_t)$，该分类器专门针对不同时间步长 $t$ 的噪声数据进行训练，这可能不易获得或不易有效训练。
- 生成样本的质量可能高度依赖于分类器的质量。性能不佳或不稳定的分类器会引入伪影。
- 分类器梯度有时可能存在噪声或对抗性，从而降低样本质量。
- 引导尺度 $s$ 引入了另一个需要仔细调整的超参数。

## 无分类器引导（CFG）

无分类器引导（CFG）作为一种无需依赖独立分类器模型即可实现条件生成的方式而出现。它已成为一种被广泛采用且非常有效的方法，尤其在用于文本到图像合成的大规模模型中表现突出。

**机制：** 核心思想是训练一个单一的条件扩散模型，通常通过 $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y)$ 进行参数 (parameter)化，该模型将条件信息 $y$（例如，类别标签、文本嵌入 (embedding)）作为额外输入。在训练期间，条件输入 $y$ 会以一定概率（例如，10-20%的时间）随机替换为一个特殊的空标记 (token) $\emptyset$（代表无条件生成）。这迫使模型在同一组权重 (weight) $\theta$ 内学习条件噪声预测 $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y)$ 和无条件噪声预测 $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \emptyset)$。

在采样期间，每一步都会计算条件和无条件噪声预测。用于去噪步骤的最终噪声预测通过从无条件预测中朝条件预测的方向推断得到：


$$
\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t, y) = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \emptyset) + s \cdot (\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \emptyset))
$$


同样，$s$ 是引导尺度（在文献中通常表示为 $w$）。

- 如果 $s = 0$，则 $\hat{\boldsymbol{\epsilon}}_\theta = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \emptyset)$，导致无条件生成。
- 如果 $s = 1$，则 $\hat{\boldsymbol{\epsilon}}_\theta = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y)$，对应于使用学习到的条件模型的标准条件生成。
- 如果 $s > 1$，则引导效果会放大。该公式本质上是取无条件预测，并添加一个从无条件预测指向条件预测的差值向量 (vector)的缩放版本。这进一步推动结果朝向条件 $y$ 指示的方向。

**直观理解：** 项 $(\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \emptyset))$ 可以被视为隐含地表示噪声预测空间中与条件 $y$ 相关的方向。CFG 在训练期间有效地直接从数据中学习这个方向，而不是依赖外部分类器的梯度。通过 $s > 1$ 缩放此差值会增强条件 $y$ 对生成结果的影响。

> 示意图说明了无分类器引导机制在单个采样步骤中的运作。无条件（$\emptyset$）和有条件（$y$）的噪声预测都从当前状态 $\mathbf{x}_t$ 计算。最终的引导预测 $\hat{\boldsymbol{\epsilon}}_\theta$ 是基于这两个预测和引导尺度 $s$ 的推断。

**优点：**

- 消除了对独立分类器的需求，简化了整体流程。
- 与分类器引导相比，通常生成更高质量的条件样本。
- 训练仅涉及扩散模型本身，可能导致生成和条件方面之间更好的配合。
- 引导强度 $s$ 在推理 (inference)时易于调整，允许控制样本质量和对条件的遵循之间的权衡。

**缺点：**

- 需要修改训练过程以包含条件dropout。
- 如果未优化，每采样一步需要模型正向运行两次（一次使用 $y$，一次使用 $\emptyset$），与纯条件或无条件模型相比，这增加了推理时间（尽管通常比分类器引导更快，后者也需要分类器评估）。

## 比较与实际考量

| 特性 | 分类器引导 | 无分类器引导（CFG） |
| --- | --- | --- |
| **外部模型** | 是（分类器 \$p\_\phi(y | \mathbf{x}\_t)\$） |
| **训练** | 标准扩散模型 + 独立分类器训练（在噪声数据上） | 修改后的扩散模型训练（包含条件dropout） |
| **推理 (inference)速度** | 每步需要扩散模型 + 分类器评估 | 每步需要扩散模型评估两次（条件 + 无条件） |
| **常见质量** | 良好，但对分类器质量敏感，可能存在伪影 | 通常很先进，质量普遍更高，伪影更少 |
| **实现** | 需要集成两个模型 | 单个模型，修改训练循环 |
| **灵活性** | 可更换分类器（如果已训练） | 引导融入模型 |

**引导尺度 (s)：** 在这两种方法中，引导尺度 $s$（或 $w$）都扮演着重要角色。它控制着样本对条件的忠实度与样本多样性/真实性之间的权衡。

- **低 $s$（例如，0 或 1）：** 样本多样化，但可能不强烈反映条件 $y$。CFG 中 $s=0$ 会产生无条件样本。
- **高 $s$（例如，5-15）：** 样本强烈遵循条件 $y$，但多样性可能降低，样本有时会变得过饱和或出现伪影，显得不那么真实。

寻找 $s$ 的最佳值通常需要针对特定模型和任务进行经验调整。它提供了一个有力的调节器，可以在推理时调整生成行为而无需重新训练模型。

## 总结

引导技术对于引导扩散模型的输出朝向特定期望属性非常重要，超越了简单的无条件生成。分类器引导使用外部分类器通过梯度注入条件信息，而无分类器引导通过修改扩散模型本身的训练过程，更有效地实现了这一点，使其能够同时学习条件和无条件生成。CFG 因其更优异的性能并消除了对独立、可能存在问题的分类器模型的需求而成为标准方法。理解和运用这些引导机制是应用扩散模型到实际条件合成任务的基础。

## 参考资料

- [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) — Prafulla Dhariwal and Alex Nichol (2021)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2105.05233](https://doi.org/10.48550/arXiv.2105.05233)
  介绍了分类器引导，一种利用预训练分类器梯度引导扩散模型的方法。
