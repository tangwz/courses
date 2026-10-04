---
course: "applied-autoencoders-feature-extraction"
chapter: "applying-autoencoder-features-practical-guidance"
lesson: "transfer-learning-approaches-autoencoders"
sourceId: 6495
sourceUrl: "https://apxml.com/zh/courses/applied-autoencoders-feature-extraction/chapter-7-applying-autoencoder-features-practical-guidance/transfer-learning-approaches-autoencoders"
title: "自编码器的迁移学习方法"
description: "讨论预训练自编码器或其组件如何用于迁移学习场景。"
order: 7
plots: []
sourceHash: "58014f7bf8832d2c62eb1614991d04f60dd59e9997bda62aaa17151f1a2f51a3"
sourceCorrections: []
---

自编码器根据其自身特性，学习压缩和重建数据，这使其能够捕获数据潜在空间中的潜在模式和结构。这种已学知识，体现在编码器的权重 (weight)中，通常对它训练所用的原始任务或数据集有价值。迁移学习 (transfer learning)提供了一些方法，可以重复使用这些学习到的表示或自编码器的学习能力来解决新的、相关的问题，特别是在新任务的数据量稀缺时。

### 将编码器用作固定特征提取器

将迁移学习 (transfer learning)应用于自编码器的最直接方式之一是使用预训练 (pre-training)编码器作为固定特征提取器。该过程通常包含这些步骤：

1. **预训练自编码器：** 首先，在一个大型通用数据集（源数据集）上训练一个自编码器（编码器和解码器）。理想情况下，这个数据集应能代表您预期对目标任务有用的特征类型。例如，如果您的目标任务涉及特定对象的图像，那么在大型多样化图像数据集（如ImageNet，尽管通常用于监督模型，但此原则也适用于自编码器的无监督预训练）或大量未标记 (token)的特定领域图像集合上进行预训练是有效可行的。
2. **独立出编码器：** 自编码器训练完成后，分离或独立出编码器部分。解码器在此方法中不再需要。
3. **提取特征：** 将您的新数据（来自目标任务）通过这个预训练的编码器。编码器的输出（瓶颈层的激活）将作为目标数据的新特征表示。这些特征是“固定的”，因为编码器的权重 (weight)在此步骤中不更新。
4. **训练后续模型：** 使用这些提取的特征来训练一个新的、通常更简单的机器学习 (machine learning)模型（例如，逻辑回归、支持向量 (vector)机或小型神经网络 (neural network)）来完成您的目标任务。由于特征是预先计算好的，并且通常维度更低且信息量更大，因此后续模型可以不那么复杂，并且用更少的标记数据进行训练。

> 使用预训练编码器作为固定特征提取器的工作流程。

当您的目标数据集较小时，此方法特别有用，因为它可以防止后续模型过拟合 (overfitting)。提取特征的质量在很大程度上取决于源数据集与目标任务的相关性。

### 微调 (fine-tuning)预训练 (pre-training)自编码器

另一种有效方法是在目标数据集上微调预训练自编码器（或仅其编码器部分）。您不再固定编码器权重 (weight)，而是允许它们在新任务训练期间更新。

1. **预训练自编码器：** 和之前一样，首先在大型源数据集上训练一个自编码器。
2. **适配目标任务：**
   - **编码器微调：** 取出预训练的编码器，并将其连接到一个适合您目标任务的新输出层（例如，如果您的目标是分类，则为分类层）。编码器的权重将作为这个新、更大模型的初始化。
   - **整个自编码器微调（直接用于特征提取较不常见）：** 如果目标任务仍然基于重建，但数据分布略有不同，您可以微调整个自编码器。
3. **继续训练：** 在您的目标数据集上训练这个新模型。重要的是，使用的学习率要小于初始预训练时所用的学习率。这有助于防止模型过快地“遗忘”从源数据中学到的有用特征。

您可以选择微调：

- **所有层：** 更新预训练编码器和新任务特定层中的权重。
- **仅部分层：** 保持编码器的初始层冻结（因为它们可能学习到非常通用的特征），仅微调编码器的后续层和新的任务特定层。当目标数据集明显小于或不同于源数据集时，这很常见。

> 为新目标任务微调预训练编码器的工作流程。

如果目标数据集足够大且与源数据集相关，微调通常比固定特征提取带来更好的性能，因为它允许特征更具体地适应目标任务的细节。

### 自编码器迁移学习 (transfer learning)的重要考量

成功应用自编码器迁移学习需要仔细考量几个因素：

- **数据相似性：** 迁移学习在源数据集和目标数据集共享一些潜在相似性时效果最佳。从自然图像训练的自编码器学习到的特征，很可能比用于处理文本数据的任务，更能迁移到其他图像任务上。数据分布越相似，迁移效果越好。
- **数据集大小：**
  - **源大，目标小：** 在这种情况下，将编码器用作固定特征提取器通常是一个好策略。微调 (fine-tuning)可能导致在小目标数据集上过拟合 (overfitting)。
  - **源大，目标中/大：** 微调变得更可行，并且通过让模型适应，可以带来更好的性能。
- **特征提取的层选择：** 当使用固定特征提取器时，特别是对于深度自编码器，您可以尝试从编码器的中间层获取特征，而不仅仅是最终的瓶颈层。较早的层倾向于捕获更通用、低级的特征，而较深的层捕获更抽象、特定的特征。最佳选择取决于目标任务的性质。
- **微调的学习率：** 这是一个重要的超参数 (parameter) (hyperparameter)。微调时，使用的学习率应显著小于初始预训练 (pre-training)时所用的学习率（例如，1/10或1/100）。这有助于温和地调整预训练权重 (weight)，而不会破坏已学信息。您也可以考量差异化学习率：对早期预训练层使用较小学习率，对新添加的任务特定层使用稍大学习率。
- **架构兼容性：** 确保目标任务的输入数据格式与预训练编码器的输入要求兼容（例如，图像尺寸、通道数、归一化 (normalization)）。

自编码器迁移学习提供了一种实用方式来利用大型未标记 (token)数据集中的知识，以提升在标记数据可能有限或获取成本高的任务上的性能。了解这些方法和考量，您可以更有效地将自编码器派生的特征和模型整合到机器学习 (machine learning)流程中。

## 参考资料

- [Reducing the Dimensionality of Data with Neural Networks](https://www.science.org/doi/10.1126/science.1127647) — Geoffrey E. Hinton, Ruslan R. Salakhutdinov (2006)
  Journal: Science; Publisher: American Association for the Advancement of Science; Volume: 313; Pages: 504-507; DOI: [10.1126/science.1127647](https://doi.org/10.1126/science.1127647)
  提出了训练深度自编码器进行无监督预训练的开创性方法，为迁移学习中学习有效数据表示奠定了基础。
- [A Survey on Transfer Learning](https://ieeexplore.ieee.org/document/5288526) — Sinno Jialin Pan, Qiang Yang (2010)
  Journal: IEEE Transactions on Knowledge and Data Engineering; Publisher: IEEE; Volume: 22; Pages: 1345-1361; DOI: [10.1109/TKDE.2009.191](https://doi.org/10.1109/TKDE.2009.191)
  一项基础性调查，对迁移学习范式进行分类和定义，为跨不同领域重用学习模型提供了重要的理论背景。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  一本全面的教材，涵盖了自编码器、表示学习以及深度神经网络中迁移学习的通用原理。第14章和第15章尤为相关。
- [Transfer Learning for Deep Learning: A Survey](https://www.jair.org/index.php/jair/article/view/12151) — Fuzhen Zhuang, Xiangliang Zhang, Pengfei Du, et al. (2021)
  Journal: Journal of Artificial Intelligence Research; Publisher: Journal of Artificial Intelligence Research; Volume: 71; Pages: 1-52; DOI: [10.1613/jair.1.12151](https://doi.org/10.1613/jair.1.12151)
  提供了针对深度学习的迁移学习技术的最新概述，包括讨论与使用预训练深度模型作为特征提取器或进行微调相关的各种策略。
