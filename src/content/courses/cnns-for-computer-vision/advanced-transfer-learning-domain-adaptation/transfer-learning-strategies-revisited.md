---
course: "cnns-for-computer-vision"
chapter: "advanced-transfer-learning-domain-adaptation"
lesson: "transfer-learning-strategies-revisited"
sourceId: 2599
sourceUrl: "https://apxml.com/zh/courses/cnns-for-computer-vision/chapter-6-advanced-transfer-learning-domain-adaptation/transfer-learning-strategies-revisited"
title: "回顾迁移学习策略"
description: "对基本迁移学习原理（特征提取、微调）的回顾，为进阶方法做准备。"
order: 1
plots: []
sourceHash: "b7f99630e75f53afd856fcfb8020e3a6faefabfc0b48e8fff07734fcf2567ab7"
sourceCorrections: []
---

如前所述，通过迁移学习 (transfer learning)使用预训练 (pre-training)模型是计算机视觉中的常见做法，这大幅减少了对大量数据集和长时间训练的需求。您可能已熟悉基本方法，但简要回顾可以提供一个坚实的基础，以便我们考察应对实际应用中复杂情况所需的更高级适应方法。

核心思路直观：一个模型，通常是卷积神经网络 (neural network) (CNN)，首先在一个大型通用数据集（如 ImageNet）上进行训练。这一预训练阶段使模型能够学习到丰富的视觉特征分层，从早期层中的简单边缘和纹理，到更后面的层中更复杂的物体部分和形状。这些学到的特征通常能很好地泛化到其他视觉任务。对于新任务，我们不从随机权重 (weight)开始学习过程，而是使用这些预训练权重初始化模型，从而传递已学到的知识。

两种主要策略在迁移学习的应用中占主导地位：

### 特征提取

这种方法中，预训练 (pre-training)模型（不包括其最终的分类层，即“头部”）被用作固定的特征提取器。卷积基础的已学权重 (weight)被冻结，这意味着在对新数据集进行训练时，它们不会被更新。

1. **创建基础模型**：加载不带分类头部的预训练模型。
2. **冻结基础模型**：将卷积基础的层设为不可训练。
3. **添加新头部**：连接一个适用于目标任务的新分类（或回归、分割等）头部。这个新头部将具有随机初始化的权重。
4. **训练**：在新数据集上训练模型。只有新添加的头部的权重会被更新。

这种方法在训练时计算效率高，因为梯度只需为小型的新头部计算。当目标数据集较小且与原始模型训练所用数据集相似时（例如，使用 ImageNet 预训练模型对不同种类的花进行分类），这种方法尤其有效。假定预训练期间学到的通用特征足以代表新任务。

> 特征提取策略的示意图。预训练的卷积基础层被冻结，只有新添加的针对特定任务的头部被训练。

### 微调 (fine-tuning)

微调使迁移学习 (transfer learning)过程更进一步。它与特征提取的开始方式相似，即用预训练 (pre-training)权重 (weight)初始化模型并添加新头部。然而，与保持整个卷积基础冻结不同，预训练基础的一些顶层被解冻，并与新头部一同训练。

1. **创建并添加头部**：加载预训练模型，替换头部，与特征提取方法相同。
2. **冻结初始层**：通常，卷积基础的早期层（学习边缘和纹理等通用特征的层）保持冻结。
3. **解冻顶层**：解冻卷积基础的后续层（学习更抽象、针对特定任务的特征的层）。
4. **训练**：在新数据集上训练模型。新头部和基础模型的解冻层都将被更新。在微调过程中使用非常低的学习率是常见做法。

微调使模型能够更具体地适应新数据集和任务的细节特点。当目标数据集足够大且可能与原始预训练数据集有所不同时，通常更倾向于使用此方法。低学习率很重要，以防止随机初始化头部产生的大梯度过快地破坏基础层中有价值的预训练权重。调整这些更高层次的特征使模型能够更好地进行专门化。

> 微调策略的示意图。预训练基础的较低层保持冻结，而顶层和新头部一同训练（微调），通常使用较低的学习率。

### 基本原理

将纯特征提取和完全微调 (fine-tuning)（所有层都解冻）视为一个范围的两端很有用。常见做法通常介于两者之间，涉及根据数据集大小、任务相似性和计算预算选择性地解冻层块。

本次回顾为本章的其余部分设定了背景。尽管这些标准策略效果好，但它们通常假定源（预训练 (pre-training)）和目标（新任务）数据分布相对相似。我们将涵盖的进阶方法，例如域适应、域泛化、少样本学习 (few-shot learning)和自监督预训练，处理这种假定不成立或标记 (token)数据稀缺的情况。理解特征提取和微调的机制及权衡，有助于我们更好地理解这些更进阶的适应策略的运作原理和方式。

## 参考资料

- [ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) — Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton (2012)
  Journal: Advances in Neural Information Processing Systems 25; Publisher: Curran Associates, Inc.; Pages: 1097-1105
  本文介绍了AlexNet，一个深度卷积神经网络，在2012年ImageNet大规模视觉识别挑战赛中获胜，促进了CNN的运用，并推动了在大型数据集上进行预训练以实现迁移学习的实践。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  一本广受认可的教材，为深度学习提供了理论和实践基础，其中详细讨论了迁移学习、微调和特征提取。
- [CS231n: Convolutional Neural Networks for Visual Recognition, Lecture Notes](http://cs231n.github.io/) — Fei-Fei Li, Justin Johnson, Serena Yeung, et al. (2017)
  Publisher: Stanford University
  提供了对CNN架构、训练技术和实际应用的易懂解释，包括有关特征提取和微调等迁移学习策略的章节。
- [Deep Transfer Learning for Computer Vision: A Survey](https://dl.acm.org/doi/10.1145/3371607) — Fuzhen Zhuang, Zhiyuan Luo, Huaxiu Lei, Fei Xu, Yong Xie, Qing He (2020)
  Journal: ACM Computing Surveys; Publisher: ACM; Volume: 53; Pages: Article 35; DOI: [10.1145/3371607](https://doi.org/10.1145/3371607)
  本综述全面介绍了计算机视觉领域深度迁移学习技术的近期进展，涵盖了特征提取和微调等基本策略以及更高级的方法。
