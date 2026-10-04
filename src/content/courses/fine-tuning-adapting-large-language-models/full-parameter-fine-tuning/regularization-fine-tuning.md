---
course: "fine-tuning-adapting-large-language-models"
chapter: "full-parameter-fine-tuning"
lesson: "regularization-fine-tuning"
sourceId: 3006
sourceUrl: "https://apxml.com/zh/courses/fine-tuning-adapting-large-language-models/chapter-3-full-parameter-fine-tuning/regularization-fine-tuning"
title: "防止过拟合的正则化方法"
description: "在全参数微调期间应用随机失活、权重衰减及其他方法来缓解过拟合。"
order: 4
plots: ["plots/3006-0.json"]
sourceHash: "a1fb1a33041a442b7b1fa2140b5e99042d2e02ca8c06ba9a6d5dc7861be13bce"
sourceCorrections: []
---

当你使用全参数 (parameter)微调 (fine-tuning)来调整一个大型预训练 (pre-training)模型时，你是在依据你特定的、通常小得多的数据集来更新可能数十亿的权重 (weight)。这种能力伴随着一个较大的风险：**过拟合 (overfitting)**。模型凭借其庞大的容量，可能会直接记住微调的例子，而不是掌握与你任务相关的潜在规律。这导致它在训练数据上表现出色，但对新的、未见过的数据泛化能力差，从而失去了微调的意义。

正则化 (regularization)方法是应对过拟合的重要手段。它们在训练过程中引入限制或惩罚，阻止模型学习只存在于微调集中的过于复杂或针对噪声的模式。我们来审视大型语言模型全参数微调背景下最相关的方法。

### 权重 (weight)衰减（L2正则化 (regularization)）

正则化技术有助于防止模型在全参数 (parameter)微调 (fine-tuning)时出现过拟合 (overfitting)。权重衰减是最常见的正则化形式之一，它在数学上等同于L2正则化。它在标准的任务损失函数 (loss function)（$L_{task}$）中添加一个惩罚项，该惩罚项与模型权重（$\theta$）的平方大小成比例。

修改后的损失函数变为：
$L_{total} = L_{task} + \frac{\lambda}{2} ||\theta||^2_2$
这里，$||\theta||^2_2$ 代表所有模型权重的平方L2范数（平方和），$\lambda$（lambda）是正则化强度，这是一个你需要调整的超参数 (hyperparameter)。

通过惩罚大权重，权重衰减促使模型将学习分散到许多参数上，而不是过度依赖少数几个。这通常会产生更简单的模型，对训练数据中的特定噪声不那么敏感。

实际使用中，现代优化器如AdamW（带有权重衰减的Adam）将权重衰减直接纳入权重更新规则，这通常比简单地将其添加到损失函数中更有效，尤其是在其与自适应学习率的配合方面。找到最优的$\lambda$很重要；在大型语言模型微调中常用值可能在0.01到0.1之间，但这很大程度上取决于模型、数据集大小和其他超参数。

### 随机失活

随机失活是另一种广泛使用的正则化 (regularization)方法，专门为神经网络 (neural network)设计。在每个训练步骤中，随机失活会随机将一部分神经元（或Transformer中的注意力头）的输出设置为零。舍弃一个单元的概率 $p$ 是一个超参数 (parameter) (hyperparameter)，通常范围在0.1到0.5之间。

这有何帮助？通过随机禁用网络的一部分，随机失活阻止神经元过度依赖特定的其他神经元。它强制网络学习更多冗余的表示，使其对任何单一单元的缺失不那么敏感。你可以把它看作是隐式地训练了许多共享权重 (weight)的较小网络的集成。

在推理 (inference)（评估或预测）期间，随机失活会被关闭，通常，剩余神经元的激活值会按 $(1-p)$ 的因子进行缩放，以弥补训练时相比，现在更多神经元处于活跃状态的事实。许多深度学习 (deep learning)框架会自动处理这种缩放。

虽然随机失活通常存在于原始预训练 (pre-training)大型语言模型架构中，但在微调 (fine-tuning)期间保持其活跃（并可能调整概率 $p$）仍然可以提供正则化效益，特别是当你的微调数据集与预训练数据显著不同，或者当你观察到过拟合 (overfitting)迹象时。

### 提前停止

也许最直观的正则化 (regularization)方法是提前停止。不像训练固定数量的轮次或步骤那样，你会监控模型在独立的*验证集*上的表现，这是你微调 (fine-tuning)数据的一部分，模型不会在此部分上进行训练。

你会定期在验证集上评估模型（例如，每几百步或每个轮次结束时）。最初，训练损失和验证损失都可能下降。然而，如果模型开始过拟合 (overfitting)，训练损失将继续下降（因为模型记住了训练数据），但验证损失将开始上升。这是模型泛化能力开始变差的点。

提前停止意味着当验证性能停止提升或开始变差时，你停止训练过程，并保存对应于所达到的*最佳*验证性能的模型检查点。



![基于验证损失的提前停止](plots/3006-0.json)



> 验证损失在500个训练步数后开始增加，这表明过拟合的出现。训练应停止，并应使用第500步的模型检查点。

这需要细心准备一个有代表性的验证集，并定义要监控的评估指标（例如，损失、准确率、F1分数）。

### 其他考量

- **学习率：** 虽然不严格算是正则化 (regularization)，但在微调 (fine-tuning)期间使用比预训练 (pre-training)时更小的学习率，相当于一种隐式正则化形式。它限制了权重 (weight)更新的大小，阻止模型基于有限的微调数据大幅偏离有用的预训练权重。学习率调度（例如，带预热的线性衰减）对于稳定收敛也很重要。
- **标签平滑：** 如果你的微调任务是分类，标签平滑会有帮助。不像训练模型以100%的概率预测目标类别那样（例如，使用独热编码标签如`[0, 1, 0]`），你会使用稍微柔和的目标（例如，`[0.05, 0.9, 0.05]`）。这会阻止模型变得过于自信，并促使正确和不正确类别对数之间的有限距离，有时能提高泛化能力。平滑的程度由一个小的超参数 (parameter) (hyperparameter) $\alpha$ 控制。

应用这些正则化方法需要细心的实验和调整。最佳组合和强度取决于特定的大型语言模型、微调数据集的大小和性质，以及目标任务。监控验证性能很重要，以便判断你选择的正则化策略是否有效地防止了过拟合 (overfitting)，并产生了对新数据具有良好泛化能力的模型。请记住，这些方法对于全参数微调尤其相关，因为有大量的参数需要更新；我们稍后会看到参数高效方法本身是如何提供一定程度的正则化效果的。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本全面的教材，涵盖深度学习的数学和概念方面，包括L2、dropout和早停等多种正则化技术。
- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://jmlr.org/papers/v15/srivastava14a.html) — Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov (2014)
  Journal: Journal of Machine Learning Research; Publisher: Journal of Machine Learning Research; Volume: 15; Pages: 1929−1958
  引入dropout作为神经网络正则化技术的开创性论文，解释了其机制和优势。
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) — Ilya Loshchilov, Frank Hutter (2019)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1711.05101](https://doi.org/10.48550/arXiv.1711.05101)
  本文提出了AdamW，它将权重衰减与Adam等优化器中的自适应学习率机制正确分离，从而改善了正则化效果。
- [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567) — Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna (2016)
  Journal: IEEE Conference on Computer Vision and Pattern Recognition (CVPR); Pages: 2818-2826; DOI: [10.48550/arXiv.1512.00567](https://doi.org/10.48550/arXiv.1512.00567)
  在Inception-v2和Inception-v3架构中引入标签平滑作为正则化组件，展示了其对模型泛化能力的优势。
