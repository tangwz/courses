---
course: "llm-alignment-safety"
chapter: "advanced-alignment-algorithms"
lesson: "contrastive-methods-alignment"
sourceId: 3155
sourceUrl: "https://apxml.com/zh/courses/llm-alignment-safety/chapter-3-advanced-alignment-algorithms/contrastive-methods-alignment"
title: "对齐中的对比方法"
description: "应用对比学习技术以改进LLM对齐。"
order: 4
plots: []
sourceHash: "9b604a47c89e84de8a942c63814748956543723634ed3c1e9302f5eef2b6cd8a"
sourceCorrections: []
---

对比学习是一种已在计算机视觉和表征学习等方面取得显著成效的技术，它也为大型语言模型对齐 (alignment)提供了一种有效途径。对比方法不单纯依赖于预测下一个词元 (token)或显式建模奖励分数，而是着重于教导模型区分给定输入语境下的期望（正向）输出和非期望（负向）输出。这与人类偏好数据的结构非常吻合，这些数据常以选择和拒绝响应对的形式出现。

主要思想很简单：给定一个提示或语境 $x$，我们希望模型 $\pi$ 为偏好的补全 $y_{\text{chosen}}$ 分配比不偏好的补全 $y_{\text{rejected}}$ 更高的概率。学习过程旨在最大化这些正负例之间概率的边际或差异。

### 对齐 (alignment)中的对比原则

设想您拥有与RLHF类似方式收集的偏好数据：对于特定的提示，人类标注员（或可能是AI，如RLAIF中所示）已经选择了一个偏好的响应并拒绝了一个或多个备选项。对比方法直接使用这种成对偏好。

模型被训练以满足条件 $P_{\pi}(y_{\text{chosen}} | x) > P_{\pi}(y_{\text{rejected}} | x)$。这是通过优化一个旨在强制执行这种偏好的损失函数 (loss function)来实现的。从本质上讲，对比损失 $L_{\text{contrastive}}$ 可能会类似于最大化所选响应的对数似然，同时最小化被拒绝响应的对数似然，通常会加入一个边际或使用基于对数比率的公式。

> 对比对齐的简化视图。在对比损失函数的引导下，模型学习对于相同的输入提示，增加偏好输出（正向示例）的似然，并降低非偏好输出（负向示例）的似然。

### 与DPO和RLHF的关系

您可能会注意到此描述与直接偏好优化（DPO）之间有很强的相似性，我们将在下一节中详细讨论。实际上，DPO是对LLM对齐 (alignment)应用对比方法的一个显著且有效的实例。DPO通过直接从RLHF目标推导出一个特定的损失函数 (loss function)来形式化对比目标，将对齐问题构建为偏好对上的二元分类任务。它有效地最大化了所选响应和被拒绝响应之间的对数概率比，并由隐式奖励差异加权。

与标准RLHF流程相比，像DPO这样的对比方法提供了一个潜在的优势：它们通常可以绕过训练一个显式、独立奖励模型的需要。偏好信号直接用于更新语言模型策略。这可以简化整体训练过程，潜在地减少计算开销，并避免与奖励模型校准和利用（奖励作弊）相关的难题。

然而，这种直接优化并非没有权衡。

- **数据依赖：** 对齐质量在很大程度上取决于偏好对 $(x, y_{\text{chosen}}, y_{\text{rejected}})$ 的质量和覆盖范围。
- **隐式奖励：** 如果没有显式奖励模型，解释模型确切的优化目标可能会不那么直接。
- **负例采样：** 负例 ($y_{\text{rejected}}$) 的选择和多样性显著影响学习动态。仅使用略微次优的拒绝可能会导致与使用明显有害或无意义的拒绝不同的结果。

### 对比对齐 (alignment)的实施

在实践中，实施对比对齐方法包括：

1. **数据准备：** 将数据集构建为 $(x, y_{\text{chosen}}, y_{\text{rejected}})$ 的元组。
2. **模型前向传播：** 在当前模型策略 $\pi$ 下，计算给定输入提示 $x$ 时所选响应和被拒绝响应的对数概率。这通常涉及对两种补全自回归 (autoregressive)地运行模型。
3. **损失计算：** 基于这些对数概率计算对比损失。对于DPO，这涉及计算相对于参考策略（通常是初始的监督微调 (fine-tuning)模型）的对数比率。
4. **反向传播 (backpropagation)：** 使用从对比损失导出的梯度更新 $\pi$ 的模型参数 (parameter) $\theta$。

正则化 (regularization)技术，类似于RLHF的PPO步骤中使用的KL散度惩罚，通常被纳入像DPO这样的对比方法中，以防止策略偏离初始模型太远，从而保持其通用语言能力。

对比方法代表了一种强大的LLM对齐技术族，相较于传统的RLHF奖励建模，它提供了从偏好数据到策略优化的更直接途径。它们与既定学习模式的联系以及DPO等方法的成功，使它们成为高级对齐工具包的重要组成部分。

## 参考资料

- [Direct Preference Optimization: Your Language Model Is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) — Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn (2023)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2305.18290](https://doi.org/10.48550/arXiv.2305.18290)
  介绍了直接偏好优化（DPO），这是一种无需单独奖励模型，直接从偏好数据对LLM进行对齐的方法。
- [SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://arxiv.org/abs/2104.08821) — Tianyu Gao, Xingcheng Yao, Danqi Chen (2021)
  Journal: Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP); DOI: [10.48550/arXiv.2104.08821](https://doi.org/10.48550/arXiv.2104.08821)
  展示了SimCSE，一种将对比学习应用于NLP中生成句子嵌入的方法，演示了在语言任务中区分正负例的策略。
