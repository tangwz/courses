---
course: "llm-constitutional-ai-rlaif"
chapter: "advanced-rlaif-implementation"
lesson: "training-rlaif-preference-model"
sourceId: 4513
sourceUrl: "https://apxml.com/zh/courses/llm-constitutional-ai-rlaif/chapter-5-advanced-rlaif-implementation/training-rlaif-preference-model"
title: "偏好模型训练"
description: "关于使用AI标注数据训练偏好模型的实现方法。"
order: 3
plots: ["plots/4513-0.json"]
sourceHash: "1783b7a343ef5b7ef4aa9fe3dc71544130649e3a91a83827305ebd6ffb9f35b6"
sourceCorrections: []
---

训练偏好模型是RLAIF流程的一个主要步骤。这个模型是RLAIF流程的基础，它能学习对数据集中所包含的偏好进行量化 (quantization)，最终为强化学习 (reinforcement learning)阶段提供奖励信号。AI生成的偏好数据集$(x, y_w, y_l)$通常包括提示$x$、偏好的（获胜）响应$y_w$和次优的（失败）响应$y_l$。

### 模型架构选择

偏好模型的主要作用是为给定的提示-响应对分配一个标量分数$r_\theta(x, y)$，此分数表示该响应根据所学习的AI偏好有多“受欢迎”。一种常见且有成效的方法是调整你打算对齐 (alignment)的基础LLM的架构，或一个相关的预训练 (pre-training)Transformer模型。

惯常的输入格式是将提示和响应连接起来，通常用一个特殊标记 (token)分隔，然后将这个序列输入到Transformer模型中。对于与提示$x$对应的偏好对$(y_w, y_l)$，你通常会进行两次前向传播：一次针对$(x, y_w)$，另一次针对$(x, y_l)$。

```
输入 1: [提示标记] [SEP] [获胜响应标记]
输入 2: [提示标记] [SEP] [失败响应标记]
```

通常会在对应于某个特定标记（例如，序列的最后一个标记，或在使用BERT风格模型时的专门分类标记）的最终隐藏状态之上添加一个线性层。该层将高维表示映射为一个单一的标量值，表示偏好分数。

使用预训练的基础LLM（或初始SFT或CAI阶段产生的模型）的权重 (weight)来初始化偏好模型通常有益。这借助了模型现有的语言理解能力，使其能够侧重于学习偏好的具体情况，而不是从零开始学习语言建模。使用基础LLM的较小、精炼版本，也可以是降低计算开销的可行方法，尽管可能牺牲一些表达能力。

### 损失函数 (loss function)：学习偏好

偏好模型的标准训练目标与排序学习（learning-to-rank）和RLHF中使用的技术相似。目标是训练模型，使得分配给获胜响应$y_w$的分数高于分配给失败响应$y_l$的分数。这通常被构想为响应对上的二元分类问题。

借鉴Bradley-Terry模型等将成对比较概率与背后强度参数 (parameter)关联起来的模型，我们可以使用它们分数差值通过Sigmoid函数后的结果，对在给定$x$下$y_w$优于$y_l$的概率进行建模：


$$
P_\theta(y_w \succ y_l | x) = \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))
$$


这里，$\sigma(z) = 1 / (1 + e^{-z})$是逻辑Sigmoid函数。训练目标是最大化观察数据集中$D = \{(x^{(i)}, y_w^{(i)}, y_l^{(i)})\}_{i=1}^N$所包含偏好的对数似然。这对应于最小化负对数似然损失：


$$
L(\theta) = - \sum_{i=1}^N \log(\sigma(r_\theta(x^{(i)}, y_w^{(i)}) - r_\theta(x^{(i)}, y_l^{(i)})))
$$


这个损失函数促使模型参数$\theta$为偏好响应赋予更高的分数$r_\theta$，为次优响应赋予更低的分数，从而直接优化排序目标。

### 训练过程与优化

训练过程遵循标准的监督学习 (supervised learning)方法：

1. **数据准备：** 打乱偏好数据集$D$。将其划分为训练集、验证集和测试集。最好确保来自相同提示上下文 (context)的对保留在相同的集合中，以避免数据泄露。
2. **批处理：** 创建批次，其中每个元素包含一个提示$x$、一个获胜响应$y_w$和一个失败响应$y_l$。
3. **前向传播：** 对于批次中的每个示例，通过模型传入连接后的输入，计算标量分数$r_\theta(x, y_w)$和$r_\theta(x, y_l)$。
4. **损失计算：** 使用上述成对偏好损失函数 (loss function)计算损失。
5. **反向传播 (backpropagation)与优化：** 计算梯度，并使用AdamW等优化器更新模型参数 (parameter)$\theta$。采用学习率调度（例如，线性热身接着余弦或线性衰减）和权重 (weight)衰减等常见方法进行正则化 (regularization)。
6. **评估：** 定期在验证集上评估模型，使用准确率（$r_\theta(x, y_w) > r_\theta(x, y_l)$被正确预测的对的百分比）和验证损失。

当GPU内存有限时，通常需要梯度累积来模拟更大的批次大小，尤其是在微调 (fine-tuning)大型模型时。

### 准确率评估

尽管在保留测试集上的准确率是主要指标，其他评估也能提供更深入的认识：

- **损失曲线：** 监控训练和验证损失，以发现过拟合 (overfitting)或训练不稳定。
- **校准：** 评估预测概率$P_\theta(y_w \succ y_l | x)$是否反映偏好的真实可能性。校准良好的模型分数更易理解。可靠性图可以展示校准情况。
- **分数分布：** 分析常见响应的分数$r_\theta(x, y)$分布。分数分布均匀吗？它们是否集中在狭窄区间内？
- **错误分析：** 人工检查模型预测偏好错误的示例。模型在哪些特定种类的提示或响应特征（例如，安全性上的细微差异、简洁与详细）上表现不佳？这种分析可以为数据集改进或进一步模型调整提供信息。



![偏好模型训练准确率](plots/4513-0.json)



> 偏好模型训练期间的准确率趋势，显示训练准确率可能会趋于平稳，而验证准确率可能会趋于饱和或略有下降，这表明过拟合的开始。

### 实现考量

- **初始化：** 如前所述，强烈建议从预训练 (pre-training)LLM检查点初始化。
- **参数 (parameter)高效微调 (fine-tuning)（PEFT）：** 像低秩适应（LoRA）这样的技术可以显著降低训练偏好模型的计算和内存要求，尤其是在调整非常大的基础模型时。这涉及只训练少量适配器参数，而不是整个模型。
- **数据质量影响：** AI偏好标注器的质量和一致性直接制约偏好模型的表现。偏好标注中的噪声或系统性偏差必然会被偏好模型学习到，从而为后续RL阶段造成有问题的奖励信号。应投入精力确保AI标注器很好地符合预期原则（例如，如果与CAI结合，则符合宪法原则）。
- **归一化 (normalization)：** 分数$r_\theta(x, y)$的绝对值不如它们的相对差值重要。然而，在RL阶段，奖励信号的量级有关。常见做法是在将偏好模型分数用作奖励之前，对其进行归一化（例如，通过减去批次或数据集的均值并除以标准差）。

一旦偏好模型训练并评估满意，其主要作用是为PPO算法提供奖励信号$r(x, y) = r_\theta(x, y)$（可能经过归一化或转换），指导LLM策略生成符合所学AI偏好的响应。这个重要环节使我们从监督偏好学习转向在线强化学习 (reinforcement learning)优化。

## 参考资料

- [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862) — Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, Ben Mann, Jared Kaplan (2022)
  Journal: arXiv preprint arXiv:2204.05862; DOI: [10.48550/arXiv.2204.05862](https://doi.org/10.48550/arXiv.2204.05862)
  本文详细介绍了基于人类反馈的强化学习（RLHF）框架，包括基于人类偏好训练奖励模型（偏好模型）的架构、损失函数和训练方法。
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2209.07858) — Deep Ganguli, Liane Lovitt, Jackson Kernion, Amanda Askell, Yuntao Bai, Saurav Kadavath, Ben Mann, Ethan Perez, Nicholas Schiefer, Kamal Ndousse, Andy Jones, Sam Bowman, Anna Chen, Tom Conerly, Nova DasSarma, Dawn Drain, Nelson Elhage, Sheer El-Showk, Stanislav Fort, Zac Hatfield-Dodds, Tom Henighan, Danny Hernandez, Tristan Hume, Josh Jacobson, Scott Johnston, Shauna Kravec, Catherine Olsson, Sam Ringer, Eli Tran-Johnson, Dario Amodei, Tom Brown, Nicholas Joseph, Sam McCandlish, Chris Olah, Jared Kaplan, Jack Clark (2022)
  Journal: arXiv preprint arXiv:2209.07858; DOI: [10.48550/arXiv.2209.07858](https://doi.org/10.48550/arXiv.2209.07858)
  这项工作描述了如何通过由“宪法”指导的AI反馈机制生成偏好数据，这些数据随后用于训练偏好模型，构成RLAIF的基础。
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2021)
  Journal: arXiv preprint arXiv:2106.09685; DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  介绍了低秩适应（LoRA），这是一种参数高效的微调技术，显著减少了可训练参数的数量，对于将大型语言模型适应为偏好模型非常有用。
- [Learning to Rank using Gradient Descent](https://dl.acm.org/doi/10.1145/1102351.1102432) — Christopher J. C. Burges, Tal Shaked, Erin Renshaw, Amit Lazier, Matt Deeds, Nicole Holman, Ding Zhou (2005)
  Journal: Proceedings of the 22nd International Conference on Machine Learning (ICML); Publisher: ACM; Pages: 89-96; DOI: [10.1145/1102351.1102432](https://doi.org/10.1145/1102351.1102432)
  这篇基础论文介绍了RankNet，它采用的成对排序损失函数与训练偏好模型所用的基于Bradley-Terry模型的客观函数在数学上相似。
