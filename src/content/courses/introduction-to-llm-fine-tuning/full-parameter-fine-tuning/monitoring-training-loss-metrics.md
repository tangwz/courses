---
course: "introduction-to-llm-fine-tuning"
chapter: "full-parameter-fine-tuning"
lesson: "monitoring-training-loss-metrics"
sourceId: 7278
sourceUrl: "https://apxml.com/zh/courses/introduction-to-llm-fine-tuning/chapter-3-full-parameter-fine-tuning/monitoring-training-loss-metrics"
title: "监控训练：损失与指标"
description: "如何解读训练和验证损失曲线，以诊断过拟合或欠拟合。"
order: 5
plots: ["plots/7278-0.json"]
sourceHash: "18f2ecd1ce6d80b71e09dd619e2e278a2e2c4db062cedbd6b0fd6c7af91d1b01"
sourceCorrections: []
---

一旦你启动微调 (fine-tuning)过程，你眼前的任务就从准备转向观察。你不能简单地开始训练，然后几个小时后才回来，期望一切顺利。积极监控训练运行对于构建有效的模型、节省计算资源以及及早发现问题不可或缺。你了解训练过程最重要的途径是损失曲线。

训练损失衡量了模型预测与训练数据集中真实标签的符合程度。随着模型在梯度下降 (gradient descent)的每一步更新其权重 (weight) $\theta$，这个值应该稳步下降。持续下降的训练损失表明模型成功学到了数据中存在的模式。

然而，单独的低训练损失是不够的。模型可以非常擅长记忆训练数据，但这并不能保证它在新颖、未见过的例子上表现良好。这就是验证损失不可或缺之处。通过定期在一个独立的验证集上评估模型，这个验证集不用于权重更新，你可以衡量其泛化能力。

### 解读损失和验证曲线

训练损失和验证损失之间的关系展现了你模型学习行为的特点。通过将这两个值与训练步数或迭代次数进行绘图，你可以诊断微调 (fine-tuning)任务的状态。你会遇到几种常见模式。

一次良好的训练运行显示训练损失和验证损失都在下降并收敛。验证损失几乎总是略高于训练损失，但它们的曲线应该遵循相似的趋势。这表明模型正在学习可泛化的模式。

微调中最常见的问题是过拟合 (overfitting)。当模型过于擅长学习训练数据，包括其噪声和特殊性，而牺牲泛化能力时，就会发生这种情况。在损失曲线上，过拟合通过明显的背离来识别：训练损失持续下降，而验证损失趋于平缓并开始上升。当你看到验证损失开始增加时，这是一个停止训练的信号。超过这个点继续训练只会使模型在新数据上的表现更差。

另一个潜在问题是欠拟合 (underfitting)。当模型未能捕捉数据中潜在的模式时，就会发生这种情况。当训练损失和验证损失都在高值处趋于平稳时，你可以发现欠拟合。这可能意味着模型需要更多的训练时间、更高的学习率，或者基础模型本身不具备完成该任务的能力。

最后，高度不规律或“噪声大”的损失曲线，即值上下跳动、无法预测的情况，通常指向不稳定的训练过程。最常见的原因是学习率过高，导致优化过程越过最小值。较小的批量大小也会导致这种不稳定。



![解读训练和验证损失](plots/7278-0.json)



> 实线显示了良好的拟合，两种损失都在下降并趋于稳定。虚线说明了过拟合，训练损失持续下降，而验证损失在某一点后开始增加。

### 关于损失：任务特定指标

虽然损失函数 (loss function)指导优化过程，但它是一种间接的性能衡量。较低的损失通常更好，但它不直接等同于更高质量的文本生成、更好的摘要或更准确的分类。

因此，你也应该监控验证集上的任务特定指标。例如：

- **准确率：** 对于分类任务，这衡量了正确预测的百分比。
- **ROUGE 或 BLEU：** 对于摘要或翻译，这些指标将模型的输出与参考文本进行比较。
- **困惑度：** 较低的困惑度分数表明模型对其预测更有信心，这通常与更流畅、连贯的文本相关。

这些指标提供了对模型在所关心的实际任务上表现的更直接、更易于理解的评估。现代训练库，包括 Hugging Face 的 `Trainer` API，与 TensorBoard 和 Weights & Biases 等日志工具配合良好。这些工具自动生成损失和你选择跟踪的任何其他指标的交互式图表，使监控过程大大简化。这使得你能够实时观察趋势并做出明智的决定，例如何时停止训练以防止过拟合 (overfitting)，这种技术被称为早停。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  深度学习的权威教材，涵盖了损失函数、优化、过拟合和验证集。
- [Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  第四版在线草稿，提供了BLEU、ROUGE和困惑度等NLP评估指标的详细信息。
- [Early Stopping - But When?](https://doi.org/10.1007/3-540-69043-4_5) — Lutz Prechelt (1998)
  Journal: Neural Networks: Tricks of the Trade; Publisher: Springer Berlin Heidelberg; Volume: 1524; Pages: 55-69; DOI: [10.1007/3-540-69043-4_5](https://doi.org/10.1007/3-540-69043-4_5)
  一篇讨论早期停止（Early Stopping）原则和实际策略以防止训练期间过拟合的经典论文。
- [Hugging Face Transformers Documentation - Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer) — Hugging Face (2024)
  Publisher: Hugging Face
  Hugging Face Trainer API的官方文档，用于微调模型以及监控损失和指标。
