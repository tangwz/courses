---
course: "introduction-to-llm-fine-tuning"
sourceUrl: "https://apxml.com/zh/courses/introduction-to-llm-fine-tuning/chapter-3-full-parameter-fine-tuning"
sourceId: 1265
chapter: "full-parameter-fine-tuning"
title: "全参数微调"
order: 3
description: "了解全参数微调的原理、资源管理、超参数配置以及训练过程监控。"
hasQuiz: false
---

准备好数据集后，你就可以开始训练过程了。本章介绍全参数微调，这是一种更新预训练模型中所有参数，使其适应你新任务的方法。这种方式直接修改模型的全部权重。

此过程基于梯度下降方法。模型参数，记作 $\theta$，根据你数据集计算出的损失进行调整。在每个训练步骤中，更新遵循以下一般形式：

$$ \theta_{new} = \theta_{old} - \eta \cdot \nabla L(\theta_{old}) $$

其中，$\eta$ 表示学习率，$\nabla L(\theta_{old})$ 是损失函数对模型参数的梯度。与更高效的方法不同，此更新应用于模型的所有数百万或数十亿个参数。

在本章中，我们将介绍应用这项技术的实际操作方面。你将学习如何：

*   选择合适的基础模型架构。
*   处理大量的计算负担和内存占用。
*   配置训练参数，包括学习率、批量大小和训练轮数。
*   解读训练和验证损失曲线，以诊断性能。
*   保存所得模型，并正确加载模型进行推理。

本章最后包含一个实践练习，你将应用这些步骤从头到尾微调一个小型模型。
