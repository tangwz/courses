---
course: "meta-learning-foundation-models"
sourceUrl: "https://apxml.com/zh/courses/meta-learning-foundation-models/chapter-3-advanced-metric-based-meta-learning"
sourceId: 849
chapter: "advanced-metric-based-meta-learning"
title: "进阶度量元学习"
order: 3
description: "研习用于少样本分类和适配的精巧度量学习方法。"
hasQuiz: false
---

基于度量的元学习将重心从梯度方法中直接调整模型参数，转移到学习一个嵌入空间 $f_\phi(\cdot)$。在这种空间中，新的分类任务通过简单的距离计算就能轻松完成。本章会审视此框架下的进阶技术。

你将学习原型网络、关系网络和匹配网络等主要算法的原理，包括它们的架构变体以及注意力机制的结合。讨论的一个主要部分是如何调整这些方法，使其能与大型基础模型生成的高维嵌入有效配合。我们也将回顾相关深度度量学习目标，例如对比损失和三元组损失 ($ \mathcal{L}_{triplet} $)，这些损失函数常用于训练嵌入函数。本章包含实现这些想法的实用指导，特别是使用从预训练模型中提取的特征来配合原型网络。
