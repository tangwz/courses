---
course: "autoencoders-representation-learning"
sourceUrl: "https://apxml.com/zh/courses/autoencoders-representation-learning/chapter-6-latent-space-manipulation"
sourceId: 653
chapter: "latent-space-manipulation"
title: "理解和操纵潜在空间"
order: 6
description: "用于可视化、分析和操纵自编码器学习到的潜在表示的方法，包括解耦。"
hasQuiz: false
---

在构建了各种自编码器架构之后，我们的重点现在转向了这些模型学习到的核心所在：潜在空间。这种通常表示为 $z$ 的压缩表示，捕获了输入数据中的主要变化。

本章提供分析和与这个学习到的空间进行交互的方法。我们将涉及：

*   用于可视化高维潜在空间的技术，例如 t-SNE 和 UMAP，以帮助理解其结构。
*   解耦的原理，即潜在空间的各个维度对应于数据中不同的变化因素，以及像 $\beta$-VAE 这样鼓励这种分离的方法。
*   通过插值和算术运算操纵潜在向量的实用方式，以生成有意义的输出或理解语义关系。
*   评估所学表示质量的度量标准和方法。

理解潜在空间有助于解释自编码器行为，并能有效应用这些模型进行生成、修改和分析任务。我们将研究如何使用常用的深度学习框架来探查这些内部表示。
