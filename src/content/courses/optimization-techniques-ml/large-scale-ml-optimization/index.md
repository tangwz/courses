---
course: "optimization-techniques-ml"
sourceUrl: "https://apxml.com/zh/courses/optimization-techniques-ml/chapter-4-large-scale-ml-optimization"
sourceId: 438
chapter: "large-scale-ml-optimization"
title: "大规模数据集的优化"
order: 4
description: "学习包含方差降低（SAG, SVRG）的随机优化方法以及针对海量数据集的策略。"
hasQuiz: false
---

训练机器学习模型通常会涉及大到无法在单个批次中处理的数据集。需要计算整个数据集梯度的标准梯度下降，在计算上变得不可行。虽然随机梯度下降（SGD）通过使用来自小数据样本（小批量）的梯度提供了一种可扩展的替代方案，但这会给优化过程带来显著的方差，可能减慢收敛速度。

本章解决使用海量数据集优化模型的具体难题。你将学习以下内容：

*   **方差降低方法：** 比如随机平均梯度（SAG）和随机方差缩减梯度（SVRG）等方法，旨在通过降低随机梯度中固有的噪声来加速收敛。
*   **小批量梯度下降分析：** 理解选择不同小批量大小时的实际权衡。
*   **并行与分布式策略：** 例如异步SGD和数据并行等思路，将计算负载分配到多个处理器或机器上。

我们将审视这些方法如何实现在否则难以处理的数据集上进行高效训练。实际实现以及它们收敛行为的分析将是重要组成部分。
