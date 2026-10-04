---
course: "foundations-transformers-architecture"
sourceUrl: "https://apxml.com/zh/courses/foundations-transformers-architecture/chapter-1-revisiting-sequence-modeling-limitations"
sourceId: 525
chapter: "revisiting-sequence-modeling-limitations"
title: "重温序列建模的局限性"
order: 1
description: "了解传统序列模型（如RNNs和LSTMs）中长距离依赖和并行化处理的挑战。"
hasQuiz: false
---

为理解Transformer架构的设计动因，我们首先回顾了之前用于序列建模的主要方法。循环神经网络（RNNs）、长短期记忆网络（LSTMs）和门控循环单元（GRUs）在处理序列数据方面代表了重要的进展。然而，其固有的结构存在一些局限性，阻碍了进展，尤其是在序列长度不断增加的情况下。

本章将审视这些具体的挑战。我们将讨论：

*   循环模型固有的顺序处理特性及其对计算效率的影响。
*   训练深度循环网络时遇到的数学上的难题，特别是梯度消失和梯度爆炸问题（例如，早期时间步 $t$ 时梯度消失，表现为 $ \frac{\partial L}{\partial \theta_t} \approx 0 $）。
*   LSTMs和GRUs中的门控机制如何尝试缓解梯度问题。
*   循环模型在有效建模输入序列中很长距离的依赖关系方面持续存在的问题。

认识到这些限制，能为理解Transformer模型在后续章节中引入的架构创新提供必要的背景信息。
