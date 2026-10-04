---
course: "foundations-transformers-architecture"
sourceUrl: "https://apxml.com/zh/courses/foundations-transformers-architecture/chapter-3-multi-head-self-attention"
sourceId: 531
chapter: "multi-head-self-attention"
title: "多头自注意力"
order: 3
description: "了解多头自注意力、线性投影、注意力头拼接及其在 Transformer 中的益处。"
hasQuiz: false
---

基于缩放点积注意力机制，本章侧重于**自注意力**，这是一种特殊应用，序列在该应用中关联自身不同位置以计算其表示。然而，仅使用单个注意力函数会限制其同时捕获多样化关联关系的能力，可能导致不理想的平均化效果。

为克服这一局限，我们介绍**多头注意力**。这种方法使模型能够同时关注来自不同表示子空间、不同位置的信息，有效实现注意力机制的多次并行运行。

本章内容包括：
*   自注意力的定义：查询 ($Q$)、键 ($K$) 和值 ($V$) 都源自相同的输入序列。
*   使用多个并行注意力层（称为“头”）的理由。
*   使用不同线性变换为每个头投影 $Q$、$K$ 和 $V$ 的机制。
*   注意力如何在多个头之间并行计算。
*   拼接各头输出并应用最终线性投影的过程。
*   不同头如何能够学习到不同的关联模式。
*   多头注意力模块的实现方法。

通过理解这些组成部分，您将对 Transformer 架构的一个基本构建块有所认识。
