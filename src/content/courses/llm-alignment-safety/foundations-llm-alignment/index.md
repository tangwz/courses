---
course: "llm-alignment-safety"
sourceUrl: "https://apxml.com/zh/courses/llm-alignment-safety/chapter-1-foundations-llm-alignment"
sourceId: 681
chapter: "foundations-llm-alignment"
title: "大型语言模型对齐的基本原理"
order: 1
description: "回顾对齐大型语言模型中的主要思想和挑战，为学习高级技术做好准备。"
hasQuiz: false
---

本章确立了理解大型语言模型对齐所必需的主要思想。在处理高级技术之前，我们需要扎实掌握这些基本要点。

我们首先给出专门针对大型语言模型的对齐操作定义，并概述对齐问题的主要目标和相关难题。你将回顾指令遵循和标准微调如何作为实现对齐行为的初步步骤。我们接着介绍衡量对齐的初步方法，并讨论这些方法为何常常不足。

本章也区分了内部对齐与外部对齐。理解这种区别有助于诊断失败模式。最后，我们考察常见问题，例如目标规避——即模型优化的是代理目标$R_{\text{proxy}}$而非预期目标$R_{\text{intended}}$，以及奖励作弊——说明模型如何钻定义不清的目标的空子。这些基本思想为你学习本课程后续章节中讨论的更复杂方法做好准备。
