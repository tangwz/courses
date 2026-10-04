---
course: "llm-compression-acceleration"
sourceUrl: "https://apxml.com/zh/courses/llm-compression-acceleration/chapter-1-foundations-llm-efficiency-challenges"
sourceId: 845
chapter: "foundations-llm-efficiency-challenges"
title: "大语言模型效率挑战：背景与基本原理"
order: 1
description: "理解促使LLM优化必要性的计算成本、瓶颈和硬件考量。"
hasQuiz: false
---

大语言模型（LLMs）展现出令人瞩目的能力，但其计算需求给部署带来了不小的难题。本章将说明为何需要优化。我们将考察模型大小与资源需求之间的关系，这通常由缩放定律描述，例如 $Performance \propto N^\alpha$，其中 $N$ 代表参数量或数据大小。

您将学会识别推理过程中的主要瓶颈，特别是内存带宽限制和计算制约。我们将分析Transformer架构的特定组成部分如何影响效率，并介绍用于评估模型压缩和速度的标准指标。此外，我们将概览用于大语言模型的常见硬件平台（CPU、GPU、TPU），并提及优化工作中固有的理论权衡。学完本章，您将对驱动本课程其余部分所讲技术的根本效率问题有扎实的理解。
