---
course: "how-to-build-a-large-language-model"
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-16-implementing-distributed-training-frameworks"
sourceId: 1094
chapter: "implementing-distributed-training-frameworks"
title: "实现分布式训练框架"
order: 16
description: "学习使用 DeepSpeed 和 Megatron-LM 等流行框架，进行复杂的分布式训练配置。"
hasQuiz: false
---

训练真正的大模型，不只是要理解像数据并行 ($DP$)、张量并行 ($TP$) 和流水线并行 ($PP$) 这样的并行策略；还需要专为处理其复杂性而设计的工具。本章将从之前讨论的理论思路转向使用专门框架的实际运用。

你将学习如何通过配置和使用 DeepSpeed 和 Megatron-LM 等流行库来运用这些策略。我们将介绍 DeepSpeed 的 ZeRO 内存优化设置（阶段 $1$、$2$ 和 $3$），以及使用 Megatron-LM 配置张量并行和流水线并行。本章结束时，你将能够把分布式训练理论付诸实践，应用到你自己的大模型项目中。
