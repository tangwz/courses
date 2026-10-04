---
course: "agentic-llm-memory-architectures"
sourceUrl: "https://apxml.com/zh/courses/agentic-llm-memory-architectures/chapter-6-evaluation-optimization-agentic-systems"
sourceId: 878
chapter: "evaluation-optimization-agentic-systems"
title: "智能体系统的评估与优化"
order: 6
description: "建立用于评估智能体表现和优化系统组成部分的衡量标准和方法。"
hasQuiz: false
---

既然已经构建了包含推理、记忆和工具能力的智能体系统，那么接下来的需求就是系统地评估它们的表现并改进其运行。本章介绍此阶段所需的方法。你将学会建立适用于智能体任务的有意义的成功衡量标准，这些标准通常超出简单的准确率度量，例如 $Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$。

我们将考察评估核心组成部分的技术：评估推理和规划过程的质量、验证工具使用的可靠性和效率，以及使用检索准确率和召回率等指标来衡量集成记忆系统的表现。此外，本章还涵盖了使用既定基准进行对比分析，针对复杂智能体行为的实用调试策略，并介绍旨在提升系统速度、成本效率和整体可靠性的优化技术，包括针对特定智能体角色的LLM微调。
