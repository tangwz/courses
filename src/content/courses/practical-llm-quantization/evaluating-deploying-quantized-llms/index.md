---
course: "practical-llm-quantization"
sourceUrl: "https://apxml.com/zh/courses/practical-llm-quantization/chapter-6-evaluating-deploying-quantized-llms"
sourceId: 896
chapter: "evaluating-deploying-quantized-llms"
title: "量化大型语言模型的评估与部署"
order: 6
description: "学习如何评估量化大型语言模型的性能和准确率，以及部署时的考量。"
hasQuiz: true
---

应用量化技术以减小模型大小和计算成本后，下一步是严格评估结果并为模型的实际使用做准备。量化通常涉及效率和预测性能之间的权衡，使得细致的评估变得必不可少。

本章侧重于这些重要的最后阶段。你将学习如何衡量量化对模型质量的影响，使用困惑度（perplexity）和特定任务准确率基准等指标。我们将介绍实用的方法，用于测试在相关硬件上的推理速度和内存消耗。此外，我们将讨论影响性能的硬件考量，在各种环境（云端、边缘设备）中部署量化模型的策略，以及处理常见量化相关问题的技巧。一个主要成果是明白如何分析模型性能提升（如延迟 $L$ 降低或内存占用 $M$ 减小）与潜在准确率下降之间的关系，以便为实际使用做出明智的决定。
