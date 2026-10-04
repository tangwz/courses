---
course: "llm-compression-acceleration"
sourceUrl: "https://apxml.com/zh/courses/llm-compression-acceleration/chapter-4-knowledge-distillation-large-models"
sourceId: 854
chapter: "knowledge-distillation-large-models"
title: "大模型知识蒸馏"
order: 4
description: "设计并实现知识蒸馏流程，以将大型教师LLM的能力迁移到更小的学生模型。"
hasQuiz: false
---

大型语言模型（LLMs）具备强大的能力，但其庞大的体积和计算要求常妨碍在资源受限的环境中部署。本章介绍知识蒸馏（KD）这一技术来解决此问题。其主要思想是将大型、复杂的“教师”模型所学到的知识迁移到更小、更高效的“学生”模型上，旨在保持性能的同时减小模型体积并降低推理成本。

您将掌握知识蒸馏的基本原理，从使用软目标（教师模型的输出概率）的最初思路，到涉及中间特征匹配和注意力迁移的更进一步的方法。我们将介绍：

*   设计不同的蒸馏目标，例如最小化教师和学生模型输出分布之间的Kullback-Leibler散度，通常表示为$L_{KD} = D_{KL}(p_{student}||p_{teacher})$。
*   知识蒸馏策略，例如任务专用型和任务无关型方法以及自蒸馏。
*   处理与蒸馏生成式LLM相关的特殊难点。
*   评估得到的学生模型的忠实度和性能的方法。
*   实现一个实用蒸馏流程。

本章结束时，您将明白如何设计、实现和评估专门为压缩大型语言模型而定制的知识蒸馏过程。
