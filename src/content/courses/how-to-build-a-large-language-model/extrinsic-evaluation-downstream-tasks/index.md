---
course: "how-to-build-a-large-language-model"
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-22-extrinsic-evaluation-downstream-tasks"
sourceId: 1100
chapter: "extrinsic-evaluation-downstream-tasks"
title: "下游任务外部评估"
order: 22
description: "通过评估大型语言模型在各种下游NLP基准测试（如GLUE、SuperGLUE）上的表现，间接评定其效用。"
hasQuiz: false
---

尽管困惑度等内部指标能衡量模型预测下一个词元的能力，但它们不总能完整体现模型的实际作用。单纯基于模型内部的语言建模能力进行评估，可能不足以了解其在具体应用中的表现。

本章将重心转向外部评估方法。我们将研究如何通过衡量大型语言模型（LLM）在具体的下游自然语言处理（NLP）任务上的表现来评定它们。这种方法能让人们更实在地了解模型在部署后可能遇到的情况中的表现。

您将学到:
*   为何要使用下游任务进行评估。
*   常见的基准任务，例如文本分类、问答和摘要。
*   专门为这些评估任务微调预训练模型的标准步骤。
*   广泛使用的基准测试集，例如GLUE和SuperGLUE。
*   在少样本和零样本设置中评估模型，考察它们在只有少量任务相关示例时泛化的能力。
*   设计根据独特需求定制的自定义评估任务时需要考虑的方面。

在本章结束时，您将了解如何通过将LLM应用于具体的、实际的NLP问题来衡量其有效性，这将补充从内部指标获得的信息。
