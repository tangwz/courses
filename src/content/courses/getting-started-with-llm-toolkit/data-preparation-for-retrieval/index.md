---
course: "getting-started-with-llm-toolkit"
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-llm-toolkit/chapter-4-data-preparation-for-retrieval"
sourceId: 1348
chapter: "data-preparation-for-retrieval"
title: "数据检索的准备工作"
order: 4
description: "了解如何从 PDF 和网页加载数据，应用文本预处理，并为 RAG 使用分块策略。"
hasQuiz: false
---

语言模型回答特定文档相关问题的能力，很大程度上取决于信息如何被处理和呈现。本章介绍检索增强生成（RAG）数据管道的初始阶段，主要说明如何将原始材料转换为适合处理的结构化形式。

你将了解如何：
*   使用 `document` 模块，从不同来源加载数据，包括 PDF 文件和网页。
*   将大型文档 $D$ 分割成一系列较小的文本块 $\{c_1, c_2, ..., c_n\}$，并应用不同的分块策略。
*   使用 `preprocessing` 模块清洗和标准化文本，从而提升后续步骤的数据质量。

完成这些部分后，你将拥有一个清洗过的、已分块的数据集，可用于下一章将要介绍的嵌入处理。
