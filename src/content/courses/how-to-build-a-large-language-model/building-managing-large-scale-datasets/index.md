---
course: "how-to-build-a-large-language-model"
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-8-building-managing-large-scale-datasets"
sourceId: 1086
chapter: "building-managing-large-scale-datasets"
title: "建立和管理大规模数据集"
order: 8
description: "应对存储、索引、版本控制以及高效访问LLM训练所需海量数据集的挑战。"
hasQuiz: false
---

在获取并预处理了潜在的数太字节乃至拍字节文本数据后，下一个工程步骤是有效地存储、组织和访问这些海量数据集合。本章介绍管理数据集所需的基础设施和技术，以适应大型语言模型训练的规模需求。

我们将讨论实际的考量，例如：
*   选择合适的数据存储格式（如文本、Apache Arrow 或 Parquet）。
*   使用分布式文件系统，例如 HDFS 或云对象存储。
*   实施数据索引以实现高效检索。
*   建立数据集版本控制实践以保证可复现性。
*   设计流式数据加载器，将数据有效地输入到分布式训练流程中。
