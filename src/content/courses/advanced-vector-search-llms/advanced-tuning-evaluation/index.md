---
course: "advanced-vector-search-llms"
sourceUrl: "https://apxml.com/zh/courses/advanced-vector-search-llms/chapter-5-advanced-tuning-evaluation"
sourceId: 787
chapter: "advanced-tuning-evaluation"
title: "高级调优与评估"
order: 5
description: "微调向量搜索参数，并使用合适的指标和方法严格评估性能。"
hasQuiz: false
---

构建和部署向量搜索系统，需要仔细考虑性能与相关性之间的权衡。本章介绍系统化方法，用于微调系统参数并进行严格评估，以确保您的实现达成目标。

您将学习使用重要的评估指标，包括Recall@k、Precision@k和查询延迟，来衡量系统效能。我们将介绍构建合适真实数据集的方法，这对于可靠评估非常重要。本章详细说明了分析主要索引参数敏感性的技术，例如HNSW中的`efSearch`或IVF索引中的`nprobe`，以了解它们对性能的影响。

此外，我们将考察设置A/B测试的实际策略，以比较不同的搜索配置；诊断常见的相关性问题；区分离线和在线评估方法；并根据特定应用需求调整调优方法，例如与更宽泛的语义搜索任务相比，针对检索增强生成（RAG）的应用。目标是为您提供优化和验证您先进向量搜索解决方案的工具。
