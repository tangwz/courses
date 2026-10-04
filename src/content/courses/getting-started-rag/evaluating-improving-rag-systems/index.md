---
course: "getting-started-rag"
sourceUrl: "https://apxml.com/zh/courses/getting-started-rag/chapter-6-evaluating-improving-rag-systems"
sourceId: 752
chapter: "evaluating-improving-rag-systems"
title: "RAG系统的评估与改进"
order: 6
description: "了解评估RAG系统性能的常用指标、方法以及基本的改进策略。"
hasQuiz: true
---

构建检索增强生成（RAG）管道是重要一步，但确定其有效性并优化其性能是后续的必要行动。一个检索到不相关信息或生成不准确回复的RAG系统，其实用价值有限。

本章侧重于评估RAG系统质量的方法。您将了解：

*   评估RAG输出时遇到的常见难题。
*   评估各个组成部分的方法：检索器查找相关上下文的能力，以及生成器基于该上下文生成忠实且相关答案的能力。
*   用于评估检索（例如命中率或平均倒数排名，$MRR$）和生成质量的指标。
*   识别RAG系统中的常见故障点。
*   提高性能的基本技术，例如调整数据分块策略或优化提示词。

本章结束时，您将对如何衡量RAG系统性能并应用初步的改进策略有基本认识。
