# 第 6 章：RAG系统的评估与改进

来源：[原章节](https://apxml.com/zh/courses/getting-started-rag/chapter-6-evaluating-improving-rag-systems)

[返回课程目录](../README.md)

构建检索增强生成（RAG）管道是重要一步，但确定其有效性并优化其性能是后续的必要行动。一个检索到不相关信息或生成不准确回复的RAG系统，其实用价值有限。

本章侧重于评估RAG系统质量的方法。您将了解：

*   评估RAG输出时遇到的常见难题。
*   评估各个组成部分的方法：检索器查找相关上下文的能力，以及生成器基于该上下文生成忠实且相关答案的能力。
*   用于评估检索（例如命中率或平均倒数排名，$MRR$）和生成质量的指标。
*   识别RAG系统中的常见故障点。
*   提高性能的基本技术，例如调整数据分块策略或优化提示词。

本章结束时，您将对如何衡量RAG系统性能并应用初步的改进策略有基本认识。

## 小节

- 1. [评估RAG的挑战](01-%E8%AF%84%E4%BC%B0RAG%E7%9A%84%E6%8C%91%E6%88%98.md)
- 2. [组件层面的评估：检索](02-%E7%BB%84%E4%BB%B6%E5%B1%82%E9%9D%A2%E7%9A%84%E8%AF%84%E4%BC%B0%EF%BC%9A%E6%A3%80%E7%B4%A2.md)
- 3. [组件级别评估：生成](03-%E7%BB%84%E4%BB%B6%E7%BA%A7%E5%88%AB%E8%AF%84%E4%BC%B0%EF%BC%9A%E7%94%9F%E6%88%90.md)
- 4. [端到端RAG评估框架](04-%E7%AB%AF%E5%88%B0%E7%AB%AFRAG%E8%AF%84%E4%BC%B0%E6%A1%86%E6%9E%B6.md)
- 5. [常见故障模式](05-%E5%B8%B8%E8%A7%81%E6%95%85%E9%9A%9C%E6%A8%A1%E5%BC%8F.md)
- 6. [改进的基本策略](06-%E6%94%B9%E8%BF%9B%E7%9A%84%E5%9F%BA%E6%9C%AC%E7%AD%96%E7%95%A5.md)
- 7. [实践：分析RAG输出质量](07-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%88%86%E6%9E%90RAG%E8%BE%93%E5%87%BA%E8%B4%A8%E9%87%8F.md)

章节测验：[在线测验](https://apxml.com/zh/courses/getting-started-rag/chapter-6-evaluating-improving-rag-systems/quiz)
