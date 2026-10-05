# 第 7 章：可扩展性、可靠性和可维护性

来源：[原章节](https://apxml.com/zh/courses/optimizing-rag-for-production/chapter-7-rag-scalability-reliability-maintainability)

[返回课程目录](../README.md)

在讨论了RAG组件的优化和整体系统效率之后，我们现在将注意力转向生产环境的持续运行需求。目标是构建和维护RAG系统，使其能持续良好运行，随负载增加有效扩展，在各种条件下可靠运作，并可通过实用、可重复的流程进行管理。

本章将指导您如何架构高可用性的RAG系统，确保即使系统部分出现问题，也能保持正常运行。我们将介绍容错机制，以帮助您的系统从故障中恢复。您将学习管理知识库更新和刷新周期的方法，这对于保持RAG系统的信息最新和相关很重要。我们还将考虑多租户问题，使用CI/CD管道自动化部署流程，建立数据治理，以及调试复杂生产问题的方法，并创建有效的操作文档。这些做法对于您的RAG应用在实际环境中的长期可行性和成功非常重要。

## 小节

- 1. [RAG系统的高可用架构设计](01-RAG%E7%B3%BB%E7%BB%9F%E7%9A%84%E9%AB%98%E5%8F%AF%E7%94%A8%E6%9E%B6%E6%9E%84%E8%AE%BE%E8%AE%A1.md)
- 2. [在RAG中实现容错](02-%E5%9C%A8RAG%E4%B8%AD%E5%AE%9E%E7%8E%B0%E5%AE%B9%E9%94%99.md)
- 3. [管理知识库更新与刷新周期](03-%E7%AE%A1%E7%90%86%E7%9F%A5%E8%AF%86%E5%BA%93%E6%9B%B4%E6%96%B0%E4%B8%8E%E5%88%B7%E6%96%B0%E5%91%A8%E6%9C%9F.md)
- 4. [多租户与多RAG实例管理](04-%E5%A4%9A%E7%A7%9F%E6%88%B7%E4%B8%8E%E5%A4%9ARAG%E5%AE%9E%E4%BE%8B%E7%AE%A1%E7%90%86.md)
- 5. [使用CI/CD流水线自动化RAG部署](05-%E4%BD%BF%E7%94%A8CI-CD%E6%B5%81%E6%B0%B4%E7%BA%BF%E8%87%AA%E5%8A%A8%E5%8C%96RAG%E9%83%A8%E7%BD%B2.md)
- 6. [RAG系统中的数据治理与血缘追溯](06-RAG%E7%B3%BB%E7%BB%9F%E4%B8%AD%E7%9A%84%E6%95%B0%E6%8D%AE%E6%B2%BB%E7%90%86%E4%B8%8E%E8%A1%80%E7%BC%98%E8%BF%BD%E6%BA%AF.md)
- 7. [生产RAG问题的高级调试](07-%E7%94%9F%E4%BA%A7RAG%E9%97%AE%E9%A2%98%E7%9A%84%E9%AB%98%E7%BA%A7%E8%B0%83%E8%AF%95.md)
- 8. [RAG 系统的运行文档](08-RAG%20%E7%B3%BB%E7%BB%9F%E7%9A%84%E8%BF%90%E8%A1%8C%E6%96%87%E6%A1%A3.md)
- 9. [实践：设计可扩展的RAG架构](09-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E8%AE%BE%E8%AE%A1%E5%8F%AF%E6%89%A9%E5%B1%95%E7%9A%84RAG%E6%9E%B6%E6%9E%84.md)
