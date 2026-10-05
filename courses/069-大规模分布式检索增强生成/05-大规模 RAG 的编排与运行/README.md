# 第 5 章：大规模 RAG 的编排与运行

来源：[原章节](https://apxml.com/zh/courses/large-scale-distributed-rag/chapter-5-orchestration-operationalization-large-scale-rag)

[返回课程目录](../README.md)

在了解了大规模 RAG 系统的各个组件后，下一步便是在生产环境中高效地部署和管理它们。本章将讲解如何使你的 RAG 解决方案能够投入实际运行。你将学习如何使用 Airflow 或 Kubeflow 等工具实现工作流编排，并将 RAG 组件作为由 Kubernetes 管理的微服务进行部署。我们还将讨论 MLOps 实践，包括建立 CI/CD 流水线、全面的监控以及 A/B 测试框架。最后，我们将讨论优化基于云的 RAG 部署运行成本的方案。

## 小节

- 1. [使用 Airflow 或 Kubeflow 进行工作流编排](01-%E4%BD%BF%E7%94%A8%20Airflow%20%E6%88%96%20Kubeflow%20%E8%BF%9B%E8%A1%8C%E5%B7%A5%E4%BD%9C%E6%B5%81%E7%BC%96%E6%8E%92.md)
- 2. [RAG 组件的微服务设计模式](02-RAG%20%E7%BB%84%E4%BB%B6%E7%9A%84%E5%BE%AE%E6%9C%8D%E5%8A%A1%E8%AE%BE%E8%AE%A1%E6%A8%A1%E5%BC%8F.md)
- 3. [RAG 部署的容器化与 Kubernetes 应用](03-RAG%20%E9%83%A8%E7%BD%B2%E7%9A%84%E5%AE%B9%E5%99%A8%E5%8C%96%E4%B8%8E%20Kubernetes%20%E5%BA%94%E7%94%A8.md)
- 4. [分布式RAG系统的高级监控、日志记录与告警](04-%E5%88%86%E5%B8%83%E5%BC%8FRAG%E7%B3%BB%E7%BB%9F%E7%9A%84%E9%AB%98%E7%BA%A7%E7%9B%91%E6%8E%A7%E3%80%81%E6%97%A5%E5%BF%97%E8%AE%B0%E5%BD%95%E4%B8%8E%E5%91%8A%E8%AD%A6.md)
- 5. [RAG 系统的 CI/CD 流水线](05-RAG%20%E7%B3%BB%E7%BB%9F%E7%9A%84%20CI-CD%20%E6%B5%81%E6%B0%B4%E7%BA%BF.md)
- 6. [RAG系统的A/B测试和实验框架](06-RAG%E7%B3%BB%E7%BB%9F%E7%9A%84A-B%E6%B5%8B%E8%AF%95%E5%92%8C%E5%AE%9E%E9%AA%8C%E6%A1%86%E6%9E%B6.md)
- 7. [云端RAG的成本优化策略](07-%E4%BA%91%E7%AB%AFRAG%E7%9A%84%E6%88%90%E6%9C%AC%E4%BC%98%E5%8C%96%E7%AD%96%E7%95%A5.md)
- 8. [动手实践：在Kubernetes上部署RAG并进行监控](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%9C%A8Kubernetes%E4%B8%8A%E9%83%A8%E7%BD%B2RAG%E5%B9%B6%E8%BF%9B%E8%A1%8C%E7%9B%91%E6%8E%A7.md)
