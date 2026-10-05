# 第 5 章：模型部署要点

来源：[原章节](https://apxml.com/zh/courses/applied-data-science/chapter-5-model-deployment-fundamentals)

[返回课程目录](../README.md)

建立一个在历史数据上表现良好的机器学习模型是一个重要的成就，但其价值通常只有在模型能够对新的、未见过的数据进行预测时才能体现。本章侧重于将训练好的模型变得可用并投入实际操作所需的必要步骤。

您将学习部署模型的核心方法，首先是保存和加载训练好的模型以确保其持久化的方法。接着，我们将介绍模型服务的思路，并指导您使用常见的Python Web框架（如Flask或FastAPI）构建一个简单的REST API，以公开模型的预测能力。最后，我们将讲解使用Docker进行容器化，这是一种打包模型及其依赖项以实现一致部署的方式，同时还将介绍模型投入生产后的基本监控思路。本章提供将模型开发与实际应用衔接起来所需的基本知识。

## 小节

- 1. [保存和加载训练好的模型](01-%E4%BF%9D%E5%AD%98%E5%92%8C%E5%8A%A0%E8%BD%BD%E8%AE%AD%E7%BB%83%E5%A5%BD%E7%9A%84%E6%A8%A1%E5%9E%8B.md)
- 2. [模型服务框架简介](02-%E6%A8%A1%E5%9E%8B%E6%9C%8D%E5%8A%A1%E6%A1%86%E6%9E%B6%E7%AE%80%E4%BB%8B.md)
- 3. [构建模型预测的REST API](03-%E6%9E%84%E5%BB%BA%E6%A8%A1%E5%9E%8B%E9%A2%84%E6%B5%8B%E7%9A%84REST%20API.md)
- 4. [使用 Docker 容器化应用](04-%E4%BD%BF%E7%94%A8%20Docker%20%E5%AE%B9%E5%99%A8%E5%8C%96%E5%BA%94%E7%94%A8.md)
- 5. [模型监控基本要点](05-%E6%A8%A1%E5%9E%8B%E7%9B%91%E6%8E%A7%E5%9F%BA%E6%9C%AC%E8%A6%81%E7%82%B9.md)
- 6. [实践：创建模型API并将其容器化](06-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%88%9B%E5%BB%BA%E6%A8%A1%E5%9E%8BAPI%E5%B9%B6%E5%B0%86%E5%85%B6%E5%AE%B9%E5%99%A8%E5%8C%96.md)

章节测验：[在线测验](https://apxml.com/zh/courses/applied-data-science/chapter-5-model-deployment-fundamentals/quiz)
