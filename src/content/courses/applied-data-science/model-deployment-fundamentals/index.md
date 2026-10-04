---
course: "applied-data-science"
sourceUrl: "https://apxml.com/zh/courses/applied-data-science/chapter-5-model-deployment-fundamentals"
sourceId: 523
chapter: "model-deployment-fundamentals"
title: "模型部署要点"
order: 5
description: "学习将机器学习模型准备并部署为API的基本步骤。"
hasQuiz: true
---

建立一个在历史数据上表现良好的机器学习模型是一个重要的成就，但其价值通常只有在模型能够对新的、未见过的数据进行预测时才能体现。本章侧重于将训练好的模型变得可用并投入实际操作所需的必要步骤。

您将学习部署模型的核心方法，首先是保存和加载训练好的模型以确保其持久化的方法。接着，我们将介绍模型服务的思路，并指导您使用常见的Python Web框架（如Flask或FastAPI）构建一个简单的REST API，以公开模型的预测能力。最后，我们将讲解使用Docker进行容器化，这是一种打包模型及其依赖项以实现一致部署的方式，同时还将介绍模型投入生产后的基本监控思路。本章提供将模型开发与实际应用衔接起来所需的基本知识。
