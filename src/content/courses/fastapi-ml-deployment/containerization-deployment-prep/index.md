---
course: "fastapi-ml-deployment"
sourceUrl: "https://apxml.com/zh/courses/fastapi-ml-deployment/chapter-6-containerization-deployment-prep"
sourceId: 985
chapter: "containerization-deployment-prep"
title: "容器化和部署准备"
order: 6
description: "使用 Docker 打包您的 FastAPI 机器学习应用，并为其部署做好准备，使用 Gunicorn 等生产服务器。"
hasQuiz: true
---

您已经构建了一个功能完备的FastAPI应用，它能够提供机器学习模型预测服务，并通过测试验证了其行为。现在将把重心放在准备该应用，以便在您的本地开发环境之外进行部署。确保在不同环境中运行一致是一个普遍的难题，而容器化能有效应对此问题。

本章着重讲解如何使用Docker打包您的应用。您将学习创建针对FastAPI服务定制的`Dockerfile`，包含应用代码、Python依赖项以及所需的机器学习模型构件。我们将按照步骤构建Docker镜像，并将其作为容器运行。此外，您将学习在容器内部使用环境变量管理应用配置的方法，并理解如何设置生产级别的ASGI服务器（例如Uvicorn，通常由Gunicorn管理）来高效地提供您的应用服务。在本章结束时，您将拥有一个容器化的机器学习API版本，可用于部署流程。
