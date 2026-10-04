# 第 6 章：容器化和部署准备

来源：[原章节](https://apxml.com/zh/courses/fastapi-ml-deployment/chapter-6-containerization-deployment-prep)

[返回课程目录](../README.md)

您已经构建了一个功能完备的FastAPI应用，它能够提供机器学习模型预测服务，并通过测试验证了其行为。现在将把重心放在准备该应用，以便在您的本地开发环境之外进行部署。确保在不同环境中运行一致是一个普遍的难题，而容器化能有效应对此问题。

本章着重讲解如何使用Docker打包您的应用。您将学习创建针对FastAPI服务定制的`Dockerfile`，包含应用代码、Python依赖项以及所需的机器学习模型构件。我们将按照步骤构建Docker镜像，并将其作为容器运行。此外，您将学习在容器内部使用环境变量管理应用配置的方法，并理解如何设置生产级别的ASGI服务器（例如Uvicorn，通常由Gunicorn管理）来高效地提供您的应用服务。在本章结束时，您将拥有一个容器化的机器学习API版本，可用于部署流程。

## 小节

- 1. [Docker 应用打包介绍](01-Docker%20%E5%BA%94%E7%94%A8%E6%89%93%E5%8C%85%E4%BB%8B%E7%BB%8D.md)
- 2. [编写 FastAPI 应用的 Dockerfile](02-%E7%BC%96%E5%86%99%20FastAPI%20%E5%BA%94%E7%94%A8%E7%9A%84%20Dockerfile.md)
- 3. [在 Docker 镜像中打包机器学习模型](03-%E5%9C%A8%20Docker%20%E9%95%9C%E5%83%8F%E4%B8%AD%E6%89%93%E5%8C%85%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E6%A8%A1%E5%9E%8B.md)
- 4. [构建和运行 Docker 容器](04-%E6%9E%84%E5%BB%BA%E5%92%8C%E8%BF%90%E8%A1%8C%20Docker%20%E5%AE%B9%E5%99%A8.md)
- 5. [在 Docker 中管理 Python 依赖](05-%E5%9C%A8%20Docker%20%E4%B8%AD%E7%AE%A1%E7%90%86%20Python%20%E4%BE%9D%E8%B5%96.md)
- 6. [使用环境变量配置应用程序](06-%E4%BD%BF%E7%94%A8%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E9%85%8D%E7%BD%AE%E5%BA%94%E7%94%A8%E7%A8%8B%E5%BA%8F.md)
- 7. [生产部署准备 (Gunicorn/Uvicorn)](07-%E7%94%9F%E4%BA%A7%E9%83%A8%E7%BD%B2%E5%87%86%E5%A4%87%20%28Gunicorn-Uvicorn%29.md)
- 8. [动手实践：机器学习API的容器化](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0API%E7%9A%84%E5%AE%B9%E5%99%A8%E5%8C%96.md)

章节测验：[在线测验](https://apxml.com/zh/courses/fastapi-ml-deployment/chapter-6-containerization-deployment-prep/quiz)
