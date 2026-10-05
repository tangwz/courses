# 第 5 章：机器学习模型推理的容器化

来源：[原章节](https://apxml.com/zh/courses/docker-for-ml-projects/chapter-5-containerizing-ml-inference)

[返回课程目录](../README.md)

模型训练完成后，下一步就是准备模型以对新数据进行预测。本章主要讲解如何将训练好的机器学习模型和必要的推理代码打包到 Docker 容器中。目的是构建自包含、高效的单元，用于提供预测服务。

你将掌握以下实用方法：

*   在容器内设计和构建简单的推理服务，通常使用 Flask 或 FastAPI 等 Web 框架。
*   使用多阶段构建和精细的依赖管理等方法，优化生成的 Docker 镜像，以减小尺寸并提高效率。
*   配置用于推理的容器，包括暴露网络端口和实现健康检查，以确保服务正常运行。
*   在容器环境中有效打包模型和相关文件。

学完本章后，你将能够构建优化的 Docker 容器，用于为训练好的机器学习模型提供服务。

## 小节

- 1. [设计推理服务](01-%E8%AE%BE%E8%AE%A1%E6%8E%A8%E7%90%86%E6%9C%8D%E5%8A%A1.md)
- 2. [构建推理API (Flask/FastAPI)](02-%E6%9E%84%E5%BB%BA%E6%8E%A8%E7%90%86API%20%28Flask-FastAPI%29.md)
- 3. [优化镜像大小：多阶段构建](03-%E4%BC%98%E5%8C%96%E9%95%9C%E5%83%8F%E5%A4%A7%E5%B0%8F%EF%BC%9A%E5%A4%9A%E9%98%B6%E6%AE%B5%E6%9E%84%E5%BB%BA.md)
- 4. [减少推理时的依赖项](04-%E5%87%8F%E5%B0%91%E6%8E%A8%E7%90%86%E6%97%B6%E7%9A%84%E4%BE%9D%E8%B5%96%E9%A1%B9.md)
- 5. [对外开放端口用于API访问](05-%E5%AF%B9%E5%A4%96%E5%BC%80%E6%94%BE%E7%AB%AF%E5%8F%A3%E7%94%A8%E4%BA%8EAPI%E8%AE%BF%E9%97%AE.md)
- 6. [推理容器的健康检查](06-%E6%8E%A8%E7%90%86%E5%AE%B9%E5%99%A8%E7%9A%84%E5%81%A5%E5%BA%B7%E6%A3%80%E6%9F%A5.md)
- 7. [实操：容器化简单推理API](07-%E5%AE%9E%E6%93%8D%EF%BC%9A%E5%AE%B9%E5%99%A8%E5%8C%96%E7%AE%80%E5%8D%95%E6%8E%A8%E7%90%86API.md)

章节测验：[在线测验](https://apxml.com/zh/courses/docker-for-ml-projects/chapter-5-containerizing-ml-inference/quiz)
