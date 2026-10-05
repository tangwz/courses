# 第 4 章：容器化机器学习训练流程

来源：[原章节](https://apxml.com/zh/courses/docker-for-ml-projects/chapter-4-containerizing-ml-training)

[返回课程目录](../README.md)

你已经掌握了如何构建 Docker 镜像以及在容器中处理数据。本章将重点转向把这些技术应用于机器学习训练过程本身。直接运行训练脚本常常导致环境差异，使得结果难以复现。容器化通过将训练代码、其依赖项和配置打包成一个单一的可移植单元，提供了一个解决方案。

在本章中，你将学习容器化机器学习训练流程的实用方法。我们将会讨论如何组织训练脚本以在容器中执行，如何传递超参数等配置，以及如何使用 `docker run` 运行训练任务。管理训练日志、使用 NVIDIA GPU 进行加速以及使用 Docker Compose 进行基本的多容器训练设置等技术也将会讲解。

## 小节

- 1. [容器训练脚本的组织方式](01-%E5%AE%B9%E5%99%A8%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E7%9A%84%E7%BB%84%E7%BB%87%E6%96%B9%E5%BC%8F.md)
- 2. [传递配置和超参数](02-%E4%BC%A0%E9%80%92%E9%85%8D%E7%BD%AE%E5%92%8C%E8%B6%85%E5%8F%82%E6%95%B0.md)
- 3. [使用 \`docker run\` 运行训练任务](03-%E4%BD%BF%E7%94%A8%20%60docker%20run%60%20%E8%BF%90%E8%A1%8C%E8%AE%AD%E7%BB%83%E4%BB%BB%E5%8A%A1.md)
- 4. [管理训练日志](04-%E7%AE%A1%E7%90%86%E8%AE%AD%E7%BB%83%E6%97%A5%E5%BF%97.md)
- 5. [训练的GPU加速](05-%E8%AE%AD%E7%BB%83%E7%9A%84GPU%E5%8A%A0%E9%80%9F.md)
- 6. [Docker Compose在训练环境中的应用简介](06-Docker%20Compose%E5%9C%A8%E8%AE%AD%E7%BB%83%E7%8E%AF%E5%A2%83%E4%B8%AD%E7%9A%84%E5%BA%94%E7%94%A8%E7%AE%80%E4%BB%8B.md)
- 7. [动手实践：容器化并运行训练脚本](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%B9%E5%99%A8%E5%8C%96%E5%B9%B6%E8%BF%90%E8%A1%8C%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC.md)

章节测验：[在线测验](https://apxml.com/zh/courses/docker-for-ml-projects/chapter-4-containerizing-ml-training/quiz)
