# 第 2 章：使用 Dockerfile 构建定制的机器学习环境

来源：[原章节](https://apxml.com/zh/courses/docker-for-ml-projects/chapter-2-building-ml-dockerfiles)

[返回课程目录](../README.md)

在掌握 Docker 基础知识后，本章将侧重于使用 `Dockerfile` 指令为机器学习项目构建定制的容器镜像。您将学习如何组织 `Dockerfile` 的结构，以提高清晰度和效率，选择合适的基础镜像（例如官方 Python 或支持 CUDA 的镜像），使用 `pip` 和 `Conda` 等工具管理复杂的 Python 依赖项，将项目代码和相关文件整合到镜像中，并使用 `WORKDIR`、`ENTRYPOINT` 和 `CMD` 定义容器的运行时行为。目标是创建一致且可重现的环境，以确保机器学习开发的可靠性和执行。

## 小节

- 1. [Dockerfile 的结构化方法](01-Dockerfile%20%E7%9A%84%E7%BB%93%E6%9E%84%E5%8C%96%E6%96%B9%E6%B3%95.md)
- 2. [选择恰当的起始映像](02-%E9%80%89%E6%8B%A9%E6%81%B0%E5%BD%93%E7%9A%84%E8%B5%B7%E5%A7%8B%E6%98%A0%E5%83%8F.md)
- 3. [管理 Python 依赖项 (pip)](03-%E7%AE%A1%E7%90%86%20Python%20%E4%BE%9D%E8%B5%96%E9%A1%B9%20%28pip%29.md)
- 4. [管理 Python 依赖项 (Conda)](04-%E7%AE%A1%E7%90%86%20Python%20%E4%BE%9D%E8%B5%96%E9%A1%B9%20%28Conda%29.md)
- 5. [使用环境变量](05-%E4%BD%BF%E7%94%A8%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F.md)
- 6. [复制代码和文件](06-%E5%A4%8D%E5%88%B6%E4%BB%A3%E7%A0%81%E5%92%8C%E6%96%87%E4%BB%B6.md)
- 7. [设置工作目录和入口点](07-%E8%AE%BE%E7%BD%AE%E5%B7%A5%E4%BD%9C%E7%9B%AE%E5%BD%95%E5%92%8C%E5%85%A5%E5%8F%A3%E7%82%B9.md)
- 8. [动手实践：构建 Scikit-learn 环境](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9E%84%E5%BB%BA%20Scikit-learn%20%E7%8E%AF%E5%A2%83.md)

章节测验：[在线测验](https://apxml.com/zh/courses/docker-for-ml-projects/chapter-2-building-ml-dockerfiles/quiz)
