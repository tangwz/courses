# 第 3 章：在容器中管理数据和模型

来源：[原章节](https://apxml.com/zh/courses/docker-for-ml-projects/chapter-3-managing-ml-data-containers)

[返回课程目录](../README.md)

既然你已经能为机器学习环境构建容器镜像，我们现在来解决一个实际需求：数据管理。机器学习应用需要访问数据集进行训练，并需要方法来存储生成的模型。由于容器通常是短暂的，处理这些持久化数据需要特定的方法。

本章将介绍连接容器与数据的方法。我们将介绍用于管理持久存储的 Docker 卷、在开发过程中有用的绑定挂载，以及与云存储服务交互的办法。你还将了解处理模型产物的不同方式，比如比较将其包含在镜像中与动态加载的做法。最后，你将明白如何为你的容器化机器学习项目选择和实施合适的数据管理策略。

## 小节

- 1. [理解容器存储](01-%E7%90%86%E8%A7%A3%E5%AE%B9%E5%99%A8%E5%AD%98%E5%82%A8.md)
- 2. [用于开发的绑定挂载](02-%E7%94%A8%E4%BA%8E%E5%BC%80%E5%8F%91%E7%9A%84%E7%BB%91%E5%AE%9A%E6%8C%82%E8%BD%BD.md)
- 3. [使用 Docker 卷实现数据持久化](03-%E4%BD%BF%E7%94%A8%20Docker%20%E5%8D%B7%E5%AE%9E%E7%8E%B0%E6%95%B0%E6%8D%AE%E6%8C%81%E4%B9%85%E5%8C%96.md)
- 4. [比较绑定挂载和卷](04-%E6%AF%94%E8%BE%83%E7%BB%91%E5%AE%9A%E6%8C%82%E8%BD%BD%E5%92%8C%E5%8D%B7.md)
- 5. [从容器访问云存储](05-%E4%BB%8E%E5%AE%B9%E5%99%A8%E8%AE%BF%E9%97%AE%E4%BA%91%E5%AD%98%E5%82%A8.md)
- 6. [将模型打包到镜像中与卷的使用](06-%E5%B0%86%E6%A8%A1%E5%9E%8B%E6%89%93%E5%8C%85%E5%88%B0%E9%95%9C%E5%83%8F%E4%B8%AD%E4%B8%8E%E5%8D%B7%E7%9A%84%E4%BD%BF%E7%94%A8.md)
- 7. [动手实践：挂载数据集和保存模型](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%8C%82%E8%BD%BD%E6%95%B0%E6%8D%AE%E9%9B%86%E5%92%8C%E4%BF%9D%E5%AD%98%E6%A8%A1%E5%9E%8B.md)

章节测验：[在线测验](https://apxml.com/zh/courses/docker-for-ml-projects/chapter-3-managing-ml-data-containers/quiz)
