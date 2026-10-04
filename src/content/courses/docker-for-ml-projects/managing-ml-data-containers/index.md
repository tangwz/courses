---
course: "docker-for-ml-projects"
sourceUrl: "https://apxml.com/zh/courses/docker-for-ml-projects/chapter-3-managing-ml-data-containers"
sourceId: 976
chapter: "managing-ml-data-containers"
title: "在容器中管理数据和模型"
order: 3
description: "掌握在使用机器学习容器时处理数据集、训练好的模型以及其他产物的方法。"
hasQuiz: true
---

既然你已经能为机器学习环境构建容器镜像，我们现在来解决一个实际需求：数据管理。机器学习应用需要访问数据集进行训练，并需要方法来存储生成的模型。由于容器通常是短暂的，处理这些持久化数据需要特定的方法。

本章将介绍连接容器与数据的方法。我们将介绍用于管理持久存储的 Docker 卷、在开发过程中有用的绑定挂载，以及与云存储服务交互的办法。你还将了解处理模型产物的不同方式，比如比较将其包含在镜像中与动态加载的做法。最后，你将明白如何为你的容器化机器学习项目选择和实施合适的数据管理策略。
