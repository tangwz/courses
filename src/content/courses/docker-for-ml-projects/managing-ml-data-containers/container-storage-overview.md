---
course: "docker-for-ml-projects"
chapter: "managing-ml-data-containers"
lesson: "container-storage-overview"
sourceId: 5270
sourceUrl: "https://apxml.com/zh/courses/docker-for-ml-projects/chapter-3-managing-ml-data-containers/container-storage-overview"
title: "理解容器存储"
description: "介绍Docker容器存储和访问数据的方法。"
order: 1
plots: []
sourceHash: "9cfc02ed0959e5bb1a5345901288ee1bba5c5bd3f71a3b8c4e3ceb8cfd08afcc"
sourceCorrections: []
---

正如我们所说，机器学习 (machine learning)工作流程对数据有很强的依赖。你需要数据集来训练模型，在长时间训练过程中可能会生成中间检查点，最终还需要保存训练好的模型文件。在使用容器时，了解数据如何处理非常重要，因为其默认行为不适合数据持久化。

容器运行时，Docker 在只读镜像层之上创建了一个可写层。可以将镜像视为蓝图，容器则是根据该蓝图构建的运行实例。运行中的应用程序所做的任何更改，例如创建新文件、修改现有文件或下载数据，都会写入到这个容器特有的可写层。

这里的重点是，这个可写层是*临时的*。它与单个容器实例的生命周期关联紧密。当容器停止并移除时（这经常发生，例如更新应用程序或仅仅清理资源时），这个可写层及其包含的所有数据都会被永久删除。想象一下，下载了一个大型数据集或完成了数小时的模型训练，结果却在容器移除时所有成果都消失了。这显然不适用于大多数机器学习任务。

我们需要独立于容器生命周期持久化存储数据的方式。我们需要以下机制：

1. **数据持久化：** 确保数据集、日志、模型检查点和最终模型在容器重启或移除后仍然存在。
2. **数据共享：** 让宿主机的数据在容器*内部*可用（例如开发阶段的源代码或本地数据集）。
3. **数据解耦：** 将应用程序逻辑（在容器镜像中）与它操作的数据分离。

Docker 提供了两种主要机制来实现这些目标，它们是容器化机器学习应用中数据管理的主要方式：

- **绑定挂载：** 这些机制将宿主机文件系统中的文件或目录直接映射到容器中。在容器中进行的更改会反映到宿主机上，反之亦然。
- **卷：** 这些是由 Docker 自身管理的存储区域。它们存储在宿主机文件系统中的特定部分，由 Docker 管理，但其确切位置通常无需直接交互。卷是持久化容器生成数据的推荐方式。

下图展示了容器的临时层与这些持久化存储选项之间的关系。

> 容器可写层中的数据在容器移除时会丢失。绑定挂载直接链接到宿主机文件系统，而卷提供Docker管理的持久化存储。

理解容器内部临时存储与外部挂载持久化存储之间的这种区别，是有效管理容器化机器学习项目中的数据集、模型及其他文件的第一步。在后续章节中，我们将详细研究绑定挂载和卷，讨论它们在不同机器学习场景中的用例、优点和缺点。

## 参考资料

- [Manage data in Docker](https://docs.docker.com/storage/) — Docker Documentation (2024)
  官方文档，说明 Docker 的存储机制，包括容器可写层的特性、绑定挂载和卷在数据持久化方面的作用。
- [Building Machine Learning Powered Applications: Going from Idea to Product](https://www.barnesandnoble.com/w/building-machine-learning-powered-applications-emmanuel-ameisen/1136458284) — Emmanuel Ameisen (2020)
  Publisher: O'Reilly Media; Pages: 257
  讨论部署机器学习应用程序的实践方面，包括在容器化环境中管理数据和模型工件的策略。
