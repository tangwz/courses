# Docker Compose在训练环境中的应用简介

来源：[原文](https://apxml.com/zh/courses/docker-for-ml-projects/chapter-4-containerizing-ml-training/docker-compose-training)

[返回章节目录](README.md) · [返回课程目录](../README.md)

使用`docker run`运行单独的训练作业为可复现性提供了很大的好处。然而，许多机器学习 (machine learning)工作流程不仅仅涉及一个训练脚本容器。你可能需要一个数据库来获取训练数据或记录指标，一个消息队列来触发作业，或者一个单独的服务来管理实验配置。仅仅使用`docker run`命令来管理多个相互连接的容器的生命周期、网络和数据持久性，很快就会变得繁琐且容易出错。

这正是Docker Compose发挥作用的地方。Docker Compose是一个专门用于定义和运行多容器Docker应用的工具。你无需为每个容器发出复杂的`docker run`命令，而是将整个应用堆栈，包括其服务、网络和卷，在一个名为`docker-compose.yml`的声明式YAML文件中进行描述。

### 定义多容器训练环境

设想一个场景，你的训练脚本需要：

1. 从一个由简单Web服务提供的配置文件中读取超参数 (parameter) (hyperparameter)。
2. 从数据库中获取训练数据标识符。
3. 运行训练过程本身。
4. 将输出指标和模型工件写回到共享卷，或者将它们记录到数据库中。

尝试用单独的`docker run`命令来编排这一切，将需要手动设置网络进行通信，管理容器启动顺序，以及在容器间一致地处理卷挂载。Docker Compose极大地简化了这些操作。

使用Compose，你可以在`docker-compose.yml`文件中将每个组件（训练脚本容器、数据库容器、配置服务容器）定义为一个单独的*服务*。Compose处理底层的Docker操作，例如：

- **创建共享网络：** 在同一Compose文件中定义的服务会自动放置在共享网络上，使它们能够使用服务名称作为主机名，轻松发现彼此并进行通信。
- **管理卷：** 你可以在Compose文件中定义卷，并将它们附加到相关服务，以确保数据持久性和共享。
- **编排启动和停止：** 像`docker-compose up`这样的命令会按照正确的依赖顺序（如果已指定）启动所有定义的服务，而`docker-compose down`则会干净地停止并删除它们。

### 一个基本示例结构

考虑一个简化的训练设置，只包含训练容器和一个用于记录结果的PostgreSQL数据库。一个`docker-compose.yml`文件可能如下所示：

```yaml
version: '3.8' # 指定Compose文件版本

services:
  # 训练脚本的服务
  training:
    build: . # 指示Compose从当前目录的Dockerfile构建镜像
    volumes:
      - ./data:/app/data # 将本地数据目录挂载到容器中
      - model_output:/app/output # 挂载一个命名卷用于模型输出
    environment:
      - DB_HOST=db # 使用服务名“db”作为主机名
      - DB_NAME=training_logs
      - DB_USER=trainer
      - DB_PASSWORD=secretpass
    depends_on: # 确保数据库在训练服务之前启动
      - db
    # command: python train.py --config /app/config.yaml # 可选的命令覆盖

  # PostgreSQL数据库服务
  db:
    image: postgres:14-alpine # 使用预构建的PostgreSQL镜像
    volumes:
      - db_data:/var/lib/postgresql/data # 挂载一个命名卷用于持久化数据库数据
    environment:
      - POSTGRES_DB=training_logs
      - POSTGRES_USER=trainer
      - POSTGRES_PASSWORD=secretpass

# 定义命名卷用于持久化存储
volumes:
  db_data:
  model_output:
```

在这个示例中：

- 我们定义了两个`services`：`training`和`db`。
- `training`服务使用本地的`Dockerfile`构建其镜像。
- `db`服务使用公共的`postgres`镜像。
- `volumes`用于输入数据（`./data`绑定挂载）、输出模型（`model_output`命名卷）和持久化数据库存储（`db_data`命名卷）。
- `environment`变量配置了两个服务，特别是为`training`服务提供了数据库连接详情。`DB_HOST`设置为`db`，这是PostgreSQL容器的服务名称，Compose使其在内部网络上可解析。
- `depends_on`确保`db`服务在`training`服务尝试连接之前启动。

> 此图描绘了`docker-compose.yml`文件中定义的服务。`training`和`db`等服务通过Compose管理的共享网络进行通信。卷（`model_output`，`db_data`）提供持久化存储。图中还显示了一个可选的配置服务。

### 运行训练堆栈

定义好`docker-compose.yml`文件后，启动整个堆栈就非常简单，只需运行：

```bash
docker-compose up --build
```

`--build`标志告诉Compose在启动容器之前为`training`服务构建镜像。如果镜像已存在且`Dockerfile`或其上下文 (context)未更改，Compose将重用现有镜像，除非指定了`--build`。

要停止并移除Compose文件中定义的容器、网络以及可能的卷，请使用：

```bash
docker-compose down
```

使用Docker Compose显著简化了更成熟的机器学习 (machine learning)训练管道中常见的多容器设置的管理。它提供了一种标准化的方式来定义和共享开发或测试环境，这些环境能紧密反映生产部署的某些方面，从而提高可复现性。尽管像Kubernetes这样强大的编排系统常用于大规模分布式训练，但Docker Compose对于本地开发、测试以及管理中等复杂度的训练堆栈来说是一个非常有价值的工具。接下来的部分将详细介绍容器化训练脚本的具体技术，以及如何处理配置和GPU使用等相关方面。

## 参考资料

- [Docker Compose overview](https://docs.docker.com/compose/) — Docker (2024)
  Publisher: Docker
  Docker Compose的官方文档，涵盖其核心概念、YAML文件结构、服务、网络、卷以及常用命令。
- [Docker: Up & Running: Building Portable Linux Container Applications](https://www.oreilly.com/library/view/docker-up-running/9781098135899/) — Sean P. Kane, Karl Matthias (2023)
  Publisher: O'Reilly Media
  提供Docker的深入指南，包括详细解释如何使用Docker Compose定义和运行多容器应用。
- [Building Machine Learning Pipelines](https://www.oreilly.com/library/view/building-machine-learning/9781492053151/) — Hannes Hapke and Catherine Nelson (2020)
  Publisher: O'Reilly Media; Pages: 364
  探讨容器化作为可重现机器学习管道的基础组成部分，为在MLOps中使用Docker Compose等工具提供背景。

---

[上一节](05-%E8%AE%AD%E7%BB%83%E7%9A%84GPU%E5%8A%A0%E9%80%9F.md) · [下一节](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%B9%E5%99%A8%E5%8C%96%E5%B9%B6%E8%BF%90%E8%A1%8C%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC.md)
