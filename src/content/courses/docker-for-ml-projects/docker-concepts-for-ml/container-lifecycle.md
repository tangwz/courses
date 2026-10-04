---
course: "docker-for-ml-projects"
chapter: "docker-concepts-for-ml"
lesson: "container-lifecycle"
sourceId: 5233
sourceUrl: "https://apxml.com/zh/courses/docker-for-ml-projects/chapter-1-docker-concepts-for-ml/container-lifecycle"
title: "容器生命周期管理"
description: "学习用于运行、停止、启动和移除 Docker 容器的命令。"
order: 3
plots: []
sourceHash: "42372c3d31338ce76e84a88a4fd7d5e6a36b48a8d34f5b405ac0157de34f021c"
sourceCorrections: []
---

将 Docker 镜像作为容器运行是一项基本操作。容器是镜像的一个活跃、可运行的实例。管理这些容器的生命周期，包括运行、停止、查看和移除它们，是有效使用 Docker 的根本，特别是对于执行训练脚本或部署推理 (inference)服务等机器学习 (machine learning)任务而言。

### 创建和运行容器：`docker run`

创建和启动新容器的主要命令是 `docker run`。最简单的用法是提供镜像名称：

```bash
docker run <image_name>
```

然而，对于实际的机器学习 (machine learning)流程，你通常会使用一些选项（标志）来控制容器的行为：

- **在后台运行（`-d` 或 `--detach`）：** 大多数机器学习训练任务或推理 (inference)服务器需要独立于你的终端会话运行。`-d` 标志以分离模式启动容器，打印容器 ID 并将控制权返回给你的终端。

  ```bash
  # 示例：在后台运行一个基于自定义 'my-ml-app' 镜像的容器
  docker run -d my-ml-app
  ```
- **交互式会话（`-it`）：** 有时你需要直接与容器的 shell 互动，例如调试脚本、查看数据或手动运行命令。结合 `-i`（交互式）和 `-t`（伪终端）可以做到这一点。你通常会指定一个 shell，比如 `/bin/bash` 作为要运行的命令。

  ```bash
  # 示例：在 Python 容器中启动交互式 Bash shell
  docker run -it python:3.9-slim /bin/bash
  ```

  在容器内部，你将看到一个提示符（例如 `root@<container_id>:/#`），你可以在其中执行命令。输入 `exit` 退出并停止容器。
- **自动移除（`--rm`）：** 对于运行一次训练脚本等短期任务，让容器在完成时自动移除非常方便。`--rm` 标志处理这种清理工作。

  ```bash
  # 示例：运行训练脚本并在之后移除容器
  docker run --rm my-training-image python train.py --epochs 10
  ```
- **分配名称（`--name`）：** Docker 为容器分配随机名称（如 `pensive_mcnulty`）。分配特定名称使容器更易于识别和管理。

  ```bash
  # 示例：运行一个带有特定名称的推理服务器容器
  docker run -d --name inference_api my-inference-image
  ```
- **端口映射（`-p`）：** 要从你的宿主机访问容器内部运行的应用程序（例如用于推理的 Web API），你需要将宿主机上的端口映射到容器内部的端口。（格式：`host_port:container_port`）

  ```bash
  # 示例：将宿主机上的 8080 端口映射到容器内的 80 端口
  docker run -d -p 8080:80 --name web_server nginx
  ```
- **卷和环境变量（`-v`，`-e`）：** 你经常需要将宿主机上的目录挂载到容器中（使用 `-v` 用于数据或代码），并传递配置参数 (parameter)（使用 `-e` 用于环境变量，如超参数 (hyperparameter)或 API 密钥）。这些将在第 2 章和第 3 章中更详细地介绍。

### 列出容器：`docker ps`

要查看哪些容器正在运行，请使用 `docker ps` 命令：

```bash
docker ps
```

这会输出一个表格，显示容器 ID、使用的镜像、正在运行的命令、创建时间、状态、暴露的端口和分配的名称。

要查看*所有*容器，包括已停止的容器（例如已完成的训练任务或崩溃的进程），请添加 `-a` 标志：

```bash
docker ps -a
```

这对于检查已完成任务的退出状态或查找需要移除的容器特别有用。

### 停止和启动容器：`docker stop` 和 `docker start`

你可以使用 `docker stop` 命令通过容器的名称或 ID 停止正在运行的容器。这会向容器内部的主进程发送一个信号（SIGTERM），允许其优雅地关闭（在默认超时时间内，之后它会被 SIGKILL 强制终止）。

```bash
# 停止名为 'inference_api' 的容器
docker stop inference_api
```

一个已停止的容器仍然存在，可以使用 `docker start` 重新启动，并保留其之前的状态：

```bash
# 重新启动 'inference_api' 容器
docker start inference_api
```

虽然 `docker start` 对于临时性训练任务（通常使用 `--rm` 运行）来说不太常用，但它对于重新启动开发服务器或手动管理的长时服务很有用。

### 移除容器：`docker rm`

已停止的容器会占用磁盘空间。当你使用完一个容器（例如，训练运行结束后或调试完成后），你应该使用 `docker rm` 加上容器名称或 ID 来移除它。除非使用强制标志 (`-f`)，否则你只能移除已停止的容器。

```bash
# 移除名为 'old_experiment' 的已停止容器
docker rm old_experiment

# 强制移除一个正在运行的容器（请谨慎使用）
# docker rm -f some_running_container
```

要一次性移除多个已停止的容器，你可以列出它们的 ID 或名称：

```bash
docker rm container_id1 container_name2
```

一个方便的命令，用于移除*所有*已停止的容器是：

```bash
docker container prune
```

Docker 在继续之前会要求确认。这对于日常清理很有帮助。

### 查看日志：`docker logs`

了解容器内部正在发生的事情，特别是以分离模式运行的容器，主要依赖于检查其日志（其主进程的标准输出和标准错误流）。`docker logs` 命令用于获取这些日志。

```bash
# 显示 'inference_api' 容器的日志
docker logs inference_api
```

这对于以下方面很重要：

- 监控机器学习 (machine learning)训练任务的进度（例如，迭代次数、损失值）。
- 调试容器内部运行脚本中的错误或崩溃。
- 查看推理 (inference)服务器的输出（例如，请求日志）。

要实时查看日志（类似于 Linux 上的 `tail -f`），请使用 `-f` 标志：

```bash
# 跟踪 'training_job_1' 容器的日志
docker logs -f training_job_1
```

按 `Ctrl+C` 停止跟踪。

### 容器生命周期图

下图说明了容器的典型状态以及用于在它们之间转换的命令：

> Docker 容器的基本状态以及触发它们之间转换的命令。

掌握这些生命周期命令（`run`、`ps`、`stop`、`start`、`logs`、`rm`）对于在机器学习 (machine learning)项目中有效使用 Docker 是必要的。它们使你能够执行任务、监控进度、管理资源并在容器化环境中进行调试。随着课程的进行，当你构建、运行和管理自己的机器学习容器时，这些命令将变得轻车熟路。

## 参考资料

- [docker run](https://docs.docker.com/reference/cli/docker/run/) — Docker Inc. (2024)
  官方文档，全面介绍了 `docker run` 命令及其用于容器创建和执行的各种选项。
- [docker container](https://docs.docker.com/reference/cli/docker/container/) — Docker Inc. (2024)
  Publisher: Docker Docs
  官方文档，涵盖了管理 Docker 容器的子命令，例如 `ls` (ps)、`stop`、`start`、`rm` 和 `logs`，这些对于容器生命周期管理至关重要。
- [Docker in Action, Third Edition](https://www.manning.com/books/docker-in-action-third-edition) — Jeff Nickoloff, Stephen Kuenzli (2019)
  Publisher: Manning Publications
  一本关于 Docker 的综合性书籍，提供了实用的指导，包括容器生命周期管理命令的详细解释和最佳实践。
