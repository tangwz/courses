---
course: "docker-for-ml-projects"
chapter: "building-ml-dockerfiles"
lesson: "workdir-entrypoint-cmd"
sourceId: 5265
sourceUrl: "https://apxml.com/zh/courses/docker-for-ml-projects/chapter-2-building-ml-dockerfiles/workdir-entrypoint-cmd"
title: "设置工作目录和入口点"
description: "定义机器学习容器的执行环境和默认命令。"
order: 7
plots: []
sourceHash: "1b45d42becf77b2828f1820011aebdd15774d3a7c3b085cb3a8cf6a29e34e895"
sourceCorrections: []
---

在为机器学习 (machine learning)项目创建 Docker 镜像时，在完成基础镜像的建立、依赖的安装以及使用 `COPY` 或 `ADD` 等指令将项目文件复制到镜像中等任务之后，有必要定义容器启动时的行为方式。具体来说，您需要告知 Docker 命令应在容器内部何处执行，以及默认应运行哪个命令。这通过 Dockerfile 中的 `WORKDIR`、`ENTRYPOINT` 和 `CMD` 指令来实现。它们对于让您的容器化机器学习应用可运行且易于使用非常重要。

### 使用 WORKDIR 设置执行环境

`WORKDIR` 指令为 Dockerfile 中所有后续的 `RUN`、`CMD`、`ENTRYPOINT`、`COPY` 和 `ADD` 指令设置工作目录。可以将其理解为在构建和运行镜像的过程中更改当前目录 (`cd`)。

如果未指定 `WORKDIR`，Docker 会在根目录 (`/`) 中执行后续命令。明确设置 `WORKDIR` 可以使您的 Dockerfile 指令更清晰，并减少对绝对路径的依赖。

**优点：**

1. **可读性：** 避免在 Dockerfile 中分散使用绝对路径。
2. **可移植性：** 便于以后在镜像中重组项目结构。
3. **一致性：** 确保运行 Python 脚本等命令相对于已知项目目录执行。

**示例：**

假设您的宿主机上有一个机器学习 (machine learning)项目，其结构如下：

```
my_ml_project/
├── Dockerfile
├── requirements.txt
├── src/
│   ├── train.py
│   └── predict.py
└── data/
    └── dataset.csv
```

与编写 `COPY src /app/src` 然后 `RUN python /app/src/train.py` 相比，您可以使用 `WORKDIR`：

```dockerfile
# Dockerfile 片段
# ... (FROM，安装依赖)

# 设置工作目录
WORKDIR /app

# 相对于 WORKDIR 复制应用程序代码
COPY src/ ./src/
COPY requirements.txt .

# 安装依赖 (pip install -r requirements.txt)

# 复制数据（可能通过卷处理更好，参阅第 3 章）
COPY data/ ./data/

# 后续命令如 CMD 或 ENTRYPOINT 将从 /app 运行
CMD ["python", "src/train.py"]
```

在此示例中，`WORKDIR /app` 会创建 `/app` 目录（如果不存在），并将其设置为当前目录。`COPY` 指令现在使用相对目标路径（`./src/`、`.`、`./data/`），将文件复制*到* `/app` 中。最终的 `CMD` 在 `/app` 作为当前目录的假定下执行 `python src/train.py`。

### 使用 ENTRYPOINT 定义容器的可执行程序

`ENTRYPOINT` 指令将容器配置为可执行程序运行。它指定了容器启动时*始终*会执行的命令。提供给 `docker run <image> [arguments]` 的任何命令行参数 (parameter)都会附加到 `ENTRYPOINT` 命令之后。

`ENTRYPOINT` 常用于您希望容器的主要目的是运行特定应用程序或脚本的场景。

有两种形式：

1. **Exec 形式（推荐）：** `ENTRYPOINT ["executable", "param1", "param2"]`
   - 这是推荐的形式。它直接执行命令，不调用命令 shell。通过 `docker run` 传递的参数会直接附加在指定参数之后。
2. **Shell 形式：** `ENTRYPOINT command param1 param2`
   - 此形式在 shell (`/bin/sh -c`) 中运行命令。它对于环境变量替换很有用，但在信号处理或命令包含 shell 元字符时，可能会出现意外行为。

**示例 (Exec 形式)：**

假设您希望容器始终执行 Python 脚本。

```dockerfile
# Dockerfile 片段
# ... (FROM、WORKDIR、COPY 等)

ENTRYPOINT ["python"]

# 如果未向 docker run 提供参数，则运行此默认脚本
CMD ["src/train.py"]
```

当您运行此容器时：

- `docker run my-ml-image`：执行 `python src/train.py`（`CMD` 为 `ENTRYPOINT` 提供默认参数）。
- `docker run my-ml-image src/predict.py --model /app/models/model.pkl`：执行 `python src/predict.py --model /app/models/model.pkl`（提供的参数会覆盖 `CMD` 并附加到 `ENTRYPOINT` 之后）。

`ENTRYPOINT` 确保 `python` 始终是主命令，使容器表现得像一个专用的 Python 脚本运行器。

### 使用 CMD 提供默认命令或参数 (parameter)

`CMD` 指令为运行中的容器提供默认值。这些默认值可以包含可执行程序，但它们通常用于为 `ENTRYPOINT` 指定默认参数。如果单独使用，`CMD` 则指定要运行的默认命令。

值得注意的是，如果用户向 `docker run <image> [arguments]` 提供了参数，则整个 `CMD` 指令都会被覆盖。

`CMD` 也有三种形式：

1. **Exec 形式（ENTRYPOINT 推荐）：** `CMD ["param1","param2"]`（用作 `ENTRYPOINT` 的默认参数）
2. **Exec 形式（独立使用）：** `CMD ["executable","param1","param2"]`（如果没有 `ENTRYPOINT`，则作为默认命令）
3. **Shell 形式：** `CMD command param1 param2`（通过 `/bin/sh -c` 运行的默认命令）

**用例：**

1. **`ENTRYPOINT` 的默认参数（最常见的组合使用方式）：**

   ```dockerfile
   WORKDIR /app
   COPY src/ ./src/
   ENTRYPOINT ["python", "src/run_experiment.py"]
   # 如果未通过 'docker run' 提供参数，则使用默认参数
   CMD ["--data", "data/dataset.csv", "--epochs", "10"]
   ```

   - 运行 `docker run my-ml-image` 会执行：`python src/run_experiment.py --data data/dataset.csv --epochs 10`
   - 运行 `docker run my-ml-image --data data/other.csv --epochs 20` 会执行：`python src/run_experiment.py --data data/other.csv --epochs 20`（`CMD` 被完全替换）。
2. **默认命令（无 `ENTRYPOINT`）：**

   ```dockerfile
   WORKDIR /app
   COPY src/ ./src/
   # 未定义 ENTRYPOINT
   CMD ["python", "src/train.py"]
   ```

   - 运行 `docker run my-ml-image` 会执行：`python src/train.py`
   - 运行 `docker run my-ml-image python src/predict.py` 会执行：`python src/predict.py`（整个默认 `CMD` 被提供的命令替换）。

### 有效地结合使用 ENTRYPOINT 和 CMD

最常用且功能强大的模式是使用 `ENTRYPOINT` 指定固定的可执行程序（如 `python` 或包装脚本），并使用 `CMD` 指定默认的、可覆盖的参数 (parameter)（如脚本名称或默认超参数 (hyperparameter)）。

**最佳实践：**

- **使用 `WORKDIR`：** 始终设置 `WORKDIR` 以在容器内建立明确的执行环境。`/app` 是一个常见的约定。
- **偏好 Exec 形式：** 对 `ENTRYPOINT` 和 `CMD` 都使用 JSON 数组语法（`["executable", "param1"]`），以避免 shell 处理的怪异行为，并使参数处理更明确。
- **`ENTRYPOINT` 用于可执行程序：** 如果您的容器旨在运行特定命令或表现得像一个可执行程序，请使用 `ENTRYPOINT`。示例：`ENTRYPOINT ["python"]`，`ENTRYPOINT ["./my_inference_server"]`。
- **`CMD` 用于默认值：** 使用 `CMD` 为 `ENTRYPOINT` 提供默认参数，或指定一个易于覆盖的默认命令。示例：`CMD ["my_script.py", "--default-flag"]`。

通过仔细定义 `WORKDIR`、`ENTRYPOINT` 和 `CMD`，您可以创建不仅可重现而且行为可预测、用户友好的 Docker 镜像，无论是运行训练任务还是部署模型。它们定义了与容器化机器学习 (machine learning)应用程序交互的界面。

## 参考资料

- [Dockerfile reference](https://docs.docker.com/reference/dockerfile/) — Docker, Inc. (2024)
  Publisher: Docker, Inc.
  官方文档，详细介绍了所有 Dockerfile 指令，包括 WORKDIR、ENTRYPOINT 和 CMD 的语法和用法示例。
- [Docker: Up & Running: Shipping Reliable Containers in Production](https://www.oreilly.com/library/view/docker-up-running/9781492087529/) — Sean P. Kane, Karl Matthias (2018)
  Publisher: O'Reilly Media, Inc.
  一份实用指南，解释了 Docker 概念、Dockerfile 和生产环境中的容器化，有助于理解指令的上下文。
- [Best practices for writing Dockerfiles](https://docs.docker.com/build/building/best-practices/) — Docker, Inc. (2024)
  Publisher: Docker, Inc.
  提供关于 Dockerfile 结构、提高效率以及有效管理 WORKDIR、ENTRYPOINT 和 CMD 等指令的官方建议和模式。
