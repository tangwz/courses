---
course: "docker-for-ml-projects"
chapter: "docker-compose-ml-workflows"
lesson: "compose-building-images"
sourceId: 5346
sourceUrl: "https://apxml.com/zh/courses/docker-for-ml-projects/chapter-6-docker-compose-ml-workflows/compose-building-images"
title: "使用 Compose 构建镜像"
description: "指示 Compose 从 Dockerfile 构建镜像作为应用程序设置的一部分。"
order: 6
plots: []
sourceHash: "d6edf031def42f271194ee5afad6c50ddbc1d0499f8f9a0cf671b9d6c17c9696"
sourceCorrections: []
---

Docker Compose 擅长通过 `image` 指令定义和运行使用预构建镜像的多容器应用程序，它还提供了一种直接方法，将容器镜像的构建作为应用程序设置的一部分。这种集成简化了开发流程，尤其当您的应用程序组件需要 Dockerfile 定义的自定义环境时。您无需手动使用 `docker build` 构建每个镜像并在 `docker-compose.yml` 中引用，而是可以直接让 Compose 处理构建过程。

实现此功能的主要方法是在 `docker-compose.yml` 文件中的服务定义中使用 `build` 关键字。它替代或补充了 `image` 关键字。

### 使用 `build` 关键字

最简单来说，您可以为 `build` 关键字提供一个字符串值，表示包含 `Dockerfile` 的目录路径。此目录被称为 *构建上下文 (context)*。Docker Compose 会在该目录中查找名为 `Dockerfile` 的文件，并使用它来为服务构建镜像。

考虑一个为模型推理 (inference)而设计的自定义 Flask API 服务：

```yaml
# docker-compose.yml
version: '3.8' # 或更高版本

services:
  inference_api:
    build: ./inference_service # 包含 Dockerfile 文件的目录路径
    ports:
      - "5001:5000" # 将主机端口 5001 映射到容器端口 5000
    volumes:
      - ./models:/app/models # 挂载本地模型目录
    environment:
      - MODEL_PATH=/app/models/my_model.pkl
```

在此示例中，当您运行 `docker-compose up` 时，Compose 会进入相对于 `docker-compose.yml` 文件的 `inference_service` 目录。然后它将执行一个类似于 `docker build ./inference_service` 的构建过程，并使用生成的镜像启动 `inference_api` 容器。

### 明确指定构建上下文 (context)和 Dockerfile

有时，您的 `Dockerfile` 可能有不同的名称，或者您可能希望更明确地指定构建上下文路径，也许是相对于不同的位置。对于这些情况，`build` 关键字接受一个包含 `context` 和 `dockerfile` 键的对象：

```yaml
# docker-compose.yml
version: '3.8'

services:
  data_processor:
    build:
      context: ./processing # 构建上下文目录
      dockerfile: Dockerfile.processor # 指定自定义的 Dockerfile 名称
    volumes:
      - ./data:/data
    command: python process.py /data/input /data/output
```

在这里，Compose 使用 `./processing` 目录作为构建上下文，但会专门查找该目录中名为 `Dockerfile.processor` 的文件，以便为 `data_processor` 服务构建镜像。

### 传递构建参数 (parameter)

Dockerfile 通常使用 `ARG` 指令，允许在构建过程中进行自定义（例如，指定版本或启用可选功能）。Docker Compose 允许您使用 `build` 对象中的 `args` 键为这些参数传递值。

想象一下您的训练环境的 `Dockerfile` 接受一个 Python 版本：

```dockerfile
# 用于训练的 Dockerfile
ARG PYTHON_VERSION=3.9
FROM python:${PYTHON_VERSION}-slim

# ... Dockerfile 的其余部分 ...
```

您可以在 `docker-compose.yml` 中指定 `PYTHON_VERSION`：

```yaml
# docker-compose.yml
version: '3.8'

services:
  training_env:
    build:
      context: ./training
      dockerfile: Dockerfile.train
      args:
        PYTHON_VERSION: 3.10 # 将此值传递给 Dockerfile ARG
    # ... 其他服务配置 ...
```

当 Compose 构建 `training_env` 服务时，它会将 `PYTHON_VERSION=3.10` 传递给 Docker 构建过程，覆盖 `Dockerfile` 中指定的默认值。

### 结合 `build` 和 `image`

您可以为服务同时指定 `build` 和 `image`。如果这样做，Compose 将使用 `build` 部分中的指令构建镜像，然后使用 `image` 关键字中提供的名称（以及可选的标签）为生成的镜像打上标签。

```yaml
# docker-compose.yml
version: '3.8'

services:
  inference_api:
    build: ./inference_service
    image: my_registry/my_ml_api:1.0 # 为构建的镜像打标签
    ports:
      - "5001:5000"
    # ...
```

如果您打算稍后将 Compose 构建的镜像推送到容器注册表，这会特别有用。运行 `docker-compose build` 或 `docker-compose up --build` 将创建镜像并将其标记 (token)为 `my_registry/my_ml_api:1.0`。然后您可以使用 `docker push my_registry/my_ml_api:1.0` 来分享它。

### 触发构建

Docker Compose 努力提高效率。它通常只在不存在具有预期特征的镜像时才构建，或者当构建上下文 (context)（指定目录中的文件）似乎已更改时。但是，依赖项内部的深层更改可能不会总是自动检测到。

要明确强制 Compose 为您的服务重建镜像，请在 `up` 命令中使用 `--build` 标志：

```bash
docker-compose up --build
```

或者，您可以在不启动容器的情况下构建镜像，方法是：

```bash
docker-compose build
```

将镜像构建直接集成到您的 `docker-compose.yml` 文件中，提供了一种简化的方法来管理自定义容器环境。它确保镜像构建过程与应用程序的服务定义紧密关联，提高了操作一致性，并简化了项目团队的设置。只需定义构建上下文和可选参数 (parameter)，其余的交给 Docker Compose 处理即可。

## 参考资料

- [Compose file \`build\` reference](https://docs.docker.com/compose/compose-file/build/) — Docker (2024)
  Publisher: Docker
  官方文档详细说明了Docker Compose中的`build`关键字，包括上下文、Dockerfile规范、构建参数以及与`image`指令的结合使用。此资源对于理解如何在Compose设置中直接构建自定义镜像至关重要。
- [Dockerfile reference](https://docs.docker.com/reference/dockerfile/) — Docker (2024)
  Publisher: Docker
  Dockerfile语法和指令的官方指南。尽管本节侧重于Compose，但`build`关键字依赖Dockerfile进行镜像创建，因此这是理解底层构建过程的基础参考文献。
- [Overview of Docker Compose](https://docs.docker.com/compose/overview/) — Docker (2024)
  Publisher: Docker
  提供了Docker Compose的高层次介绍，解释了其用于定义和运行多容器Docker应用程序的用途。它建立了对Compose服务的基础理解，而`build`功能则对此进行了增强。
- [Docker: Up & Running: Building, Scaling, and Structuring Applications with Containers](https://www.oreilly.com/library/view/docker-up-running/9781492062723/) — Karl Matthias, Sean P. Kane (2022)
  Publisher: O'Reilly Media; Pages: 105-132
  一本关于Docker基础知识的综合书籍，包括镜像构建和使用Docker Compose管理多容器应用程序。关于Compose的章节为定义服务和集成自定义构建提供了更广阔的背景。
