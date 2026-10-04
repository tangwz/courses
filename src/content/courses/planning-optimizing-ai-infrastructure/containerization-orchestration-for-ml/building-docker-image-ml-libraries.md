---
course: "planning-optimizing-ai-infrastructure"
chapter: "containerization-orchestration-for-ml"
lesson: "building-docker-image-ml-libraries"
sourceId: 7010
sourceUrl: "https://apxml.com/zh/courses/planning-optimizing-ai-infrastructure/chapter-4-containerization-orchestration-for-ml/building-docker-image-ml-libraries"
title: "构建包含机器学习库的Docker镜像"
description: "编写Dockerfile以构建包含Python、CUDA以及TensorFlow或PyTorch等机器学习库的自定义镜像。"
order: 2
plots: []
sourceHash: "d52cbb934e43d158a2417d4c014b69bcb58c6dbd18e0c40c4bf136fd6f4cad3b"
sourceCorrections: []
---

Docker镜像定义了应用程序的蓝图，以创建稳定、可移动的环境。这个蓝图是一个简单的文本文件，叫做`Dockerfile`。可以把它想象成一个食谱。文件中的每一行都是一条指令，它告诉Docker如何一层一层地组装镜像，从一个基础操作系统开始，并在其之上添加库、代码和配置。

对于机器学习 (machine learning)应用，这种分层结构显得尤其重要。一个典型的机器学习镜像不只包含你的代码和Python。它还必须包含与宿主机GPU通信所需的特定且通常很大的CUDA工具包和cuDNN库。

> 典型的机器学习Docker镜像的分层结构。每一层都在前一层的基础上构建，并由Docker缓存，当只有顶层变化时，可以实现更快的重新构建。

### Dockerfile基本指令

`Dockerfile`可以包含许多指令，但有几条核心指令承担了大部分工作。我们来审视那些与构建机器学习 (machine learning)环境最相关的指令。

#### FROM: 选择你的基础镜像

每个`Dockerfile`都必须以`FROM`指令开头。它指定了你所构建的父镜像。基础镜像的选择对机器学习应用来说是一个重要的决定。

- **标准Python镜像：** 你可以从官方Python镜像开始，例如`python:3.9-slim`。它很轻量，非常适合仅使用CPU的应用。然而，它不包含NVIDIA库，无法用于GPU加速。
- **NVIDIA CUDA镜像：** 对于GPU加速的任务，建议使用NVIDIA的官方基础镜像。像`nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04`这样的镜像预装了特定版本的CUDA工具包和cuDNN库。这能让你省去自行安装NVIDIA驱动和工具包的复杂且易错的过程。`runtime`标签表明它包含运行预编译CUDA应用所需的库，而`devel`标签则包含用于编译CUDA代码的完整SDK。对于大多数机器学习训练，`runtime`镜像就足够了。

> **关于可重复性**
> 始终为你的基础镜像使用特定的版本标签（例如`11.8.0-cudnn8-runtime-ubuntu22.04`），而不是像`latest`这样的通用标签。`latest`标签随时可能被发布者更新，这可能在未来破坏你的构建。固定到特定版本可确保你的环境是可重复的。

#### WORKDIR 和 COPY: 将你的代码放入镜像

`WORKDIR`指令设置了后续`RUN`、`CMD`、`ENTRYPOINT`、`COPY`和`ADD`指令的工作目录。尽早设置`WORKDIR`是一个好做法，以保持容器文件系统的整洁。

`COPY`指令将文件和目录从你的本地机器复制到容器的文件系统。一个常见的方法是首先`COPY`只列出依赖项的文件，安装它们，然后再`COPY`其余的应用程序代码。这使用了Docker的层缓存。如果你的应用程序代码发生变化但依赖项没有变化，Docker可以重用安装库的缓存层，从而使后续构建速度快很多。

```dockerfile
# 设置工作目录
WORKDIR /app

# 首先只复制requirements文件
COPY requirements.txt .
```

#### RUN: 安装依赖

`RUN`指令在当前镜像之上执行命令，并提交结果。这是你安装Python包的方式。你可以使用`&&`和`\`将命令链接在一起，以实现行连续。这会在单个`RUN`指令内执行所有命令，只创建一个新层，并使你的最终镜像大小更小。

```dockerfile
# 在一个层中安装系统依赖项和Python包
RUN apt-get update && \
    apt-get install -y python3-pip && \
    pip3 install --no-cache-dir -r requirements.txt
```

在`pip`中使用`--no-cache-dir`标志可以防止它存储包缓存，这在最终镜像中是不必要的，并有助于减小其大小。

#### CMD: 定义默认命令

`CMD`指令提供从你的镜像启动容器时要执行的默认命令。一个`Dockerfile`中只能有一条`CMD`指令。如果你想运行一个名为`train.py`的训练脚本，你的`CMD`会像这样：

```dockerfile
CMD ["python3", "train.py", "--epochs", "10"]
```

这是`CMD`的“exec形式”，是首选的语法。它不会调用命令行shell，并避免了信号处理的潜在问题。

### 实际示例：PyTorch训练应用的Docker化

我们把这些都结合起来。假设我们有一个简单的项目结构：

```
.
├── Dockerfile
├── requirements.txt
└── train.py
```

我们的`requirements.txt`文件列出了我们需要的库：

```
torch==1.13.1
torchvision==0.14.1
numpy==1.23.5
```

这是一个完整的`Dockerfile`，用于将此应用容器化以进行GPU训练：

```dockerfile
# 1. 使用特定的NVIDIA CUDA运行时镜像作为基础
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 2. 设置环境变量以防止安装期间出现交互式提示
ENV DEBIAN_FRONTEND=noninteractive

# 3. 设置容器内的工作目录
WORKDIR /app

# 4. 复制requirements文件以借助Docker的构建缓存
COPY requirements.txt .

# 5. 更新包列表，安装Python和Pip，然后安装Python包
# 链式命令减少镜像层数
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    pip3 install --no-cache-dir -r requirements.txt && \
    rm -rf /var/lib/apt/lists/*

# 6. 将其余应用程序代码复制到工作目录
COPY . .

# 7. 指定容器启动时运行的命令
CMD ["python3", "train.py"]
```

### 构建和运行镜像

有了`Dockerfile`，你就可以在终端中使用`docker build`命令构建镜像。`-t`标志会用一个易于识别的名称和版本来标记 (token)镜像。

```bash
# 从当前目录构建Docker镜像
docker build -t pytorch-training-app:1.0 .
```

构建完成后，你就拥有了一个自包含、可移动的镜像。要运行它并让它访问宿主机的GPU，你可以使用带有`--gpus all`标志的`docker run`命令。此标志由宿主机上的NVIDIA Container Toolkit处理，它会自动将必要的GPU驱动和库挂载到容器中。

```bash
# 运行容器，并授予它访问所有可用宿主机GPU的权限
docker run --gpus all pytorch-training-app:1.0
```

你现在已经成功地将一个机器学习 (machine learning)应用及其特定的CUDA和Python依赖项打包成一个Docker镜像，可以随时在任何安装了Docker和NVIDIA GPU的机器上运行。这为构建我们接下来将要讨论的可扩展且可重复的系统奠定了基础。

## 参考资料

- [Dockerfile reference](https://docs.docker.com/reference/dockerfile/) — Docker Documentation (2024)
  Dockerfile指令、语法和镜像创建原则的官方指南。
- [Best practices for writing Dockerfiles](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/) — Docker Documentation (2024)
  提供创建高效、可维护和安全Docker镜像的指导。
- [NVIDIA Container Toolkit Documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html) — NVIDIA (2024)
  Publisher: NVIDIA
  在NVIDIA硬件上使用Docker运行GPU加速容器的官方文档。
- [NVIDIA CUDA container images](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda) — NVIDIA (2024)
  Publisher: NVIDIA
  NVIDIA提供的官方CUDA容器镜像，针对GPU计算进行了优化。
