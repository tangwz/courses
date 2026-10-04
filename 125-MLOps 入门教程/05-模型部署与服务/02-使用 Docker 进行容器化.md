# 使用 Docker 进行容器化

来源：[原文](https://apxml.com/zh/courses/introduction-to-mlops/chapter-5-model-deployment-and-serving/containerization-with-docker)

[返回章节目录](README.md) · [返回课程目录](../README.md)

模型训练完成后，紧接着面临的挑战是如何让它在任何地方都能稳定运行，而不仅仅是在开发它的机器上。你可能遇到过这样的典型问题：代码在你的笔记本电脑上运行正常，但在同事的机器上却因为操作系统、库版本或 Python 安装环境的不同而运行失败。对于通常依赖于 `scikit-learn`、`pandas` 和 `numpy` 等特定库组合的机器学习 (machine learning)模型来说，这个问题表现得尤为突出。

容器化为这一挑战提供了一种有效的解决方式。它是将应用程序及其完整的运行时环境打包成一个单一、隔离且可移植的单元（称为容器）的过程。这个容器包含了模型文件、预测代码、所有必需的库以及系统依赖项。

可以将容器想象成标准的集装箱。箱子里装的是电子产品还是生鲜蔬菜并不重要，只要它装在标准的箱子里，任何配备合适设备的港口都能处理它。Docker 就是为软件提供这种“标准箱子”的技术。只要安装了 Docker 软件，Docker 容器就可以在任何机器上运行，而不受底层操作系统的影响。这从根本上解决了“在我的机器上能运行”的问题。

容器经常被拿来与虚拟机（VM）做比较，但它们本质上更加轻量化 (quantization)。虚拟机模拟一整台计算机，包括完整的客户操作系统，这会消耗大量资源。相比之下，容器共享宿主机的操作系统内核，只打包应用程序及其依赖项。这使得它们体积更小、启动更快且效率更高。

> 虚拟机与容器架构的比较。虚拟机包含完整的客户操作系统，因此比较笨重。容器共享宿主操作系统，因此轻量且高效。

### 为什么在机器学习 (machine learning)中使用 Docker？

对于 MLOps 而言，使用 Docker 进行容器化是一项基础实践，具有以下几个显著优势：

- **环境一致性：** 容器确保了运行模型的环境在开发、测试和生产环节中是完全相同的。这消除了由依赖项不匹配引起的错误。
- **可复现性：** 通过打包准确的模型版本及其所有依赖项，你可以完美地复现特定时间点的预测结果，这对于调试和合规性检查非常必要。
- **可移植性：** 容器化后的模型可以部署在个人笔记本电脑、私有数据中心或任何主流云服务商（AWS、Google Cloud、Azure）上，而无需对应用程序本身进行任何更改。
- **隔离性：** 每个容器都在自己独立的进程空间中运行。这可以防止模型的依赖项与同一服务器上运行的其他应用程序发生冲突。

### Docker 工作流：镜像与容器

使用 Docker 涉及几个核心组件，它们组合成一个简单直接的工作流。

- **Dockerfile：** 这是一个简单的文本文件，包含构建 Docker 镜像的一系列指令。它是容器环境的蓝图。你可以在这里指定所有内容：基础操作系统、安装库的命令、要复制的文件以及容器启动时运行的命令。
- **镜像 (Image)：** 镜像是一个只读模板，通过对 Dockerfile 执行 `docker build` 命令创建而成。它包含你的应用程序代码、依赖项和运行时环境。你可以将镜像存储在本地，也可以使用 Docker Hub 等注册表与他人共享。
- **容器 (Container)：** 容器是镜像的一个可运行实例。当你使用 `docker run` 命令运行镜像时，就创建了一个容器。它是一个正在运行的进程，执行你的应用程序。你可以从同一个镜像启动、停止并创建多个容器。

> 基础 Docker 工作流始于 Dockerfile，用于构建镜像。随后运行该镜像以创建一个活跃的容器。

### 用于机器学习 (machine learning)模型的 Dockerfile 示例

让我们看一个典型的 Dockerfile，用于打包一个简单的 scikit-learn 模型。假设你的项目目录包含三个文件：`model.pkl`（训练好的模型）、`app.py`（预测脚本）和 `requirements.txt`（Python 库列表）。

你的 `requirements.txt` 文件可能如下所示：

```
scikit-learn==1.1.0
pandas==1.4.2
flask==2.2.0
```

以下是打包此应用程序的简单 `Dockerfile`：

```dockerfile
# 1. 从官方 Python 基础镜像开始
FROM python:3.9-slim

# 2. 设置容器内的执行目录
WORKDIR /app

# 3. 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 复制其余的应用程序文件
COPY . .

# 5. 定义运行应用程序的命令
CMD ["python", "app.py"]
```

让我们逐行解析：

1. `FROM python:3.9-slim`：每个 Dockerfile 都以基础镜像开始。这里我们使用预装了 Python 3.9 的官方 Python 镜像。`-slim` 标签表示这是一个精简版，有助于减小最终镜像的大小。
2. `WORKDIR /app`：这为后续命令在容器内设置了名为 `/app` 的工作目录。为应用程序创建一个专门的文件夹是一个良好的习惯。
3. `COPY requirements.txt .` 和 `RUN pip install ...`：我们先只复制 `requirements.txt` 文件，然后安装依赖项。Docker 采用分层构建。通过将依赖安装与复制应用程序代码分开，Docker 可以缓存已安装的库。如果你更改了应用代码但没改依赖，Docker 会重用现有层，使后续构建速度大幅提升。
4. `COPY . .`：此命令将本地项目目录（构建上下文 (context)）中的所有剩余文件复制到容器的工作目录（`/app`）中。这包括 `app.py` 和 `model.pkl`。
5. `CMD ["python", "app.py"]`：这指定了从该镜像启动容器时执行的默认命令。在本例中，它运行我们的 Python 预测脚本。

在项目目录中准备好这个 `Dockerfile` 后，你可以用一条命令构建镜像：

```bash
docker build -t sentiment-model:v1 .
```

`-t` 标志为镜像添加名称（`sentiment-model`）和版本号（`v1`），以便引用。末尾的 `.` 告诉 Docker 使用当前目录作为构建上下文。

构建完成后，你的模型、代码和所有依赖项都被打包进一个自包含、可移植的镜像中。这个镜像就是你要部署的资产。你已成功将模型容器化，为下一步通过网络提供预测服务做好了准备。

## 参考资料

- [Get started with Docker](https://docs.docker.com/get-started/) — Docker Documentation (2024)
  提供关于 Docker 核心概念、架构和工作流的全面介绍，与本节内容直接相关。
- [Docker: Up & Running](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFj-0bQPQB_Pf6ZKH-a40RRxCgl776SvnKW2U0G7_LdrGQ5Pd-36y4rpQJXxz4tHq9l1dvdBAGWqoigqkz-0smYBKhEy-Ihd5dZRj5JH8IIEyZESnethrJYuZ161FMFBiv1dfqWFIXpS2uOZ3FB4TdALsk_2bP3wmmScjm8QYeaySf3OsovwA==) — Sean P. Kane, Karl Matthias (2018)
  Publisher: O'Reilly Media, Inc.; Pages: 354
  提供 Docker 的实用指南，涵盖其架构、命令行界面以及如何有效地构建、运行和管理容器。
- [Practical MLOps: How to Get Models into Production](https://learning.oreilly.com/library/view/practical-mlops/9781098103006/) — Noah Gift, Alfredo Deza (2021)
  Publisher: O'Reilly Media
  讨论 Docker 容器化作为 MLOps 基础实践的作用，侧重于生产环境中的模型部署、可复现性和环境一致性。
- [Custom containers for predictions](https://cloud.google.com/vertex-ai/docs/predictions/using-custom-containers) — Google Cloud Documentation (2024)
  Publisher: Google Cloud
  说明 Google Cloud 等主要云平台如何使用 Docker 容器部署机器学习模型以提供预测服务，强化容器化在 MLOps 中的实际应用。

---

[上一节](01-%E4%BB%80%E4%B9%88%E6%98%AF%E6%A8%A1%E5%9E%8B%E9%83%A8%E7%BD%B2%EF%BC%9F.md) · [下一节](03-%E4%BD%BF%E7%94%A8%20Flask%20%E5%88%9B%E5%BB%BA%E6%A8%A1%E5%9E%8B%20API.md)
