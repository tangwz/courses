# Docker 注册中心与仓库

来源：[原文](https://apxml.com/zh/courses/docker-for-ml-projects/chapter-1-docker-concepts-for-ml/docker-registries)

[返回章节目录](README.md) · [返回课程目录](../README.md)

在本地构建 Docker 镜像通常是第一步；将该镜像发布到其他地方或获取他人已构建好的镜像，是一项基本能力。Docker 注册中心和仓库能够满足这一需求。注册中心是 Docker 镜像的集中存储和分发系统，很像 PyPI 为 Python 包提供服务，或者 GitHub 托管代码仓库。

## 什么是注册中心和仓库？

A **Docker 注册中心** 是一种存储和分发 Docker 镜像的服务。它可以是公共的，也可以是私有的。最著名的公共注册中心是 Docker Hub，它是 Docker 命令行工具的默认注册中心。

在一个注册中心内，镜像被组织成**仓库**。一个仓库包含特定镜像的不同版本（标签）。例如，Docker Hub 托管着官方 Python 镜像（`python`）、流行数据库如 PostgreSQL（`postgres`）的仓库，以及对机器学习 (machine learning)很重要的 TensorFlow（`tensorflow/tensorflow`）和 PyTorch（`pytorch/pytorch`）等核心框架的仓库。

Docker 镜像的标准命名约定遵循此模式：

`[registry_host/][username_or_organization/]repository_name[:tag]`

- `registry_host`: 指定注册中心（如果不是 Docker Hub），例如 Google Container Registry 的 `gcr.io`，NVIDIA GPU Cloud 的 `nvcr.io`。如果省略，Docker 会认为是 Docker Hub。
- `username_or_organization`: 标识注册中心上仓库的所有者（例如 `tensorflow/tensorflow` 中的 `tensorflow`）。对于 Docker Hub 上的官方镜像（如 `python` 或 `ubuntu`），这部分通常省略。
- `repository_name`: 镜像集的实际名称（例如 `tensorflow`, `python`）。
- `tag`: 特定的版本或变体标识符（例如 `latest`, `3.9-slim`, `2.11.0-gpu`）。如果省略，Docker 默认为 `latest` 标签。

例如：

- `python:3.9`: 官方 Python 镜像，版本 3.9 标签，来自 Docker Hub。
- `tensorflow/tensorflow:2.11.0-gpu`: TensorFlow 镜像，版本 2.11.0，带 GPU 支持，来自 Docker Hub 上的 `tensorflow` 组织。
- `nvcr.io/nvidia/pytorch:23.10-py3`: PyTorch 镜像，标签 `23.10-py3`，来自 NVIDIA GPU Cloud 注册中心（`nvcr.io`）上的 `nvidia` 组织。

## Docker Hub：公共注册中心

Docker Hub 是默认且最大的公共注册中心。它包含大量镜像，包括：

- **官方基础镜像：** 由 Docker 或上游项目维护的镜像，如 `ubuntu`、`debian`、`alpine` 和 `python`。它们常作为定制 ML 环境的起点。
- **框架镜像：** 由框架开发者维护的仓库，例如 `tensorflow/tensorflow`、`pytorch/pytorch`，提供带有特定框架版本和有时带有 GPU 支持的即用型环境。
- **社区镜像：** 由个人用户或组织共享的镜像。

使用 Docker Hub 上的镜像对机器学习 (machine learning)实践者而言十分重要。您不必从头开始构建所有内容，可以借助已包含 Python、特定机器学习库甚至 CUDA 驱动以实现 GPU 加速的基础镜像。

## 标签：标识镜像版本

标签对于可复现性不可或缺。`latest` 标签很方便，但通常指向最新推送的版本，这可能意外地发生变化。在机器学习 (machine learning)中，一致性非常重要。依赖 `latest` 可能导致构建失败或模型行为不同，如果底层镜像发生变化。

最佳实践建议使用表示版本或特定配置的明确标签：

- `python:3.9.18-slim-bookworm`: 指定 Python 版本 3.9.18，`slim` 变体（更小尺寸），基于 Debian `bookworm`。
- `tensorflow/tensorflow:2.12.0-gpu`: 指定 TensorFlow 版本 2.12.0，带 GPU 支持。
- `pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime`: 指定 PyTorch 2.0.1，为 CUDA 11.7 和 cuDNN 8 构建，仅包含运行时组件。

总是在您的 Dockerfile（`FROM python:3.9-slim`）和运行容器时使用明确的标签，以确保您使用的是预期的准确环境。

## 拉取镜像：`docker pull` 命令

要将镜像从注册中心下载到您的本地机器，请使用 `docker pull` 命令。Docker 会自动先检查您的本地存储；如果镜像（带有特定标签）不存在，它会联系相应的注册中心（默认为 Docker Hub）并下载它。

```bash
# 从 Docker Hub 拉取官方 Python 3.9 slim 镜像
docker pull python:3.9-slim

# 从 Docker Hub 拉取特定的 TensorFlow GPU 镜像
docker pull tensorflow/tensorflow:2.11.0-gpu

# 从不同的注册中心（NVIDIA）拉取镜像
docker pull nvcr.io/nvidia/cuda:12.1.1-base-ubuntu22.04
```

如果您使用本地不存在的镜像名称运行 `docker run`，Docker 会先隐式地执行 `docker pull`。

## 推送镜像：`docker push` 命令

在为您的机器学习 (machine learning)项目构建了定制镜像（例如，包含您的特定依赖项甚至训练好的模型）之后，您可能希望将其分享给同事或部署到服务器。这通过使用 `docker push` 命令完成。

在推送之前，您需要：

1. **正确标记 (token)您的镜像：** 镜像名称必须包含目标注册中心（如果不是 Docker Hub）以及您在该注册中心上的用户名或组织名称。使用 `docker tag` 来重命名或为现有本地镜像创建别名。

   ```bash
   # 假设您在本地构建了一个名为 'my-sklearn-app:1.0' 的镜像
   # 为推送到 Docker Hub 的用户名 'myusername' 下进行标记
   docker tag my-sklearn-app:1.0 myusername/my-sklearn-app:1.0
   ```
2. **向注册中心进行身份验证：** 使用 `docker login` 命令。对于 Docker Hub，它会提示您输入用户名和密码（或访问令牌）。

   ```bash
   docker login
   # 出现提示时输入用户名和密码/令牌
   ```

   对于其他注册中心，提供注册中心主机：`docker login myregistry.example.com`。

标记并验证身份后，您就可以推送镜像了：

```bash
# 将已标记的镜像推送到 Docker Hub
docker push myusername/my-sklearn-app:1.0
```

镜像层将被上传到注册中心，使其可供他人（或您的部署系统）拉取。

> 使用注册中心共享定制 Docker 镜像的流程。

## 私有注册中心

尽管 Docker Hub 非常适合公共和开源项目，但您通常会需要一个**私有注册中心**来处理专有代码、敏感数据处理配置或公司内部使用。主要的云服务提供商提供托管注册中心服务：

- Amazon Elastic Container Registry (ECR)
- Google Artifact Registry（或 Container Registry）
- Azure Container Registry (ACR)

您还可以使用 Harbor 或官方 Docker Registry 镜像等软件来托管自己的注册中心。访问控制和安全性是在商业或研究环境中使用私有注册中心的主要原因。身份验证（`docker login`）的工作方式类似，但您将使用特定于私有注册中心提供商的凭据。

了解注册中心和仓库对在机器学习 (machine learning)中有效使用 Docker 来说非常重要。它们有助于共享标准化环境、获取预构建工具，并构成基于容器的部署策略的主干，确保在本地构建的环境与在其他地方运行的环境一致。

## 参考资料

- [About registries and Docker Hub](https://docs.docker.com/docker-hub/) — Docker Documentation (2024)
  说明了Docker注册表、仓库和标签的基本概念，并以Docker Hub作为主要示例。
- [docker pull](https://docs.docker.com/reference/cli/docker/image/pull/) — Docker Documentation (2024)
  提供了`docker pull`命令的详细文档，对于从注册表下载Docker镜像至关重要。
- [docker push](https://docs.docker.com/reference/cli/docker/image/push/) — Docker Documentation (2024)
  提供了`docker push`命令的详细文档，用于将Docker镜像上传到注册表。
- [Docker in Action](https://www.manning.com/books/docker-in-action-second-edition) — Jeff Nickoloff and Stephen Kuenzli (2019)
  Publisher: Manning Publications
  一本内容广泛的书籍，涵盖了Docker概念，包括镜像管理、注册表和容器的实际应用。

---

[上一节](04-Dockerfile%20%E7%AE%80%E4%BB%8B.md) · [下一节](06-%E5%AE%9E%E8%B7%B5%E6%93%8D%E4%BD%9C%EF%BC%9A%E8%BF%90%E8%A1%8C%E9%A2%84%E6%9E%84%E5%BB%BA%E7%9A%84%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E9%95%9C%E5%83%8F.md)
