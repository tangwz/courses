---
course: "basics-ml-deployment"
chapter: "intro-containerization-docker"
lesson: "writing-simple-dockerfile"
sourceId: 4015
sourceUrl: "https://apxml.com/zh/courses/basics-ml-deployment/chapter-4-intro-containerization-docker/writing-simple-dockerfile"
title: "编写一个简单的Dockerfile"
description: "了解Dockerfile中的基本结构和常用指令，以定义镜像。"
order: 5
plots: []
sourceHash: "caeff449fff9c6360f29416f99f2b82c220ef4430d0b6a18db8d08ca49f774c1"
sourceCorrections: []
---

使用Docker进行容器化的目的是将应用及其环境打包在一起。要实现这一点，我们如何告诉Docker要打包*什么*呢？这时就需要`Dockerfile`了。

可以将`Dockerfile`看作是一份食谱或一系列指令。它是一个简单的文本文件，通常命名为`Dockerfile`（首字母大写'D'，无文件扩展名），其中包含一系列命令，Docker会使用这些命令来构建一个特定的镜像。文件中的每条指令都会在镜像中创建一个新层，在前一层的基础上构建。

让我们来看一些你经常会用到的基本指令，尤其是在打包像我们Flask预测服务这样的Python Web应用时。

### 常见Dockerfile指令

- **`FROM`**: 每个`Dockerfile`都必须以`FROM`指令开头。它指定了你正在构建所依据的**基础镜像**。基础镜像提供了一个起始的操作系统和环境。对于Python应用，你通常会从官方Python基础镜像开始，其中已包含一个Linux发行版和特定版本的Python。

  - 示例：`FROM python:3.9-slim` 指定使用官方镜像的轻量级（“slim”）版本，该镜像包含Python 3.9。
- **`WORKDIR`**: 这条指令设置了`Dockerfile`中任何后续`RUN`、`CMD`、`ENTRYPOINT`、`COPY`和`ADD`指令的工作目录。如果目录不存在，Docker会创建它。这有助于整理容器文件系统中的文件。

  - 示例：`WORKDIR /app` 将容器内的当前目录设置为`/app`。
- **`COPY`**: 这条指令将文件或目录从你的本地机器（构建上下文 (context)）复制到容器镜像的文件系统中。你需要用它将应用代码、模型文件和依赖列表放入镜像中。

  - 示例：`COPY requirements.txt .` 将`requirements.txt`文件从项目目录复制到镜像内的当前工作目录（如果通过`WORKDIR`设置为`/app`）。`.`表示当前的`WORKDIR`。
  - 示例：`COPY . .` 将项目目录（构建上下文）中的*所有内容*复制到镜像内的当前工作目录。请注意你复制了什么；通常，最好复制特定文件或使用`.dockerignore`文件（类似于`.gitignore`）来排除不必要的文件，例如虚拟环境或本地配置。
- **`RUN`**: 这条指令在镜像**构建**过程中执行命令。它常用于安装软件包，包括Python库。每条`RUN`指令都会创建一个新的镜像层。

  - 示例：`RUN pip install --no-cache-dir -r requirements.txt` 使用`pip`（由于我们从Python镜像开始，所以可用）来安装`requirements.txt`中列出的包。`--no-cache-dir`标志在Dockerfile中常被建议使用，因为它能阻止pip存储下载缓存，从而减小镜像大小。
- **`EXPOSE`**: 这条指令告知Docker容器在运行时将监听指定的网络端口。它主要用作构建镜像者和运行容器者之间的文档。它并*没有*实际发布端口或使其可从宿主机访问；这需要你在运行容器时使用`docker run`命令的`-p`或`-P`标志来完成。

  - 示例：`EXPOSE 5000` 表示容器内的应用预期将在端口5000上监听（Flask开发服务器的默认端口）。
- **`CMD`**: 这条指令指定了从镜像**启动**容器时要执行的默认命令。一个`Dockerfile`中只能有一条`CMD`指令。如果你列出多条`CMD`，只有最后一条会生效。通过`CMD`指定的命令可以在启动容器时被覆盖。

  - 示例：`CMD ["python", "app.py"]` 将默认命令设置为使用Python解释器运行Flask应用脚本`app.py`。这是`CMD`首选的“exec”形式。

### 准备依赖项：`requirements.txt`

在编写`Dockerfile`之前，列出应用所有依赖的Python库是必要的。标准做法是在项目根目录创建一个名为`requirements.txt`的文件。对于我们的Flask预测服务，这个文件可能看起来像这样：

```text
# requirements.txt
flask
scikit-learn
joblib
numpy
# 添加模型或预处理可能需要的其他库
```

这个文件使得`Dockerfile`中的`RUN pip install -r requirements.txt`命令能够在容器镜像内安装所需的准确依赖项，确保一致性。

### Flask服务的`Dockerfile`示例

让我们把这些指令组合成一个简单的`Dockerfile`，它适用于我们在上一章开发的Flask预测服务。假设你的项目目录包含你的Flask应用脚本（例如`app.py`）、保存的模型文件（例如`model.joblib`）、任何已保存的预处理器以及`requirements.txt`文件。

```dockerfile
# Dockerfile

# 从一个轻量级Python基础镜像开始
FROM python:3.9-slim

# 设置容器内的工作目录
WORKDIR /app

# 先复制requirements文件，以便利用Docker缓存
COPY requirements.txt .

# 安装Python依赖项
# 使用--no-cache-dir可以减小镜像大小
RUN pip install --no-cache-dir -r requirements.txt

# 将其余的应用代码复制到工作目录
COPY . .

# 告知Docker容器在端口5000上监听
EXPOSE 5000

# 定义容器启动时运行应用的命令
CMD ["python", "app.py"]
```

**解释：**

1. `FROM python:3.9-slim`: 我们从Python 3.9环境开始。
2. `WORKDIR /app`: 我们将容器内的默认目录设置为`/app`。
3. `COPY requirements.txt .`: 我们首先*只*复制`requirements.txt`文件。Docker分层构建镜像并缓存它们。如果`requirements.txt`没有变化，Docker可以重用由下一条`RUN`指令创建的层，从而在仅应用代码变化时加快后续构建的速度。
4. `RUN pip install ...`: 我们安装`requirements.txt`中列出的依赖项。
5. `COPY . .`: 现在，我们将项目其余文件（例如`app.py`和`model.joblib`）复制到镜像内的`/app`目录中。
6. `EXPOSE 5000`: 我们记录了我们的Flask应用将在端口5000上运行。
7. `CMD ["python", "app.py"]`: 我们指定，当容器从该镜像启动时，运行`python app.py`应为默认操作。

这个简单的文本文件为Docker提供了所有必要步骤，以创建一个包含我们Python环境、依赖项和应用代码的自包含镜像。下一步是使用这个`Dockerfile`来实际构建Docker镜像。

## 参考资料

- [Dockerfile reference](https://docs.docker.com/engine/reference/builder/) — Docker (2024)
  详细解释所有 Dockerfile 指令，包括 FROM、WORKDIR、COPY、RUN、EXPOSE 和 CMD。
- [Best practices for writing Dockerfiles](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/) — Docker (2024)
  提供创建高效且可维护 Dockerfile 的指南，涵盖镜像分层和缓存策略等。
