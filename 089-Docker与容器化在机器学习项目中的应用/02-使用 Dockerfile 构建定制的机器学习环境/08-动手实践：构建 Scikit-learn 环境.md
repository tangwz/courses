# 动手实践：构建 Scikit-learn 环境

来源：[原文](https://apxml.com/zh/courses/docker-for-ml-projects/chapter-2-building-ml-dockerfiles/practice-build-sklearn-env)

[返回章节目录](README.md) · [返回课程目录](../README.md)

这个动手练习将构建一个简单的 Docker 镜像，其中包含使用 Python 和常用的 Scikit-learn 库，以及用于数据处理的 pandas 库的标准机器学习 (machine learning)环境。这个镜像将为开发和运行基于 Scikit-learn 的模型提供一个统一的基础。

**目标：** 创建一个安装了 Python 3.9、pip、Scikit-learn 和 pandas 的 Docker 镜像。

**前提条件：**

- 你的机器上已安装并运行 Docker Desktop 或 Docker Engine。
- 用于创建文件的文本编辑器。
- 终端或命令提示符。

### 步骤 1：设置项目目录

首先，为本次练习创建一个专用目录。这有助于我们保持文件整洁。

```bash
mkdir sklearn-env
cd sklearn-env
```

在此目录中，我们将放置 `Dockerfile` 和 `requirements.txt` 文件。

### 步骤 2：在 `requirements.txt` 中定义依赖项

使用 `requirements.txt` 文件是管理 Python 包依赖项的常用方式。Docker 可以使用此文件来高效安装所需的库。

在 `sklearn-env` 目录中创建一个名为 `requirements.txt` 的文件，内容如下：

```text
# requirements.txt
scikit-learn==1.2.2
pandas==1.5.3
```

*注意：我们指定了特定版本以实现可重现性。你可能需要根据项目需求调整这些版本，但通常建议使用特定版本以获得统一的环境。*

### 步骤 3：创建 `Dockerfile`

现在，在 `sklearn-env` 目录中创建核心文件：`Dockerfile`（无扩展名）。添加以下指令：

```dockerfile
# 用于基本 Scikit-learn 环境的 Dockerfile

# 1. 选择基础镜像
FROM python:3.9-slim

# 2. 设置容器内的工作目录
WORKDIR /app

# 3. 首先复制 requirements 文件以利用 Docker 缓存
COPY requirements.txt .

# 4. 安装 Python 依赖项
RUN pip install --no-cache-dir -r requirements.txt

# 5. （可选）设置容器启动时运行的默认命令
CMD ["python"]
```

让我们分析一下这个 `Dockerfile`：

- `FROM python:3.9-slim`：我们从官方的 Python 基础镜像开始。我们选择了 3.9 版本和 `slim` 变体，它比默认标签更小，减少了最终镜像的大小，同时仍提供了一个可用的 Python 环境。
- `WORKDIR /app`：这设置了容器内后续命令（`COPY`、`RUN`、`CMD`、`ENTRYPOINT`）的工作目录。如果目录不存在，`WORKDIR` 会创建它。设置一个专用工作目录是一个好习惯。
- `COPY requirements.txt .`：这将 `requirements.txt` 文件从你的构建上下文 (context)（`sklearn-env` 目录）复制到容器的工作目录（`/app`）。我们是在运行 `pip install` *之前*复制的。Docker 以层的方式构建镜像。如果 `requirements.txt` 自上次构建以来没有更改，Docker 可以重用后续 `RUN` 指令的缓存层，显著加快重新构建的速度。
- `RUN pip install --no-cache-dir -r requirements.txt`：这在容器内执行 `pip install` 命令。
  - `--no-cache-dir`：此选项禁用 pip 的缓存，这有助于减小镜像大小，因为最终镜像中不需要缓存。
  - `-r requirements.txt`：告诉 pip 安装指定文件中列出的软件包。
- `CMD ["python"]`：这定义了在没有指定命令的情况下从此镜像启动容器时运行的默认命令。在这种情况下，它将启动一个交互式 Python 解释器。

### 步骤 4：构建 Docker 镜像

现在，在你的终端中导航到 `sklearn-env` 目录（如果你不在那里），并运行 `docker build` 命令：

```bash
docker build -t sklearn-env:1.0 .
```

让我们分解这个命令：

- `docker build`：从 `Dockerfile` 构建镜像的命令。
- `-t sklearn-env:1.0`：`-t` 标志为镜像指定名称和可选的标签（版本）。这里，我们将其命名为 `sklearn-env` 并标记 (token)为 `1.0`。添加标签使得以后引用该镜像更方便。
- `.`：这指定了构建上下文 (context)。Docker 会在当前目录（`.`）中寻找 `Dockerfile`，并将该目录（及其子目录）中的文件发送到 Docker 守护进程以执行构建。

你将看到 Docker 执行 `Dockerfile` 中定义的每个步骤。它会下载基础镜像（如果尚未存在），然后运行每条指令，为镜像创建层。

### 步骤 5：验证镜像

构建成功完成后，你可以验证环境是否已正确设置。从镜像运行一个临时容器并检查已安装的软件包版本：

```bash
docker run --rm -it sklearn-env:1.0 python -c "import sklearn; import pandas; print(f'Scikit-learn version: {sklearn.__version__}'); print(f'Pandas version: {pandas.__version__}')"
```

- `docker run`：从镜像运行容器的命令。
- `--rm`：容器退出时自动移除。这对于短暂的验证任务很有用。
- `-it`：以交互模式（`-i`）运行容器并分配一个伪 TTY（`-t`），让你能够与它交互（尽管对于这个特定命令并非严格必需，但这是常见做法）。
- `sklearn-env:1.0`：我们刚刚构建的镜像的名称和标签。
- `python -c "..."`：这会覆盖默认的 `CMD` 并在容器内执行一个特定的 Python 命令。该命令会导入库并打印它们的版本。

你应该看到类似于此的输出（如果你更改了 `requirements.txt`，确切版本可能会略有不同）：

```
Scikit-learn version: 1.2.2
Pandas version: 1.5.3
```

如果你看到版本号正确打印出来，恭喜你！你已成功构建了一个包含可重现 Scikit-learn 环境的 Docker 镜像。这个镜像现在可以与同事共享，或用作特定训练或推理 (inference)脚本容器化的基础，确保每个人都使用相同的依赖项集。

## 参考资料

- [Overview of a Dockerfile](https://docs.docker.com/reference/dockerfile/) — Docker (2024)
  全面介绍 Dockerfile 语法、指令以及构建 Docker 镜像的最佳实践。
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html) — scikit-learn developers (2023)
  scikit-learn 官方用户指南，涵盖机器学习任务的使用，为围绕其构建的环境提供了背景信息。
- [Installing Python packages](https://packaging.python.org/en/latest/guides/installing/) — Python Packaging Authority (2024)
  Publisher: Python Packaging Authority
  Python 包管理官方指南，包含 pip 和 requirements.txt 的使用。

---

[上一节](07-%E8%AE%BE%E7%BD%AE%E5%B7%A5%E4%BD%9C%E7%9B%AE%E5%BD%95%E5%92%8C%E5%85%A5%E5%8F%A3%E7%82%B9.md) · [下一节](../03-%E5%9C%A8%E5%AE%B9%E5%99%A8%E4%B8%AD%E7%AE%A1%E7%90%86%E6%95%B0%E6%8D%AE%E5%92%8C%E6%A8%A1%E5%9E%8B/01-%E7%90%86%E8%A7%A3%E5%AE%B9%E5%99%A8%E5%AD%98%E5%82%A8.md)
