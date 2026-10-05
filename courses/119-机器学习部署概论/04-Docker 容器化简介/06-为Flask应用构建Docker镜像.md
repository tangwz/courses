# 为Flask应用构建Docker镜像

来源：[原文](https://apxml.com/zh/courses/basics-ml-deployment/chapter-4-intro-containerization-docker/building-docker-image-flask)

[返回章节目录](README.md) · [返回课程目录](../README.md)

一个`Dockerfile`定义了Flask预测服务的环境和设置指令。构建Docker镜像就是使用这个文件来创建一个独立的打包文件。这个镜像将应用程序代码、依赖项（如Flask和scikit-learn）、保存的模型文件以及所需的系统库打包在一起。

构建镜像包括指示Docker守护程序读取`Dockerfile`，执行其中的指令，并创建生成的镜像文件系统层。主要的命令是`docker build`。

### `docker build` 命令

`docker build` 命令将构建上下文 (context)的路径（即包含你的`Dockerfile`和应用程序文件的目录）作为主要参数 (parameter)。它从这个上下文中读取`Dockerfile`并开始构建过程。

`docker build` 命令的一个常见选项是 `-t` 标志，它代表“标签”（tag）。添加标签可以让你给镜像一个易读的名称，并且可以选择性地添加版本号，从而更方便管理和识别。一个好的做法是使用反映应用程序的名称和版本标签，例如 `my-app:1.0` 或 `prediction-service:0.1`。如果你省略版本标签（例如，只使用 `prediction-service`），Docker会默认使用 `latest` 标签。

### 构建你的镜像：分步说明

让我们为我们准备好的Flask应用程序构建镜像。

1. **准备构建上下文 (context)：** 确保你在一个目录下有以下文件。这个目录将是你的构建上下文。

   - `Dockerfile`：你在上一节创建的文件。
   - `app.py` （或你的Flask应用程序脚本名称）：包含用于加载模型和提供预测的Flask代码。
   - `requirements.txt`：列出所需的Python库（例如，`flask`、`joblib`、`scikit-learn`、`numpy`）。
   - `model.joblib` （或你保存的模型文件名称）：序列化的机器学习 (machine learning)模型。
2. **进入目录：** 打开你的终端或命令提示符，切换到包含这些文件的目录。

   ```bash
   cd /path/to/your/flask_app_directory
   ```

   将 `/path/to/your/flask_app_directory` 替换为你系统上的实际路径。
3. **运行构建命令：** 执行 `docker build` 命令。我们将把镜像标记 (token)为 `prediction-service`，版本标签为 `0.1`。末尾的 `.` 告诉Docker使用当前目录作为构建上下文。

   ```bash
   docker build -t prediction-service:0.1 .
   ```

### 理解构建输出

当Docker执行此命令时，你将在终端中看到与`Dockerfile`中定义的每个步骤对应的输出：

```
Sending build context to Docker daemon  5.8MB # 步骤 1：上下文已发送
Step 1/7 : FROM python:3.9-slim             # 拉取基础镜像
 ---> a1b2c3d4e5f6
Step 2/7 : WORKDIR /app                    # 设置工作目录
 ---> Using cache
 ---> f6e5d4c3b2a1
Step 3/7 : COPY requirements.txt .         # 复制 requirements.txt
 ---> Using cache
 ---> 1a2b3c4d5e6f
Step 4/7 : RUN pip install --no-cache-dir -r requirements.txt # 安装依赖项
 ---> Running in abcdef123456
 Collecting flask...                         # pip 安装输出
 ... (installation output) ...
 Removing intermediate container abcdef123456
 ---> 7f8e9d0c1b2a
Step 5/7 : COPY . .                        # 复制剩余上下文
 ---> 3a4b5c6d7e8f
Step 6/7 : EXPOSE 5000                     # 暴露端口
 ---> Running in fedcba987654
 Removing intermediate container fedcba987654
 ---> 9d8c7b6a5f4e
Step 7/7 : CMD ["flask", "run", "--host=0.0.0.0"] # 设置运行命令
 ---> Running in 123456abcdef
 Removing intermediate container 123456abcdef
 ---> e1f2a3b4c5d6

Successfully built e1f2a3b4c5d6
Successfully tagged prediction-service:0.1
```

Docker逐层执行每条指令。如果指令和上下文 (context)没有改变，它通常会使用之前构建的缓存层，这会显著加快后续构建的速度（由 `---> Using cache` 指示）。当执行 `RUN` 或 `COPY` 等指令时，Docker可能会创建一个临时的中间容器，执行操作，然后将结果提交为新的镜像层，最后删除中间容器。最终，它会确认构建和标记 (token)成功。

### 验证镜像

构建成功完成后，你可以使用 `docker images` 或 `docker image ls` 命令验证镜像是否存在于你的系统上：

```bash
docker images
```

你应该看到你的新镜像已列出，类似于这样：

```
REPOSITORY            TAG       IMAGE ID       CREATED          SIZE
prediction-service    0.1       e1f2a3b4c5d6   2 minutes ago    150MB
python                3.9-slim  a1b2c3d4e5f6   2 weeks ago      115MB
... (其他镜像) ...
```

注意 `prediction-service` 仓库、`0.1` 标签、一个唯一的 `镜像 ID`、创建时间及其大小。

你现在已经成功地将你的Flask预测服务及其所有要求打包成一个Docker镜像。这个镜像是一个便携式组件，它包含在不同环境中一致运行应用程序所需的一切。下一个合理的步骤是使用这个镜像并将其作为容器运行。

## 参考资料

- [\`docker build\`](https://docs.docker.com/reference/cli/docker/build/) — Docker Documentation (2024)
  提供关于 `docker build` 命令的官方权威参考，包括其语法、用于标记的 `-t` 等选项以及构建上下文的处理，这些都是本节涵盖的核心内容。
- [Best practices for writing Dockerfiles](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/) — Docker Documentation (2024)
  提供关于编写高效、安全和可维护 Dockerfile 的官方指南和建议。尽管本节侧重于从现有 Dockerfile 构建镜像，但理解这些实践可以提高 Docker 镜像的质量和效率。
- [Docker in Action](https://www.manning.com/books/docker-in-action-second-edition) — Jeff Nickoloff, Stephen Kuenzli (2019)
  Publisher: Manning Publications
  一本综合性书籍，提供使用 Docker 的实用实例。它涵盖了构建镜像、使用 Dockerfile 以及在容器中管理应用程序，提供了比特定命令参考更广泛和详细的 Docker 生态系统理解。
- [Designing Machine Learning Systems: An Iterative Approach](https://learning.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  本书讨论了构建和部署机器学习系统的实践方面，包括容器化的作用。它为 Docker 镜像构建为何是可靠、一致地部署机器学习模型的基础步骤提供了背景。

---

[上一节](05-%E7%BC%96%E5%86%99%E4%B8%80%E4%B8%AA%E7%AE%80%E5%8D%95%E7%9A%84Dockerfile.md) · [下一节](07-%E5%9C%A8%20Docker%20%E5%AE%B9%E5%99%A8%E4%B8%AD%E8%BF%90%E8%A1%8C%E5%BA%94%E7%94%A8.md)
