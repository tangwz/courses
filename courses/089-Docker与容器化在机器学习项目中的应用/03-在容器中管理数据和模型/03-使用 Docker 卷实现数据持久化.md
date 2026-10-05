# 使用 Docker 卷实现数据持久化

来源：[原文](https://apxml.com/zh/courses/docker-for-ml-projects/chapter-3-managing-ml-data-containers/docker-volumes-persistence)

[返回章节目录](README.md) · [返回课程目录](../README.md)

将应用程序代码容器化可以提供一致性，但容器的临时性给机器学习 (machine learning)任务带来了问题：数据集从何而来，训练好的模型又将存放何处？默认情况下，容器在被移除后会丢失其中写入的所有数据。对于机器学习而言，数据集可能很大，而训练好的模型代表着大量的计算投入，这种方式是行不通的。我们需要一种方法来独立于容器的生命周期来保存数据。

Docker 卷是持久化 Docker 容器生成和使用的数据的首选机制。可以将卷看作是主机文件系统上一个专用的、由 Docker 管理的目录。其主要好处在于 Docker 管理卷的存储区域、生命周期和权限，从而将数据完全与任何特定容器分离。

与绑定挂载（我们接下来会讨论）不同，绑定挂载是将主机上的特定目录映射到容器中，而卷是由 Docker 自己创建和管理的。您可以通过名称来与它们交互。这种抽象使得它们在不同环境和平台上的管理更加简便。

### 创建和管理卷

您可以使用 Docker 命令行显式创建一个卷：

```bash
docker volume create my-ml-data
```

此命令会创建一个名为 `my-ml-data` 的新卷。Docker 会处理该卷在主机文件系统上 *实际位于何处*。您通常不需要知道确切的主机路径，只需知道卷的名称即可。

您可以列出现有卷：

```bash
docker volume ls
```

输出：

```
驱动    卷名称
local     my-ml-data
```

要获取特定卷的更多详情，包括其在主机上的挂载点（尽管您很少直接与此交互）：

```bash
docker volume inspect my-ml-data
```

当您不再需要卷中的数据时，可以将其移除（请谨慎使用！）：

```bash
docker volume rm my-ml-data
```

重要的一点是，您不总是需要在事先显式创建卷。如果在运行容器时指定了一个命名卷且该卷不存在，Docker 会自动为您创建它。

### 容器与卷的使用

要使卷中的数据可供容器访问，您可以使用 `docker run` 命令的 `-v` 或 `--mount` 标志来挂载它。使用 `-v` 的最常见语法指定卷的名称以及它在容器内部应挂载的路径：

```bash
# 语法: -v <卷名称>:<容器内路径>
docker run -d --name my-training-app -v my-ml-data:/app/data my-image
```

在此示例中：

- `my-ml-data` 是 Docker 卷的名称（如果该卷不存在，Docker 会创建它）。
- `/app/data` 是容器内卷内容将出现的绝对路径。应用程序写入 `/app/data` 的任何数据都将存储在主机上的 `my-ml-data` 卷中。

如果您停止并移除 `my-training-app` 容器：

```bash
docker stop my-training-app
docker rm my-training-app
```

`my-ml-data` 卷及其内部存储的所有数据 *保持不变*。然后您可以启动一个新的容器并挂载相同的卷来访问持久化的数据。

```bash
# 启动一个新容器，可能用于推理，挂载相同的卷
docker run --name my-inference-app -v my-ml-data:/app/input-models my-other-image
```

现在，第二个容器可以从其 `/app/input-models` 目录读取由第一个容器保存的模型或数据。

### 机器学习 (machine learning)工作流的优点

使用卷提供了多项好处，尤其是对于机器学习任务而言：

1. **数据持久性：** 这是主要好处。存储在卷中的大型数据集、训练好的模型文件（`.h5`、`.pkl`、SavedModel 格式）、检查点、日志和评估指标在容器重启、移除和更新后依然存在。
2. **存储与容器解耦：** 数据的生命周期独立于您的应用程序容器。您可以更新训练脚本的 Docker 镜像，而不会丢失数据集或训练好的模型。
3. **容器间数据共享：** 卷可以在多个容器之间轻松共享。一种常见模式是让训练容器将模型写入卷中，而独立的推理 (inference)或评估容器则挂载相同的卷来读取模型。
4. **性能：** Docker 管理卷的位置和 I/O。在 Docker Desktop（macOS、Windows）等系统上，由于文件系统虚拟化层，卷通常比从主机用户空间绑定挂载目录提供更好的磁盘密集型操作 I/O 性能。
5. **备份和迁移：** Docker 提供命令和工具，可以方便地备份卷数据或在主机之间迁移卷，尽管备份管理通常涉及针对卷数据内容的特定策略。
6. **平台一致性：** Docker 管理卷存储位置，与依赖主机特定路径的绑定挂载相比，这在不同主机操作系统（Linux、macOS、Windows）上提供了更一致的行为。

> 此图展示了 Docker 卷如何在主机文件系统上 Docker 管理的区域内存在。容器，例如“训练容器”和“推理容器”，将这些卷挂载到其内部的特定路径（例如 `/data_in_cont`、`/model_in_cont`），从而实现了独立于容器生命周期的持久化存储和数据共享。

### 卷的常见机器学习 (machine learning)应用场景

卷非常适合管理机器学习项目中的各种持久化数据：

- **数据集：** 存储训练容器需要读取的大型数据集（图像、文本语料库、表格数据）。
- **训练好的模型：** 保存由训练任务生成的模型工件（例如权重 (weight)、模型架构文件）。
- **日志和指标：** 持久化训练日志、TensorBoard 事件文件或运行期间生成的评估结果。
- **检查点：** 在长时间训练过程中保存中间模型状态，允许在中断后恢复。
- **预处理数据：** 存储由一个容器生成供另一个容器使用的特征或中间数据表示。

### 实际示例：保存模型工件

让我们用一个简单的例子来说明。设想一个容器内有一个 Python 脚本（`save_model.py`），它模拟保存一个模型工件。

**`save_model.py`：**

```python
import os
import time
import datetime

# 定义容器内的输出目录
output_dir = "/app/output"
# 根据时间创建唯一文件名
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = os.path.join(output_dir, f"model_{timestamp}.txt")

# 确保输出目录存在（容器内的良好做法）
os.makedirs(output_dir, exist_ok=True)

print(f"模拟模型训练...")
time.sleep(1) # 模拟工作
print(f"正在将模拟模型工件保存到: {model_filename}")

# 向文件写入一些内容
with open(model_filename, "w") as f:
    f.write("这是一个模拟的模型工件。\n")
    f.write(f"保存时间: {datetime.datetime.now()}\n")

print("模型工件已保存。")
```

**`Dockerfile`：**

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY save_model.py .
# 运行脚本的默认命令
CMD ["python", "save_model.py"]
```

现在，让我们构建并运行镜像，使用名为 `model-storage` 的卷来持久化输出。

```bash
# 1. 构建 Docker 镜像
docker build -t simple-saver .

# 2. 创建一个命名卷（可选，docker run 也可以创建）
# docker volume create model-storage

# 3. 运行容器，将卷挂载到 /app/output
#    --rm 标志会在容器退出时自动移除它。
echo "正在运行容器以保存模型..."
docker run --rm \
  -v model-storage:/app/output \
  --name saver-instance \
  simple-saver

# 容器运行，打印消息，将文件保存到 /app/output，然后退出。
# 容器被移除，但数据保留在 'model-storage' 卷中。

# 4. 通过从另一个容器检查卷来验证数据是否存在
echo "正在检查卷内容..."
docker run --rm \
  -v model-storage:/data \
  python:3.9-slim ls -l /data
```

您应该会看到类似以下的输出（确切的文件名会不同）：

```
正在运行容器以保存模型...
模拟模型训练...
正在将模拟模型工件保存到: /app/output/model_20231027_103045.txt
模型工件已保存。
正在检查卷内容...
total 4
-rw-r--r-- 1 root root 81 Oct 27 10:30 model_20231027_103045.txt
```

这表明，即使 `saver-instance` 容器被移除，它在 `/app/output` 内创建的模型文件也已持久化到 `model-storage` 卷中，并且稍后可以被访问（在本例中，由一个挂载相同卷的临时 `python:3.9-slim` 容器访问）。

总之，Docker 卷提供了一种可靠且易于管理的方式来处理容器化机器学习 (machine learning)应用程序中的持久化数据需求。它们允许您将数据问题与应用程序代码分离，确保数据集、模型和日志在单个容器之外仍然存在，这对于高效的机器学习开发和部署流程非常重要。

## 参考资料

- [Use volumes](https://docs.docker.com/storage/volumes/) — Docker, Inc. (2024)
  Publisher: Docker, Inc.
  Docker卷的官方指南，涵盖了它们的创建、管理以及用于数据持久化的方法。
- [Introducing MLOps: How to go from Model to Production](https://www.oreilly.com/library/view/introducing-mlops/9781492083283/) — Mark Treveil, Nicolas Omont, Rishabh Bahri, Aurélien Géron, Dmitry Kandalov, Costa Mirian, Josh Newman, Chetan Parikh, Tom Phelan, Michael Redman (2022)
  Publisher: O'Reilly Media
  提供了MLOps工作流中数据持久化挑战和解决方案的背景知识，将容器化与机器学习领域需求相结合。

---

[上一节](02-%E7%94%A8%E4%BA%8E%E5%BC%80%E5%8F%91%E7%9A%84%E7%BB%91%E5%AE%9A%E6%8C%82%E8%BD%BD.md) · [下一节](04-%E6%AF%94%E8%BE%83%E7%BB%91%E5%AE%9A%E6%8C%82%E8%BD%BD%E5%92%8C%E5%8D%B7.md)
