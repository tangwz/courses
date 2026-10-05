# 配置 MLflow

来源：[原文](https://apxml.com/zh/courses/data-versioning-experiment-tracking/chapter-3-tracking-experiments-mlflow/setting-up-mlflow)

[返回章节目录](README.md) · [返回课程目录](../README.md)

MLflow 为系统地记录机器学习 (machine learning)实验提供了一种实用且直接的方法。MLflow 的上手过程很简单。

### 安装

MLflow 是一个 Python 包，可以使用 Python 标准包管理器 pip 进行安装。打开您的终端或命令行界面，然后运行：

```bash
pip install mlflow
```

此命令会安装 MLflow 核心库，其中包括本章的重点——追踪组件。根据您的项目需求，您之后可能还会安装额外的 MLflow 扩展（例如用于数据库后端或特定机器学习 (machine learning)库集成的 `mlflow[extras]`），但基础安装目前足够使用。

要验证安装，您可以运行：

```bash
mlflow --version
```

这会打印出已安装的 MLflow 版本号。

### MLflow 追踪环境

MLflow 追踪设计灵活，允许您根据配置和需求以不同方式记录运行。当您的 Python 脚本使用 MLflow 日志函数（例如 `mlflow.log_param()` 或 `mlflow.log_metric()`，我们稍后会讲到）时，它需要知道*将*这些信息发送到哪里。这个目标地点被称为**追踪后端**。MLflow 支持多种后端类型：

1. **本地文件系统（默认）：** 这是最简单的选项，也是您未进行其他配置时的默认行为。当您运行一个已集成 MLflow 的脚本时，MLflow 会自动在脚本执行的当前工作目录下创建一个名为 `mlruns` 的目录。`mlruns` 内部会以结构化的文件格式存储运行元数据、参数 (parameter)、指标和工件。这非常适合个人实验和快速上手。
2. **本地追踪服务器：** 您可以运行一个本地 MLflow 追踪服务器。这提供一个专门的用户界面 (UI)，您可以通过网页浏览器访问它来查看和比较运行，即使这些运行来自您机器上的不同目录或项目。服务器默认仍将数据存储在本地文件系统（使用指定目录）上，但集中了访问权限。
3. **远程追踪服务器：** 为了协作或更持久的数据存储，您可以配置 MLflow 以记录到远程追踪服务器。此服务器可以将追踪数据存储在关系型数据库（例如 PostgreSQL、MySQL）等后端中，并将工件存储在共享存储（例如 S3、Azure Blob Storage、GCS 或 NFS）中。这种设置在团队环境或 MLOps 流水线中很常见。

我们来看看如何使用前两个选项。

#### 选项 1：记录到本地文件系统

如前所述，安装 MLflow 后，此选项无需额外配置。如果您运行包含 MLflow 日志命令的 Python 脚本，它会自动创建 `mlruns` 目录。

```python
# 示例：simple_script.py
import mlflow
import os

# 无需服务器配置，MLflow 默认记录到 ./mlruns
mlflow.start_run()

# 记录一些示例数据
mlflow.log_param("learning_rate", 0.01)
mlflow.log_metric("accuracy", 0.95)

# 创建一个示例工件文件
os.makedirs("outputs", exist_ok=True)
with open("outputs/model.txt", "w") as f:
    f.write("This is a dummy model file.")
mlflow.log_artifact("outputs/model.txt")

mlflow.end_run()

print("MLflow 运行已记录到本地 './mlruns' 目录。")
```

如果您执行 `python simple_script.py`，您会发现出现了一个新的 `mlruns` 目录。

要在 MLflow UI 中查看这些本地运行，请在终端中导航到*包含* `mlruns` 文件夹的目录，然后运行：

```bash
mlflow ui
```

此命令会启动一个本地 Web 服务器（通常在 `http://127.0.0.1:5000` 或 `http://localhost:5000`），该服务器提供 MLflow 追踪 UI。在浏览器中打开此地址，您将看到脚本运行的结果。在终端中按 `Ctrl+C` 可以停止 UI 服务器。

#### 选项 2：使用本地追踪服务器

虽然 `mlflow ui` 直接从 `mlruns` 读取，但有时您需要一个持久的服务器进程，可以接受您机器上（甚至本地网络中的其他机器上）任何位置运行的脚本的运行记录。

首先，启动追踪服务器。选择一个目录作为服务器的数据存储位置（此目录将包含 `mlruns` 的等效内容）。我们使用一个名为 `mlflow_server_data` 的目录。

```bash
# 为服务器数据创建目录
mkdir mlflow_server_data

# 启动 MLflow 服务器
# --backend-store-uri: 指定存储运行元数据、指标、参数、标签的位置。
# --default-artifact-root: 指定存储工件（如模型、文件）的位置。
# --host: 要监听的网络接口（127.0.0.1 是本地主机）
# --port: 要使用的端口（默认是 5000，这里使用 8080）
mlflow server \
    --backend-store-uri ./mlflow_server_data \
    --default-artifact-root ./mlflow_server_data \
    --host 127.0.0.1 \
    --port 8080
```

此命令会在您的终端中启动服务器进程。它会持续运行，直到您停止它（例如，通过 `Ctrl+C`）。

现在，您需要让您的 Python 脚本记录到此服务器，而不是本地文件系统。您可以通过在启动运行*之前*，使用 `mlflow.set_tracking_uri()` 函数设置**追踪 URI** 来实现。

```python
# 示例：script_log_to_server.py
import mlflow
import os

# 设置追踪 URI，指向正在运行的服务器
mlflow.set_tracking_uri("http://127.0.0.1:8080") 

# 现在，当我们启动一个运行，它会记录到服务器
mlflow.start_run()

mlflow.log_param("learning_rate", 0.02)
mlflow.log_metric("accuracy", 0.98)

# 创建另一个示例工件
os.makedirs("outputs", exist_ok=True)
with open("outputs/advanced_model.txt", "w") as f:
    f.write("This is another dummy model file.")
mlflow.log_artifact("outputs/advanced_model.txt")

mlflow.end_run()

print("MLflow 运行已记录到服务器 http://127.0.0.1:8080")
```

如果您运行 `python script_log_to_server.py`（在另一个终端中运行着 `mlflow server` 命令时），运行数据将被发送到服务器并存储在 `mlflow_server_data` 目录中。您可以通过在网页浏览器中打开 `http://127.0.0.1:8080` 来查看这些运行。

或者，您可以使用 `MLFLOW_TRACKING_URI` 环境变量来设置追踪 URI，这通常更适合自动化工作流：

```bash
export MLFLOW_TRACKING_URI="http://127.0.0.1:8080"
python your_training_script.py 
```

如果设置了此环境变量，`mlflow.start_run()` 将自动使用它，您的代码中就不需要 `mlflow.set_tracking_uri()` 了。

> 不同的 MLflow 追踪设置配置。Python 脚本可以直接记录到本地 `mlruns` 目录，到本地 `mlflow server` 进程（集中存储），或到通常用于团队或生产环境的远程服务器。

#### 选择您的设置

- **简单开始：** 对于初步尝试或个人项目，直接记录到本地文件系统（`mlruns`）并使用 `mlflow ui` 通常是最简单的上手方法。
- **本地集中：** 如果您在同一台机器上从不同项目目录运行实验并希望获得统一的 UI 视图，运行一个指向共享数据目录的本地 `mlflow server` 是一个不错的下一步。
- **团队协作 / 持久性：** 当在团队中工作，需要独立于本地机器的持久运行历史，或集成到 CI/CD 时，配置带数据库后端和共享工件存储的远程追踪服务器是标准做法。远程服务器的配置细节很大程度上取决于所选的后端（数据库类型、云存储供应商），并遵循设置追踪 URI 的类似原则。

对于本章的练习，记录到本地文件系统或使用本地 `mlflow server` 将足够使用。既然 MLflow 已安装，并且您已明白基本配置选项，那么我们继续调整您的训练代码，使其能够实际记录有用信息。

## 参考资料

- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html) — Databricks, Inc. (2024)
  Publisher: Databricks, Inc.
  MLflow实验追踪组件的官方文档，涵盖安装、API以及各种服务器配置。
- [MLflow: A Machine Learning Lifecycle Platform](https://dl.acm.org/doi/10.1145/3361098.3361099) — Matei Zaharia, Andy Konwinski, Ben Lorica, Jules S. Damji, Ali Ghodsi, and Aaron Davidson (2019)
  Journal: ACM Queue; Publisher: ACM; Volume: 17; Pages: 84-93; DOI: [10.1145/3361098.3361099](https://doi.org/10.1145/3361098.3361099)
  介绍MLflow平台的原始出版物，详细阐述了其架构和包括追踪系统在内的各个组件的设计原则。

---

[上一节](02-MLflow%20%E8%BF%BD%E8%B8%AA%E5%8A%9F%E8%83%BD%E4%BB%8B%E7%BB%8D.md) · [下一节](04-%E8%AE%B0%E5%BD%95%E5%8F%82%E6%95%B0%E5%92%8C%E6%8C%87%E6%A0%87.md)
