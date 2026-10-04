---
course: "docker-for-ml-projects"
chapter: "containerizing-ml-training"
lesson: "managing-training-logs"
sourceId: 5299
sourceUrl: "https://apxml.com/zh/courses/docker-for-ml-projects/chapter-4-containerizing-ml-training/managing-training-logs"
title: "管理训练日志"
description: "捕获和持久化容器化训练期间生成的日志的策略。"
order: 4
plots: []
sourceHash: "82e918ef30a7fc64b4f59294f826d027955564d97c5065b3404e867e03eac925"
sourceCorrections: []
---

当您的机器学习 (machine learning)训练脚本在容器内部运行时，它会生成输出，这些输出对于监视进度、调试问题以及确保实验的可复现性非常重要。有效管理这些日志是容器化机器学习工作流的一个重要组成部分。如果管理不当，当容器停止时，有用的信息可能会丢失。

### 捕获容器输出：`stdout` 和 `stderr`

默认情况下，在 Docker 容器内部运行的应用程序会将其标准输出 (`stdout`) 和标准错误 (`stderr`) 流发送到容器的日志驱动。Docker 会捕获这些流，以便您后续检查它们。

访问这些日志最直接的方式是使用 `docker logs` 命令，后跟容器 ID 或名称：

```bash
# 在后台启动一个训练容器
docker run -d --name training_job my_ml_image python train.py --epochs 10

# 从容器中获取日志
docker logs training_job

# 实时追踪日志输出（类似于 tail -f）
docker logs -f training_job
```

这种方法对于简单情况或快速调试来说很直接。但是，仅仅依赖默认的日志驱动存在一些限制：

1. **持久性：** 以这种方式捕获的日志通常存储在 Docker 主机上。如果容器被移除 (`docker rm`)，与之相关的日志通常会丢失，除非配置了不同的日志驱动。
2. **扩展性：** 当容器数量众多或在分布式环境中时，通过 `docker logs` 管理日志会变得很繁琐。
3. **格式：** 原始流输出可能不容易解析或搜索。

### 将日志写入容器内的文件

一种常见且更直接的做法是，配置您的训练脚本将日志写入容器 *内部* 的特定文件，而不是仅仅打印到 `stdout`/`stderr`。大多数编程语言都为此提供了标准日志库。

例如，在 Python 中，您可以使用内置的 `logging` 模块：

```python
# train.py 中的示例代码片段
import logging
import sys

# 配置日志输出到文件和控制台
log_file = '/app/logs/training.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout) # 同时输出到控制台
    ]
)

logging.info("Starting training process...")
# ... 训练代码 ...
logging.info("Training finished.")
```

这个脚本现在会将带时间戳的日志消息写入容器文件系统内的 `/app/logs/training.log` 文件，*并且* 将它们复制到 `stdout`。

### 使用卷或绑定挂载持久化日志文件

将日志写入容器内的文件解决了格式问题，但没有解决持久性问题。如果容器被移除，`/app/logs/training.log` 文件也会随之消失。为了保留这些日志，您需要使用 Docker 卷或绑定挂载将它们存储在容器的临时文件系统之外，这些内容您在上一章中已经了解过。

**使用绑定挂载：**

绑定挂载将您主机上的目录直接映射到容器内部。这在开发期间通常很方便。

```bash
# 在主机上创建一个目录来存储日志
mkdir -p /path/on/host/training_logs

# 运行容器，将主机目录挂载到容器的日志目录
docker run -d \
  --name training_job \
  -v /path/on/host/training_logs:/app/logs \
  my_ml_image \
  python train.py --epochs 10
```

现在，容器内部 `/app/logs/training.log` 生成的 `training.log` 文件实际上将写入到您主机上的 `/path/on/host/training_logs/training.log`，即使容器停止并移除后也能持久存在。

**使用 Docker 卷：**

卷由 Docker 管理，是在生产环境中持久化数据或当您不想将数据存储绑定到特定主机目录结构时的首选方法。

```bash
# 创建一个 Docker 卷（可选，如果不存在 Docker 会自动创建）
docker volume create ml_training_logs

# 运行容器，将命名卷挂载到容器的日志目录
docker run -d \
  --name training_job \
  -v ml_training_logs:/app/logs \
  my_ml_image \
  python train.py --epochs 10
```

日志现在存储在 `ml_training_logs` 卷中，由 Docker 管理。您可以检查卷的内容，或独立于容器或主机文件系统的具体情况来备份它。

> 此图说明了绑定挂载和 Docker 卷如何将容器内部的日志文件连接到 Docker 主机上的存储。

### 结构化日志

为了更容易进行自动化处理和分析，可以考虑以结构化的方式（例如 JSON）格式化您的日志。许多日志库都支持自定义格式器。

```python
# 使用 Python 的 logging 模块和 JSON 格式器示例（需要 python-json-logger）
# pip install python-json-logger

import logging
from pythonjsonlogger import jsonlogger

log_handler = logging.FileHandler('/app/logs/training.log')
formatter = jsonlogger.JsonFormatter('%(asctime)s %(levelname)s %(message)s')
log_handler.setFormatter(formatter)

logger = logging.getLogger()
logger.addHandler(log_handler)
logger.setLevel(logging.INFO)

logger.info("Training started", extra={'learning_rate': 0.01, 'epochs': 10})
# training.log 中的输出: {"asctime": "...", "levelname": "INFO", "message": "Training started", "learning_rate": 0.01, "epochs": 10}
```

结构化日志更容易导入到日志分析平台中。

### 集中式日志系统

尽管通过卷或绑定挂载将日志持久化到主机文件对于单个实验或小型设置很有效，但在管理跨多个容器或在生产环境中的日志时，通常会涉及**集中式日志系统**。

ELK Stack（Elasticsearch、Logstash、Kibana）、Splunk、Graylog、Grafana Loki 等工具，或云提供商服务（AWS CloudWatch Logs、Google Cloud Logging、Azure Monitor Logs）旨在聚合来自多个来源（包括 Docker 容器）的日志。它们提供强大的搜索、可视化和警报功能。

配置 Docker 将日志直接转发到这些系统，通常涉及为 Docker 守护程序全局设置或使用 `docker run` 命令配合 `--log-driver` 和 `--log-opt` 标志为每个容器设置特定的日志驱动。虽然配置这些系统超出了本中级课程的范围，但重要的是要了解它们作为大规模日志管理的标准解决方案而存在。

总之，管理容器化训练作业的日志，不仅仅是使用基本的 `docker logs` 命令。配置您的应用程序将日志写入文件，然后使用 Docker 卷或绑定挂载将这些文件持久化到容器外部，这提供了一种可靠的方式来捕获和保留训练历史，以便进行调试、分析和可复现性。对于更复杂的设置，请考虑采用结构化日志并研究集中式日志解决方案。

## 参考资料

- [Configure logging drivers](https://docs.docker.com/config/containers/logging/configure/) — Docker, Inc. ()
  Journal: Docker Documentation
  官方文档，解释 Docker 如何捕获容器输出、`docker logs` 命令以及如何配置各种日志驱动程序用于日志管理和转发。
- [Manage data in Docker](https://docs.docker.com/storage/) — Docker, Inc. ()
  Journal: Docker Documentation; Publisher: Docker, Inc.
  Docker 数据管理选项的官方指南，包括绑定挂载和卷，它们对于在容器的临时文件系统外部持久化日志文件至关重要。
- [logging - Logging facility for Python](https://docs.python.org/3/library/logging.html) — Python Software Foundation ()
  Journal: Python Documentation
  Python 内置日志模块的官方文档，提供配置应用程序级日志到文件和流的详细信息。
- [Learn Docker in a Month of Lunches, Second Edition](https://www.manning.com/books/learn-docker-in-a-month-of-lunches-second-edition) — Elton Stoneman (2025)
  Publisher: Manning Publications; Pages: 488
  一本实用的 Docker 指南，提供了容器化、数据管理和操作方面的见解，与机器学习工作流的持久化日志记录策略相关。
