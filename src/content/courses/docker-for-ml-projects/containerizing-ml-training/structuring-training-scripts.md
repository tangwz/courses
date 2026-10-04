---
course: "docker-for-ml-projects"
chapter: "containerizing-ml-training"
lesson: "structuring-training-scripts"
sourceId: 5291
sourceUrl: "https://apxml.com/zh/courses/docker-for-ml-projects/chapter-4-containerizing-ml-training/structuring-training-scripts"
title: "容器训练脚本的组织方式"
description: "调整您的训练脚本以在容器中高效运行，并正确处理输入和输出。"
order: 1
plots: []
sourceHash: "b4965af90b968cc3e5573b1bbd87712826455ef8457b7b25942f44e19b7c3c3c"
sourceCorrections: []
---

调整机器学习 (machine learning)训练代码以使其在 Docker 容器内高效运行，对于可靠的机器学习部署是必不可少的。尽管开发人员可以简单地将现有脚本复制到镜像中，但将容器环境视为普通服务器通常会导致不稳定的配置。硬编码文件路径或依赖特定用户配置的脚本，在移入容器的隔离、标准化环境中时将无法工作。

为了成功地将训练过程容器化，您的脚本需要围绕容器的生命周期和执行环境进行设计或重构。这涉及使它们更具弹性、可配置，并明白如何在容器化环境中接收输入和生成输出。

### 面向容器执行的设计

将您的容器化训练脚本视为一个自包含的可执行单元。它接收指令和数据，执行其任务，并输出结果。为了可靠地做到这一点，请考虑以下原则：

1. **参数 (parameter)化输入和输出：** 避免在脚本中硬编码数据集、模型保存位置或配置文件的路径。这些路径可能因容器运行时卷或绑定挂载的配置方式而异。相反，脚本应将这些位置作为输入接收。
2. **外部化配置：** 超参数 (hyperparameter)、训练设置（如 epoch 或 batch 大小）以及资源标志不应直接嵌入 (embedding)代码中。它们应在运行时传递给脚本。
3. **标准化日志：** 将日志输出定向到标准流（`stdout` 和 `stderr`）。这使得 Docker 的日志驱动程序能够轻松捕获输出，使监控和调试变得简单，无需脚本显式管理日志文件位置。
4. **假定无状态（通常）：** 每个 `docker run` 命令通常都会启动一个全新的容器实例。除非您专门为使用持久卷的增量训练而设计，否则您的脚本不应依赖于同一容器镜像中先前运行遗留的状态。在每次运行时，它应在给定相同输入和配置的情况下生成相同的结果。

### 实现灵活的脚本

让我们看看实现这些原则的实用方法，主要关注 Python 脚本，这是机器学习 (machine learning)训练的常见选择。

#### 使用命令行参数 (parameter)

在许多环境中（包括 Docker 容器），向脚本传递参数的标准方法是通过命令行参数。Python 的 `argparse` 模块非常适合此用途。

考虑一个需要输入数据路径、输出模型路径和学习率的脚本。

```python
# training_script.py
import argparse
import os
import pandas as pd
# 假设 scikit-learn 或其他机器学习库已安装在镜像中
# from sklearn.linear_model import LogisticRegression 

def train_model(data_path, model_path, learning_rate):
    print(f"Loading data from: {data_path}")
    # 模拟数据加载
    # df = pd.read_csv(data_path) 
    print(f"Training model with learning rate: {learning_rate}")
    # 模拟模型训练
    # model = LogisticRegression(C=1.0/learning_rate) # 示例用法
    # model.fit(df[['feature1']], df['target'])
    print(f"Saving model to: {model_path}")
    # 如有必要，确保输出目录存在
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    # 模拟模型保存
    with open(model_path, 'w') as f:
        f.write(f"dummy model trained with lr={learning_rate}")
    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练一个简单的机器学习模型。")

    parser.add_argument('--data-path', 
                        type=str, 
                        required=True, 
                        help='输入训练数据文件的路径（例如：/data/train.csv）')

    parser.add_argument('--model-path', 
                        type=str, 
                        required=True, 
                        help='保存训练好的模型文件的路径（例如：/output/model.pkl）')

    parser.add_argument('--lr', 
                        type=float, 
                        default=0.01, 
                        help='训练的学习率。')

    args = parser.parse_args()

    train_model(args.data_path, args.model_path, args.lr)
```

构建 Docker 镜像时，您通常会设置 `ENTRYPOINT` 或 `CMD` 来执行此脚本。例如，在您的 Dockerfile 中：

```dockerfile
# ... (基础镜像，依赖项安装) ...

WORKDIR /app
COPY training_script.py .

# 选项 1：使用 CMD（允许轻松覆盖命令）
# CMD ["python", "training_script.py"] 
# 用户将提供参数，例如：docker run my-image --data-path /data/input.csv --model-path /output/final_model.pkl --lr 0.005

# 选项 2：使用 ENTRYPOINT（使容器像脚本可执行文件一样运行）
ENTRYPOINT ["python", "training_script.py"]
# 用户直接提供参数：docker run my-image --data-path /data/input.csv --model-path /output/final_model.pkl --lr 0.005 
```

通过使用 `argparse`，脚本清晰地定义了其所需的输入以及如何提供这些输入。然后运行容器涉及在镜像名称后传递这些参数，并将主机目录或卷映射到预期的容器路径（如 `/data` 和 `/output`）。

#### 使用环境变量

环境变量提供了另一种传递配置的方式，通常适用于 API 密钥、数据库连接字符串或在不同部署环境（开发、测试、生产）中可能敏感或不同的标志。Python 的 `os` 模块可以访问它们。

```python
# training_script_env.py
import os
import argparse

def load_config():
    config = {}
    # 从环境变量获取路径，如果未设置则使用默认值或引发错误
    config['data_path'] = os.environ.get('TRAINING_DATA_PATH')
    config['model_path'] = os.environ.get('MODEL_OUTPUT_PATH')

    if not config['data_path'] or not config['model_path']:
        raise ValueError("Missing required environment variables: TRAINING_DATA_PATH, MODEL_OUTPUT_PATH")

    # 从环境变量获取超参数，并带有默认值
    config['learning_rate'] = float(os.environ.get('LEARNING_RATE', 0.01))
    config['epochs'] = int(os.environ.get('EPOCHS', 10))

    return config

def train_model(config):
    print(f"Loading data from: {config['data_path']}")
    print(f"Training model with learning rate: {config['learning_rate']} for {config['epochs']} epochs.")
    print(f"Saving model to: {config['model_path']}")
    # ... (其余训练逻辑) ...
    print("Training complete.")

if __name__ == "__main__":
    # 对于不太可能通过环境变量更改的参数，您可能仍然使用 argparse，
    # 或者完全依赖环境变量。
    # parser = argparse.ArgumentParser() 
    # parser.add_argument('--some-other-arg', default='value')
    # args = parser.parse_args()

    configuration = load_config()
    train_model(configuration)
```

然后，您可以使用 `-e` 或 `--env` 标志传递环境变量来运行容器：

```bash
docker run \
  -e TRAINING_DATA_PATH=/data/train.csv \
  -e MODEL_OUTPUT_PATH=/output/model.joblib \
  -e LEARNING_RATE=0.005 \
  -e EPOCHS=50 \
  -v /path/on/host/data:/data \
  -v /path/on/host/models:/output \
  my-training-image 
  # 假设镜像的 CMD 或 ENTRYPOINT 运行 training_script_env.py
```

*选择参数与环境变量：*

- **参数：** 通常更适合与特定运行直接相关的事项，例如文件路径或在不同实验之间频繁更改的主要超参数 (hyperparameter)。它们在 `docker run` 命令中是明确的。
- **环境变量：** 适用于在多次运行中可能保持一致或在不同环境（开发/生产）之间存在差异的配置，例如数据库凭据、API 端点或公司标准期望的固定配置路径。它们也可以从文件（`--env-file`）加载。

#### 正确处理文件路径

无论路径是通过参数还是环境变量传递，脚本都需要正确使用它们。

- **使用提供的路径：** 始终使用从参数或环境变量填充的变量来引用数据和输出位置。
- **安全地拼接路径：** 如果需要构建子路径（例如，输出目录内的日志或检查点），请使用 `os.path.join()` 来确保跨平台兼容性（尽管在 Linux 容器内不那么关键，但这仍是良好习惯）。
- **创建输出目录：** 指定用于保存模型或日志的目录可能不存在，尤其是在映射卷时。在写入文件之前，使用 `os.makedirs(path, exist_ok=True)` 来确保目标目录存在。

#### 日志记录到标准流

配置 Python 的标准 `logging` 模块（或者在简单情况下只使用 `print`）来输出消息。除非绝对必要，否则避免在脚本内配置文件处理器。

```python
import logging
import sys

# 配置基本日志输出到标准输出
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    stream=sys.stdout) # 显式定向到标准输出

# 使用日志记录器
logging.info("Starting training process...")
# ... 执行训练步骤 ...
logging.warning("Encountered a minor issue...")
logging.info("Training finished.") 
```

Docker 会自动捕获 `stdout` 和 `stderr`，使得这些日志可以通过 `docker logs <container_id>` 访问。

通过根据这些原则组织您的训练脚本，您可以创建可移植、可配置并与 Docker 的数据管理和执行机制良好结合的代码。这为使用容器构建可重现和可扩展的机器学习训练流程奠定了基础，我们将在后续章节中讨论配置、执行和 GPU 使用时，在此基础上进行进一步阐述。

## 参考资料

- [Dockerfile reference](https://docs.docker.com/engine/reference/builder/) — Docker (2024)
  Docker Dockerfile 的官方文档，解释了 Dockerfile 的核心指令（如 CMD 和 ENTRYPOINT），以及对容器执行至关重要的卷挂载和环境变量等概念。
- [Effective MLOps: Create, Deploy, and Manage Production-Ready Machine Learning Systems](https://www.oreilly.com/library/view/effective-mlops-create/9781801072922/) — Vignesh Sukumaran (2021)
  Publisher: Packt Publishing
  一本关于生产就绪机器学习系统设计与管理的实用书籍，涵盖容器化、可复现性和部署策略等实践。
- [argparse - Parser for command-line options, arguments and sub-commands](https://docs.python.org/3/library/argparse.html) — Python Software Foundation (2024)
  Python argparse 模块的官方文档，详细介绍了如何为脚本创建用户友好的命令行接口，这对于容器化应用程序的参数化至关重要。
