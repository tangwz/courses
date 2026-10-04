---
course: "docker-for-ml-projects"
chapter: "containerizing-ml-training"
lesson: "practice-containerize-run-training"
sourceId: 5308
sourceUrl: "https://apxml.com/zh/courses/docker-for-ml-projects/chapter-4-containerizing-ml-training/practice-containerize-run-training"
title: "动手实践：容器化并运行训练脚本"
description: "将一个机器学习训练脚本示例打包成Docker镜像，并以容器形式运行，同时管理数据和输出。"
order: 7
plots: []
sourceHash: "995422efda7ac642ed43e5478beb9d4d3ec39a3dd63c0568a85100c0094fe2de"
sourceCorrections: []
---

将一个简单的机器学习 (machine learning)训练脚本容器化，为其构建一个Docker镜像，并在容器内执行训练过程。数据输入和模型输出使用绑定挂载进行管理。这种方法有助于巩固创建可复现训练环境的过程。

### 前提条件

- Docker已安装并在您的系统上运行。
- 本地已安装Python 3（用于准备示例）。
- 一个文本编辑器或IDE。

### 训练脚本（示例：train.py）

首先，我们来创建一个基本的Python脚本，用于在Iris数据集上训练一个Scikit-learn逻辑回归模型。我们会将其设计为通过命令行参数 (parameter)接收输入/输出目录路径和一个超参数 (hyperparameter)（正则化 (regularization)强度`C`）。

将以下代码保存为`src/train.py`:

```python
# src/train.py
import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

def train_model(input_dir, output_dir, C):
    """加载数据，训练模型，并保存模型。"""
    print(f"Loading data from: {input_dir}")
    # 假设 iris.csv 位于输入目录中
    data_path = os.path.join(input_dir, 'iris.csv')
    try:
        iris_df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Could not find {data_path}. Make sure iris.csv is mounted correctly.")
        return

    print("数据加载成功。")
    X = iris_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = iris_df['species']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"正在使用 C={C} 训练逻辑回归模型...")
    model = LogisticRegression(C=C, max_iter=200, random_state=42)
    model.fit(X_train, y_train)

    # 评估
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"模型在测试集上的准确率: {acc:.4f}")

    # 保存模型
    os.makedirs(output_dir, exist_ok=True) # 确保输出目录存在
    model_path = os.path.join(output_dir, 'iris_model.joblib')
    joblib.dump(model, model_path)
    print(f"模型已保存到: {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="在Iris数据上训练一个逻辑回归模型。")
    parser.add_argument('--input-dir', type=str, required=True, help='包含iris.csv的目录')
    parser.add_argument('--output-dir', type=str, required=True, help='用于保存训练后模型的目录')
    parser.add_argument('--C', type=float, default=1.0, help='正则化强度的倒数')

    args = parser.parse_args()

    train_model(args.input_dir, args.output_dir, args.C)
```

此脚本使用`argparse`处理命令行参数，使用pandas加载数据，使用Scikit-learn训练模型，并使用`joblib`保存训练后的模型。它明确要求提供输入数据和输出模型的路径。

### 项目结构

您的项目文件应按以下方式组织：

```
ml-training-project/
├── Dockerfile
├── requirements.txt
├── src/
│   └── train.py
├── data/
│   └── iris.csv  # 您需要下载/创建此文件
└── output/       # 此目录将被创建或用于模型输出
```

您可以在线轻松找到Iris数据集（例如，从Kaggle或UCI机器学习 (machine learning)库），或者创建一个示例`iris.csv`文件并将其放入`data/`目录。请确保它包含`sepal_length`、`sepal_width`、`petal_length`、`petal_width`和`species`等列。

### 依赖项（`requirements.txt`）

创建一个`requirements.txt`文件，列出所需的Python库：

```
pandas
scikit-learn==1.2.2  # 固定版本以确保可复现性
joblib
```

*注意：固定版本（例如`scikit-learn==1.2.2`）是确保可复现性的一种良好做法。*

### 创建Dockerfile

现在，在项目根目录（`ml-training-project/`）中创建`Dockerfile`：

```dockerfile
# 使用官方Python运行时作为基础镜像
FROM python:3.9-slim

# 设置容器内的工作目录
WORKDIR /app

# 先复制requirements文件以利用Docker缓存
COPY requirements.txt .

# 安装requirements.txt中指定的任何所需软件包
# 使用 --no-cache-dir 减少镜像大小
RUN pip install --no-cache-dir -r requirements.txt

# 将源代码复制到容器中
COPY src/ ./src/

# 定义容器的入口点
# 这使得容器像一个可执行文件一样运行
ENTRYPOINT ["python", "src/train.py"]

# 默认命令（可以在运行时覆盖）
# 这里，如果没有提供参数，我们设置默认的help标志
CMD ["--help"]
```

让我们分析此Dockerfile：

- `FROM python:3.9-slim`: 使用轻量级的Python 3.9镜像作为起点。
- `WORKDIR /app`: 将容器内的默认目录设置为`/app`。随后的命令（`COPY`、`RUN`、`CMD`、`ENTRYPOINT`）将相对于此目录运行。
- `COPY requirements.txt .`: 仅复制requirements文件。
- `RUN pip install ...`: 安装依赖项。如果`requirements.txt`没有变化，此层通常会被缓存，从而加快后续构建的速度。
- `COPY src/ ./src/`: 将我们的训练脚本目录复制到镜像中的`/app/src/`下。
- `ENTRYPOINT ["python", "src/train.py"]`: 指定使用此镜像运行的容器默认会执行`python src/train.py`。在镜像名称之后提供给`docker run`的任何参数 (parameter)都将附加到此命令。
- `CMD ["--help"]`: 为`ENTRYPOINT`提供一个默认参数。如果您在不带参数的情况下运行容器，它将执行`python src/train.py --help`。

### 构建Docker镜像

在您的终端中导航到项目根目录（`ml-training-project/`）并运行构建命令：

```bash
docker build -t ml-training-app:latest .
```

- `-t ml-training-app:latest`: 为镜像添加名为`ml-training-app`和标签`latest`的标记 (token)。
- `.`: 指定构建上下文 (context)（Docker查找`Dockerfile`和要复制文件的位置）是当前目录。

Docker将执行您`Dockerfile`中的步骤，下载基础镜像，安装依赖项，并复制您的代码。

### 运行容器化训练

现在，我们将在容器内部运行训练脚本。我们需要做到：

1. 将本地`data`目录挂载到容器的`/app/data`路径，以便脚本可以读取`iris.csv`。
2. 将本地`output`目录挂载到容器的`/app/output`路径，以便脚本可以将`iris_model.joblib`文件保存回我们的宿主机。
3. 将所需的命令行参数 (parameter)（`--input-dir`、`--output-dir`以及可选的`--C`）传递给容器，这些参数将附加到`ENTRYPOINT`。

从项目根目录执行以下`docker run`命令：

```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/output:/app/output" \
  ml-training-app:latest \
  --input-dir /app/data \
  --output-dir /app/output \
  --C 0.5
```

让我们分析此命令：

- `docker run`: 用于创建并启动新容器的命令。
- `--rm`: 容器退出时自动删除容器。这对于训练等一次性任务很有用。
- `-v "$(pwd)/data:/app/data"`: 将宿主机的当前目录（`$(pwd)`）加上`/data`挂载到容器内的`/app/data`目录。在Windows PowerShell上使用`${PWD}`或在需要时将`$(pwd)`替换为完整路径。这使得`iris.csv`可供脚本使用。
- `-v "$(pwd)/output:/app/output"`: 将宿主机的`output`目录（如果不存在，Docker将创建它）挂载到容器内的`/app/output`。这是脚本保存模型的位置。
- `ml-training-app:latest`: 容器要使用的镜像。
- `--input-dir /app/data --output-dir /app/output --C 0.5`: 这些是传递给`ENTRYPOINT`（`python src/train.py`）的参数。请注意，这里我们使用的是*容器路径*（`/app/data`、`/app/output`），而不是宿主机路径。我们还指定了一个超参数 (hyperparameter)`C=0.5`。

您应该在终端中看到类似以下的输出：

```
从: /app/data 加载数据
数据加载成功。
正在使用 C=0.5 训练逻辑回归模型...
模型在测试集上的准确率: 1.0000
模型已保存到: /app/output/iris_model.joblib
```

### 验证输出

容器运行结束后，检查您的本地`output`目录。您应该会发现`iris_model.joblib`文件，它是从容器内部成功保存的。

```bash
ls output/
```

```
iris_model.joblib
```

### 总结

在本次实践练习中，您成功地：

1. 创建了一个Python训练脚本，通过接受输入/输出路径，使其设计为在容器化环境中工作。
2. 编写了一个`Dockerfile`来定义环境、安装依赖项并复制训练代码。
3. 构建了一个包含训练脚本及其环境的Docker镜像。
4. 使用`docker run`在隔离的容器中执行了训练过程。
5. 使用绑定挂载（`-v`）管理了数据输入和模型输出，将宿主机目录与容器目录关联起来。
6. 通过命令行参数 (parameter)向脚本传递了配置（超参数 (hyperparameter)）。

此过程演示了机器学习 (machine learning)训练容器化的核心工作流程。通过将代码和依赖项打包在一起，您可以确保训练环境的一致性和可复现性，无论Docker镜像在哪里运行。使用卷允许与宿主机文件系统进行数据和结果的交互，弥合了隔离容器与外部环境之间的差异。

## 参考资料

- [Docker Documentation: Get Started, Overview, and Dockerfile Reference](https://docs.docker.com/) — Docker Inc. (2024)
  官方文档提供了理解Docker概念、Dockerfile语法、镜像构建和带卷容器运行时的基础知识。
- [Scikit-learn: Machine Learning in Python](https://scikit-learn.org/stable/) — Scikit-learn Developers (2025)
  Scikit-learn库的官方用户指南和API参考，该库用于训练脚本中的机器学习模型。
- [Practical MLOps: How to Get Your Machine Learning Models into Production](https://www.oreilly.com/library/view/practical-mlops/9781098110300/) — Noah Gift, Alfred DeSimone (2022)
  Publisher: O'Reilly Media
  一本实践指南，其中包含如何使用Docker进行容器化，以创建可复现的机器学习训练和部署工作流。
