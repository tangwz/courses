---
course: "data-versioning-experiment-tracking"
chapter: "integrating-dvc-mlflow"
lesson: "logging-dvc-metadata-mlflow"
sourceId: 4098
sourceUrl: "https://apxml.com/zh/courses/data-versioning-experiment-tracking/chapter-4-integrating-dvc-mlflow/logging-dvc-metadata-mlflow"
title: "在 MLflow 中记录 DVC 元数据"
description: "自动将 DVC 数据版本或流水线输出记录为 MLflow 运行中的参数或标签的方法。"
order: 3
plots: []
sourceHash: "865a9c136fe6878baf45f8767cb17910165595258d940754031da0d878c125da"
sourceCorrections: []
---

实现机器学习 (machine learning)项目的完整追溯能力，需要将数据版本控制和实验追踪紧密结合，而不仅仅是分别管理。DVC 负责数据版本控制，MLflow 负责实验追踪。为了获得完整的记录，必须将具体的数据版本与每个实验运行关联起来。MLflow 提供了灵活的机制，可以直接记录与 DVC 相关的信息，并将其与标准的实验参数 (parameter)和指标一同保存。

### 为何关联数据版本与实验？

设想您训练了两个模型，它们产生了不同的结果。这种差异是由于超参数 (parameter) (hyperparameter)、代码还是基础数据的变化造成的？如果未明确关联每个 MLflow 运行中使用的数据版本，回答这个问题就会变得困难，并且需要依赖手动记录或猜测。将 DVC 元数据直接记录到 MLflow 中，可以建立清晰、自动化的关联，确保您始终可以将实验结果追溯到产生它们的准确数据快照。这提高了可复现性，并简化了调试和比较。

### 应记录哪些 DVC 信息？

有几项 DVC 元数据在 MLflow 运行中记录会很有帮助：

- **数据路径：** 运行中使用的特定数据集文件或目录的路径（例如，`data/processed_features`）。这确认了预期使用的数据集文件。
- **数据版本哈希：** 代表数据特定版本的唯一标识符（哈希）。DVC 将此哈希存储在相应的 `.dvc` 文件中，或者为目录计算此哈希。这可以说是确保数据可复现性的最重要信息，因为它唯一标识了数据文件的状态。
- **远程存储 URL（可选）：** 了解特定数据版本存储在哪里（例如，S3 桶路径）有时对理解情况有帮助，特别是在分布式团队或复杂的存储设置中。

### 记录 DVC 元数据的方法

您可以使用几种方法将这些信息整合到您的 MLflow 运行中。请选择最符合您的工作流复杂性和自动化需求的方法。

#### 使用 MLflow API 手动记录

最直接的方法是在您的训练脚本中添加对 `mlflow.log_param()` 或 `mlflow.set_tag()` 的调用。您将相关的 DVC 信息明确地作为参数 (parameter)或标签传递。标签通常更适合路径或哈希等标识符，而参数通常保留给可能影响模型行为的数值（如超参数 (hyperparameter)）。

例如，如果您的脚本从 DVC 追踪的目录 `data/prepared` 访问数据，您可以将其路径记录为标签：

```python
import mlflow
import os

# 假设 'data_path' 指向您的 DVC 追踪的数据
data_path = "data/prepared" 

with mlflow.start_run():
    # 将路径作为标签记录，用于信息目的
    mlflow.set_tag("data_path", data_path)

    # 记录其他参数
    mlflow.log_param("learning_rate", 0.01)

    # ... 训练代码的其余部分，使用来自 data_path 的数据 ...

    print(f"MLflow 运行 ID: {mlflow.active_run().info.run_id}")
    print(f"已记录的 data_path 标签: {data_path}")
```

尽管简单，手动记录哈希要求您事先知道哈希，或者通过检查 `.dvc` 文件手动提取它，这对于数据版本可能频繁更改的自动化工作流来说并不理想。

#### 使用 DVC Python API

一种更自动化的方法是使用 `dvc.api` 模块。这使得您的 Python 脚本能够以编程方式查询 DVC 获取有关追踪文件或目录的信息，而无需通过命令行操作。

首先，请确保您已安装必要的 DVC 组件。基础的 `dvc` 包可能就足够了，或者根据您与远程交互的方式或是否需要特定 API 功能，您可能需要额外组件：

```bash
# 如果尚未安装，则安装基础 DVC
pip install dvc

# 可选：如果需要，安装 API 额外组件或特定远程组件
# pip install dvc[api] 
# pip install dvc[s3] # S3 远程交互示例
```

尽管 `dvc.api` 提供了 `get_url`（用于获取缓存路径或远程 URL）或 `read`（用于读取文件内容）等函数，但以编程方式获取 `.dvc` 文件中记录的特定*版本哈希*则需要更多工作。一种常见的实用方法是直接读取和解析 `.dvc` 文件，因为它们通常是小型文本文件（通常为 YAML 或 JSON）。

让我们通过从 `data/features.csv.dvc` 读取哈希来演示，假设它是一个由 DVC 追踪的单输出文件：

```python
import mlflow
import os
import yaml # 用于解析 .dvc 文件（假设为 YAML 格式）
import json # 如果您的 .dvc 文件是 JSON 格式，则使用 JSON

# 指向代表您的数据集版本的 .dvc 文件路径
dvc_file_path = "data/features.csv.dvc" 
# 脚本实际使用的数据路径
data_path = "data/features.csv" 

data_version_hash = None

# 检查 .dvc 文件是否存在
if os.path.exists(dvc_file_path):
    try:
        with open(dvc_file_path, 'r') as f:
            # 尝试作为 YAML 加载，如果需要，回退到 JSON 或处理纯文本
            try:
                dvc_content = yaml.safe_load(f)
            except yaml.YAMLError:
                f.seek(0) # 重置文件指针
                try:
                    dvc_content = json.load(f)
                except json.JSONDecodeError:
                    print(f"警告: 无法将 {dvc_file_path} 解析为 YAML 或 JSON。")
                    dvc_content = {} # 赋空字典以避免下方出错

            # 提取哈希：结构取决于 DVC 版本和配置
            # 常见位置：'outs' 列表 -> 第一个项目 -> 'hash' (DVC >= 3.0) 或 'md5'
            if 'outs' in dvc_content and isinstance(dvc_content['outs'], list) and len(dvc_content['outs']) > 0:
                 output_info = dvc_content['outs'][0]
                 if isinstance(output_info, dict):
                     if 'hash' in output_info: # DVC >= 3.0 使用 'hash'
                         data_version_hash = output_info.get('hash')
                     elif 'md5' in output_info: # 较旧的 DVC 版本使用 'md5'
                         data_version_hash = output_info.get('md5')

            if not data_version_hash:
                 print(f"警告: 未在 {dvc_file_path} 中找到哈希（'hash' 或 'md5'）。")

    except Exception as e:
        print(f"警告: 读取 DVC 文件 {dvc_file_path} 时出错: {e}")

# 启动 MLflow 运行并记录信息
with mlflow.start_run():
    mlflow.set_tag("data_path", data_path) # 记录使用的数据路径

    if data_version_hash:
        # 将哈希作为标签记录 - 它是一个标识符
        mlflow.set_tag("dvc_data_version_hash", data_version_hash) 
    else:
        mlflow.set_tag("dvc_data_version_status", "Hash unavailable or not found")

    # 照常记录其他参数
    mlflow.log_param("learning_rate", 0.01)

    # ... 训练代码的其余部分，使用 data_path ...

    run_id = mlflow.active_run().info.run_id
    print(f"MLflow 运行 ID: {run_id}")
    if data_version_hash:
      print(f"已记录 DVC 数据版本哈希: {data_version_hash}")
    else:
      print("DVC 数据版本哈希未被记录。")
```

> **注意：** `.dvc` 文件的结构会演变。上述解析逻辑涵盖了常见的 YAML/JSON 格式，但可能需要根据不同的 DVC 版本或配置（例如，多输出文件、不同的哈希类型如 `etag`）进行调整。务必检查您的 `.dvc` 文件以确认其结构。对于哈希和路径等标识符，通常更推荐使用标签（`mlflow.set_tag`）。

#### 通过配置或环境变量记录

另一种策略是，通过配置文件（例如，`config.yaml`）或环境变量，将数据路径甚至描述性版本标识符（例如，与所需数据版本关联的 Git 标签）传递给您的脚本。您的脚本随后简单地读取此配置值并记录它，使用 `mlflow.log_param` 或 `mlflow.set_tag`。这种方法将特定版本信息与核心训练逻辑分离，使脚本更具可重用性。

```python
# 示例：从环境变量读取数据路径和版本标签
import mlflow
import os

# 从环境变量读取配置，提供默认值
data_path = os.getenv("INPUT_DATA_PATH", "data/features.csv") 
data_version_tag = os.getenv("DATA_VERSION_TAG", "unknown") # 例如，“v1.2-processed”

with mlflow.start_run():
    # 记录本次运行使用的配置
    mlflow.set_tag("configured_data_path", data_path)
    mlflow.set_tag("configured_data_version_tag", data_version_tag)

    # 记录其他参数
    mlflow.log_param("batch_size", 64)

    # ... 训练代码从 data_path 加载数据 ...
    print(f"使用数据源: {data_path}")
    print(f"假定的数据版本标签: {data_version_tag}")

    # ... 训练的其余部分 ...
```

然后，您将在执行脚本之前设置这些环境变量：

```bash
# 为本次运行设置环境变量
export INPUT_DATA_PATH="data/processed_features_v2"
export DATA_VERSION_TAG="release-2024-q1"

# 运行训练脚本
python train.py 
```

此方法依赖于*运行*脚本的进程（例如，CI/CD 流水线、DVC 阶段或手动执行）来提供与检出数据版本对应的正确环境变量。

### 将记录集成到您的工作流中

添加此 DVC 元数据记录的理想位置是在您的训练或处理脚本的早期，通常在初始化 MLflow 运行（`mlflow.start_run()`）后立即进行，并且通常作为您的数据加载或参数 (parameter)设置阶段的一部分。这确保了在主要计算工作开始之前，捕获实验运行与数据版本之间重要的关联。

通过在 MLflow 运行中持续记录您的 DVC 追踪数据的路径，更重要的是，其版本哈希或一个有意义的标签，您可以在实验与所使用的准确数据文件之间创建明确且可验证的关联。这大大增强了可复现性，让您和您的团队能够自信地回顾过去的实验，理解所有输入（代码、参数*和*数据），并可靠地复现结果或调试差异。这种做法将您的 MLflow 实验追踪从单纯关注模型性能指标，转变为为您的整个 ML 工作流提供全面的来源记录。

## 参考资料

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html) — MLflow Project (2024)
  提供 MLflow 实验追踪、日志 API 和模型管理的详细信息。
- [DVC Documentation](https://dvc.org/doc) — Iterative, Inc. (2024)
  解释 DVC 在数据版本控制、管道管理和远程存储集成方面的功能。
- [Machine Learning Engineering](https://books.google.com/books/about/Machine_Learning_Engineering.html?id=z1bPDwAAQBAJ) — Andriy Burkov (2020)
  Publisher: True Positive Incorporated; Pages: 310
  一本权威指南，涵盖 MLOps 原则，包括实验追踪和数据处理的重要性。
