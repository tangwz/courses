---
course: "data-versioning-experiment-tracking"
chapter: "integrating-dvc-mlflow"
lesson: "dvc-pipelines-mlflow-tracking"
sourceId: 4106
sourceUrl: "https://apxml.com/zh/courses/data-versioning-experiment-tracking/chapter-4-integrating-dvc-mlflow/dvc-pipelines-mlflow-tracking"
title: "结合 DVC 流水线与 MLflow 追踪"
description: "在 DVC 流水线阶段中加入 MLflow 日志记录，以便进行全面的记录和追踪。"
order: 7
plots: []
sourceHash: "551e2122f02ca9b0919fde2edadcc8dc958b205293719766eaa18dda7e8ad89a"
sourceCorrections: []
---

在 DVC 流水线阶段中直接嵌入 (embedding) MLflow 追踪功能，提供了一种有效的方法，可以在 DVC 流水线每次执行时自动收集详细的实验元数据。DVC 流水线（使用 `dvc.yaml` 或 `dvc run` 定义）用于创建自动化工作流程。这种结合将流水线的结构可复现性与 MLflow 丰富的追踪能力关联起来。

核心思路很简单：在 DVC 流水线阶段中执行的脚本或命令将包含用于日志记录的标准 MLflow API 调用。DVC 负责编排，确保阶段以正确的顺序运行，并使用正确的依赖项，而脚本本身则将其参数 (parameter)、指标和工件报告给您配置的 MLflow 追踪服务器。

### 在流水线阶段中嵌入 (embedding) MLflow 日志记录

考虑一个由 DVC 管理的典型机器学习 (machine learning)流水线，它可能在 `dvc.yaml` 文件中定义。你可能包含数据处理、训练和评估阶段。我们来关注训练阶段。

之前，你已学会如何使用 `dvc run` 或直接在 `dvc.yaml` 中定义一个阶段。这个阶段通常会执行一个脚本，例如 `train.py`。为了结合 MLflow，你需要修改这个脚本（`train.py`），使其包含 MLflow 日志记录调用。

以下是一个 `train.py` 脚本的简化示例，它旨在作为 DVC 流水线的一部分运行：

```python
# train.py
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import argparse

# 设置参数解析，用于接收来自 dvc.yaml 的参数
parser = argparse.ArgumentParser()
parser.add_argument('--n_estimators', type=int, default=100)
parser.add_argument('--max_depth', type=int, default=10)
parser.add_argument('--input_data', type=str, required=True)
parser.add_argument('--output_model', type=str, required=True)
args = parser.parse_args()

# 加载数据（由 DVC 管理依赖）
data = pd.read_csv(args.input_data)
X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 启动一个 MLflow 运行
# 如果在 Git 仓库中运行，MLflow 会自动检测 Git 提交
with mlflow.start_run():
    # 记录从 DVC 阶段定义接收的参数
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)
    # 记录有关输入数据的信息（由 DVC 追踪）
    mlflow.log_param("input_data_path", args.input_data)

    # 训练模型
    model = RandomForestClassifier(n_estimators=args.n_estimators,
                                   max_depth=args.max_depth,
                                   random_state=42)
    model.fit(X_train, y_train)

    # 评估模型
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # 记录指标
    mlflow.log_metric("accuracy", accuracy)
    print(f"准确率: {accuracy:.4f}")

    # 保存模型（输出由 DVC 管理）
    joblib.dump(model, args.output_model)
    print(f"模型已保存到 {args.output_model}")

    # 同时将模型工件记录到 MLflow
    # 这通过 MLflow UI/注册表提供了更丰富的模型管理
    mlflow.sklearn.log_model(model, "random-forest-model")

    # 如果需要，记录其他相关工件
    # 例如，特征重要性图、混淆矩阵
    # mlflow.log_artifact("feature_importance.png")

print("训练脚本执行完毕。")
```

接下来，我们来看看这个脚本如何被整合到 `dvc.yaml` 中的 DVC 流水线阶段：

```yaml
# dvc.yaml
stages:
  prepare:
    # ... 数据准备阶段定义 ...
    cmd: python src/prepare.py --input data/raw/data.csv --output data/prepared/features.csv
    deps:
      - data/raw/data.csv
      - src/prepare.py
    outs:
      - data/prepared/features.csv

  train:
    # 这个阶段运行我们的脚本并包含 MLflow 日志记录
    cmd: python src/train.py
           --input_data data/prepared/features.csv
           --output_model models/rf_model.joblib
           --n_estimators 150
           --max_depth 15
    deps:
      - data/prepared/features.csv
      - src/train.py
    params: # DVC 追踪这些参数
      - n_estimators
      - max_depth
    outs: # DVC 追踪这个输出文件
      - models/rf_model.joblib
    metrics: # DVC 也可以追踪主要指标
      - metrics.json: # 假设 train.py 也在此处输出指标（可选）
          cache: false
```

在此配置中：

1. `dvc.yaml` 文件定义了 `train` 阶段。
2. `cmd` 指定了如何执行 `train.py` 脚本，将 `n_estimators` 和 `max_depth` 等参数 (parameter)作为命令行参数传递。这些参数也可以在 `params.yaml` 文件中定义并在此处引用。
3. `deps` 列出了依赖项：准备好的数据文件和训练脚本本身。如果其中任何一个发生变化，`dvc repro` 就会知道需要重新运行此阶段。
4. `params` 明确告诉 DVC 追踪特定参数（例如，可能在 `params.yaml` 中定义的 `n_estimators`、`max_depth`）。这些参数的变化也会触发重新运行。
5. `outs` 列出了主要输出文件（`models/rf_model.joblib`），DVC 将追踪其哈希值。
6. 当你运行 `dvc repro train`（或仅 `dvc repro`）时，DVC 会执行 `train` 阶段的 `cmd` 命令。
7. `train.py` 脚本运行并执行 `mlflow.start_run()`、`mlflow.log_param()`、`mlflow.log_metric()` 和 `mlflow.sklearn.log_model()` 调用。
8. MLflow 将所有记录的信息（参数、指标、模型工件、Git 提交等）记录为一个新的运行，并存储到其后端（本地 `mlruns` 或远程服务器）。
9. DVC 完成阶段执行，并用输出文件 `models/rf_model.joblib` 的哈希值更新 `dvc.lock` 文件。

### 将流水线执行与实验关联起来

这种结合为您提供了强有力的关联：

- **DVC 确保流水线完整性**：它保证运行 `dvc repro` 时使用您的 Git 提交和 DVC 追踪定义的代码（`src/train.py`）和数据（`data/prepared/features.csv`）的正确版本。它管理执行流程并缓存输出。
- **MLflow 记录执行详情**：对于 `dvc repro` 触发的每次执行，MLflow 会记录使用的特定参数 (parameter)（即使是通过 `cmd` 传递的参数）、结果指标以及训练好的模型等相关工件。

当您在 MLflow UI 中查看实验历史时，您会看到与 DVC 流水线阶段每次执行相对应的运行。因为 MLflow 通常会自动记录与运行相关的 Git 提交哈希值，您可以将 MLflow 运行直接关联到生成它的 DVC 管理仓库（代码、`dvc.yaml`、`dvc.lock`、`params.yaml`）的特定状态。

### 整合方法的益处

- **自动化追踪**：每次流水线通过 `dvc repro` 运行时，实验详情都会自动记录，无需手动干预。
- **完整的来源信息**：您将 DVC 流水线的结构定义和依赖项与 MLflow 中详细的运行元数据关联起来。您确切地知道是哪个数据版本、代码版本和参数 (parameter)产生了特定的结果。
- **增强的比较**：使用 MLflow UI 比较由参数（`params.yaml`）、代码（`src/train.py`）或数据依赖项变化而触发的不同流水线运行。
- **简化工作流程**：开发人员主要使用 Git 和 DVC 命令（`git commit`、`dvc repro`），实验追踪作为流水线执行的自然副产品而发生。

通过在 DVC 流水线阶段执行的脚本中嵌入 (embedding) MLflow 调用，您创建了一个系统，其中 DVC 管理的可复现性得到了 MLflow 详细追踪和比较能力的增强，从而使机器学习 (machine learning)项目更易于管理和理解。

## 参考资料

- [DVC Documentation](https://dvc.org/doc) — Iterative.ai (2024)
  数据版本控制（DVC）的官方文档，提供了关于DVC管道、数据管理和可复现性的指南。
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html) — MLflow Project (2024)
  MLflow跟踪组件的官方指南，详细介绍了如何记录参数、指标、工件和模型，这是本节内容的核心。
- [MLflow Experiment Tracking with DVC](https://dvc.org/doc/use-cases/mlflow-experiment-tracking) — Iterative.ai (2024)
  此官方指南提供了将MLflow跟踪与DVC管道集成的说明和示例，直接涉及本节的核心主题。
