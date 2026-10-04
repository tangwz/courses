---
course: "data-versioning-experiment-tracking"
chapter: "tracking-experiments-mlflow"
lesson: "organizing-mlflow-runs-experiments"
sourceId: 4087
sourceUrl: "https://apxml.com/zh/courses/data-versioning-experiment-tracking/chapter-3-tracking-experiments-mlflow/organizing-mlflow-runs-experiments"
title: "使用实验管理运行"
description: "通过将相关运行分组到 MLflow 实验中来组织你的跟踪工作。"
order: 6
plots: []
sourceHash: "291aa1b52c3015b39c75e66ac22fa7ad01f8f1f368f99f854794dd0de79ee9df"
sourceCorrections: []
---

使用 `mlflow.log_param()`、`mlflow.log_metric()` 和 `mlflow.log_artifact()` 等函数跟踪多个机器学习 (machine learning)试验时，运行记录集合会迅速增长。设想你正在训练数十个用于预测客户流失的模型，每个模型都使用略有不同的超参数 (parameter) (hyperparameter)或特征集。简单地将所有这些运行列在一起，将难以找出与特定目标或方法相关的运行。MLflow 提供了一种简单的组织结构来处理这种复杂情况：**实验**。

MLflow 实验本质上是一个命名的容器，用于存放一组相关运行。你可以把它看作是一个文件夹，或者一个专门用于你正在研究的特定机器学习问题或目标的“项目空间”。例如，你可以为以下目的创建独立的实验：

- `customer-churn-prediction` (客户流失预测)
- `product-recommendation-engine` (产品推荐引擎)
- `sentiment-analysis-reviews` (评论情感分析)

所有与开发客户流失预测模型相关的运行都会被记录在 `customer-churn-prediction` 实验下，将它们与产品推荐相关的运行区分开来。

### 默认行为

当你开始记录运行而没有明确指定实验时，MLflow 会使用一个默认实验。如果你记录到本地 `mlruns` 目录（默认后端），MLflow 会创建一个名为 `Default`、ID 为 `0` 的实验。虽然对于快速测试来说是可用的，但仅依赖默认实验会很快导致来自可能不相关项目的运行记录变得杂乱无章。

### 创建和设置实验

为了有效组织你的工作，你应该明确地创建并分配实验。你可以使用 MLflow Python API 管理实验。

#### 创建实验

要以编程方式创建新实验，请使用 `mlflow.create_experiment()` 函数。它需要一个实验名称，并可选地接受一个 `artifact_location`。如果你不提供 `artifact_location`，MLflow 将使用相对于跟踪后端的一个默认位置（例如，对于本地文件系统后端，是 `mlruns` 内的一个子目录）。

```python
import mlflow

# 定义新实验的名称
experiment_name = "house-price-prediction-xgboost"

try:
    # 创建实验。它返回创建的实验的ID。
    experiment_id = mlflow.create_experiment(name=experiment_name)
    print(f"实验 '{experiment_name}' 已创建，ID 为: {experiment_id}")
except mlflow.exceptions.MlflowException as e:
    # 处理实验可能已存在的情况
    print(f"实验 '{experiment_name}' 已存在。")
    # （可选）获取现有实验的ID
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
    print(f"正在使用现有实验ID: {experiment_id}")
```

这段代码尝试创建一个实验。如果同名实验已存在，它会捕获异常并使用 `mlflow.get_experiment_by_name()` 获取现有实验的ID。

#### 为运行设置活动实验

一旦你有了实验（无论是新创建的还是已有的），你需要告诉 MLflow 将后续运行记录在该实验的上下文 (context)中。以编程方式执行此操作的主要方法是使用 `mlflow.set_experiment()`。你可以提供实验名称或实验ID。如果你提供的名称不对应任何现有实验，`mlflow.set_experiment()` 会自动为你创建它。

```python
import mlflow
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

# 定义实验名称
experiment_name = "simple-linear-regression"

# 设置活动实验。如果不存在，则创建。
mlflow.set_experiment(experiment_name)

# 样本数据生成（请替换为你的实际数据加载）
X = np.random.rand(100, 1) * 10
y = 2.5 * X.squeeze() + np.random.randn(100) * 2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 在活动实验中启动一个 MLflow 运行
with mlflow.start_run(run_name="baseline_run") as run:
    print(f"正在实验 {run.info.experiment_id} 中启动运行 {run.info.run_id}")

    # 实例化并训练模型
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # 进行预测
    predictions = lr.predict(X_test)

    # 计算指标
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    # 记录参数和指标
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("train_test_split_random_state", 42)
    mlflow.log_metric("rmse", rmse)

    print(f"已记录均方根误差 (RMSE): {rmse}")
    print(f"运行已完成。在 MLflow UI 中查看详情。")
```

在这个例子中，`mlflow.set_experiment("simple-linear-regression")` 确保了后续的 `mlflow.start_run()` 块会将参数 (parameter)和指标记 (token)录在 “simple-linear-regression” 实验下。如果在运行脚本之前该实验不存在，MLflow 会自动创建它。

另外，你可以使用 `experiment_id` 参数直接在 `mlflow.start_run()` 中指定实验，但这要求实验必须事先存在。

### 在 MLflow UI 中可视化实验

将运行组织到实验中的真正好处在使用 MLflow UI 时会变得很明显。当你启动 UI 时（通常是通过在包含 `mlruns` 文件夹的目录中运行 `mlflow ui` 命令），你会在左侧导航面板中看到你的实验列表。

> MLflow 中的层次结构：UI 显示实验，每个实验包含多个运行。

选择一个实验会过滤主视图，只显示与该实验相关的运行。这使你能够：

1. **聚焦：** 专注于与特定问题相关的结果。
2. **比较：** 使用 UI 的比较功能（例如，参数 (parameter)和指标图）轻松比较同一实验中的运行。
3. **导航：** 快速找到与特定任务相关的过往工作。

### 命名约定和最佳实践

尽管 MLflow 不强制命名规则，但采用一致的约定有助于保持清晰度：

- **基于项目：** 根据实验所属的项目或组件命名实验（例如，`fraud-detection-v1`，`recommendation-api-model`）。
- **基于任务：** 如果一个项目涉及不同的任务，请使用反映任务的名称（例如，`churn-prediction-feature-engineering`，`churn-prediction-hyperparameter-tuning`）。
- **描述性：** 使用能清楚表明其中运行目的的名称。避免使用 `test` 或 `my-experiment` 等通用名称。

通过审慎地将运行归入实验，你将 MLflow 从一个简单的记录器转变为一个强大的组织工具。这种结构化的方法对于管理模型开发的迭代过程非常重要，有助于更轻松地分析、比较和重现你的机器学习 (machine learning)工作。

## 参考资料

- [MLflow Tracking Concepts](https://mlflow.org/docs/latest/concepts.html) — MLflow Authors (2024)
  官方文档，解释了MLflow追踪的基本概念，包括实验在组织机器学习运行中的作用。
- [Practical MLOps: How to Get Your Machine Learning Models into Production](https://www.oreilly.com/library/view/practical-mlops/9781098103002/) — Nicholas Renotte (2022)
  Publisher: O'Reilly Media
  MLOps 实践指南，提供了关于为什么实验追踪和结构化运行组织是稳健机器学习工作流程必要组成部分的背景。
