---
course: "data-versioning-experiment-tracking"
chapter: "tracking-experiments-mlflow"
lesson: "logging-params-metrics-mlflow"
sourceId: 4083
sourceUrl: "https://apxml.com/zh/courses/data-versioning-experiment-tracking/chapter-3-tracking-experiments-mlflow/logging-params-metrics-mlflow"
title: "记录参数和指标"
description: "使用 MLflow API 从训练脚本中记录超参数和评估指标。"
order: 4
plots: ["plots/4083-0.json"]
sourceHash: "1cb8fdd7e337f5f8a2a1f5bac3f7a42b14bb1f24f4f1d3f274826e6dddd7a354"
sourceCorrections: []
---

要使用 MLflow 有效地跟踪机器学习 (machine learning)实验，调整代码以记录每次训练运行的重要信息至关重要。实验跟踪的中心是*参数 (parameter)*和*指标*。参数代表运行的输入配置，例如超参数 (hyperparameter)或特征选择。指标代表输出或结果，通常是评估分数或损失值，用于衡量运行的效果。

MLflow 提供简洁的 Python API 来记录这些信息。您会用到两个主要函数：`mlflow.log_param()` 和 `mlflow.log_metric()`。

### 记录参数 (parameter)

参数是您在开始训练运行*之前*定义的设置。它们通常包括：

- **超参数 (hyperparameter)：** 学习率、批大小、层数、正则化 (regularization)强度 (alpha)、树深度等。
- **输入特征：** 哪些特征用于训练。
- **环境信息：** 有时记录库版本或数据集标识符很有用（尽管我们稍后会看到更完善的方法来关联数据版本）。

参数通常在 MLflow 运行开始时记录一次。您可以使用 `mlflow.log_param(key, value)` 记录单个参数，其中 `key` 是参数的字符串名称，`value` 是其值（支持字符串、数字或布尔类型）。

要一次记录多个参数，您采用 `mlflow.log_params(params_dict)`，此处的 `params_dict` 是一个包含参数名称和值的字典。

我们来看看实际操作中是怎样的。假设您正在训练一个分类模型，您可能希望记录正则化强度 (`alpha`) 和所用的求解器类型。

```python
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# 加载数据（示例）
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定义参数
alpha_val = 0.1
solver_type = 'liblinear'

# 启动一个 MLflow 运行
with mlflow.start_run():
    # 记录参数
    mlflow.log_param("alpha", alpha_val)
    mlflow.log_param("solver", solver_type)
    print(f"记录参数: alpha={alpha_val}, solver={solver_type}")

    # 实例化并训练模型
    lr = LogisticRegression(C=1/alpha_val, solver=solver_type, random_state=42)
    lr.fit(X_train, y_train)

    # 进行预测
    y_pred = lr.predict(X_test)

    # 计算准确率（指标）
    accuracy = accuracy_score(y_test, y_pred)
    print(f"模型准确率: {accuracy:.4f}")

    # 记录指标（下文会讲到）
    mlflow.log_metric("accuracy", accuracy)

print("MLflow 运行完成。")
```

在这个代码片段中，在 `mlflow.start_run()` 上下文 (context)内部，我们明确地为 `alpha` 和 `solver` 调用了 `mlflow.log_param()`。这些值现在将与 MLflow 跟踪系统中的此特定运行关联起来。

### 记录指标

指标是运行的结果数值，用于衡量性能或表现。常见示例如下：

- **评估指标：** 准确率、精确率、召回率、F1 分数、AUC、均方误差 (MSE)、R 平方。
- **损失值：** 训练损失、验证损失。
- **资源使用：** 训练时间、GPU 内存使用（对于基本跟踪不太常见，但可以记录）。

指标可以在运行期间的任何时候使用 `mlflow.log_metric(key, value, step=None)` 记录。

- `指标名称`：指标的字符串名称。
- `值`：指标的数值。
- `step`：一个可选整数，表示序列或时间步长（例如 epoch 编号）。如果提供，MLflow 将记录指标在这些步长中的历史数据。

与参数 (parameter)类似，您采用一个字典使用 `mlflow.log_metrics(metrics_dict, step=None)` 来记录多个指标。

让我们扩展之前的示例，来记录最终的准确率分数：

```python
# （之前的代码：导入、数据加载、参数定义）

# 启动一个 MLflow 运行
with mlflow.start_run():
    # 记录参数
    mlflow.log_param("alpha", alpha_val)
    mlflow.log_param("solver", solver_type)

    # 实例化并训练模型
    lr = LogisticRegression(C=1/alpha_val, solver=solver_type, random_state=42)
    lr.fit(X_train, y_train)

    # 进行预测
    y_pred = lr.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"模型准确率: {accuracy:.4f}")

    # 记录最终的准确率指标
    mlflow.log_metric("accuracy", accuracy)
    print(f"已记录指标: accuracy={accuracy:.4f}")

print("MLflow 运行完成。")
```

这里，`mlflow.log_metric("accuracy", accuracy)` 记录了最终的性能分数。

#### 随时间记录指标

通常，您会希望跟踪指标在训练期间的变化情况，例如每个 epoch 后的损失。`step` 参数就是为此设计的。

考虑一个简化的训练循环，我们模拟基于 epoch 的训练，并在每个步长记录损失：

```python
import mlflow
import time
import random

# 启动一个 MLflow 运行
with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("epochs", 5)

    print("模拟训练循环中...")
    # 模拟多个 epoch 的训练
    initial_loss = 1.0
    for epoch in range(5):
        # 模拟训练过程
        time.sleep(0.5)
        # 计算模拟损失（随机递减）
        current_loss = initial_loss * (1 - random.uniform(0.1, 0.3))
        initial_loss = current_loss

        # 记录该 epoch 的损失指标
        mlflow.log_metric("train_loss", current_loss, step=epoch)
        print(f"Epoch {epoch}: 已记录 train_loss={current_loss:.4f}")

    # 记录最终指标（例如验证准确率）
    final_val_accuracy = 0.85 + random.uniform(-0.05, 0.05)
    mlflow.log_metric("validation_accuracy", final_val_accuracy)
    print(f"最终 validation_accuracy={final_val_accuracy:.4f}")

print("MLflow 运行完成。")
```

在这个示例中，`mlflow.log_metric("train_loss", current_loss, step=epoch)` 记录了每个 epoch 的损失值。当您在 MLflow UI 中查看此运行记录时，您将看到一个图表，显示训练损失如何随 epoch 递减。



![每 Epoch 模拟训练损失](plots/4083-0.json)



> 示例：使用步长记录的训练损失在 MLflow UI 中可能显示的效果。

通过持续记录参数和指标，您为每个实验创建了详细记录。这极大地方便了您理解哪些设置产生了哪些结果，比较不同运行，并在之后重现成功的成果。使用这些函数记录的信息成为在 MLflow UI 中分析实验的依据，我们将在后续章节中研究这些内容。

## 参考资料

- [MLflow Tracking](https://www.mlflow.org/docs/latest/tracking.html) — MLflow Documentation (2024)
  Publisher: The MLflow Project
  MLflow的官方文档，介绍了其实验跟踪功能，包括记录参数、指标和其他工件的详细API参考。
- [MLOps Engineering at Scale: A Guide to Building and Deploying Machine Learning Systems](https://www.oreilly.com/library/view/mlops-engineering-at/9781098114418/) — Carl Osipov (2023)
  Publisher: O'Reilly Media
  提供了MLOps的全面视角，其中包含实验跟踪、超参数管理以及通过指标监控模型性能的章节。
