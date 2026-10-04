---
course: "data-versioning-experiment-tracking"
chapter: "tracking-experiments-mlflow"
lesson: "using-mlflow-ui"
sourceId: 4088
sourceUrl: "https://apxml.com/zh/courses/data-versioning-experiment-tracking/chapter-3-tracking-experiments-mlflow/using-mlflow-ui"
title: "使用 MLflow 用户界面"
description: "浏览 MLflow 网页界面，查看、比较和分析实验结果。"
order: 7
plots: ["plots/4088-0.json"]
sourceHash: "16ca65b08b24c2a428b2cfb7f1816c765dfdff5d11616a98f6decbb87b60a5a7"
sourceCorrections: []
---

MLflow 用户界面（UI）提供了一个功能强大的基于网页的控制面板，用于可视化和分析训练脚本使用 MLflow 追踪 API 记录的实验信息。该控制面板支持实验运行的交互式查看、比较和管理，为通过 API 进行的程序化访问提供了用户友好的替代方案。

### 启动 MLflow 用户界面

要启动用户界面，请在您的终端中进入项目根目录（如果您使用默认的本地文件存储，也可以是包含 `mlruns` 文件夹的任何目录），然后运行以下命令：

```bash
mlflow ui
```

默认情况下，此命令会启动一个本地网页服务器，通常可以通过 `http://127.0.0.1:5000` 访问。MLflow 将在您的终端中打印准确的 URL。

如果您在记录运行时，将 MLflow 配置为使用不同的后端存储（例如数据库或远程追踪服务器），您需要将用户界面指向同一位置。请使用 `--backend-store-uri` 标志：

```bash
# PostgreSQL 后端的例子
mlflow ui --backend-store-uri postgresql://user:password@host:port/database

# 远程追踪服务器的例子
mlflow ui --backend-store-uri http://your-mlflow-server.com:5000
```

在网页浏览器中打开提供的 URL 即可访问该界面。

### 浏览界面

MLflow 用户界面以简洁有序的方式展示您的实验和运行。

#### 实验视图

默认视图通常在左侧导航面板中列出所有实验。实验将相关的运行分组，例如针对特定任务训练模型的不同尝试。点击实验名称会过滤主视图，使其只显示与该实验关联的运行。如果您在记录时未指定实验，则会自动创建“Default”（默认）实验。

#### 运行列表

选中一个实验后，主区域会显示一个列出单个运行的表格。每一行代表一次对您追踪的脚本 (`mlflow.start_run()`) 的独立执行。以下重要信息会呈现在各列中：

- **开始时间：** 运行开始的时间。
- **持续时间：** 运行花费的时间。
- **运行名称：** 一个可选的、易于阅读的名称（如果未指定，通常会自动生成）。
- **源：** 启动运行的入口脚本或笔记本。
- **版本：** 与运行关联的 Git 提交哈希值（如果在 Git 仓库中运行）。
- **参数 (parameter)：** 每项记录的超参数 (hyperparameter)的列（例如 `learning_rate`、`n_estimators`）。
- **指标：** 每项记录的指标的列（例如 `accuracy`、`rmse`、`val_loss`）。您通常会在此处看到最终记录的值。

您可以自定义显示的列，按任意列排序（例如，按 `accuracy` 降序排序以找到最佳运行），以及使用搜索框根据参数值或指标得分来筛选运行。

#### 运行详情页

点击特定运行的开始时间或名称，即可进入其详情页。此页面提供对该次运行所有记录内容的完整查看：

- **运行 ID：** 运行的唯一标识符。
- **状态：** 指示运行是已完成、失败还是仍在进行中。
- **参数：** 一个专用部分，列出了所有已记录的超参数及其值。
- **指标：** 一个显示已记录指标的部分。对于多次记录的指标（例如，每个周期的损失），MLflow 通常会显示交互式图表，展示指标随步骤或时间的变化历史。
- **标签：** 与运行关联的任何自定义标签。标签对于添加定性信息或标记 (token)很有用。
- **工件：** 所有已记录工件的文件浏览器视图。您将在此处找到保存的模型、输出文件、图像、图表等。

### 查看运行详情

运行详情页是您查看单个实验尝试具体信息的地方。

- **参数 (parameter)：** 快速核实特定运行使用的确切配置。
- **指标：** 分析性能结果。如果您在训练步骤中记录了指标（例如，在每个周期后记录验证准确率），您可以直接在用户界面中将训练进度可视化。将鼠标悬停在图表上通常会显示特定点的精确值。
- **工件：** 这对于访问运行的输出特别有用。您可以：
  - 下载已保存的模型文件（例如 `.pkl`、`.h5`、`saved_model` 目录）。
  - 直接在浏览器中查看已记录的图像或图表（例如混淆矩阵、特征重要性图）。
  - 下载运行期间记录的任何其他文件，例如处理后的数据样本或评估报告。

### 比较实验运行

MLflow 用户界面最有价值的功能之一是能够并排比较多个运行。

1. 返回实验的运行列表视图。
2. 选择您想要比较的运行旁边的复选框。
3. 点击出现的“比较”按钮。

这将打开一个专用的比较视图：

- **参数 (parameter)比较：** 显示所选运行的不同参数值，并突出显示差异。
- **指标比较：** 显示每次运行已记录指标的值，从而方便查看哪次运行根据特定标准表现最佳。
- **可视化工具：**
  - **散点图：** 有助于将参数与指标（例如 `learning_rate` 与 `accuracy`）之间或两个指标之间的关系可视化。
  - **等高线图：** 有助于将两个参数与一个指标之间的关系可视化。
  - **平行坐标图：** 非常适用于可视化多个超参数 (hyperparameter)的组合如何影响众多运行的产出指标。



![准确率与学习率对比](plots/4088-0.json)



> 散点图比较了五次不同实验运行的验证准确率与学习率。

这些比较工具通过视觉化地整理多次试验的结果，大大加快了超参数调整和模型选择的速度。

### 搜索和筛选

对于有大量运行的实验，搜索功能是不可或缺的。您可以根据以下内容构建查询：

- **指标：** `metrics.accuracy > 0.9`
- **参数 (parameter)：** `params.learning_rate = '0.01'` 或 `params.optimizer = 'Adam'`
- **标签：** `tags.data_version = 'v2'`
- **属性：** `attributes.status = 'FINISHED'`

使用 `AND` 组合这些条件以创建特定的筛选器，从而帮助您快速找到符合复杂条件的运行。

MLflow 用户界面将实验追踪从一种被动的日志记录活动，转变为一个主动的分析和决策制定工具。通过提供直观的方法来查看、排序、筛选、比较和检查单次运行，它帮助您理解模型的行为，找到最佳配置，并最终更高效地构建更好的机器学习 (machine learning)模型。

## 参考资料

- [MLflow UI](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHBgM3sBpTY_AeJ07dcuQ7jCsphmHvKGVpsaXXiMqFLHKRQxYM77IVep8wwnOjZa-VoqrLuGCgyx8YXP6NV-ErYiDYE2wDHCSWXIr-in5TliRTiLeKTu6rOTdfabsTW64Ck9BCJd0odHRfA5wE3jzA5eounSVHjlj7tpUCNc13CCM8W24CPy2bK828KgIYA17BmOxhRLuDRkpGevQoKCWHNTNNJO_KSEUXgmbYn) — MLflow Project (2024)
  Publisher: MLflow Project
  详细介绍了MLflow用户界面的功能和操作，用于记录、比较和检查实验运行。
- [Practical MLOps: Operationalizing Machine Learning Models into Production](https://www.packtpub.com/product/practical-mlops/9781801815822) — Krishna Kumar (2022)
  Publisher: Packt Publishing
  MLOps的实用指南，包含实验跟踪和模型管理的原则与工具，为MLflow UI提供背景信息。
