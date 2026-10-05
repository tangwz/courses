# 追踪 DVC 流水线指标

来源：[原文](https://apxml.com/zh/courses/data-versioning-experiment-tracking/chapter-4-integrating-dvc-mlflow/tracking-dvc-pipeline-metrics)

[返回章节目录](README.md) · [返回课程目录](../README.md)

DVC 流水线在 `dvc.yaml` 中定义。这些流水线会生成各种输出，其中性能指标是一种主要的输出类型。当流水线阶段（例如模型训练或评估）完成时，它通常会生成包含重要性能指标的文件，如准确率、精确率、召回率、损失或业务特定衡量标准。DVC 提供了一种机制，可显式追踪这些指标文件，将它们直接关联到流水线的执行状态和底层的 Git 提交。

### 为何直接通过 DVC 追踪指标？

您可能想知道，既然我们有 MLflow 用于全面的实验记录，为何还需要 DVC 指标追踪。在 DVC 流水线内追踪指标有几个显著的优点，尤其对于工作流自动化以及与代码和数据更改相关的快速比较：

1. **直接与流水线关联：** 在 `dvc.yaml` 中定义的指标与生成它们的特定阶段以及该次运行中使用的确切输入（数据、代码、依赖项）内部关联。
2. **版本控制集成：** 由于 `dvc.yaml` 和生成的 `dvc.lock` 文件由 Git 追踪，使用 DVC 命令比较不同 Git 提交间的指标变得直接。这能让您快速评估代码更改或不同数据版本对性能的影响。
3. **命令行访问便利：** DVC 提供简单命令（`dvc metrics show`、`dvc metrics diff`）直接从终端查看和比较指标，相比每次小改动都要操作用户界面，这提供了更快的反馈周期。
4. **自动化潜力：** 追踪的指标可用于自动化 CI/CD 工作流，在合并更改前检查性能退步情况。

虽然 MLflow 在详细实验比较、可视化和跨多个可能不相关运行的工件管理方面表现出色，但 DVC 指标提供了一种专注的、受版本控制的视图，直接关联到 Git 仓库中管理的流水线结构。

### 在 `dvc.yaml` 中声明指标

要告诉 DVC 哪些文件包含由流水线阶段生成的指标，您需要在 `dvc.yaml` 中该阶段的定义里添加一个 `metrics` 部分。DVC 期望这些文件是它可以解析的简单格式，例如 JSON、YAML、CSV 或 TSV（制表符分隔值）。

以一个使用 `dvc stage add` 或 `dvc run` 定义的典型训练阶段为例。假设此阶段运行 `train.py` 脚本，该脚本完成后会将评估结果写入名为 `results/metrics.json` 的文件。

以下是您如何修改 `dvc.yaml` 中的阶段定义，以将此文件声明为指标输出：

```yaml
stages:
  train:
    cmd: python src/train.py --data data/processed --model-out models/model.pkl --metrics-out results/metrics.json
    deps:
      - src/train.py
      - data/processed
    params:
      - training.epochs
      - training.learning_rate
    outs:
      - models/model.pkl
    metrics: # 在此处声明指标文件
      - results/metrics.json:
          cache: false # 通常，指标文件较小，无需 DVC 缓存
```

在此示例中：

- `cmd` 执行训练脚本，该脚本接受 `--metrics-out` 参数 (parameter)，指定保存指标的位置。
- `metrics` 键列出了 DVC 应作为此阶段指标输出追踪的文件。
- `cache: false` 常用于指标文件。由于它们通常是小型文本文件，并且其内容是主要信息（而不是像大型数据集那样需要高效存储/传输），禁用 DVC 的默认缓存机制可以简化操作。如果文件不在 `.gitignore` 中，它仍会由 Git 追踪。如果您确实希望 DVC 缓存并版本化指标文件本身（如果文件较大或为二进制文件，则很有用，尽管不常见），您可以省略 `cache: false` 或将其设置为 `true`。

`results/metrics.json` 文件可能看起来像这样：

```json
{
  "accuracy": 0.875,
  "precision": 0.85,
  "recall": 0.90,
  "f1_score": 0.874,
  "validation_loss": 0.25
}
```

DVC 可以解析这种简单的键值结构。对于 CSV/TSV 文件，它期望一个标题行，并将每列视为一个指标。

### 使用 DVC 命令查看和比较指标

在 `dvc.yaml` 中定义指标并使用 `dvc repro` 运行流水线后，您可以使用 DVC 命令查看结果。

1. **显示当前指标：** `dvc metrics show` 命令显示 `dvc.yaml` 中为当前工作区和最近 Git 提交指定文件中的指标。

   ```bash
   $ dvc metrics show
   Path                accuracy    precision    recall    f1_score    validation_loss
   results/metrics.json    0.875       0.85         0.90      0.874       0.25

   # 如果有可用数据，输出也可能显示来自先前提交的指标
   # （例如，来自 'main' 分支或特定标签）
   ```
2. **比较不同修订版本间的指标：** 真正的优势在于比较。`dvc metrics diff` 命令比较当前工作区与特定 Git 修订版本（如分支、标签或提交哈希）之间的指标，或比较两个修订版本之间的指标。

   ```bash
   # 比较工作区与 main 分支
   $ dvc metrics diff main

   Path                  Metric           main    workspace    Change
   results/metrics.json  accuracy         0.860   0.875        0.015
   results/metrics.json  precision        0.84    0.85         0.01
   results/metrics.json  recall           0.88    0.90         0.02
   results/metrics.json  f1_score         0.859   0.874        0.015
   results/metrics.json  validation_loss  0.28    0.25         -0.03

   # 比较两个特定提交或标签
   $ dvc metrics diff v1.0 v1.1
   ```

   此命令立即显示了由于指定 Git 修订版本之间追踪的修改，性能如何发生了变化。

### 使用 DVC 绘图功能可视化指标演变

虽然 `dvc metrics show` 和 `dvc metrics diff` 提供表格数据，但可视化趋势能提供更多信息。DVC 通过 `dvc plots` 提供基本的绘图功能，以监测指标在您的 Git 历史中如何变化。

如果您的指标文件遵循特定结构（如 CSV/TSV 或 JSON 对象列表），DVC 可以自动生成图表。您也可以在 `dvc.yaml` 中使用 Vega-Lite 模板定义自定义图表配置，以进行更多控制。

对于简单情况，如果您的 `results/metrics.json` 在不同提交中始终包含相同的键，您可以生成比较修订版本的图表：

```bash
# 使用默认图表显示 main 与工作区之间的准确率变化
$ dvc plots diff main -o accuracy_changes.html --show-vega

# 更常见的是，在 dvc.yaml 中定义图表配置以便于渲染
```

在 `dvc.yaml` 中添加 `plots` 部分允许定义命名图表：

```yaml
plots:
  # 使用默认模板的简单图表，显示跨修订版本的准确率
  - results/metrics.json:
      x: revision # Git 修订版本的特殊值
      y: accuracy
      title: 准确率趋势
      template: linear # 使用线性图表模板

  # 使用特定属性比较多个指标的图表
  - id: performance_metrics
    template: linear
    x: revision
    y:
      results/metrics.json: [accuracy, f1_score] # 绘制准确率和 F1 分数
    title: 跨修订版本的性能指标
    x_label: Git 修订版本
    y_label: 分数
```

通过这些定义，您可以使用诸如 `dvc plots show performance_metrics` 或 `dvc plots diff main performance_metrics` 等命令来生成可视化内容（通常输出 HTML 文件或用于渲染的 JSON 规范）。

> DVC 指标和图表将指标文件版本直接关联到 Git 提交，从而能够伴随代码和流水线结构演进追踪性能变化。

虽然 DVC 为追踪和比较与流水线运行和 Git 历史相关的指标提供了有用的能力，但它主要侧重于已定义、可复现阶段的输出。对于更全面的视图，包括超参数 (parameter) (hyperparameter)调整、任意脚本运行、详细的工件记录以及更丰富的用户界面以供查看，MLflow 仍是首选工具。下一节将展示如何连接这两个系统，确保您的 DVC 流水线追踪的指标也能在 MLflow 实验中被记录，以便进行统一分析。

## 参考资料

- [DVC Documentation: Metrics](https://dvc.org/doc/command-reference/metrics) — Iterative (2024)
  解释如何在 DVC 管道中声明、显示、比较和绘制指标，是讨论概念的主要来源。
- [DVC Documentation: Plots](https://dvc.org/doc/command-reference/plots) — Iterative (2024)
  Publisher: Iterative
  详细介绍了如何定义、生成和自定义图表，以使用 DVC 可视化指标演变和 Git 历史中的其他数据。
- [Building Machine Learning Pipelines](https://books.google.com/books/about/Building_Machine_Learning_Pipelines.html?id=oP8BEAAAQBAJ) — Hannes Hapke, Catherine Nelson (2020)
  Publisher: O'Reilly Media
  涵盖了构建可复现机器学习管道的最佳实践，包括使用 DVC 等工具进行数据和模型版本控制以及实验跟踪。

---

[上一节](05-%E5%A4%8D%E7%8E%B0%20DVC%20%E6%B5%81%E6%B0%B4%E7%BA%BF.md) · [下一节](07-%E7%BB%93%E5%90%88%20DVC%20%E6%B5%81%E6%B0%B4%E7%BA%BF%E4%B8%8E%20MLflow%20%E8%BF%BD%E8%B8%AA.md)
