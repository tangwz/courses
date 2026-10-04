---
course: "data-versioning-experiment-tracking"
chapter: "integrating-dvc-mlflow"
lesson: "best-practices-integrated-workflows"
sourceId: 4107
sourceUrl: "https://apxml.com/zh/courses/data-versioning-experiment-tracking/chapter-4-integrating-dvc-mlflow/best-practices-integrated-workflows"
title: "集成工作流程的最佳实践"
description: "总结 DVC 和 MLflow 配合使用时维护可复现机器学习工作流程的推荐做法。"
order: 8
plots: []
sourceHash: "24853137ff68fd340e4c9f4403962109d64a07d1fb68709e8a2138a0c1983873"
sourceCorrections: []
---

整合数据版本控制（DVC）与 MLflow Tracking 为机器学习 (machine learning)项目管理提供了强大的组合。采纳一致的做法可确保您的工作流程长期保持可理解、可维护和真正可复现。高效配合使用 DVC 和 MLflow 的建议得到归纳。

### 维护一致的项目结构

定义良好的项目结构是实现清晰度和自动化的基础。虽然存在灵活性，但采纳标准布局有助于团队成员理解项目并简化集成脚本的编写。

考虑类似这样的结构：

```
project-root/
├── .dvc/                 # DVC 内部文件
├── .git/                 # Git 内部文件
├── data/
│   ├── raw/              # 原始的、不可变数据（可能由 DVC 跟踪）
│   ├── processed/        # 处理过的数据（DVC 阶段的输出）
│   └── features/         # 特征数据（DVC 阶段的输出）
├── models/               # 训练好的模型（可能由 DVC 跟踪）
├── notebooks/            # 探索性分析笔记本
├── src/                  # 用于数据处理、训练等的源代码
│   ├── process_data.py
│   └── train_model.py
├── tests/                # 单元和集成测试
├── .dvcignore            # 指定 DVC 应忽略的文件/目录
├── .gitignore            # 指定 Git 应忽略的文件/目录
├── dvc.yaml              # DVC 流水线定义
├── mlflow_utils.py       # 用于 MLflow 日志记录的辅助函数
├── params.yaml           # 项目参数（超参数、路径）
└── requirements.txt      # Python 包依赖
```

要点：

- 将原始数据与处理过的数据分开。使用 `dvc.yaml` 中定义的 DVC 阶段来管理转换过程。
- 将源代码 (`src/`) 与笔记本 (`notebooks/`) 分开存放。`src/` 中的代码通常由 DVC 流水线或脚本执行。
- 将参数 (parameter)集中到 `params.yaml` 中。此文件由 Git 跟踪，其值可被 DVC 阶段和 MLflow 日志记录使用。
- 使用 `.gitignore` 来指定 Git 应忽略的文件（例如，虚拟环境、未被 DVC 跟踪的大型数据文件），使用 `.dvcignore` 来指定 DVC 应忽略的文件（例如，跟踪目录中的临时文件）。

### 使用原子提交连接代码、数据和配置

可复现性依赖于了解任何给定实验运行中代码、数据、配置和环境的精确状态。构建提交以一起记录这些元素。

典型的工作流程：

1. 修改代码 (`src/train_model.py`) 或参数 (parameter) (`params.yaml`)。
2. 如果数据处理发生变化，运行 `dvc repro <stage_name>` 来更新派生数据/特征。这会更新 `dvc.lock`。
3. 在 Git 中暂存更改：`git add src/train_model.py params.yaml dvc.lock data/.dvc` (或相关的 `.dvc` 文件/目录)。
4. 提交更改：`git commit -m "feat: Tune hyperparameters for model X"`
5. 推送代码和 DVC 元数据：`git push`
6. 推送由 DVC 跟踪的数据/模型产物：`dvc push`

每个 Git 提交应表示一个连贯的更改，连接所使用的代码版本、数据版本（通过 `.dvc` 文件和 `dvc.lock`）以及配置 (`params.yaml`)。MLflow 会自动记录与运行关联的 Git 提交哈希值，从而创建回溯到此快照的链接。

### 明确连接 DVC 数据版本与 MLflow 运行

虽然 MLflow 会记录 Git 提交，这通过 `.dvc` 文件间接指向由 DVC 跟踪的数据，但在 MLflow 中明确记录数据版本信息可以增加清晰度。

策略包括：

- **将数据哈希值作为参数 (parameter)记录：** 在训练之前，使用 DVC 命令检索输入数据的哈希值，并将其作为 MLflow 参数记录。

  ```python
  import dvc.api
  import mlflow

  # 获取处理过的数据目录的哈希值
  data_version = dvc.api.read('data/processed/features.csv.dvc') # 读取 .dvc 文件内容
  # 或者如果使用较新的 DVC 版本，获取目录本身的哈希值
  # data_status = dvc.api.status(['data/processed'], show_checksum=True)
  # data_hash = data_status['data/processed'].get('checksum')

  with mlflow.start_run():
      # 记录 DVC 哈希值（示例，需要从 dvc.api.read 输出中解析）
      # 假设从 read() 输出中提取了 'md5'
      # mlflow.log_param("dvc_data_md5", data_version['outs'][0]['md5'])
      mlflow.log_param("dvc_input_data", "data/processed/features.csv") # 记录路径
      # 记录 params.yaml 中的参数
      # ...
      # 训练模型
      # ...
      # 记录指标
      # ...
  ```

  *注：可能需要从 `dvc.api.read` 中解析特定的哈希值或使用其他方法，具体取决于您的 DVC 版本和设置。*
- **将 `dvc.lock` 作为产物记录：** `dvc.lock` 文件包含所有流水线输出的精确哈希值。将此文件作为 MLflow 产物记录可提供流水线驱动实验的数据依赖关系的完整快照。

  ```python
  mlflow.log_artifact("dvc.lock")
  ```
- **使用标签：** 使用 `mlflow.set_tag()` 记录与数据版本相关的 Git 分支或描述性标签。

### 借助 DVC 流水线实现自动化

将您的工作流程步骤（数据处理、特征工程、训练）定义为 `dvc.yaml` 中的阶段，而不是仅仅依赖手动运行的独立脚本或笔记本。

好处：

- **自动化：** 当依赖项（代码、数据、参数 (parameter)）发生变化时，`dvc repro` 只自动重新运行必要的阶段。
- **依赖跟踪：** DVC 管理阶段之间的关系，确保正确的执行顺序。
- **可复现性：** `dvc.yaml` 和 `dvc.lock` 明确定义了工作流程及其结果。

将 MLflow 日志记录直接集成到 DVC 阶段执行的命令中：

```yaml
# dvc.yaml
stages:
  process_data:
    cmd: python src/process_data.py --config=params.yaml
    deps:
      - data/raw/input.csv
      - src/process_data.py
    params:
      - processing.param1
    outs:
      - data/processed/features.csv

  train:
    cmd: python src/train_model.py --config=params.yaml
    deps:
      - data/processed/features.csv
      - src/train_model.py
    params:
      - training.learning_rate
      - training.epochs
    metrics:
      - metrics.json: # DVC 可以跟踪文件中的指标
          cache: false
    plots:
      - plots/confusion_matrix.png: # DVC 可以跟踪绘图
          cache: false
          x: actual
          y: predicted
    outs:
      - models/model.pkl # 使用 DVC 跟踪最终模型
```

您的 `src/train_model.py` 脚本将包含必要的 `mlflow.start_run()`、`mlflow.log_param()`、`mlflow.log_metric()` 和 `mlflow.log_artifact()` 调用。如果需要，它还应将指标写入 `metrics.json`，将绘图写入 `plots/` 以供 DVC 跟踪。

### 集中配置

使用 `params.yaml` 这样的配置文件（由 Git 跟踪）来存储超参数 (parameter) (hyperparameter)、文件路径和其他设置。

- 在您的 `dvc.yaml` 阶段中，使用 `params` 部分引用 `params.yaml` 中的参数。
- 在您的 Python 脚本 (`src/*.py`) 中加载 `params.yaml`，以便在执行期间访问参数。
- 使用 `mlflow.log_params()` 将 `params.yaml` 中的相关参数记录到 MLflow。

这使得配置在 DVC 流水线定义、脚本执行和 MLflow 跟踪中保持一致。

### 决定大型产物的跟踪位置

DVC 和 MLflow 都可以存储产物，但它们的目的略有不同：

- **DVC：** 针对大型文件（数据集、模型）进行了优化，与 Git 紧密集成以进行版本控制，并使用远程存储（S3、GCS 等）。是可复现流水线阶段输入/输出的理想选择。
- **MLflow：** 存储与特定*运行*相关的产物（模型、绘图、任意文件），通常与指标/参数 (parameter)一起存储在其后端（本地文件、S3、数据库等）。适用于运行特有的输出和模型注册集成。

**建议：**

- 使用 **DVC** 跟踪大型输入数据集以及您主流水线 (`dvc.yaml`) 输出的最终“生产候选”模型。
- 使用 **MLflow** 记录与*实验运行*直接相关的产物，例如评估绘图、超参数 (hyperparameter)调优期间生成的模型文件（在选择最终模型之前）或诊断信息。您可以将 DVC 跟踪的模型的*路径*作为 MLflow 参数/标签进行记录，以供参考。

### 确保环境一致性

可复现性也包括软件环境。

- 在 `requirements.txt`（针对 pip）或环境文件（针对 Conda）中固定依赖项。使用 Git 跟踪此文件。
- 考虑将 `requirements.txt` 文件本身作为 MLflow 产物记录到每次运行中。
- 为了更严格的控制，使用 Docker 等容器化工具，可以在由 Git 跟踪的 `Dockerfile` 中定义环境。

遵循这些做法，您可以创建一个系统，其中 DVC 管理您的数据生命周期和流水线执行，而 MLflow 则提供您实验的详细记录。这种集成方法显著提升了机器学习 (machine learning)项目的可复现性、协作性和可维护性。

## 参考资料

- [DVC Documentation](https://dvc.org/doc) — Iterative, Inc. (2023)
  Data Version Control (DVC) 的官方指南，对于理解机器学习项目中的数据版本控制、管道管理和工件存储至关重要。
- [MLflow Documentation](https://mlflow.org/docs) — Databricks (2023)
  MLflow 的官方文档，涵盖机器学习工作流的实验跟踪、模型日志记录和工件管理，以及其可重现性。
- [Designing Machine Learning Systems](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  一本关于构建和操作机器学习系统的综合书籍，提供项目组织、MLOps 和确保可重现性的最佳实践。
- [The Turing Way: A handbook for reproducible research](https://the-turing-way.netlify.app/) — The Turing Way Community (2022)
  Publisher: The Alan Turing Institute
  一本由社区驱动的手册，提供可重现研究的指导和最佳实践，与建立稳健且可验证的机器学习工作流直接相关。
