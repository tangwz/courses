---
course: "data-versioning-experiment-tracking"
chapter: "integrating-dvc-mlflow"
lesson: "structuring-projects-integration"
sourceId: 4095
sourceUrl: "https://apxml.com/zh/courses/data-versioning-experiment-tracking/chapter-4-integrating-dvc-mlflow/structuring-projects-integration"
title: "为集成构建项目结构"
description: "组织项目仓库以适应DVC和MLflow的最佳实践。"
order: 2
plots: []
sourceHash: "aa79148810490f8d83bd0a144805231d2aa636601764a018c31f8615778d191a"
sourceCorrections: []
---

组织良好的项目结构是成功集成DVC数据版本控制和MLflow实验追踪的重要前提。虽然没有单一的强制布局，但采用常规结构能大幅提升清晰度、可维护性和协作性，尤其当项目复杂度增加时。这种结构有助于分离数据、源代码、配置和输出等职责，使得人工和自动化工具都能更方便地管理。

接下来，我们来了解一种在同时使用DVC和MLflow时组织机器学习 (machine learning)项目的常见且有效的方式。

### 主要目录和文件

典型的项目布局通常包含以下组件：

- **`data/`**: 此目录存放您的数据集。它可能包含子目录，如用于存放原始、未经修改数据的`data/raw`，以及用于存放经过清洗或转换、可用于建模数据的`data/processed`。DVC主要用于追踪此目录的内容，通常通过创建`.dvc`文件，这些文件充当指向实际存储在其他地方（如云存储）的数据的指针。大型数据文件本身通常会添加到`.gitignore`中。
- **`src/`**: 包含您的源代码，通常是Python (`.py`) 文件。这包括用于数据加载、预处理、特征工程、模型训练、评估以及可能的实用工具函数的脚本或模块。此代码使用 Git 进行版本控制。
- **`models/`**: 这是保存训练好的模型制品（例如，序列化的模型文件如`.pkl`或`.h5`）的潜在位置。如果模型文件较大，可以使用DVC追踪它们，或者在实验运行期间使用MLflow直接将其作为制品记录。选择取决于您的工作流程需求。
- **`notebooks/`**: 通常用于使用Jupyter笔记本 (`.ipynb`) 进行探索性数据分析 (EDA)、初步实验和可视化。虽然对探索有帮助，但通常建议将笔记本中可重用或生产环境的代码重构到`src/`目录下的Python脚本中，以便更好地进行测试、模块化和在自动化流水线中使用。
- **`dvc.yaml`**: 此文件是定义DVC流水线的核心。它概述了您工作流程的各个阶段（例如，处理数据、训练模型）、它们的依赖项（输入数据、代码）、要执行的命令以及输出（处理后的数据、模型、指标）。我们将在后续章节中详细查看此文件。
- **`params.yaml`**: 一种常见做法是将项目参数 (parameter)，特别是模型训练的超参数 (hyperparameter)或数据处理的配置设置，存储在一个专用的YAML文件中。这使得参数明确且易于追踪。DVC流水线和MLflow都可以从此文件中读取，从而确保一致性。对`params.yaml`的更改由Git追踪。
- **`requirements.txt` 或 `environment.yml`**: 定义Python包依赖的标准文件。指定依赖项对确保项目环境能够准确重现非常重要。
- **`mlruns/`**: 如果您尚未配置远程追踪服务器，这是MLflow存储实验追踪数据（参数、指标、制品）的默认本地目录。此目录几乎总应包含在`.gitignore`中。
- **`.dvc/`**: 包含DVC的内部文件，包括配置、缓存目录结构等。此目录由DVC管理，也应在`.gitignore`中。
- **`.gitignore`**: 对于任何使用Git的项目来说，这是一个重要文件。它告诉Git哪些文件或目录应被忽略。当使用DVC和MLflow时，忽略由DVC追踪的大型数据文件、DVC缓存、MLflow的本地追踪目录 (`mlruns/`)、虚拟环境目录以及其他不适用于Git版本控制的生成文件非常重要。

### 项目结构示例

以下是典型项目布局的可视化表示：

> 一个常见的项目布局，将数据、源代码、模型、笔记本、配置（`params.yaml`）、DVC流水线定义（`dvc.yaml`）和被忽略的目录（`.dvc/`、`mlruns/`）进行分离。

### 对工作流程的意义

此结构有助于集成工作流程：

1. **数据管理：** 原始数据放入`data/raw`中。DVC追踪更改（例如，`dvc add data/raw/your_data.csv`）。处理后的数据可能放入`data/processed`中，同样由DVC追踪。实际的大文件被Git忽略，但小型`.dvc`指针文件会提交。
2. **代码开发：** `src/`中的脚本执行诸如数据处理、模型训练和评估等任务。这些脚本的更改由Git追踪。
3. **配置：** `params.yaml`中的参数 (parameter)由`src/`中的脚本读取。参数的更改由Git追踪。
4. **实验追踪：** 当`src/train.py`运行时，它使用MLflow记录参数（可能从`params.yaml`读取）、指标和制品（如模型，这些模型可能保存到`models/`或直接记录）。MLflow会记录此信息，并将其链接到所用代码的特定Git提交。
5. **流水线编排：** `dvc.yaml`定义了执行`src/`中脚本的阶段，使用由`.dvc`文件（或`params.yaml`）指定的数据，并产生输出（如由DVC追踪的处理数据或模型，或指标文件）。

### 理由和最佳实践

- **职责分离：** 使数据、代码、配置和生成输出区分开来，使项目更易于理解和管理。
- **依赖明确：** 使代码、数据和配置的位置显而易见。DVC等工具可以根据此结构自动检测依赖项。
- **协作：** 标准结构降低了新团队成员理解项目流程的障碍。
- **自动化：** 简化自动化工作流程（例如，ML的CI/CD流水线）的创建，因为脚本、数据和配置的路径是可预测的。
- **`.gitignore` 非常重要：** 正确配置`.gitignore`是阻止大型数据文件、临时MLflow日志或DVC缓存内容提交到Git仓库的重要一步。它确保Git只追踪代码、配置和DVC元数据文件。

以下是此类项目中`.gitignore`的示例片段：

```gitignore
# DVC 相关
.dvc/cache
.dvc/tmp
.dvc/lock
/data/raw/ # 如果由 DVC 管理，则忽略原始数据
/data/processed/ # 如果由 DVC 管理，则忽略处理后的数据
/models/ # 如果由 DVC 管理或通过 MLflow 记录，则忽略模型

# MLflow 相关
mlruns/

# Python 相关
__pycache__/
*.pyc
*.pyo
*.pyd
.env
venv/
env/
*.egg-info/
dist/
build/

# 笔记本相关
.ipynb_checkpoints

# 操作系统相关
.DS_Store
Thumbs.db
```

> 为使用DVC和MLflow的项目量身定制的`.gitignore`规则示例。根据您使用`dvc add`追踪的特定文件或目录，调整`/data/raw/`等路径。

从项目一开始就采用这样的结构，为集成DVC和MLflow提供了一个坚实框架，使得机器学习 (machine learning)系统更加有条理、可重现和可维护。虽然此模板可以作为一个良好的起点，但请随意根据您项目的具体要求和规模进行调整。

## 参考资料

- [DVC Documentation](https://dvc.org/doc) — Iterative.ai (2024)
  这是理解 DVC 功能（包括数据版本控制、管道和 dvc.yaml 文件结构）的主要资源。
- [MLflow Documentation](https://www.mlflow.org/docs/latest/index.html) — Databricks (2024)
  MLflow 的官方指南，涵盖实验追踪、模型日志记录以及 mlruns/ 目录的组织。
- [MLOps with DVC and MLflow: A Quickstart Guide](https://dvc.org/blog/mlops-dvc-mlflow-quickstart) — Iterative.ai (2023)
  Publisher: Iterative.ai
  本指南演示了如何将 DVC 用于数据和管道版本控制，并结合 MLflow 进行实验追踪，展示了实用的项目设置。
