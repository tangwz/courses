---
course: "introduction-to-mlops"
chapter: "versioning-in-machine-learning"
lesson: "techniques-for-model-versioning"
sourceId: 7216
sourceUrl: "https://apxml.com/zh/courses/introduction-to-mlops/chapter-3-versioning-in-machine-learning/techniques-for-model-versioning"
title: "模型版本管理技术"
description: "了解对训练好的机器学习模型进行版本管理的各种方法和工具，确保能够回滚或重现预测结果。"
order: 4
plots: []
sourceHash: "def0dde7f7dc4ee02b1a7e6f8699e8186a7be2d38cc21ee72837eaa50e2cfedd"
sourceCorrections: []
---

代码和数据是机器学习 (machine learning)项目的原料，而训练好的模型则是成品。正如厨师需要知道制作特定菜肴的确切食谱和所用食材批次一样，机器学习工程师必须能够将模型直接与其产生的代码和数据联系起来。将模型工件（如 `model.pkl` 之类的二进制文件）与代码一起存储在 Git 仓库中，通常是许多人的首选。然而，这通常不是一个好的做法。Git 是为跟踪文本文件的更改而优化的，并不适合存储大型、不透明的二进制文件。这样做会迅速增加仓库的体积，导致克隆速度变慢且难以管理。

模型版本管理的目标不仅是保存模型文件，还要创建一个永久的、可审计的记录，将模型与其完整谱系连接起来。这意味着对于任何给定的模型，你应该能够回答：

- 训练它时使用了哪个版本的代码？
- 它是基于哪个版本的数据集训练的？
- 在那次训练运行中使用了哪些超参数 (parameter) (hyperparameter)？
- 它的性能指标（如准确率或 F1 分数）是多少？

回答这些问题对于调试、审计和重现结果非常有用。让我们看几种实现这一目标的常见技术，从简单的手动方法到更自动化的工业标准系统。

> 模型版本管理的核心目标是维持模型工件与创建它的特定代码和数据版本之间的联系。

### 策略 1：命名规范与云存储

最基础的模型版本管理策略涉及将模型存储在共享文件存储系统中，如 Amazon S3、Google Cloud Storage 甚至是共享网络驱动器，并使用具有高度描述性的命名规范。文件名本身就成了元数据的主要来源。

例如，你可以采用这样的规范：

`[模型名称]_[数据版本哈希]_[Git 提交哈希]_[时间戳].pkl`

实际的文件名可能如下所示：

`sentiment-classifier_3f4e5a6_a1b2c3d_20231027T103000.pkl`

- **优点：** 这种方法易于理解和实施。它不需要云存储提供商以外的任何专门工具。
- **缺点：** 它非常脆弱，完全依赖于手动执行的严谨程度。一个细小的拼写错误就可能断开与源代码或数据的联系。它也很难查询。例如，要找到“基于数据集 `3f4e5a6` 训练的准确率最高的模型”，需要列出并解析所有文件名，这不仅效率低下且容易出错。

这种方法对于个人项目或初步尝试是可以接受的，但对于团队协作或生产级别的工作来说，扩展性较差。

### 策略 2：存储元数据文件

一个明显的改进是在每个模型工件旁边存储一个元数据文件，通常是 JSON 文件。与其将所有信息挤进文件名，不如将其存储在结构化格式中。

当你保存模型 `sentiment_model_v2.pkl` 时，你同时在同一目录下保存一个对应的 `sentiment_model_v2.json` 文件。该文件将包含血缘追踪信息。

```json
{
  "model_name": "情感分类器",
  "model_file": "sentiment_model_v2.pkl",
  "version": "2.0",
  "creation_timestamp": "2023-10-28T14:00:00Z",
  "lineage": {
    "code_commit_hash": "a1b2c3d4e5f6g7h8i9j0",
    "data_version_id": "3f4e5a6b7c8d9e0f1a2b"
  },
  "hyperparameters": {
    "learning_rate": 0.001,
    "epochs": 15,
    "optimizer": "Adam"
  },
  "performance_metrics": {
    "validation_accuracy": 0.935,
    "f1_score": 0.928
  }
}
```

- **优点：** 元数据现在是结构化的、机器可读的，且更加详细。查找所需信息更容易，你还可以编写简单的脚本来读取这些文件并对比模型。
- **缺点：** 尽管有所改进，但你仍然是在存储系统中管理一堆文件。发现、对比以及管理这些模型的生命周期（例如，哪个正在生产环境中运行？）仍然是一个手动或半手动的过程。它缺乏一个集中的记录系统。

### 策略 3：使用模型注册表

对于模型版本管理，最有效且可扩展的方案是**模型注册表**（Model Registry）。模型注册表是一个专门为存储、版本化和管理机器学习 (machine learning)模型生命周期而设计的集中式系统。它是你所有训练模型的单一事实来源。

常见的 MLOps 工具如 MLflow、DVC Studio、Amazon SageMaker 和 Google Vertex AI 都包含模型注册表组件。这些系统使模型管理过程规范化。

模型注册表提供了几个核心功能：

1. **集中存储与版本化：** 它存储模型工件并为其分配逻辑版本（例如 `sentiment-classifier:v1`, `sentiment-classifier:v2`）。
2. **元数据关联：** 它提供了一种结构化方式来附加我们讨论过的所有元数据：代码提交、数据版本、超参数 (parameter) (hyperparameter)和性能指标。这些信息直接链接到数据库中的模型版本，便于搜索和查询。
3. **生命周期管理：** 注册表允许你为模型版本分配阶段或标签，如 `测试中`、`生产中` 或 `已归档`。这对于控制部署流程非常有用。你的 CI/CD 流水线可以配置为自动部署任何晋升到 `生产中` 阶段的模型。
4. **API 访问：** 注册表提供程序化 API 来与模型交互。训练脚本可以使用 API 注册新模型版本，部署脚本可以使用它来获取最新的生产就绪模型。

> 模型注册表将训练过程与部署分离开来。当训练流水线注册了一个合格的模型时，其任务就结束了。而部署流水线的任务是从获取具有特定状态（如“生产中”）的模型开始的。

通过采用模型注册表，你从管理文件转变为管理结构化资产。这提供了构建可靠机器学习系统所需的审计能力和控制力。它确保生产环境中的每个模型都能追溯到其源头，使整个系统更加透明和易于维护。这种结构化方法是成熟 MLOps 实践的基础组成部分。

## 参考资料

- [MLflow Model Registry](https://www.mlflow.org/docs/latest/model-registry.html) — Databricks (2024)
  MLflow 集中式机器学习模型管理系统的官方指南，内容涵盖模型版本控制、阶段转换和元数据。
- [DVC: Data and Model Versioning](https://dvc.org/doc/start/versioning) — Iterative AI (2024)
  Publisher: Iterative AI
  此文档解释 DVC 如何为大型数据文件和模型提供类似 Git 的版本控制，解决 Git 在二进制工件方面的局限性。
- [Engineering MLOps: From Model to Production](https://www.oreilly.com/library/view/engineering-mlops/9781098103328/) — Emmanuel Raj, Mark Wallace (2021)
  Publisher: O'Reilly Media
  一本构建生产就绪机器学习系统的指南，其中包含专门关于模型版本控制和管理实践的章节。
- [Manage models with Model Registry in Vertex AI](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEXk8DAsQf6_H_AjrXabuvaZbWwBTVmNiNNv5kM7tGxqT8wgdkWL6ALsD7lIrN-K8qeRa_UxW2FCSc1QOQu6b43NFDIz7gI5MHPqfebdTB1J98vdixlmGMCneea-c18pHIJsPORD2lZZP0JTnPXG4N10VP7Kc2LVVd81sWYKeE=) — Google Cloud (2024)
  Publisher: Google Cloud
  Google Cloud Vertex AI 平台如何提供集中式模型注册表来管理机器学习模型生命周期的官方文档。
