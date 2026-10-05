# 编排 TFX 流水线

来源：[原文](https://apxml.com/zh/courses/advanced-tensorflow/chapter-5-production-ml-pipelines-tfx/tfx-pipeline-orchestration)

[返回章节目录](README.md) · [返回课程目录](../README.md)

TFX 流水线由 `ExampleGen`、`Transform` 和 `Trainer` 等各个组件构成。每个组件都在机器学习 (machine learning)工作流程中执行特定步骤。然而，定义这些组件及其依赖关系只是其中一部分。要在生产环境中将这些组件作为协调一致、自动化且可靠的系统运行，需要一个编排器。

编排器是负责调度、执行和监控流水线内任务（TFX 组件）的系统。它理解您在组件之间定义的依赖关系，并确保它们按正确顺序运行。如果某个组件失败，编排器可以处理重试或通知您问题。它管理流水线运行的整体生命周期。

可以将 TFX 组件视为复杂食谱中的单独指令。编排器就是厨师，负责阅读食谱，收集食材（数据制品），按顺序执行每个步骤（运行组件），并管理厨房（计算资源）。

### 为何使用编排器？

虽然 TFX 提供了 `LocalDagRunner` 用于在本地环境中直接运行流水线（适用于开发和测试），但生产环境需要更精细的管理。编排器提供：

- **调度：** 根据时间（例如每日再训练）或事件（例如新数据到达）自动触发流水线运行。
- **依赖关系管理：** 确保组件仅在其所需输入制品由上游组件成功生成后才运行。
- **可伸缩性：** 将组件的执行分布到多台机器或集群上。
- **资源管理：** 为每个组件高效分配计算资源（CPU、内存、GPU）。
- **错误处理和重试：** 自动重试失败的组件或管理复杂的故障情况。
- **监控和日志：** 提供对流水线执行状态、组件日志和制品跟踪的查看。
- **参数 (parameter)化：** 允许在不更改流水线定义代码的情况下，使用不同参数（例如超参数 (hyperparameter)、日期范围）配置流水线运行。

### TFX 常用编排平台

TFX 被设计为与编排器无关，这意味着您可以在各种平台上运行 TFX 流水线。TFX 流水线的核心定义大体保持不变，但您将使用特定的 `Runner` 类或配置来指定所选的编排器。开源社区中两个受欢迎的选择是 Apache Airflow 和 Kubeflow Pipelines。

#### Apache Airflow

Apache Airflow 是一个广泛采用的通用工作流编排器。Airflow 中的流水线使用 Python 定义为有向无环图 (DAG)。TFX 提供将基于 Python 的 TFX 流水线定义编译成 Airflow DAG 的工具。

- **优点：** 成熟的生态系统，广泛的集成，适合计划性批处理，用于监控 DAG 运行的用户界面很强大。
- **集成：** 使用 `tfx.orchestration.airflow.AirflowDagRunner` 将您的 TFX 流水线编译成 Airflow DAG 文件。

#### Kubeflow Pipelines (KFP)

Kubeflow Pipelines 专为在 Kubernetes 上编排机器学习 (machine learning)工作流而设计。它使用容器执行每个流水线步骤，提供出色的隔离和环境管理。

- **优点：** Kubernetes 原生，采用容器化，专为机器学习工作流设计，提供用于可视化流水线和制品的用户界面，与其他 Kubeflow 组件良好配合。
- **集成：** 使用 `tfx.orchestration.kubeflow.KubeflowDagRunner`（或针对新 KFP v2 API 的 `KubeflowV2DagRunner`）将您的流水线编译成 YAML 文件，该文件可以上传并在 Kubeflow Pipelines 部署上运行。

#### 其他选项

- **Vertex AI 流水线：** 谷歌云的托管机器学习流水线服务，基于 Kubeflow Pipelines 构建，提供无服务器执行环境。TFX 直接与其集成。
- **本地运行器：** 如前所述，`tfx.orchestration.LocalDagRunner` 在您的本地 Python 环境中顺序执行组件。仅适用于开发或小型实验。

### 定义和编译流水线

无论使用哪种编排器，您都可以通过实例化组件并连接它们的输入和输出来在 Python 中定义 TFX 流水线。

```python
# 示例代码段
from tfx.components import CsvExampleGen, StatisticsGen, SchemaGen, ExampleValidator, Transform, Trainer
from tfx.proto import example_gen_pb2
from tfx.orchestration import pipeline

# 假设已定义了必要的输入，例如 data_path, module_file

# 1. 实例化组件
example_gen = CsvExampleGen(input_base=data_path)
statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'], infer_feature_shape=True)
example_validator = ExampleValidator(statistics=statistics_gen.outputs['statistics'], schema=schema_gen.outputs['schema'])
transform = Transform(examples=example_gen.outputs['examples'], schema=schema_gen.outputs['schema'], module_file=transform_module_file)
trainer = Trainer(module_file=trainer_module_file, examples=transform.outputs['transformed_examples'], transform_graph=transform.outputs['transform_graph'], schema=schema_gen.outputs['schema'])
# ... 添加 Evaluator, Pusher 等

# 2. 定义组件列表
components = [
    example_gen,
    statistics_gen,
    schema_gen,
    example_validator,
    transform,
    trainer,
    # ...
]

# 3. 创建流水线定义
pipeline_definition = pipeline.Pipeline(
    pipeline_name='my_tfx_pipeline',
    pipeline_root='/path/to/pipeline/artifacts', # 由编排器管理
    components=components,
    enable_cache=True,
    metadata_connection_config=None # 配置 ML 元数据连接
    # beam_pipeline_args 与 Dataflow/Spark 运行器相关
)

# 4. 为特定编排器编译（在单独的脚本/CLI 步骤中）
# 示例编译命令：
# tfx pipeline compile --engine kubeflow --pipeline_path my_pipeline_definition.py --output_path my_pipeline.yaml
# 或在 Python 中使用运行器：
# from tfx.orchestration.kubeflow import KubeflowDagRunner
# KubeflowDagRunner().run(pipeline_definition)
```

Python 代码定义了结构和逻辑。编译步骤将此定义转换为所选编排器所需的特定格式（例如，Airflow DAG Python 文件、KFP YAML 文件）。编排器随后获取此编译后的定义并执行它。

> TFX 流水线的一个典型依赖关系图，由编排器管理。组件在其输入（制品以标签形式表示）可用后运行。

### 选择合适的编排器

选择编排器取决于几个因素：

- **现有基础设施：** 如果您的组织已广泛使用 Kubernetes，那么 Kubeflow Pipelines 是一个自然之选。如果 Apache Airflow 已用于其他 ETL 任务，将 TFX 集成到其中可能更简单。
- **团队专业知识：** 考虑您的团队对 Kubernetes、Airflow 或 Python 编程的熟悉程度。
- **规模要求：** 在 Kubernetes 上运行的 Kubeflow Pipelines 提供伸缩能力。Airflow 也能伸缩，但可能需要更多配置，具体取决于执行器设置。
- **云提供商：** 谷歌云 Vertex AI 流水线等托管服务可极大地简化基础设施管理。
- **功能需求：** 评估每个编排器在用户界面、监控、制品跟踪和机器学习 (machine learning)特定集成方面的具体功能。

通过使用编排器，您可以将 TFX 组件定义转换为一个可管理、自动化且可用于生产的机器学习系统。这使您能够专注于改进模型和数据，而编排器则处理一致且可靠地运行流水线的操作复杂性。

## 参考资料

- [TensorFlow Extended (TFX) Documentation](https://www.tensorflow.org/tfx/guide) — Google (2024)
  此官方指南提供了TFX架构、其组件以及如何将TFX管道与各种编排器集成的全面概览。
- [Apache Airflow Documentation](https://airflow.apache.org/docs/) — Apache Software Foundation (2024)
  Apache Airflow的官方文档，详细介绍了其工作流编排概念、DAGs、运算符和部署策略。
- [Kubeflow Pipelines Documentation](https://www.kubeflow.org/docs/components/pipelines/) — Kubeflow Authors (2024)
  该文档涵盖了Kubeflow Pipelines的设计原则、组件创建以及在Kubernetes上执行ML工作流的说明。
- [Vertex AI Pipelines documentation](https://cloud.google.com/vertex-ai/docs/pipelines/introduction) — Google Cloud (2024)
  Google Cloud托管式ML管道服务的官方文档，提供了运行TFX管道的无服务器环境。
- [Building Machine Learning Pipelines](https://www.oreilly.com/library/view/building-machine-learning/9781492055661/) — Hannes Hapke, Catherine Nelson, and Max Neunhöffer (2020)
  Publisher: O'Reilly Media
  本书提供了构建、部署和管理机器学习管道的实用指南，包括对生产环境中编排工具和策略的讨论。

---

[上一节](07-%E4%BD%BF%E7%94%A8%20Pusher%20%E8%BF%9B%E8%A1%8C%E6%A8%A1%E5%9E%8B%E6%9C%8D%E5%8A%A1%E4%B8%8E%E9%83%A8%E7%BD%B2.md) · [下一节](09-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9E%84%E5%BB%BA%E4%B8%80%E4%B8%AA%E7%AE%80%E5%8D%95%E7%9A%84%20TFX%20%E6%B5%81%E6%B0%B4%E7%BA%BF.md)
