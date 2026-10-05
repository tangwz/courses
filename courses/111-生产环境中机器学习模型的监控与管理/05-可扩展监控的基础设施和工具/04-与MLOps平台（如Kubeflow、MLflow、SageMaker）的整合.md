# 与MLOps平台（如Kubeflow、MLflow、SageMaker）的整合

来源：[原文](https://apxml.com/zh/courses/monitoring-managing-ml-models-production/chapter-5-scalable-monitoring-infrastructure/mlops-platform-integration)

[返回章节目录](README.md) · [返回课程目录](../README.md)

您的监控系统，无论多么精细，都不能独立运行。它必须与您的模型被开发、训练、注册和部署的整个MLOps生态系统有效配合。将监控功能直接整合到Kubeflow、MLflow或Amazon SageMaker等MLOps平台中，能将监控从被动的外部流程转变为模型生命周期中主动且不可或缺的一部分。这种整合使工作流程更高效，增强了自动化程度，并为解读监控信号提供了重要的背景信息。

目标是使监控数据和操作在您的ML工程师和数据科学家已使用的工具中变得可访问和可操作。这不仅仅是将指标简单地导入仪表盘，它还意味着借助平台特性进行编排、元数据追踪以及触发自动化响应。

### 常见整合模式

在详细查看平台特性之前，请考虑以下常见整合策略：

1. **API驱动的交互：** 大多数MLOps平台都暴露API接口（通常是RESTful），允许外部系统查询元数据（例如，模型版本、部署状态、训练参数 (parameter)），有时还允许回传信息（例如，将监控结果与特定模型产物关联）。您的监控系统可以使用这些API来获取其正在监控的模型的背景信息。
2. **元数据存储：** 平台通常维护一个元数据存储（如Kubeflow和TFX使用的ML Metadata (MLMD)，或MLflow Tracking Server的后端）。接入此存储能让监控系统理解模型谱系、用于训练的数据集以及部署历史，这对于诊断问题非常有价值。反之，监控结果（漂移报告、性能摘要）有时也可以作为元数据产物被记录。
3. **事件驱动钩子：** 平台可能会针对重要的生命周期事件（例如，模型注册、部署成功/失败）发送事件或支持Webhooks。监控系统可以订阅这些事件，以便自动为新部署配置监控或触发特定检查。
4. **编排整合：** 监控任务本身（例如，运行批处理漂移计算作业）通常可以在平台的P工作流引擎中定义并作为步骤执行（如Kubeflow Pipelines、SageMaker Pipelines，或通过编排器执行的MLflow Projects）。
5. **日志和指标标准：** 运用平台的标准机制进行应用日志记录和指标抓取（例如，与Prometheus、CloudWatch整合），提供了一种统一的方式来收集监控组件本身的操作数据。

### 与MLflow整合

MLflow提供用于追踪实验、打包代码和管理模型的组件。将您的监控系统与之整合通常涉及与MLflow Tracking Server和Model Registry的交互。

#### 使用Tracking Server

Tracking Server存储有关实验运行的信息，包括参数 (parameter)、指标和产物。虽然它主要在训练期间使用，但也可以作为记录与*已部署*模型相关联的监控结果的目标。

- **记录监控指标：** 您的监控服务可以定期为已部署的模型版本计算指标（例如，日均预测延迟、周漂移分数）。然后，它可以使用MLflow客户端库将这些指标记 (token)录回与该模型训练或部署相关联的特定MLflow运行。这有助于保留生产性能的历史记录，与训练指标并列。

  ```python
  # 示例：将漂移分数记录回MLflow运行
  import mlflow

  # 假设 'deployed_model_run_id' 是与当前已部署模型版本训练
  # 相关联的 run_id
  deployed_model_run_id = "abc123xyz789"
  drift_score = calculate_drift(...) # 您的漂移计算逻辑

  mlflow.set_tracking_uri("http://your-mlflow-server:5000")
  with mlflow.start_run(run_id=deployed_model_run_id) as run:
      mlflow.log_metric("production_drift_score", drift_score, step=int(time.time()))
  ```
- **存储监控产物：** 您的监控系统生成的复杂漂移报告、性能分析图或数据段快照，可以作为产物记录到相关的MLflow运行中。

#### 与Model Registry交互

Model Registry管理模型的生命周期（暂存、生产、归档）。

- **发现模型：** 您的监控系统可以查询Model Registry API，以发现哪些模型版本当前处于生产或暂存阶段，并获取它们的URI（例如，`models:/MyModel/Production`）或特定的产物位置。这确保了监控针对正确的模型资产。
- **Registry Webhooks：** MLflow支持注册表事件的Webhooks（例如，将模型版本转换为“生产”）。您可以配置Webhook通知您的监控系统，触发其自动为新晋升的模型版本设置监控配置（仪表盘、警报、漂移检查）。
- **将监控与谱系关联：** 通过获取与注册模型版本相关联的`run_id`，您的监控系统可以将生产行为与使用的特定训练实验、参数和数据集关联起来，从而协助根源分析。

> 监控系统与MLflow组件之间的交互流程。监控代理使用Model Registry识别目标，并使用Tracking Server回传结果，将生产行为与模型的历史关联起来。

### 与Kubeflow整合

Kubeflow提供了一个在Kubernetes上进行ML的全面平台，包括流水线、模型服务和元数据追踪。整合的重点是运用Kubeflow Pipelines进行编排，以及KFServing/KServe来获取部署背景。

#### Kubeflow Pipelines (KFP)

KFP允许您将ML工作流定义和编排为有向无环图（DAGs）。

- **将监控作为流水线步骤：** 将监控任务（数据验证、漂移检查、在近期数据窗口上进行性能评估）作为独立的组件在KFP流水线中实现。这使您能够调度周期性监控运行，或将监控检查作为CI/CD或再训练流水线的一部分。
- **编排监控设置：** KFP中的部署流水线可以在模型成功部署（通过KFServing/KServe）*之后*包含步骤，以自动配置必要的监控资源（例如，设置Prometheus抓取目标、创建Grafana仪表盘定义、在Alertmanager中注册警报规则）。
- **传递元数据：** KFP本身会追踪每个步骤的输入和输出。您的监控组件可以消费由先前步骤产生的产物（如数据集位置、模型URI），并生产自己的产物（漂移报告），供后续步骤（如触发再训练）使用。

> 示例Kubeflow流水线，在模型部署后，将监控设置和周期性检查作为独立的组件。

#### KFServing / KServe

KFServing（现为KServe）提供了一个在Kubernetes上部署模型的标准接口。

- **标准化指标与日志：** KServe部署通常会暴露标准操作指标（延迟、请求计数、CPU/内存使用率），这些指标可被Prometheus抓取。应用日志可以使用标准Kubernetes日志代理（如Fluentd）收集。您的自定义监控可以使用这些标准输出。
- **请求/响应日志记录：** KServe允许配置预测请求和响应的日志记录（通常发送到Kafka或指定的URL端点）。这为漂移和性能监控系统提供了所需的原始数据。您的监控系统可以订阅此数据流。
- **自定义监控Sidecar：** 对于更复杂或实时的监控逻辑，您可以将自定义监控代理作为Sidecar容器部署在与KServe模型服务相同的Kubernetes Pod中。此Sidecar可以根据需要直接访问预测数据或模型内部，执行分析，并暴露指标或将结果推送到外部。
- **元数据整合：** KServe资源通常包含标签和注解，将它们链接到KFP运行或其他元数据。监控系统可以查询Kubernetes API以获取这些元数据作为背景信息。

#### Kubeflow 元数据 (MLMD)

Kubeflow使用ML Metadata (MLMD)来追踪产物、执行及其关系。

- **记录监控执行：** 通过KFP执行的监控流水线运行会自动记录执行元数据。自定义监控作业也可以直接将元数据写入MLMD（例如，记录漂移检查的执行、其输入数据产物以及生成的漂移报告产物）。
- **查询谱系：** 在计算漂移或性能之前，您的监控系统可以查询MLMD，以找到与已部署模型产物相关联的确切训练数据集产物。这使得能够直接比较生产数据和原始训练数据分布。

### 与Amazon SageMaker整合

SageMaker为ML生命周期提供托管服务，其中一些与监控整合相关。

#### SageMaker Model Monitor

SageMaker提供了一个内置服务Model Monitor，用于检测数据质量问题、数据漂移、偏差漂移和特征归因漂移。

- **运用内置功能：** 对于常见的漂移类型（基于Deequ进行数据质量检查或与训练约束进行比较），Model Monitor提供托管作业和报告。您的系统可以直接运用此功能进行基线监控。
- **消费监控器输出：** Model Monitor作业会将结果（约束违规、统计数据）发布到S3和CloudWatch指标。您的自定义监控仪表盘（例如，Grafana）可以使用这些CloudWatch指标，或者可以根据S3报告触发自定义分析/警报逻辑。
- **与自定义逻辑互补：** Model Monitor可能未涵盖所有期望的漂移检测方法（例如，高级多变量漂移、特定于您领域的概念漂移）或性能指标。您可以运行自定义监控作业（或许使用SageMaker Processing Jobs），它们使用相同的捕获数据（见下文）来执行额外检查，并将其结果与Model Monitor的输出整合。

#### SageMaker 端点与数据捕获

SageMaker实时端点是主要的部署目标。

- **端点数据捕获：** 此功能允许您配置端点，自动捕获一部分请求和/或响应负载并将其存储在S3中。这是为SageMaker Model Monitor和自定义监控系统提供数据的主要机制。您的监控作业可以直接从这个S3位置读取。
- **CloudWatch整合：** 端点会自动发送操作指标（延迟、调用、错误）到CloudWatch。自定义监控仪表盘和警报可以直接基于这些指标构建。端点日志也会发送到CloudWatch Logs，这有助于调试操作问题。

#### SageMaker 流水线与模型注册表

与KFP和MLflow类似，这些组件促进编排和治理。

- **监控流水线步骤：** SageMaker Pipelines可以包含运行Model Monitor作业、自定义监控/处理作业的步骤，或者在模型部署后根据监控指标配置CloudWatch警报的步骤。
- **Model Registry整合：** SageMaker Model Registry追踪模型版本及其批准状态。您可以根据模型包组状态变化自动化监控设置（例如，当某个版本获批生产时，设置数据捕获和Model Monitor调度）。由Model Registry上的EventBridge事件触发的Lambda函数是这种自动化的常见模式。

> 使用Amazon SageMaker的整合架构。从端点捕获的数据会同时供给SageMaker Model Monitor和自定义处理作业，结果存储在S3和CloudWatch中，以驱动仪表盘和警报。EventBridge和Lambda根据Model Registry事件自动化设置。

### 跨平台考量

运用平台特定功能虽然提供便利，但也可能导致供应商锁定。请考虑以下几点：

- **抽象层：** 您可以构建内部库或接口，以抽象化与不同MLOps平台交互的细节。您的监控代码与您的抽象层交互，抽象层随后将调用转换为相应的MLflow、Kubeflow或SageMaker API。
- **平台无关工具：** 尽可能使用本身平台中立的监控工具（日志库、漂移检测器、仪表盘软件）。例如，使用Prometheus和Grafana适用于Kubernetes (Kubeflow)、虚拟机，甚至可以与SageMaker一起使用（拉取CloudWatch指标）。
- **通过事件解耦：** 采用消息总线（如Kafka、RabbitMQ、Kinesis）可以解耦监控组件。平台或自定义代理发布事件（例如，`prediction_logged`、`drift_detected`），各种监控服务订阅这些事件以执行其任务，从而减少直接依赖。CloudEvents是一种描述通用格式事件数据的新兴标准。

最终，整合深度的选择取决于您团队现有的MLOps技术栈、监控需求的复杂性以及您关于平台灵活性的策略。紧密整合提供自动化优势，而更解耦的方法则提供可移植性。有效监控通常需要务实的整合，在平台特性提供显著优势时使用它们，并尽可能以更具可移植性的方式维护核心监控逻辑。

## 参考资料

- [MLOps: Continuous delivery and automation pipelines in machine learning](https://cloud.google.com/solutions/machine-learning/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning) — Google Cloud (2021)
  Publisher: Google Cloud
  介绍了MLOps原则、机器学习的持续交付和自动化管道，包括监控在生命周期中的作用。
- [MLflow Documentation](https://www.mlflow.org/docs/latest/index.html) — Databricks (Current)
  MLflow的官方文档，详细说明了如何使用跟踪服务器、模型注册表和客户端库来集成监控结果和模型元数据。
- [Kubeflow Documentation](https://www.kubeflow.org/docs/components/) — The Kubeflow Authors (2025)
  Kubeflow的官方文档，介绍了Kubeflow Pipelines、KServe和ML Metadata (MLMD)等组件，用于在Kubernetes原生的ML工作流中构建和集成监控解决方案。
- [Amazon SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/index.html) — Amazon Web Services (Current)
  Publisher: Amazon Web Services
  Amazon SageMaker的官方文档，介绍了模型监控器、数据捕获、SageMaker Pipelines和模型注册表等服务，用于在AWS上监控和管理模型。

---

[上一节](03-%E7%9B%91%E6%8E%A7%E6%B5%81%E7%A8%8B%E7%9A%84%E5%88%86%E5%B8%83%E5%BC%8F%E6%9E%B6%E6%9E%84.md) · [下一节](05-%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%9B%91%E6%8E%A7%E4%B8%93%E7%94%A8%E5%B7%A5%E5%85%B7%E5%92%8C%E6%9C%8D%E5%8A%A1.md)
