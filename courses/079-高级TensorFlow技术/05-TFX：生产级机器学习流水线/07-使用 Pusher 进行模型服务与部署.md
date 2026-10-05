# 使用 Pusher 进行模型服务与部署

来源：[原文](https://apxml.com/zh/courses/advanced-tensorflow/chapter-5-production-ml-pipelines-tfx/tfx-serving-deployment-pusher)

[返回章节目录](README.md) · [返回课程目录](../README.md)

TensorFlow Extended (TFX) 框架中的 `Pusher` 组件在部署机器学习 (machine learning)模型方面扮演着主要角色。它的功能是使经过训练并充分验证的模型能够用于生产流水线中的预测服务。该组件充当把关者，确保只有经过批准的模型才能部署到指定的服务基础设施。

`Pusher` 组件旨在有条件地将经过验证的模型工件推送到预定义的部署目标。它的运行取决于模型验证步骤的结果。

### 基于验证的条件部署

回想一下，`Evaluator` 组件会使用在保留数据集上计算出的评估指标，将新训练的模型与基准模型（通常是当前已部署的模型）进行比较。根据预定义的阈值和标准，`Evaluator` 会生成一个 `ModelBlessing` 工件。此工件表示新模型是否达到了部署所需的质量标准。

`Pusher` 组件接收两个主要输入：

1. **Model Artifact**：由 `Trainer` 组件生成的已训练模型，通常是 SavedModel 格式。
2. **ModelBlessing Artifact**：来自 `Evaluator` 或 `ModelValidator` 的输出。

`Pusher` 的核心逻辑很简单：如果 `ModelBlessing` 工件表明模型“通过验证”（即它通过了验证），`Pusher` 就会将 `Model` 工件复制到指定的 `push_destination`。如果模型未通过验证，`Pusher` 不会采取任何行动，从而防止性能可能不佳的模型进入生产服务环境。这一条件步骤对于维护机器学习 (machine learning)系统的可靠性非常重要。

> `Pusher` 从 `Trainer` 接收模型，从 `Evaluator` 接收 `ModelBlessing`，并且只有在模型通过验证后才将其推送到部署目标。

### 部署目标与配置

`Pusher` 组件配置的 `push_destination` 决定了经过验证的模型被复制到哪里。虽然 TFX 具有灵活性，但一个常见且受良好支持的目标是与 TensorFlow Serving 兼容的文件系统目录结构。

TensorFlow Serving 监控指定目录以获取新的模型版本。它期望一个特定的结构，通常类似于 `/path/to/base/model_name/version_number/saved_model.pb|variables/`。`Pusher` 负责自动创建此结构。当推送一个通过验证的新模型时，`Pusher` 会在模型的基本路径内创建一个新的编号子目录（表示模型版本），并将 SavedModel 内容复制到其中。然后，TensorFlow Serving 可以检测并加载这个新版本以服务请求。

这是一个使用 Python 配置 TFX 流水线定义中 `Pusher` 组件的简化示例：

```python
# 假设 'trainer' 和 'evaluator' 是之前定义的 TFX 组件
# 分别生成 'model' 和 'blessing' 工件。

from tfx.components import Pusher
from tfx.proto import pusher_pb2
import os

# 定义 TF Serving 监控的位置
serving_model_dir = os.path.join('/path/to/serving/models/my_model')

pusher = Pusher(
    model=trainer.outputs['model'],
    model_blessing=evaluator.outputs['blessing'],
    push_destination=pusher_pb2.PushDestination(
        filesystem=pusher_pb2.PushDestination.Filesystem(
            base_directory=serving_model_dir
        )
    )
)

# 这个 'pusher' 组件随后将被包含在流水线的组件列表中。
```

在此配置中：

- `model=trainer.outputs['model']` 将 `Pusher` 连接到 `Trainer` 的输出模型工件。
- `model_blessing=evaluator.outputs['blessing']` 将其连接到 `Evaluator` 的验证结果。
- `push_destination` 指定了目标。这里，`pusher_pb2.PushDestination.Filesystem` 指示流水线和服务系统可访问的文件系统上的一个位置。`base_directory` 是将创建版本化模型目录的根路径（例如，`/path/to/serving/models/my_model/1/`、`/path/to/serving/models/my_model/2/` 等）。

尽管 TensorFlow Serving 是一个主要目标，但 `Pusher` 也可以配置为推送到其他位置，例如 Google Cloud Storage (GCS)、AWS S3，甚至通过实现自定义执行器来触发自定义部署逻辑，尽管基于文件系统的 TF Serving 部署是最常见的用例。如果部署到 TF Lite，`Pusher` 可能会将转换后的 `.tflite` 模型文件（通常由自定义组件或修改后的 `Trainer` 生成）推送到指定位置，用于移动应用程序打包或边缘设备部署。

### 与 ML 元数据 (MLMD) 的集成

与所有标准 TFX 组件一样，`Pusher` 与 ML 元数据进行交互。当它成功推送模型时，它会将此事件记录在 MLMD 存储中。这会创建一个可追溯的记录，连接特定的模型工件、其验证状态 (`ModelBlessing`) 以及其部署位置和版本。此元数据对于追踪模型来源、了解生产中正在服务的模型版本以及在出现问题时回滚部署很有帮助。

总而言之，`Pusher` 组件在 TFX 流水线中扮演着重要的控制点角色。它通过自动化验证模型的条件部署，将模型开发/验证与生产服务有效衔接起来。通过使用 `ModelBlessing` 工件并与 TensorFlow Serving 和 MLMD 等部署目标配合，`Pusher` 有助于确保只有高质量、经过审查的模型才能提供给最终用户，从而对生产机器学习 (machine learning)系统的可靠性和可维护性带来很大帮助。

## 参考资料

- [TFX Pusher component](https://www.tensorflow.org/tfx/guide/pusher) — TensorFlow (2023)
  提供TFX Pusher组件的官方详情，包括其输入、条件部署逻辑和配置。
- [TensorFlow Serving: A basic guide](https://www.tensorflow.org/tfx/serving/serving_basic) — TensorFlow (2023)
  解释TensorFlow Serving的工作原理，包括模型版本控制和预期的目录结构，这是TFX Pusher常见的部署目标。
- [MLOps: Continuous delivery and automation pipelines in machine learning](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning) — Google Cloud (2022)
  讨论MLOps的原则、持续交付和机器学习管道中的自动化，为Pusher等组件对可靠部署的重要性提供背景。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098122461/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media; Pages: 861
  一本全面的书籍，详细介绍了TensorFlow Extended (TFX)和TensorFlow Serving的实践，包括讨论的部署阶段。

---

[上一节](06-%E6%A8%A1%E5%9E%8B%E9%AA%8C%E8%AF%81%E4%B8%8E%E5%88%86%E6%9E%90.md) · [下一节](08-%E7%BC%96%E6%8E%92%20TFX%20%E6%B5%81%E6%B0%B4%E7%BA%BF.md)
