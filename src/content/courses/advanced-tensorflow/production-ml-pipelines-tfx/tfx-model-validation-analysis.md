---
course: "advanced-tensorflow"
chapter: "production-ml-pipelines-tfx"
lesson: "tfx-model-validation-analysis"
sourceId: 2906
sourceUrl: "https://apxml.com/zh/courses/advanced-tensorflow/chapter-5-production-ml-pipelines-tfx/tfx-model-validation-analysis"
title: "模型验证与分析"
description: "使用 TFX Evaluator 和 ModelValidator 对训练后的模型进行细致分析和验证。"
order: 6
plots: ["plots/2906-0.json"]
sourceHash: "1dfcfb149f0b5fac44c4fd5f06fe4f60f671c43052a1c3247f78067915d7f540"
sourceCorrections: []
---

即使潜在的模型候选已经通过 `Trainer` 组件进行了训练，仅仅在静态测试集上取得良好性能不足以将其部署到生产环境。生产系统要求全面的验证，以确保模型不仅整体表现良好，而且在不同数据分段中表现可预测且公平，更重要的是，与当前服务的模型相比没有性能下降。因此，TFX 管道中的模型验证和分析步骤变得非常必要。

负责这种细致性能分析的主要 TFX 组件是 `Evaluator`。它远不止计算一个单一的准确率分数。`Evaluator` 借助 Apache Beam 的能力进行可扩展分析，并与 TensorFlow Model Analysis (TFMA) 库紧密集成，以提供关于模型行为的详细信息。

### 了解 TFX Evaluator

`Evaluator` 组件通常接收：

1. **示例：** 评估数据，通常由训练期间使用的相同 `Transform` 图处理，以确保一致性。
2. **模型：** 由 `Trainer` 组件导出的新训练的候选模型。
3. **基准模型（可选但建议）：** 当前在生产中服务的模型或之前验证过的模型。这使得能够直接比较，以防止性能下降。

其核心功能是计算多种评估指标，不仅整体上，而且跨数据不同*切片*。切片分析可以帮助您查看模型是否在特定用户群、时间段、特征值或其他重要数据分段中表现不同。例如，您可能希望验证模型在不同地理区域的用户或不同产品类别上的表现是否同样良好。

```python
# 示例：在 TFX 管道定义中配置 Evaluator
from tfx.components import Evaluator
from tfx.proto import evaluator_pb2

# 假设 'trainer'、'examples_gen' 和 'schema_gen' 已在上游定义

eval_config = evaluator_pb2.EvalConfig(
    model_specs=[
        # 指定候选模型
        evaluator_pb2.ModelSpec(label_key='loan_status_binary')
    ],
    slicing_specs=[
        # 定义切片：整体数据集
        evaluator_pb2.SlicingSpec(),
        # 按特定特征 'employment_length' 进行切片
        evaluator_pb2.SlicingSpec(
            feature_keys=['employment_length']
        ),
        # 按两个特征的交叉进行切片
        evaluator_pb2.SlicingSpec(
            feature_keys=['home_ownership', 'verification_status']
        )
    ],
    metrics_specs=[
        # 定义使用 TFMA 指标定义来计算的指标
        evaluator_pb2.MetricsSpec(
            metrics=[
                evaluator_pb2.MetricConfig(class_name='AUC'),
                evaluator_pb2.MetricConfig(class_name='Precision'),
                evaluator_pb2.MetricConfig(class_name='Recall'),
                evaluator_pb2.MetricConfig(
                    class_name='ExampleCount' # 总是很有用
                ),
                evaluator_pb2.MetricConfig(
                  # 阈值设置示例：候选模型 AUC 必须比基准模型高 1%
                  class_name='AUC',
                  threshold=evaluator_pb2.MetricThreshold(
                      value_threshold=evaluator_pb2.GenericValueThreshold(
                          lower_bound={'value': 0.01}),
                      # 相对于基准模型的 AUC 进行比较
                      change_threshold=evaluator_pb2.GenericChangeThreshold(
                          direction=evaluator_pb2.MetricDirection.HIGHER_IS_BETTER,
                          absolute={'value': 0.01})
                  )
                )
            ]
        )
    ]
)

evaluator = Evaluator(
    examples=examples_gen.outputs['examples'],
    model=trainer.outputs['model'],
    # baseline_model=model_resolver.outputs['model'], # 如果与基准模型进行比较
    eval_config=eval_config,
    schema=schema_gen.outputs['schema'] # 模式有助于 TFMA 解释特征
)

# 此 'evaluator' 组件随后将添加到管道的组件列表中
```

TFMA 支持的 `Evaluator` 的重要方面包括：

- **切片评估：** 在由特征值定义的数据子集上计算指标。这对于识别性能差异和潜在的公平性问题非常重要。
- **指标计算：** 计算标准分类和回归指标（AUC、Precision、Recall、RMSE 等），以及通过 TensorFlow/Keras 定义的自定义指标。
- **基准比较：** 在相同的评估数据切片上，直接将候选模型的指标与基准模型进行比较。这有助于确保新模型确实是改进，并且不会降低某些用户分段的性能。
- **阈值设置：** 定义候选模型必须满足的特定性能标准，无论是绝对值（例如，AUC > 0.85）还是相对于基准（例如，Precision 不得降低）。
- **公平性指标：** TFMA 包含评估敏感人口统计或基于组的切片上的公平性指标的能力，有助于识别和减轻意外的模型偏差。

`Evaluator` 的输出包括详细的分析结果，通常使用 TensorBoard 的 TFMA 插件进行可视化，以及一个重要的验证结果（通常称为“祝福”）。此结果表明模型是否通过了预定义的质量阈值。



![AUC 比较：候选模型与基准模型在各切片上的表现](plots/2906-0.json)



> 此比较显示了跨不同数据切片的性能。虽然候选模型显示出整体 AUC 提高，并在区域 B 表现良好，但在区域 A 和新用户方面与基准模型相比略有退步。这种由 `Evaluator` 的切片分析提供的信息，对于做出明智的部署决策非常重要。

### 使用 ModelValidator 确保模型健全

`Evaluator` 侧重于*预测性能*，而另一个组件 `ModelValidator` 则处理模型工件本身的*技术有效性*和*一致性*。它有助于发现仅凭性能指标可能不明显的问题，但可能在服务期间引发问题或表明底层数据漂移。

`ModelValidator` 通常检查：

- **基础设施验证：** 确保模型能够被服务基础设施（例如 TensorFlow Serving）实际加载。它执行基本检查，如尝试加载 SavedModel。
- **漂移和异常检测：** 比较当前模型预测或内部激活的统计数据与先前版本或基准分布的统计数据。显著的偏差可能表明训练/服务偏差或数据模式中出现意外变化。它通常使用来自 `SchemaGen` 的模式信息和来自 `StatisticsGen` 的统计数据进行这些比较。

`Evaluator` 询问“这个模型*更好*吗？”，而 `ModelValidator` 询问“这个模型*健全*且*一致*吗？”。两者都是部署前的重要检查。

### 守门员角色

`Evaluator`（有时也包括 `ModelValidator`）生成的验证结果充当关键的关卡。下游的 `Pusher` 组件负责将模型部署到服务环境或模型注册表，它通常会使用验证结果。如果 `Evaluator` 未“祝福”模型（即，它未达到性能阈值或显示性能下降），`Pusher` 将不会部署它。这种自动化质量检查可以防止次优或性能下降的模型进入生产环境，维护机器学习 (machine learning)系统的可靠性和可信赖性。

总之，模型验证和分析阶段主要由使用 TFMA 的 `Evaluator` 组件驱动，它提供生产环境所需的细致、多方面的性能信息。通过跨数据切片评估模型、与基准模型比较以及执行质量阈值，TFX 确保只有经过充分验证且性能稳定的模型才会被考虑部署。

## 参考资料

- [TensorFlow Extended (TFX) Documentation](https://www.tensorflow.org/tfx) — Google (2024)
  提供关于TFX组件（包括评估器和模型验证器）及其在生产ML管道中作用的全面信息。
- [TensorFlow Model Analysis (TFMA) Guide](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGcwXAeJ31_ZHj5Wm-Q0JQUMPOCxzGj30Z9KRlsOV8GngolloFlpRnY4fDKuIE8nOp2fffbDwjO-sGAu_twRGFpORa7VHSYlX_G4gF_tTraqxZuuPU8cCb57cMMBZtkkdERJ1NHHGFhcSccgY-iSLkZbHbJAA==) — Google (2024)
  Publisher: Google
  深入了解TFMA功能的详细资源，特别是在切片评估、基线比较和公平性指标方面。
- [Designing Machine Learning Systems: An Iterative Process for Production-Ready AI](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  一本关于构建、部署和维护机器学习系统的综合指南，涵盖了生产环境中健全验证和分析的重要性。
