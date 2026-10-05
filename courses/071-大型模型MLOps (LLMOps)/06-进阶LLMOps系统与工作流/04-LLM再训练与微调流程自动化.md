# LLM再训练与微调流程自动化

来源：[原文](https://apxml.com/zh/courses/mlops-for-large-models-llmops/chapter-6-advanced-llmops-systems-workflows/automating-llm-retraining)

[返回章节目录](README.md) · [返回课程目录](../README.md)

大型语言模型（LLM）在动态环境中运行时，其性能可能因数据分布变化（数据漂移）或模型所基于的概念发生变化（概念漂移）而随时间推移而下降。此外，持续收集反馈通常需要定期更新模型。对LLM所需规模的再训练或微调 (fine-tuning)过程进行手动触发和管理，效率低下且容易出错。自动化这些流程对于保持模型质量、响应速度和运营效率是必需的。

本节介绍如何构建LLM再训练和微调的自动化流程，集成监控触发条件、执行步骤和评估检查，以保证模型的持续改进和可靠更新。

### 为何要自动化LLM更新？

自动化再训练和微调 (fine-tuning)循环带来多项益处：

1. **一致性：** 确保更新过程每次都遵循预定义的、经过测试的步骤，减少人工干预带来的差异。
2. **可扩展性：** 处理触发、资源管理和执行可能耗时较长任务的复杂性，这些任务涉及大量数据集和分布式计算，无需持续人工监督。
3. **响应速度：** 让模型能更快适应变化的数据模式或通过监控发现的性能问题，缩短性能不佳的时间。
4. **效率：** 将工程师从重复性操作任务中解放出来，使其能专注于改进模型和流程本身。

### 自动化流程的触发条件

自动化再训练或微调 (fine-tuning)流程不会持续运行；它需要特定的触发条件。这些触发条件通常源自前一章讨论的监控系统：

1. **定时触发：** 以固定间隔（例如每周、每月）运行流程是最简便的方法。这适用于数据变化可预测或无论监控性能如何都希望定期更新的情况。
2. **性能下降警报：** 监控推断延迟、吞吐量 (throughput)或任务特定准确性等重要性能指标（KPI）非常要紧。当某项指标在持续一段时间内低于预设阈值时，警报可以自动触发流程。
3. **漂移检测警报：** 专门的监控工具可以检测输入数据分布中的统计漂移（例如，提示主题、用户人口统计信息的变化）或概念漂移（例如，术语含义的演变）。明显的漂移常表明需要更新模型。
4. **反馈积累阈值：** 包含反馈循环（例如，用户评分、修正）的系统可以在收集到足够量的新标注数据或负面反馈时触发再训练。
5. **手动触发（结合自动化）：** 尽管目标是自动化，但为*自动化流程*提供手动触发功能通常对有意的更新很有用，例如引入新的PEFT技术或进行大型数据集修订。

### 自动化再训练/微调 (fine-tuning)流程的组成部分

一个典型的自动化流程包含多个协调步骤，通常表示为有向无环图（DAG）。

> 一个通用自动化LLM再训练/微调流程结构。

我们来分析这些阶段：

1. **触发器：** 启动流程运行的事件或计划（如前所述）。
2. **数据收集：** 收集所需数据集。这可能包括查询数据仓库、拉取日志、访问反馈数据库，或从数据湖或版本控制系统（如DVC）中选择特定版本。对于微调，这通常涉及收集近期交互数据或精心挑选的样本。
3. **数据预处理：** 执行为LLM定制的必要清洗、转换、分词 (tokenization)和格式化步骤，如果处理大量数据集，可能使用Spark或Dask等可扩展数据处理框架。此阶段必须与原始训练或之前微调运行中使用的预处理保持一致。
4. **模型训练/微调：** 启动训练或微调任务。这通常是资源消耗最大的步骤。流程协调器需要提供所需的计算资源（GPU/TPU集群），配置环境，并执行训练脚本（可能使用DeepSpeed、Megatron-LM等框架或结合PEFT的标准库）。它使用分布式训练和检查点（第三章已介绍）等技术来提高效率和容错性。超参数 (parameter) (hyperparameter)、基础模型标识符和数据集版本等参数常会传递给此步骤。
5. **模型评估：** 评估新训练/微调模型的质量。这包括使用预定义评估数据集运行模型，并计算相关指标（例如，困惑度 $PPL$、特定任务的准确性、BLEU分数、ROUGE分数或自定义业务指标）。它也可能包含使用第五章讨论的技术进行偏见、毒性或幻觉 (hallucination)倾向性检查。
6. **验证关卡：** 比较新模型的性能与当前已部署模型或预定义质量阈值。这是一个重要的决策点。新模型是否有明显提升？它是否符合安全和质量标准？此关卡可以完全自动化（基于指标比较），也可以包含人机协作（HITL）步骤，由人类专家在批准推广前审查评估结果。如果验证失败，流程可能会停止或触发警报。
7. **模型注册：** 如果验证通过，新的模型工件（包括其权重 (weight)、配置以及评估结果和谱系等元数据）会被版本化并存储到模型注册中心（例如MLflow Model Registry、SageMaker Model Registry、Vertex AI Model Registry）中。这提供了一个经批准模型的集中目录。
8. **部署触发器：** 可选地，新模型的成功注册可以自动触发一个独立的CI/CD流程，负责将模型部署到预演或生产环境，可能使用金丝雀发布等高级模式（第四章已介绍）。
9. **警报/通知：** 提供流程执行状态、成功或失败的可见性，通常与Slack、PagerDuty或电子邮件等工具集成。

### LLMOps流程的协调工具

管理这些复杂、多步骤的工作流程需要一个高效的协调工具。标准的CI/CD工具（如Jenkins、GitLab CI）可以使用，但专为机器学习 (machine learning)设计的专业工作流协调器通常更合适，因为它们更了解数据并与ML生态系统集成。

常见选择包括：

- **Kubeflow Pipelines：** Kubernetes原生工作流系统，非常适合大量投入Kubernetes的组织。将流程定义为代码（Python SDK或YAML）。支持复杂的DAG、参数 (parameter)传递和工件追踪。
- **Apache Airflow：** 成熟且高度可扩展的工作流协调器，在数据工程中很流行。使用Python定义DAG。强大的社区支持和众多集成，但与Kubeflow相比，可能需要更多配置才能处理ML特定任务。
- **Argo Workflows：** Kubernetes原生工作流引擎，常作为Kubeflow Pipelines的底层引擎。可直接用于基于容器的工作流。
- **MLflow Pipelines：** 提供一个框架，用预定义步骤（摄取、拆分、训练、评估、注册）来组织ML项目，并且可以在本地或Databricks等平台上运行这些步骤。更具主张性，但有助于标准化。
- **云服务商解决方案：**
  - **AWS SageMaker Pipelines：** 与SageMaker生态系统紧密集成、完全托管的服务，用于构建、自动化和管理ML工作流。
  - **Google Cloud Vertex AI Pipelines：** 基于Kubeflow Pipelines或TFX的托管服务，与Vertex AI服务集成。
  - **Azure Machine Learning Pipelines：** Azure ML中的集成服务，用于创建、调度和管理ML工作流。

选择取决于您现有基础设施（特别是Kubernetes使用情况）、云服务商偏好、团队专业能力以及所需的定制化程度与托管便利性之间的平衡。对于LLMOps的规模，Kubernetes原生解决方案通常在管理GPU资源和复杂依赖方面提供更好的灵活性。

### 自动化流程中大规模资产的管理

LLM流程处理异常大的工件：数TB的数据集和数百GB的模型检查点。自动化必须考虑到这一点：

- **数据版本控制：** 使用DVC或Git LFS等工具，或运用数据湖内的功能（例如Delta Lake时间旅行）来对流程引用的数据集进行版本控制。流程运行应使用特定数据版本进行参数 (parameter)化，以确保可复现性。
- **高效数据传输：** 尽量减少数据移动。在靠近数据存储的地方执行预处理（例如，在数据湖上使用Spark）。在存储和计算集群之间使用高带宽连接。
- **检查点管理：** 训练任务应使用可靠的检查点机制。流程需要逻辑以在发生故障时从最新检查点恢复。检查点本身需要高效存储，可能使用分布式文件系统（如HDFS、Ceph）或云对象存储（S3、GCS、Azure Blob Storage），这些存储可被训练集群访问。
- **模型工件存储：** 模型注册中心需要处理大型模型文件。确保注册中心和底层存储配置得当。

### 集成评估与验证

自动化评估是必不可少的，但仅仅计算指标可能不足够。

- **对比评估：** 始终使用*相同*的评估数据集和指标，将候选模型与当前已部署的生产模型进行比较。这为验证关卡提供了明确的基准。
- **多套评估集：** 在代表不同数据切片或场景的多个数据集上进行评估，以获取更全面的性能概况。
- **自动化质量检查：** 包括自动评估毒性、偏见或其他安全指标的步骤。在验证关卡中为这些检查设置阈值。
- **预演环境与影子部署：** 在替换生产模型之前，流程可能会触发部署到预演环境进行进一步测试，或以影子模式部署新模型（接收生产流量但不提供响应），以比较其在线性能和稳定性与当前模型。
- **人机协作（HITL）集成：** 对于重要应用，验证关卡可能会暂停自动化并需要人工批准。协调器应支持集成此类手动审批步骤，向审查者展示评估报告和模型比较结果。



[交互图表：模型性能与再训练周期成本对比](https://apxml.com/zh/courses/mlops-for-large-models-llmops/chapter-6-advanced-llmops-systems-workflows/automating-llm-retraining#plot-1k3mr43)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "x": [
        "Cycle 1",
        "Cycle 2",
        "Cycle 3",
        "Cycle 4",
        "Cycle 5"
      ],
      "y": [
        0.75,
        0.78,
        0.77,
        0.81,
        0.83
      ],
      "type": "scatter",
      "mode": "lines+markers",
      "name": "任务准确率",
      "line": {
        "color": "#228be6"
      }
    },
    {
      "x": [
        "Cycle 1",
        "Cycle 2",
        "Cycle 3",
        "Cycle 4",
        "Cycle 5"
      ],
      "y": [
        100,
        115,
        110,
        125,
        120
      ],
      "type": "scatter",
      "mode": "lines+markers",
      "name": "再训练成本 ($)",
      "yaxis": "y2",
      "line": {
        "color": "#fd7e14"
      }
    }
  ],
  "layout": {
    "title": "模型性能与再训练周期成本对比",
    "xaxis": {
      "title": "再训练周期"
    },
    "yaxis": {
      "title": "任务准确率",
      "range": [
        0.7,
        0.9
      ],
      "tickformat": ".0%",
      "titlefont": {
        "color": "#228be6"
      },
      "tickfont": {
        "color": "#228be6"
      }
    },
    "yaxis2": {
      "title": "成本 ($)",
      "range": [
        80,
        140
      ],
      "overlaying": "y",
      "side": "right",
      "titlefont": {
        "color": "#fd7e14"
      },
      "tickfont": {
        "color": "#fd7e14"
      }
    },
    "legend": {
      "yanchor": "top",
      "y": 0.99,
      "xanchor": "left",
      "x": 0.01
    },
    "autosize": true,
    "margin": {
      "l": 60,
      "r": 60,
      "t": 40,
      "b": 40
    }
  }
}
```

</details>



> 自动化再训练周期中的性能和成本追踪。请注意，提高准确性可能与训练成本增加之间存在潜在的权衡。

### 自动化中的成本考量

自动化流程会消耗大量计算资源，特别是在训练/微调 (fine-tuning)阶段。有效的成本管理包括：

- **资源优化：** 配置流程以请求适当的GPU/TPU资源，并在使用后及时释放。在容错允许的情况下，使用竞价实例。
- **高效训练技术：** 在不需要完全再训练时采用PEFT方法，大幅减少计算需求。
- **智能调度：** 如果可能，在非高峰时段安排资源密集型流程运行。
- **成本监控：** 将成本追踪集成到流程报告中，以了解每次自动化运行的相关开销。在协调器或云平台上设置预算警报。
- **条件执行：** 设计验证关卡以避免不必要的再训练，如果性能提升相对于所产生的成本微不足道。

构建自动化再训练和微调流程是迈向成熟LLMOps的重要一步。它将模型维护从被动的手动过程转变为主动、一致且可扩展的运营，保证您的大型语言模型能持续有效地提供价值。

## 参考资料

- [MLOps: A guide to accelerate trusted, production-ready machine learning](https://services.google.com/fh/files/misc/practitioners_guide_to_mlops_whitepaper.pdf) — Google Cloud (2021)
  Publisher: Google Cloud
  提供MLOps原则的权威概述，包括机器学习系统的自动化管道、监控和持续交付。
- [Data Drift in Natural Language Processing: A Systematic Review](https://doi.org/10.1007/s42979-023-01915-w) — Oumaima Boukkhriss, Mustapha Amhoual, Abdellatif Kobbane (2023)
  Journal: SN Computer Science; Publisher: Springer; Volume: 4; Pages: article number 427; DOI: [10.1007/s42979-023-01915-w](https://doi.org/10.1007/s42979-023-01915-w)
  一篇针对自然语言处理领域数据漂移的系统性综述，对于理解大型语言模型（LLM）环境中的漂移现象非常重要。
- [Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  一本关于构建和运行生产级机器学习系统的综合书籍，涵盖管道设计、监控和系统架构。
- [Kubeflow Pipelines Documentation](https://www.kubeflow.org/docs/components/pipelines/) — The Kubeflow Authors (2024)
  Kubeflow Pipelines的官方文档，这是一个Kubernetes原生的工作流系统，广泛用于编排机器学习管道，包括LLM再训练的管道。

---

[上一节](03-%E5%90%91%E9%87%8F%E6%95%B0%E6%8D%AE%E5%BA%93%E6%93%8D%E4%BD%9C%E4%B8%8E%E7%AE%A1%E7%90%86.md) · [下一节](05-LLMOps%E4%B8%AD%E7%9A%84%E5%AE%89%E5%85%A8%E8%80%83%E9%87%8F.md)
