---
course: "quantized-llm-deployment"
chapter: "optimizing-deploying-quantized-llms"
lesson: "monitoring-deployed-quantized-models"
sourceId: 4662
sourceUrl: "https://apxml.com/zh/courses/quantized-llm-deployment/chapter-4-optimizing-deploying-quantized-llms/monitoring-deployed-quantized-models"
title: "监控已部署的量化模型"
description: "监控已部署量化模型的性能、资源使用和运行状况的方法。"
order: 8
plots: ["plots/4662-0.json"]
sourceHash: "7c459d5845cd19a3b59160f2af61d79e69f7ecd9d0218986ccbd9458bf5b9ab5"
sourceCorrections: []
---

部署量化 (quantization)大型语言模型是一个重要进展，但工作并未就此停止。持续监控是必不可少的，以确保模型在生产环境中可靠、高效、准确地运行。量化会带来一些特定敏感性；性能可能因硬件内核执行而异，准确性可能随时间或输入数据分布的变化而发生细微偏差。有效的监控能提供必要的可见性，以便及早发现并处理这些问题。

### 主要监控方面

监控已部署的量化 (quantization)模型涉及跟踪几类指标：

1. **性能指标：** 这些指标跟踪推理 (inference)服务的运行速度和效率。
2. **准确性和输出质量指标：** 这些指标评估模型是否持续满足其功能要求。
3. **资源使用指标：** 这些指标衡量底层硬件资源的消耗情况。
4. **运行状况指标：** 这些指标提供部署稳定性与可用性的信息。

下面我们更详细地查看每个方面。

### 性能监控

性能一致性是量化 (quantization)的主要目标。监控主要性能指标（KPIs）有助于确认量化模型能够达到所需的速度和吞吐量 (throughput)。

- **延迟：** 衡量处理单个推理 (inference)请求所需的时间。跟踪不同百分位数非常重要，例如 p50（中位数）、p90 和 p99，以了解典型的用户体验以及最差情况。p99 延迟的明显增加可能表明潜在问题，即使中位数保持稳定。
- **吞吐量：** 衡量服务每单位时间可处理的请求数量（例如，每秒请求数）。在不同负载条件下监控此项，以了解扩展行为。
- **冷启动时间：** 对于无服务器或自动扩缩部署，跟踪新实例准备好服务请求所需的时间。量化模型由于体积更小，通常在此方面有优势，但仍值得监控。

标准应用程序性能监控（APM）工具，如 Prometheus 搭配 Grafana、Datadog、Google Cloud Monitoring 或 Azure Monitor，非常适合收集和可视化这些指标。对偏离既定基线的显著情况设置警报是一种标准做法。



![推理延迟监控](plots/4662-0.json)



> P95 推理延迟随时间变化的示例可视化，显示了一个超过预设警报阈值的峰值。

### 准确性和输出质量监控

量化 (quantization)本身涉及一种权衡，在性能/大小和准确性之间。虽然部署前评估（第3章已介绍）确立了初始准确性，但监控生产环境中的潜在性能下降情况也很重要。

- **直接准确性评估：** 定期使用精心策划的评估数据集（“黄金数据集”）来运行已部署的模型，其中已知真实标签。计算相关任务特定指标（例如，准确率、F1 分数、BLEU、ROUGE）。这提供了准确性的直接衡量，但计算成本高昂，且可能无法捕捉由实时数据模式变化引起的漂移。
- **代理指标：** 跟踪与输出质量相关但更易于计算的指标。示例包括：
  - 平均输出长度
  - 特定关键词或实体的存在/频率
  - 情感分数分布（用于文本生成）
  - 生词或不寻常令牌的频率
  - 拒绝回答或“我不知道”回应的频率
    这些代理指标的明显变化可能表明模型质量存在潜在问题或输入数据分布发生变化（数据漂移）。
- **数值稳定性：** 监控是否存在意外的数值输出，如 `NaN`（非数字）或 `Inf`（无穷大）。低比特量化有时会增加在某些输入条件下出现数值不稳定性的风险。记录并对此类情况发出警报。
- **人工干预：** 对于重要应用程序，纳入人工审查抽样输出的机制。这提供了对感知质量最可靠的评估，但成本高昂且速度慢。反馈机制（例如，点赞/点踩按钮）也能提供有价值的信号。

检测漂移是一个主要难题。监控输入提示的统计特性（例如，长度分布、令牌频率），并将其与训练或校准期间观察到的分布进行比较。专门的机器学习 (machine learning)监控工具通常提供检测数据漂移和含义漂移的功能。

### 资源使用监控

量化 (quantization)旨在减少资源需求，尤其是内存。监控这些资源可以验证预期收益，并有助于防止内存不足错误或性能瓶颈。

- **GPU/CPU 利用率：** 跟踪正在使用的处理能力百分比。低利用率可能表明存在 I/O 瓶颈或批处理效率低下，而持续高利用率可能预示着需要扩容。
- **内存使用：** 监控 GPU 显存 (VRAM)和系统内存的使用情况。特别是在负载下，密切关注推理 (inference)期间的峰值内存消耗。确保其保持在分配的限制内。量化模型应显示出比其全精度模型明显更低的显存使用量。
- **磁盘 I/O：** 监控磁盘读/写操作，特别是当模型权重 (weight)动态加载或记录大量数据时。
- **网络 I/O：** 跟踪推理服务发送和接收的数据量。

同样，这里也使用标准基础设施监控工具。将资源使用高峰与延迟增加或错误率关联起来，有助于诊断问题。

### 运行状况监控

这些指标侧重于已部署服务的整体可靠性和可用性。

- **运行时长/可用性：** 跟踪服务正常运行并成功响应请求的时间百分比。
- **错误率：** 监控不同 HTTP 错误代码的频率（例如，4xx 客户端错误，5xx 服务器错误）。5xx 错误的激增通常表明推理 (inference)服务本身存在问题。
- **请求日志：** 记录传入请求（时间戳、可能已清洗的输入片段）和响应（时间戳、状态、延迟）的重要信息。这对调试非常有帮助。

### 建立基线和反馈循环

有效的监控依赖于将当前指标与既定基线进行比较。在典型负载条件下成功部署后，立即捕获详细的性能、准确性和资源指标。这些基线是检测性能下降或异常情况的参考点。

从监控中获得的信息应反馈到 MLOps 生命周期中。例如：

- 延迟持续增加可能促使调查基础设施瓶颈或需要进一步优化（例如使用前面讨论的 TensorRT-LLM）。
- 检测到的准确性漂移可能需要重新训练模型或调整量化 (quantization)策略（例如，使用不同的校准数据集或研究 QAT）。
- 高错误率可能指向部署代码中的错误，或导致量化模型出现问题的特定类型输入数据。

监控并非一次性设置；它是一个持续的过程。定期审查监控仪表板，调整警报阈值，并随着应用程序和模型的演变而调整监控策略。适当的监控确保了量化带来的效率提升在生产中可靠且可持续地实现。

## 参考资料

- [Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQF2WsCnPJjsfy7NGo5zLAbRgtG5WE6diStVFp8fq6tuE7uLFqe4FX3IghU1DWOEmCMwT67c...) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  本书全面指导机器学习系统的整个生命周期，其中有一章专门介绍监控，涵盖了维护生产健康所需的多项指标、工具和策略。
- [Troubleshooting Machine Learning Deployments: A Taxonomy of Failures and Fixes](https://dl.acm.org/doi/10.1145/3476059) — Nithya Sambasivan, Heidi Lam, Sherry Qian, Khang Nguyen, Ann Yuan, Beenish Chaudhry, Leilani Battle, Greg Nelson, Anusha Muralidharan, Li Zhang, Xiang'Anthony' Chen, Michael S. Bernstein, James Landay (2021)
  Journal: Proceedings of the ACM on Human-Computer Interaction; Publisher: ACM; Volume: 5; Pages: Article 291, 1-32; DOI: [10.1145/3476059](https://doi.org/10.1145/3476059)
  本文识别并分类了已部署机器学习系统中常见的故障模式，为通过稳健监控主动检测和解决问题提供了有力依据。
