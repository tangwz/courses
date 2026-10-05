# 规模化MLOps原则

来源：[原文](https://apxml.com/zh/courses/advanced-ai-infrastructure-design-optimization/chapter-1-architectural-patterns-ai-platforms/mlops-principles-at-scale)

[返回章节目录](README.md) · [返回课程目录](../README.md)

虽然机器学习 (machine learning)工作流的自动化已是成熟做法，但将这些原则应用于大规模生产系统则带来了新的挑战。当系统涉及数十个团队、PB级数据集以及数十亿参数 (parameter)模型时，MLOps 从自动化单个管道演变为构建一个统一的多租户平台。关注点从单个模型的成功转向整个AI生态系统的效率、可靠性和治理。

在这种规模下，基础设施不仅仅是运行代码的场所；它是MLOps循环中不可或缺的部分。硬件和系统架构的选择直接影响重现性、成本和性能，使它们成为任何MLOps实践者的主要考量。

### 从管道自动化到平台抽象

在小规模环境中，MLOps管道通常是线性的定制脚本：它拉取数据、训练模型并部署端点。当支持多个项目时，这种方法就行不通了。每个新项目都需要一个新管道，导致工作重复、工具不一致以及高昂的维护负担。

规模化的解决方案是从管道自动化转向**平台抽象**。不再构建数十个独特的管道，而是构建一个单一、标准化的平台，以服务形式提供MLOps能力。数据科学家和ML工程师通过定义清晰的API或用户界面与平台交互，启动训练任务、部署模型或调配资源，而无需管理底层基础设施。

这种以平台为核心的模式使重要操作标准化：

- **环境管理：** 提供预配置的、版本化的容器镜像，包含经过验证的驱动程序（例如CUDA）和库。
- **工作流编排：** 为分布式训练或批量推理 (inference)等常见任务提供模板，隐藏Kubernetes或工作流引擎的复杂性。
- **资源调配：** 通过标准化的请求管理对计算资源池（CPU、GPU、竞价实例）的访问。

> 从线性的按项目划分的管道转向服务多个团队和工作负载的集中式平台模式。

### 跨越复杂的依赖关系图实现统一版本控制

标准`git`足以用于源代码版本控制，但对于大规模ML系统中所有类型的制品来说，它是不够的。生产模型是代码、数据和配置特定组合的产物。重现特定模型版本需要能够检出其所有依赖项的精确状态，这些依赖项包括：

- **代码：** 训练和推理 (inference)逻辑。
- **数据集：** 训练和验证数据的特定快照，大小可达TB级。
- **模型制品：** 训练后的权重 (weight)，对于大型语言模型可达GB级。
- **配置：** 超参数 (parameter) (hyperparameter)、基础设施设置（例如，实例类型）和环境变量。
- **容器环境：** 包含特定操作系统、系统库和语言依赖项的Docker镜像。

一个成熟的MLOps平台提供了一种**统一版本控制**机制，在这些组件之间建立不可变的链接。一个单一的标识符，很像git提交哈希，应该让你能够获取给定模型的整个依赖关系图。这对于调试生产问题、审计模型行为以及确保数月后重新训练的模型与之前版本完全一致至关重要。

### 异构计算环境中的可重现性

可重现性超越了代码和数据，延伸至硬件和软件环境。在配备特定CUDA驱动的NVIDIA A100 GPU上训练的模型，如果在H100 GPU上或甚至在同一GPU上使用不同驱动版本进行再训练，可能会产生略微不同的结果。

在规模化应用中，确保可重现性需要将基础设施状态以代码形式捕获。一个简单的`Dockerfile`是不够的。你还必须对以下内容进行版本控制：

- **计算实例配置：** 精确的云实例类型（例如，`p4d.24xlarge`）或本地硬件规格。
- **加速器驱动：** NVIDIA驱动和CUDA工具包的精确版本。
- **互联：** 用于分布式训练的网络结构类型和拓扑（例如，InfiniBand、NVLink），因为这会影响通信模式并可能影响模型收敛。
- **编排清单：** 定义资源请求、限制和调度策略的Kubernetes YAML文件或其他配置。

未对基础设施层进行版本控制，使得调试细微的性能退化或仅在特定硬件环境中表现出的非确定性行为几乎不可能。

### 监控模型性能

尽管监控精度衰减或数据漂移仍然重要，规模化MLOps需要更广阔的视角，将运营和财务指标纳入其中。生产中模型的健康状况是其统计性能、技术性能和成本效益的函数。

因此，一个全面的监控策略必须跟踪三类信号：

1. **模型质量指标：** 准确率、精确率/召回率、F1分数或业务特定的KPI。这包括随时间跟踪概念和数据漂移。
2. **基础设施与运营指标：**
   - **延迟：** 端到端预测时间，按百分位数（p50、p90、p99）细分。
   - **吞吐量 (throughput)：** 推理 (inference)服务每秒处理的请求数（RPS）。
   - **利用率：** GPU/CPU利用率、内存使用和网络I/O。低利用率可能表明部署过度配置且成本高昂。
   - **错误率：** HTTP 5xx错误或其他系统级故障的发生率。
3. **财务指标（财务运营）：**
   - **每次推理成本：** 单次预测的摊销成本，计算方式为 `(总基础设施成本) / (总预测数)`。
   - **每次训练任务成本：** 训练模型的总成本，归属于特定团队或项目。

这种全面的视角让你能够回答复杂问题，例如“将模型量化 (quantization)为INT8能否在不影响p99延迟或业务KPI的情况下，将每次推理成本降低30%？”。这种程度的分析对于高效运行AI系统是根本性的。

## 参考资料

- [Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems.pdf) — D. Sculley, Gary Holt, Daniel Golovin, Eugene Davydov, Todd Phillips, Dietmar Ebner, Vinay Chaudhary, Michael Young, Jean-François Crespo, Dan Dennison (2015)
  Journal: Advances in Neural Information Processing Systems (NeurIPS) 28; Publisher: Neural Information Processing Systems Foundation, Inc. (NeurIPS); Pages: 2503-2511
  指出生产环境中机器学习系统复杂性和维护负担的核心挑战。
- [Engineering MLOps: Machine Learning Operations at Scale](https://www.oreilly.com/library/view/engineering-mlops/9781098104522/) — Emmanuel Raj, Harish Lakshmanan, Anurag Agarwal (2022)
  Publisher: O'Reilly Media
  一本关于设计和实现可扩展 MLOps 平台，包含抽象、工作流和基础设施的指南。
- [Effective MLOps: Enabling Reproducibility and Auditability for Machine Learning Experiments](https://dl.acm.org/doi/10.1145/3429381.3432924) — Klaus Schelter, Simon Klinger, Michael Fechner, Tobias Schmidt, Andreas Schmidt (2020)
  Journal: Proceedings of the 3rd Workshop on MLOps Systems (MLOps'20); Publisher: ACM; Pages: 1-8; DOI: [10.1145/3429381.3432924](https://doi.org/10.1145/3429381.3432924)
  讨论了实现机器学习实验和模型可复现性及可审计性的策略与挑战。
- [Designing Machine Learning Systems: An Iterative Process for Production-Ready AI](https://www.oreilly.com/library/view/designing-machine-learning/9781098107950/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  这本书为设计机器学习系统提供了实践指导，包含关于生产工作负载的监控、测试和成本优化章节。

---

[下一节](02-%E7%AE%97%E5%8A%9B%E9%80%89%E6%8B%A9%EF%BC%9ACPU%E3%80%81GPU%20%E5%92%8C%20TPU%20%E6%9E%B6%E6%9E%84.md)
