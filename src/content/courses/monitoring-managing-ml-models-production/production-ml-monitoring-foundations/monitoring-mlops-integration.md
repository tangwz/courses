---
course: "monitoring-managing-ml-models-production"
chapter: "production-ml-monitoring-foundations"
lesson: "monitoring-mlops-integration"
sourceId: 3255
sourceUrl: "https://apxml.com/zh/courses/monitoring-managing-ml-models-production/chapter-1-production-ml-monitoring-foundations/monitoring-mlops-integration"
title: "将监控融入MLOps生命周期"
description: "了解监控如何融入更广泛的MLOps生命周期，并与其他组成部分（如CI/CD和模型注册表）互相作用。"
order: 5
plots: []
sourceHash: "5e538471a78b8b6379b655a5c11cdc4d340d95964f9ded2186db4365e5c0ba44"
sourceCorrections: []
---

监控机器学习 (machine learning)模型并非部署后才进行的孤立活动。相反，它是贯穿整个MLOps生命周期的基本组成部分。若仅仅将监控视为最终检查点，便会忽视其作为主要反馈机制的作用，它能为机器学习系统的整个运行过程提供信息并推动行动。高效的MLOps依赖于监控系统与CI/CD管道、模型注册表和自动化再训练工作流等其他组成部分的紧密结合。

### 监控作为运行反馈循环

可以将MLOps生命周期看作一个持续循环：数据准备引向模型训练，随后是验证、部署，接着是运行。监控通过观察模型在实际情况下的行为，并将观察所得回馈到早期阶段，从而闭合这个循环。若没有这种反馈，循环就会中断，系统会盲目运行，容易出现本章前面讨论过的无声故障和性能下降问题。

这种结合确保从生产中获得的信息不会丢失，而是积极用于改进系统。例如，当检测到明显的数据漂移时，不应仅仅发出警报；它还应可能引发自动化数据验证检查，通知数据工程团队，甚至启动使用更新数据的再训练管道。同样，特定性能指标（违反服务水平目标）的逐渐下降也可能触发对模型注册表中等待的新候选模型的评估。

> 监控是MLOps循环中的核心反馈机制，它根据观察到的生产行为产生信号，这些信号可以触发再训练、回滚或调查等行动。

### 结合点

有效地结合监控，需要建立监控系统与其他MLOps组件之间清晰的接口和工作流。

#### 1. 持续集成与持续部署 (CI/CD)

- **触发验证：** 监控警报（例如，明显的性能下降、严重的漂移检测）可以充当自动化验证管道的触发器。新的候选模型可能会自动与监控发现的有问题的生产数据片段进行测试。
- **为部署策略提供信息：** 监控数据，特别是来自影子部署或金丝雀部署（第4章会介绍）的数据，直接告知CI/CD系统关于将模型提升到完全生产环境或启动回滚的决定。现用模型与挑战模型之间的性能比较，很大程度上依赖于监控数据。
- **自动化回滚：** 如果监控发现新部署的模型表现明显不如其前身，或者违反了服务水平目标，它可以在CI/CD管道内自动触发回滚机制，以恢复到先前已知的稳定版本。

#### 2. 模型注册表

- **标注模型版本：** 监控系统收集的性能指标、漂移分数和运行状况指标应链接回模型注册表中存储的特定模型版本。这在选择部署模型或分析历史性能时提供了重要背景信息。
- **生命周期管理：** 监控结果可以影响模型在注册表中的状态（例如，根据生产性能将版本标记 (token)为“已降级”或“需要审查”）。治理工作流（第6章会讨论）可以运用此状态信息。

#### 3. 实验跟踪

虽然实验跟踪侧重于模型开发阶段，但将其与生产监控相结合可提供更全面的情况。
"\* **闭合循环：** 将生产性能数据链接回原始实验运行，有助于数据科学家理解离线评估指标与结果如何关联（或不关联）。这为未来的模型选择和特征工程工作提供了信息。"

- **调试性能下降：** 当模型性能下降时，通过模型注册表追溯其来源，回到原始实验和训练数据集（通过实验跟踪系统），这对根本原因分析很重要。

#### 4. 自动化再训练管道

这可能是最直接的结合点。

- **再训练触发器：** 监控系统是智能再训练触发器的主要来源。监控可以根据检测到的概念漂移、影响重要特征的明显数据漂移，或低于定义阈值的持续性能下降来触发再训练，而不仅仅是依赖固定时间表（第4章会详细介绍）。
- **再训练数据选择：** 监控系统内的漂移检测机制有助于找出用于再训练的近期生产数据的最佳时间范围，可能带来更有效的模型更新。

#### 5. 特征库

- **输入数据监控：** 监控输入数据分布和检测漂移与特征库中管理的数据密切相关。漂移警报可以表明需要更新特征生成逻辑，或调查影响提供给模型特征的上游数据质量问题。

### 数据与工作流考虑事项

实施这些结合需要仔细考虑数据流和工作流编排：

- **仪器化与日志记录：** 生产服务必须配置以记录相关信息：输入特征、模型预测、真实标签（如果可用）、时间戳、模型版本标识符等。第5章会讨论可扩展的日志记录策略。
- **监控数据存储：** 这些记录的数据会输入到监控系统，通常经过处理后存储在时序数据库或优化用于分析的专用数据存储中。
- **警报与触发机制：** 监控系统需要有机制（例如，Webhook、消息队列、API调用）将检测到的问题或事件传达给其他MLOps组件，例如CI/CD服务器或工作流编排器（例如，Kubeflow Pipelines、Airflow）。
- **双向通信：** 结合不仅仅是监控触发行动。MLOps工作流也必须告知监控系统，例如，在新模型版本部署时通知它，以便监控能够正确跟踪。

总之，将监控视为MLOps生命周期的一个集成组成部分，而非独立的模块，对构建弹性强、适应性强并持续改进的机器学习 (machine learning)系统很重要。它将监控从被动观察工具转变为主动推动运行优化的驱动力，确保模型在生产运行期间保持高性能和可靠性。后续章节会研究构建和结合这些进阶监控能力所需的具体技术和工具。

## 参考资料

- [Engineering MLOps: A Guide to the Machine Learning Operations Process](https://learning.oreilly.com/library/view/engineering-mlops/9781098104445/) — Emmanuel Raj (2021)
  Publisher: O'Reilly Media
  本书全面指导 MLOps 流程，包括将监控集成到生命周期中及其在反馈循环和自动化中的作用。
- [Machine Learning Design Patterns: Solutions to Common Challenges in Machine Learning Engineering](https://www.oreilly.com/library/view/machine-learning-design/9781098115786/) — Valliappa Lakshmanan, Sara Robinson, Michael Munn (2020)
  Publisher: O'Reilly Media
  本书提供了构建稳健 ML 系统的设计模式，涵盖了监控、数据版本控制和部署策略等主题，这些都是有效 MLOps 集成的基础。
- [MLOps: Continuous delivery and automation pipelines in machine learning](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning) — Karl Weinberger, et al. (2021)
  Journal: Google Cloud Architecture Center; Publisher: Google Cloud
  本白皮书概述了 Google 的 MLOps 框架，强调持续监控作为反馈循环的作用，以推动 ML 生命周期中的自动化、再训练和持续改进。
- [Reliable Machine Learning: Applying SRE Principles to ML in Production](https://www.oreilly.com/library/view/reliable-machine-learning/9781098106218/) — Cathy Chen, Niall Richard Murphy, Kranti Parisa, D. Sculley, Todd Underwood (2022)
  Publisher: O'Reilly Media
  本资源将站点可靠性工程原则应用于 ML 系统，详细说明了监控对于在生产环境中维护模型可靠性、性能和稳定性至关重要。
