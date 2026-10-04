---
course: "introduction-to-mlops"
chapter: "automation-and-cicd-for-ml"
lesson: "continuous-training"
sourceId: 7236
sourceUrl: "https://apxml.com/zh/courses/introduction-to-mlops/chapter-4-automation-and-cicd-for-ml/continuous-training"
title: "持续训练 (CT)"
description: "了解持续训练，这是 MLOps 特有的通过自动重新训练模型以适应新数据的流程。"
order: 4
plots: []
sourceHash: "c179679291c364729f49abca4474b981a89ab4e2a3522cbe6c998e22bd4d68f3"
sourceCorrections: []
---

持续集成 (CI) 和持续交付 (CD) 是为自动化软件工作流打下良好基础的实践方式。然而，这些实践并不能完全解决机器学习 (machine learning)系统特有的挑战。与传统软件不同，机器学习模型的表现并非一成不变；相反，随着所接触数据的变化，其性能可能会随时间下降。

**持续训练 (CT)** 是自动重新训练机器学习模型以适应这些变化的流程。它完成了 MLOps 生命周期的闭环，确保模型在初始部署后的很长一段时间内仍能保持准确性和有效性。可以将其理解为对抗**模型陈旧化**的机制。模型陈旧化是指模型因为不再反映环境的当前状态，导致预测能力下降的情况。

### 为什么持续训练必不可少

推行 CT 的主要驱动因素是一种被称为**模型漂移**的现象。当模型在生产环境中接收到的数据统计特性与训练数据发生偏离时，就会产生漂移。漂移主要有两种形式：

- **数据漂移：** 输入特征的分布发生了变化。例如，一个在经济稳定时期数据上训练的贷款审批模型，在经济衰退期间可能会看到财务状况截然不同的申请。此时模型的假设不再适用。
- **概念漂移：** 输入特征与目标变量之间的关系发生了变化。对于产品推荐系统，用户的偏好和趋势会演变。去年的热门商品在今天可能无人问津，即使人口统计学特征（输入特征）保持不变。

如果没有 CT，部署的模型就是一个价值逐渐流失的静态资产。有了 CT，它就变成了一个可以学习和适应的动态系统。

### 持续训练流水线

CT 流水线是一个自动化的工作流，负责重新训练、评估并准备新模型以供部署。虽然具体细节可能有所不同，但核心阶段是一致的。

> 自动化持续训练闭环图示。监控生产环境的表现可以触发流水线，该流水线会重新训练、评估并注册新模型，随后将其发送到 CD 流水线进行部署。

让我们来看看这个过程中的主要步骤。

#### 1. 触发器

CT 流水线并不会一直运行。它需要一个信号来启动。常见的触发方式包括：

- **定时触发：** 流水线按固定时间表运行，如每天、每周或每月。这种方式实现简单，适用于数据变化速率可预测的场景。
- **基于性能的触发：** 监控系统发现模型的性能指标（如准确率或 F1 分数）已跌破预设阈值。这是一种更具响应性且高效的方法。
- **数据可用性触发：** 一旦收集到一定数量的新标记 (token)数据，流水线就会启动。

#### 2. 数据摄取与重新训练

触发后，流水线会自动收集新数据，将其与相关的历史数据结合，并执行训练脚本。这一步与初始模型训练过程相同，但完全由程序自动完成。其目标是生成一个从最新可用信息中学习到的新候选模型。

#### 3. 评估与验证

这是一个把控质量的关口。仅仅重新训练模型并不能保证它会表现得更好。必须将新模型与当前部署的模型进行详细对比。这种评估通常使用两个模型都未接触过的预留测试数据集进行。

如果新模型没有表现出统计意义上的性能提升，流水线就会停止。将表现较差的模型推向生产环境可能比保留现有模型后果更严重。

#### 4. 模型注册与交付

如果新模型通过了验证，它会被赋予版本号并存储在**模型登记表**中。该登记表充当所有已训练模型的中心库。将模型存储在登记表中会生成一个确定的、带版本的产物，随后可由**持续交付 (CD)** 流水线接手。从这里开始，CD 系统负责包装模型并将其部署到生产环境，替换掉旧的、性能较低的版本。

通过将 CI、CT 和 CD 连接起来，你就创建了一个全自动化的系统，它不仅能验证代码，还能确保机器学习 (machine learning)模型在整个生命周期中不断适应并产生价值。

## 参考资料

- [MLOps: Continuous Delivery and Automation Pipelines in Machine Learning](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning) — Google Cloud (2020)
  Publisher: Google Cloud
  这份来自Google Cloud的官方指南详细阐述了MLOps的原则和实践，包括持续训练在维持模型性能和可靠性方面的作用。
- [Practical MLOps: Operationalizing Machine Learning Models](https://www.oreilly.com/library/view/practical-mlops/9781098103002/) — Noah Gift and Alfredo Deza (2021)
  Publisher: O'Reilly Media
  这本权威著作涵盖了整个MLOps生命周期，其中包含专门章节介绍持续训练、模型漂移和管道自动化。
- [MLOps - A systematic review](https://www.sciencedirect.com/science/article/pii/S016412122100346X) — Anila Syed, Mika Mäntylä, Kai Petersen, and Markus Borg (2022)
  Journal: Journal of Systems and Software; Publisher: Elsevier; Volume: 183; Pages: 111108; DOI: [10.1016/j.jss.2021.111108](https://doi.org/10.1016/j.jss.2021.111108)
  这是一篇近期学术综述，系统性地描绘了MLOps的研究图景，从学术角度阐述了持续训练在更广泛MLOps框架中的地位。
- [Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  这本书提供了构建稳健机器学习系统的综合指南，讨论了模型漂移等挑战以及在生产环境中持续更新模型的必要性。
