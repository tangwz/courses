---
course: "graph-neural-networks-gnns"
chapter: "gnn-implementation-tooling-optimization"
lesson: "gnns-in-production"
sourceId: 2761
sourceUrl: "https://apxml.com/zh/courses/graph-neural-networks-gnns/chapter-5-gnn-implementation-tooling-optimization/gnns-in-production"
title: "图神经网络在生产系统中的应用"
description: "部署图神经网络模型的注意事项，包括模型服务和监控。"
order: 7
plots: []
sourceHash: "585f63f91bd1b2da61bed4f4dd47dd6147d8d1164bcc9dfb0ae30b3b063e3ee3"
sourceCorrections: []
---

将训练好的图神经网络 (neural network)从研究或开发环境部署到实际生产系统，会带来一系列特定的难题和要求。图神经网络的实际部署涉及具体的步骤和考量，以确保它们能高效、稳定地支持各项应用。与处理表格或序列数据的模型不同，图神经网络通常需要处理复杂、动态的图结构，并且在推理 (inference)阶段需要专门的基础设施来高效访问数据和进行计算。

成功地将图神经网络集成到生产环境，需要围绕部署策略、数据管理、基础设施、监控以及模型生命周期等方面进行周密规划。

### 图神经网络 (neural network)部署策略

部署策略的选择主要取决于应用对预测频率和延迟的需求。

#### 批处理推理 (inference)

当不需要实时预测时，批处理是合适的。比如生成每周用户推荐、对交易图进行定期欺诈分析，或者在夜间更新风险评分。

- **工作流程：** 通常涉及一个定时任务，加载图快照（可能来自数据湖或图数据库），提取相关节点特征，在所需节点或子图上运行图神经网络模型，并将预测结果（例如更新的嵌入 (embedding)、分类、分数）存回数据库或下游系统。
- **数据处理：** 管理图快照，并确保训练和推理时使用的图结构与特征之间的数据一致性，这一点很重要。特征工程流程需要可重复。
- **基础设施：** 通常可以利用现有的批处理框架（如Apache Spark、Airflow、Kubeflow Pipelines）。计算可以横向扩展，但将大型图加载到内存中进行处理可能仍会带来难题。

#### 在线（实时）推理

对于要求即时预测的应用，需要在线推理，比如交易过程中的实时欺诈检测、用户浏览时的内容推荐更新，或识别网络流量中的异常情况。

- **工作流程：** 通常涉及将图神经网络模型部署在API端点之后。当请求到达（例如，针对特定节点的预测）时，系统需要快速获取节点的当前特征，以及其多跳邻居的特征。图神经网络随后计算预测结果，并在API响应中返回。
- **延迟限制：** 这通常是主要的难题。从可能庞大的图数据库中获取邻居数据，并执行图神经网络的消息传递步骤，都必须在严格的时间限制内完成（例如，毫秒级）。
- **基础设施：** 需要低延迟数据存储（可能是针对遍历优化的图数据库或专用特征存储）、高效的模型服务框架（如TorchServe、TensorFlow Serving或定制解决方案），以及可能优化的图神经网络推理引擎。对频繁访问的节点特征或中间计算结果采用缓存策略会有益处。

> 图神经网络批处理和在线推理工作流程的简化比较。

### 生产环境中的图数据管理

图数据表示和可访问性是生产图神经网络 (neural network)系统的基本要求。

- **图数据库：** Neo4j、TigerGraph或Amazon Neptune等系统常用于存储和查询大型图。它们能高效进行多跳遍历，这对于图神经网络在线推理 (inference)时获取邻居数据很有利。然而，将它们集成到机器学习 (machine learning)流程中，需要仔细考虑数据一致性和查询性能。
  "\* **动态图：** 图会演变。新用户加入，连接建立，特征改变。生产系统必须处理这些更新。"
  - **流式更新：** 对于高速变化，图数据库可能会持续摄取更新。图神经网络推理过程需要访问相对最新的图状态。这可能带来在邻居采样期间维护图一致视图的难题。
  - **周期性更新：** 对于不那么频繁的变化，图可能会被重新构建或批量更新（例如，每日）。这简化了一致性管理，但引入了数据陈旧问题。
- **特征存储：** 将图神经网络集成，通常意味着要将图派生特征（节点嵌入 (embedding)、结构属性）与其他特征类型结合。特征存储有助于管理这些不同类型的特征，确保训练和服务之间的一致性，处理时间旅行查询（获取特定过去时间点的特征），并为在线推理提供低延迟访问。将预计算的图神经网络嵌入存储在特征存储中是一种常见做法。

### 基础设施和性能优化

高效运行图神经网络 (neural network)推理 (inference)，特别是对于大型图或低延迟需求，需要特定的基础设施选择。

- **硬件：** 虽然GPU能显著加速图神经网络训练，但对于一些生产场景，CPU推理可能就足够了，并且更具成本效益，特别是当批次大小较小或模型复杂度较低时。对于高吞吐量 (throughput)在线服务，可能需要GPU或专用加速器。请根据应用需求评估成本-性能权衡。
- **内存：** 大型图及其相关特征/嵌入 (embedding)会消耗大量内存。策略包括：
  - 使用内存优化的服务器。
  - 即使在推理期间也采用图分区或采样技术（尽管这会引入延迟）。
  - 量化 (quantization)模型以减少内存占用。
  - 运用DGL或PyG中高效的稀疏矩阵库和内存管理功能。
- **模型服务框架：** 采用专为部署机器学习 (machine learning)模型设计的框架（例如NVIDIA Triton Inference Server、Seldon Core、KServe）。它们提供请求批处理、模型版本管理以及与监控工具的集成等功能。确保所选框架支持图神经网络中可能使用的特定库（PyTorch/TensorFlow）和自定义操作。将模型导出为ONNX等标准化格式有时可以简化部署，尽管对自定义图神经网络操作的支持可能有所不同。

### 图神经网络 (neural network)系统监控

监控已部署的图神经网络不仅仅是观察标准软件指标。它需要观察模型在图数据环境中的行为。

- **预测性能：** 在实时数据上跟踪标准分类/回归指标（准确率、F1分数、MAE）。如果适用，监控每种节点类型或边类型的指标（特别是在异构图中）。
- **运行状态：** 监控API延迟、吞吐量 (throughput)、错误率以及资源利用率（CPU/GPU/内存）。
- **数据漂移：** 这对于图而言尤其复杂。需要监控：
  - **特征漂移：** 输入节点/边特征分布的变化。
  - **结构漂移：** 图属性的变化，如度分布、密度、聚类系数，或新社区的出现。突然的变化可能表明建模的基础过程发生了改变，并可能导致模型性能下降。
- **概念漂移：** 监控输入特征/图结构与目标变量之间的关系。模型的理解是否仍然有效？这通常表现为预测性能的逐渐下降。
- **嵌入 (embedding)稳定性：** 如果图神经网络生成节点嵌入，请监控其分布和随时间变化的距离。剧烈变化可能表明存在问题。

通常需要专门的监控工具或定制仪表板来有效跟踪这些图特有的方面。

### 再训练和模型生命周期管理

图神经网络 (neural network)模型与其他机器学习 (machine learning)模型一样，会随着时间推移因数据和概念漂移而性能下降。制定一个有效的再训练策略非常重要。

- **再训练触发条件：** 明确定义何时重新训练模型的标准。这可以基于：
  - 定期间隔（例如，每周、每月）。
  - 性能下降到设定阈值以下。
  - 探测到明显的数据或结构漂移。
- **数据版本控制：** 跟踪每个模型版本训练时使用的图快照和特征集。这对于可重现性和调试极为重要。
- **模型版本控制：** 存储训练好的模型工件、相关代码和性能指标。使用MLOps平台或工具（MLflow、Kubeflow、Weights & Biases）来管理模型生命周期。
- **自动化（MLOps）：** 尽可能自动化再训练、验证和部署过程，采用适用于机器学习的CI/CD原则。这确保了一致性并减少了人工工作。该流程应包括数据摄取、图处理、特征工程、模型训练、评估和部署。

将图神经网络集成到生产系统中是一项重要的工程工作，远超模型训练本身。它需要仔细考虑部署模式、能够处理图复杂性的数据基础设施、针对延迟和吞吐量 (throughput)的性能优化、考虑图属性的全面监控策略，以及模型更新和生命周期管理的自动化流程。机器学习工程师、数据工程师和基础设施团队之间的协作，对于构建和维护成功的生产图神经网络应用非常重要。

## 参考资料

- [Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  本书为设计、构建和操作生产级机器学习系统提供了全面指导，内容涵盖部署、监控和机器学习生命周期。
- [The Case for a Unified Feature Store](https://dl.acm.org/doi/10.1145/3474347.3474354) — Mike Del Balso, Neoklis Kyriakides, Eric Schmidt, Ben Shreck, Noah Solomon, Yury Yatskiv (2021)
  Journal: ACM SIGMOD Record; Publisher: ACM; Volume: 50; Pages: 44-51; DOI: [10.1145/3474347.3474354](https://doi.org/10.1145/3474347.3474354)
  讨论了特征存储在管理机器学习特征方面的作用和优势，处理训练和服务的兼容性，并提供低延迟访问。
- [GNN-Lens: A Unified Framework for Efficient Graph Neural Network Inference on Large-Scale Graphs](https://www.vldb.org/pvldb/vol16/p182-yang.pdf) — Yujin Yang, Jianbo Liu, Min Li, Rui Zhang, Shifu Zhang, Zongqian Wu, Yi Xu, Peng Zhang, Huajun Chen (2023)
  Journal: Proceedings of the VLDB Endowment; Publisher: VLDB Endowment; Volume: 16; Pages: 182-195; DOI: [10.14778/3579040.3579049](https://doi.org/10.14778/3579040.3579049)
  提出了一个优化大型图上GNN推理的框架，涵盖了内存管理、执行调度和数据访问模式等，对生产部署很重要。
- [MLOps: A Taxonomy and Survey](https://dl.acm.org/doi/10.1145/3453140) — Xuexin Liang, Xin Peng, Yuetong Li, Yijia Zhang, Wenyang Li, Xiaohong Su (2021)
  Journal: ACM Computing Surveys; Publisher: Association for Computing Machinery (ACM); Volume: 54; Pages: 1-38; DOI: [10.1145/3453140](https://doi.org/10.1145/3453140)
  提供了MLOps实践、工具和挑战的分类和调查，涵盖机器学习生命周期中的部署、监控和版本控制。
