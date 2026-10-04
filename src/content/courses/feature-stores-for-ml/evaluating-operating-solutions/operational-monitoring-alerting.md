---
course: "feature-stores-for-ml"
chapter: "evaluating-operating-solutions"
lesson: "operational-monitoring-alerting"
sourceId: 3450
sourceUrl: "https://apxml.com/zh/courses/feature-stores-for-ml/chapter-6-evaluating-operating-solutions/operational-monitoring-alerting"
title: "操作监控与警报"
description: "为特征存储的健康、性能和数据质量设置全面监控。"
order: 4
plots: ["plots/3450-0.json", "plots/3450-1.json"]
sourceHash: "185b593f103dcdb2a333833c3d9c514f018faadb7ed3dc1366bf94b172a10def"
sourceCorrections: []
---

特征存储部署后，无论是在内部构建还是作为托管服务采用，确保其可靠高效的运行都成为首要考虑。操作监控和警报并非可有可无的附加项；它们是维护系统健康、性能、数据完整性和生产环境中成本效益的根本做法。有效的监控提供所需的可见性，以主动诊断问题、了解系统在负载下的行为，并做出有关扩展和优化的明智决策。

本节详细介绍专门为高级特征存储实现设置全面监控和警报的重要方面。我们将介绍要跟踪的指标类别、实施策略以及如何设计有效的警报机制。

## 特征存储的核心监控方面

生产中的特征存储涉及多个相互作用的组件：数据摄入管道、转换逻辑、离线存储、在线存储和服务API。监控必须涵盖所有这些方面。必要的监控可以分为几个主要方面。

### 系统健康与可用性

这一基础层侧重于底层基础设施和基本服务可用性。

- **基础设施指标：** 跟踪支撑特征存储组件的计算和存储资源的标准指标。这包括虚拟机、容器、数据库节点（在线/离线）和处理集群（例如Spark、Flink）的CPU利用率、内存使用、磁盘I/O、磁盘空间和网络带宽。高资源利用率可能导致性能下降或中断。
- **服务可用性：** 监控重要端点（特别是特征服务API（用于在线特征）以及任何元数据或注册服务）的运行时间和响应性。使用健康检查（例如，HTTP `GET /health`）来验证基本服务运行。
- **错误率：** 跟踪不同组件生成的错误率。对于服务API，监控HTTP状态码（例如，5xx表示服务器错误，4xx表示客户端错误）。对于数据管道，监控作业失败、转换错误和连接问题。错误率的突然飙升通常表示部署问题、基础设施问题或上游数据变化。

### 性能指标

了解特征存储的性能特性对于满足服务级别目标（SLO）和识别瓶颈非常重要。

- **在线存储性能：**
  - *读取/写入延迟：* 这通常是在线服务最主要的指标。衡量获取特征向量 (vector)或写入特征更新所需的时间。跟踪常见的百分位数，如p50（中位数）、p90、p95和p99，以了解延迟的分布。根据这些百分位数定义SLO（例如，p99读取延迟 < 50ms）。
  - *吞吐量 (throughput)：* 衡量在线存储可以处理的请求速率，通常以每秒查询数（QPS）或每秒请求数（RPS）表示读取和写入。对照容量限制监控吞吐量。



![在线存储读取延迟 (p99)](plots/3450-0.json)



> 在线特征存储的p99读取延迟，显示出飙升，表明潜在的性能问题。

- **离线存储性能：**
  - *作业完成时间：* 监控批处理特征计算和回填作业的持续时间。跟踪趋势以识别性能退化或扩展需求。
  - *数据处理吞吐量：* 衡量批处理计算期间每单位时间处理的数据量（例如，GB/小时，记录/秒）。这有助于容量规划和优化作业效率。
  - *资源利用率：* 跟踪特征计算期间批处理框架（Spark、Flink）的资源使用情况（CPU、内存），以优化集群大小和配置。
- **特征计算延迟：** 对于支持按需或流式计算的系统，监控从数据到达特征可用或计算完成的端到端延迟。

### 数据质量和时效性

监控应扩展到特征数据本身的质量和时效性。这与第3章中关于数据一致性讨论的内容直接相关。

- **数据时效性：**
  - *滞后：* 衡量事件发生时间（事件时间戳）与其相应特征在在线存储中可用时间之间的时间差。这对于实时用例非常重要。
  - *上次更新时间：* 对于批处理特征，监控自特征组上次成功更新以来经过的时间。过期数据可能显著降低模型性能。



![特征组时效性滞后](plots/3450-1.json)



> 不同特征组的预期更新时间与实际更新时间之间的滞后，突显批处理更新特征中潜在的过期问题。

- **数据有效性：** 在数据摄入或计算期间实施检查，以监控数据质量。跟踪以下指标：
  - 每个特征的空值或缺失值百分比。
  - 类型不匹配或约束违反的速率（例如，值超出定义范围）。
  - 自动化数据验证工具（例如Great Expectations、Pandera）的输出。
- **分布漂移：** 监控离线和在线存储中重要特征的统计特性（均值、中位数、标准差、分位数、分类特征的基数）随时间的变化。比较服务分布与训练分布，以检测偏差或漂移。意料之外的变化可能预示上游数据问题或影响模型相关性的概念漂移。

### 成本监控

特征存储，特别是大规模的特征存储，可能会产生高昂的基础设施成本。

- **资源消耗：** 跟踪与存储（在线/离线层）、计算实例（服务、批处理）、数据库服务和数据传输相关的云提供商成本。
- **成本分配：** 使用标签或标记 (token)策略将成本归因于特定的特征组、团队或项目。这有助于理解成本驱动因素并优化资源使用。

## 实施监控和警报系统

建立有效的监控涉及选择合适的工具、对系统进行仪表化以及配置有意义的警报。

### 工具选择

尽可能采用标准可观测性平台，而不是构建定制解决方案。常见的选择包括：

- **指标与可视化：** 带有Grafana的Prometheus、Datadog、Google Cloud Monitoring、AWS CloudWatch、Azure Monitor。
- **日志：** Elasticsearch/Logstash/Kibana (ELK) 堆栈、Splunk、Loki、CloudWatch Logs、Google Cloud Logging。
- **追踪：** Jaeger、Zipkin、OpenTelemetry、AWS X-Ray、Google Cloud Trace。
- **数据质量：** 将Great Expectations、Deequ或TFX Data Validation等库的输出集成到您的监控仪表板中。
- **托管服务：** 云提供商的特征存储通常带有内置监控功能，可与其各自的云监控套件集成。评估这些原生集成。

### 仪表化和数据收集

您的应用程序和基础设施必须进行仪表化，以发出必要的日志和指标。

- **导出器和代理：** 使用标准导出器（例如，Prometheus节点导出器、数据库导出器）和可观测性平台提供的监控代理。
- **客户端库：** 使用指标库（例如Prometheus客户端库、Micrometer、OpenTelemetry SDKs）对特征生成代码、服务API和数据管道进行仪表化，以发出自定义的应用程序级别指标（延迟、错误计数、数据验证结果）。
- **结构化日志：** 以结构化格式（例如JSON）记录重要事件和错误，以便在您的日志系统中更轻松地解析和分析。

### 仪表板和可视化

创建专用仪表板以提供关于特征存储健康和性能的可操作视图。

- **基于角色的仪表板：** 为不同角色设计不同的仪表板：
  - *运维/SRE：* 侧重于系统健康、可用性、延迟、错误率和资源利用率。
  - *ML工程师：* 强调管道健康、特征计算时间、数据时效性和验证结果。
  - *数据科学家：* 提供特征分布、漂移指标以及潜在的特征使用统计数据视图。
- **性能指标 (KPIs)：** 突出显示最重要的指标和SLO。使用清晰显示趋势、异常和偏离目标的视图。

> 监控组件如何与特征存储架构集成的概览。

### 设计有效的警报

警报通知操作员需要注意的潜在问题。配置不当的警报会导致疲劳和警告被忽视。

- **定义清晰的SLO：** 基于预定义服务级别目标（例如，p99延迟超过SLO阈值持续5分钟）的违规行为设置警报。
- **阈值和异常检测：** 对可预测的指标（例如，磁盘空间>90%）使用静态阈值。对具有动态模式的指标（例如，请求突然飙升、特征值异常下降）采用异常检测算法，以捕获意外行为。
- **严重级别：** 根据影响对警报进行分类（例如，CRITICAL：立即中断或数据损坏；WARNING：潜在的未来问题或性能下降；INFO：信息性）。
- **目标路由：** 将警报路由到负责的特定团队（例如，数据管道警报发送给ML工程师，在线存储延迟警报发送给SRE）。使用PagerDuty或Opsgenie等工具进行待命排班和升级策略。
- **减少噪音：** 对相关警报进行分组，在已知维护窗口期间实施静默，并持续调整阈值，以避免过多的低优先级通知。
- **可操作警报：** 确保警报包含足够上下文 (context)：哪个指标超出，哪个组件受影响，当前值与阈值的对比，以及理想情况下，指向相关仪表板或故障排除操作手册的链接。

通过实施监控和周到的警报，您建立了必要的反馈循环，以可靠高效地运行您的特征存储。这种可见性不仅仅是为了解决问题；它是关于透彻了解您的系统，实现持续改进、明智的扩展决策，并维持对提供给生产机器学习 (machine learning)模型的特征的信任。

## 参考资料

- [Site Reliability Engineering: How Google Runs Production Systems](https://sre.google/sre-book/) — Benjamin Treynor Sloss, Betsy Beyer (2016)
  Publisher: O'Reilly Media
  提供系统可靠性、监控、警报和事件响应的基本原则和实践，适用于特征存储等复杂生产系统。
- [Tecton: A Modern Feature Store for ML](https://www.vldb.org/pvldb/vol14/p2959-zadeh.pdf) — Mike Zadeh, Sam Steingold, Ben Zaitlen, Matt Gormley, Jeremy Hyrkas, Max Hjelm, Benji Cooper, Andrew Lee, Adarsha Nadig, Josh Rosen, Ryan Smith, Kevin Tian, Justin Trevor, Xiaoyong Wang, Mike Williams, David Li (2021)
  Journal: Proceedings of the VLDB Endowment; Publisher: VLDB Endowment; Volume: 14; Pages: 2959-2971; DOI: [10.14778/3476311.3476344](https://doi.org/10.14778/3476311.3476344)
  描述了现代特征存储架构，包括需要操作监控的组件，并涉及机器学习系统的可靠性。
- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  探讨构建可靠、可扩展和可维护数据系统的原则，这些系统是特征存储的基础，包括一致性、可用性及其操作影响的概念。
- [Reliable Machine Learning: Applying SRE Principles to ML in Production](https://www.oreilly.com/library/view/reliable-machine-learning/9781098106218/) — Cathy Chen, Niall Richard Murphy, Kranti Parisa, D. Sculley, Todd Underwood (2022)
  Publisher: O'Reilly Media
  专门针对机器学习系统应用站点可靠性工程（SRE）原则，包括数据质量监控、模型监控以及确保机器学习管道的运行可靠性。
