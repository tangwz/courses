---
course: "data-governance-quality-observability-production"
chapter: "data-observability-systems"
lesson: "pillars-data-observability"
sourceId: 8092
sourceUrl: "https://apxml.com/zh/courses/data-governance-quality-observability-production/chapter-3-data-observability-systems/pillars-data-observability"
title: "数据可观测性的支柱"
description: "了解数据系统中的指标、元数据、血缘和日志。"
order: 1
plots: ["plots/8092-0.json"]
sourceHash: "a3a77976b2d4db93d8e8df51b2022d2789a72f18cdcd6f237cf393e6966d0a34"
sourceCorrections: []
---

软件工程团队长期以来一直使用应用性能监控（APM）来监控系统正常运行时间与延迟。然而，数据工程却带来不同的难题：系统可能运行良好，服务器在线，有向无环图（DAG）显示正常，任务顺利完成，但数据本身却可能是无效的。任务的成功执行不保证数据准确。

数据可观测性弥补了这一不足。它将DevOps可观测性的原则应用于数据管道，使得工程师能够根据外部信号推断数据的内部质量。与在部署前确认已知条件的单元测试不同，可观测性发现生产中的未知问题。它依靠三种基本的信号类型：指标、日志和追踪（在数据背景下通常表现为血缘关系）。

### 数据健康的三种信号

为构建一个监控系统，我们必须从数据基础设施中获取特定信号。这些信号使我们能够回答三个不同的问题：“是否存在问题？”、“问题出在哪里？”以及“为什么会发生？”。

#### 指标

指标是在特定时间间隔内测量的数据的数值表示。在数据背景下，这些是追踪信息形态和流动的时间序列值。它们存储高效，易于查询以用于告警。常见的数据指标包括：

- **运营指标：** 任务持续时间、计算成本和槽位使用。
- **质量指标：** 行数（数量）、空值百分比和唯一值计数。
- **SLA指标：** 延迟和上次更新时间（新鲜度）。

#### 日志

日志是不可变的、带有时间戳的离散事件记录。指标告诉你趋势正在改变，而日志提供调试该事件所需的详细信息。当转换失败时，日志包含异常堆栈追踪或数据仓库返回的特定SQL错误代码。

#### 追踪与血缘

在微服务中，追踪负责请求在服务间的跳转过程。在数据工程中，这个想法对应于**数据血缘**。血缘描绘了上游源、转换任务和下游表之间的关系。它提供理解事件影响范围所需的背景信息。

下图说明了这些信号如何汇集，在物理基础设施之上形成一个可观测层。

> 信号从物理基础设施流入可观测性平台进行分析。

### 使用指标检测异常

主动告警的主要方式是对收集的指标进行异常检测。我们通常关注三个特定方面：新鲜度、数量和模式。

#### 新鲜度

新鲜度测量自数据集上次更新以来经过的时间。它有效地追踪管道的延迟。如果一个通常在协调世界时08:00运行的任务在09:30完成，那么数据就“过时”了90分钟。

新鲜度检查具有很高的意义，因为它们发现静默故障。一个计划好的有向无环图（DAG）可能仅仅因为调度器故障而未能触发。在这种情况下，因为没有代码运行，所以没有生成错误日志。只有观察目标表的新鲜度监控器才能发现这种静默情况。

#### 数量

数量指正在处理的数据的大小，通常以行数或字节数衡量。数据量突然下降通常表示上游数据丢失或API故障，而突然的飙升可能表示重复数据摄取。

因为数据量自然波动（例如，周末流量较低），静态阈值常常产生误报。因此，我们使用统计基线。通过计算Z分数，我们可以确定当前数量 $V_t$ 相对于标准差 $\sigma$ 偏离移动平均值 $\mu$ 的程度：

$Z = \frac{V_t - \mu}{\sigma}$

当 $|Z| > k$ 时触发告警，其中 $k$ 是一个灵敏度阈值（通常为2或3）。

下图显示了一个数量异常，其中行数显著低于预期的历史范围。



![数量异常检测](plots/8092-0.json)



> 10月5日，实际值与历史基线显著偏离，触发了异常告警。

#### 模式

模式漂移发生在数据结构改变时。这包括列的添加、移除或重命名，以及数据类型改变（例如，`整数`字段变为`字符串`）。

尽管模式演进在敏捷开发中是很常见的，但未管理的模式改变是导致管道中断的主要原因。可观测性系统监控数据仓库的信息模式或数据湖中JSON数据块的结构。当检测到改变时，系统会将新模式与已知注册表或先前状态进行比较，以判断该改变是否会造成破坏性影响。

### 元数据和血缘的作用

指标告诉你*有什么*问题，但血缘告诉你*哪里*出问题了，以及*谁*受到影响。

血缘通过解析查询日志或与Airflow等编排工具集成来构建。它为你的数据资产构建一个有向无环图（DAG）。当仪表板上触发新鲜度告警时，血缘允许你向后遍历图（上游）以找到根本原因。反之，如果源表损坏，血缘允许你向前遍历（下游）以通知相关方。

有效的可观测性需要结合这些支柱。指标告警触发事件。血缘隔离出损坏的组件。日志显示错误信息。这种工作流程缩短了平均解决时间（MTTR），并将数据治理从手动监督任务转变为自动化工程规范。

## 参考资料

- [Site Reliability Engineering: How Google Runs Production Systems](https://sre.google/books/sre-book/) — Betsy Beyer, Chris Jones, Jennifer Petoff, and Niall Richard Murphy (2016)
  Publisher: O'Reilly Media
  建立站点可靠性工程原则的基础指南，包括使用指标、日志和追踪进行系统可观察性。
- [Data Downtime: The Most Impactful Data Quality Incidents and How to Avoid Them](https://www.wiley.com/en-us/Data+Downtime%3A+The+Most+Impactful+Data+Quality+Incidents+and+How+to+Avoid+Them-p-9781119864228) — Barr Moses, Lior Gavish, Jeremy Levy (2022)
  Publisher: Wiley; DOI: [10.1002/9781119864259](https://doi.org/10.1002/9781119864259)
  定义数据停机并引入数据可观察性作为解决方案，涵盖新鲜度、容量和模式作为数据健康监控的关键维度。
- [Data Lineage for Big Data: A Survey](https://www.sciencedirect.com/science/article/pii/S074373151930030X) — Kun Ma, Yongfeng Huang, and Guoping Long (2019)
  Journal: Journal of Parallel and Distributed Computing; Publisher: Elsevier; Volume: 129; Pages: 162-178; DOI: [10.1016/j.jpdc.2019.03.003](https://doi.org/10.1016/j.jpdc.2019.03.003)
  对大数据环境中数据血缘的概念、技术和挑战的全面学术调查，详细说明其在理解数据流中的作用。
- [The Data Engineering Cookbook](https://github.com/andkret/Cookbook) — Andreas Kretz (2019)
  Publisher: The Data Engineering Academy
  提供构建可靠数据系统的实践方法，包括监控、警报和确保生产数据质量的章节。
