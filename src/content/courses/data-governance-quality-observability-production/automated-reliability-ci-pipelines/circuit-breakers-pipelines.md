---
course: "data-governance-quality-observability-production"
chapter: "automated-reliability-ci-pipelines"
lesson: "circuit-breakers-pipelines"
sourceId: 8116
sourceUrl: "https://apxml.com/zh/courses/data-governance-quality-observability-production/chapter-5-automated-reliability-ci-pipelines/circuit-breakers-pipelines"
title: "流水线中的断路器"
description: "实现断路器以在质量下降时停止执行。"
order: 3
plots: ["plots/8116-0.json"]
sourceHash: "6bc8d7bc4282fa84768a1829b485e97dd2c1d1333186f189ad45e005291d6629"
sourceCorrections: []
---

自动化流水线在传输数据方面效率很高，但在传播错误方面也同样高效。当源系统发送损坏文件或架构变化破坏了转换逻辑时，标准流水线通常仍会尝试处理不良数据。这会导致数据湖受污染，下游仪表板损坏。为避免此问题，我们引入**断路器**。

数据工程中的断路器是一种运行时机制，在不满足特定条件时停止流水线执行。与通常在部署期间验证代码或静态配置的质量门不同，断路器在实际执行期间验证*数据状态*。如果数据质量指标超出设定的安全阈值，“电路打开”，数据流被物理停止，以保护下游使用者。

### 断路器的工作原理

断路器的基本逻辑依赖于一个阻塞条件。我们定义一个指标 $M$（例如空值百分比或行数）和一个阈值 $\tau$。系统会为每个批次或微批次评估一个布尔条件。


$$
\text{状态} = \begin{cases} 
\text{打开 (停止)} & \text{如果 } M > \tau \\
\text{关闭 (继续)} & \text{如果 } M \leq \tau 
\end{cases}
$$


在“打开”状态下，依赖图被切断。依赖于当前阶段的任务被标记 (token)为 `跳过` 或 `失败`，而不是尝试执行。这通过限制数据质量事故的影响范围，避免了“垃圾进，垃圾出”的现象。

> 流水线执行流程，其中验证步骤的失败会重新路由执行到警报路径，而不是转换逻辑。

### 阻塞逻辑的实现

在像 Airflow 这样的基于 Python 的编排框架中，不同的操作符负责处理这种逻辑。`ShortCircuitOperator` 是一种常见模式。它执行一个返回 `True` 或 `False` 的 Python 可调用对象。如果为 `False`，所有下游任务都将被跳过。

在设计这些检查时，必须权衡可靠性与流水线可用性。过于敏感的断路器会导致“警报疲劳”和频繁停机。过于宽松的断路器将无法发现严重问题。

考虑一个验证 `transaction_amount` 列中负值的检查。一个严格的断路器可能如下所示：

1. **查询**: `SELECT count(*) FROM staging_transactions WHERE amount < 0`
2. **评估**: 如果 `count > 0`，返回 `False` (停止)。
3. **操作**: 后续任务 `load_to_warehouse` 将被跳过。

在使用 dbt 的以 SQL 为中心的工作流程中，断路器通过测试定义中的 `severity: error` 配置实现。默认情况下，dbt 测试会发出警告，但不会停止执行。将严重性更改为 `error` 可确保如果断言失败，运行器返回非零退出代码。这会指示 CI/CD 系统或编排器中止作业的其余部分。

### 硬性阈值与自适应断路器

将静态标量值设置为阈值是最常见的起点，但在动态环境中常常失效。像“如果行数 < 1000 则停止”这样的静态规则可能在工作日有效，但在周末流量自然下降时会错误地停止。

对于成熟的生产系统，我们使用**自适应断路器**。它们利用历史元数据定义一个动态的可接受范围。我们计算 Z 分数（标准分数）来确定当前指标与移动平均值偏离的程度。

如果 Z 分数的绝对值超过显著性水平（通常为 3，代表 3 个标准差），断路器将触发：


$$
\left| \frac{x - \mu}{\sigma} \right| > 3
$$


其中：

- $x$ 是观测指标（当前行数）。
- $\mu$ 是移动平均值（例如，特定星期几过去 30 天的平均值）。
- $\sigma$ 是标准差。



![自适应流量阈值](plots/8116-0.json)



> 阴影区域代表可接受范围（$\mu \pm 3\sigma$）。红色标记 (token)表示断路器将因异常低流量而启动的违规。

### 在 DAG 中的策略性部署

断路器的位置显著影响其有效性。我们通常遵循“左移”策略，将断路器尽可能靠近数据摄入点放置。

1. **摄入断路器：** 验证文件格式、编码和基本可用性。如果源文件为空，则没有理由启动 Spark 集群。
2. **架构断路器：** 验证列名和数据类型。如果上游架构发生漂移，后续的 SQL 查询将失败。在此处中断可避免日志中出现令人困惑的“未找到列”错误。
3. **业务逻辑断路器：** 放置在转换之后但在加载到生产表之前。这些验证高级不变式，例如“收入不能为负”或“总百分比必须等于 100”。

### 恢复与干预

当断路器跳闸时，流水线停止。这是一种“故障安全”状态。当务之急是工程干预。与通过重试可能解决的暂时性网络错误不同，数据质量故障通常会持续存在，直到数据得到修复或规则得到调整。

您的 CI 配置必须支持两种恢复路径：

1. **回填：** 上游数据修复后，可以重新触发流水线以处理失败期间的数据。
2. **绕过：** 在紧急情况下，当数据不完善但“足够好”以用于报告时，工程师需要一种机制来暂时强制关闭断路器。这通常通过向运行配置传递特定标志（例如，`--skip-quality-checks`）来处理，尽管这应谨慎使用并受到严格审计。

通过将这些控制直接嵌入 (embedding)到编排逻辑中，我们将数据可靠性视为生产可用性的一个硬性依赖，确保无声胜于误报。

## 参考资料

- [Release It!: Design and Deploy Production-Ready Software](https://books.google.com/books/about/Release_It_Design_and_Deploy_Production.html?id=oE07DwAAQBAJ) — Michael T. Nygard (2018)
  Publisher: The Pragmatic Bookshelf; Pages: 376
  此书是引入熔断器模式的开创性著作，提供了直接适用于数据管道的基础原则。
- [Fundamentals of Data Engineering: Planning and Building Robust Data Systems](https://www.oreilly.com/library/view/fundamentals-of-data/9781098108298/) — Joe Reis and Matt Housley (2022)
  Publisher: O'Reilly Media; Pages: 447
  此书涵盖现代数据管道中的数据可靠性、质量和自动化检查，为熔断器的使用提供了背景。
- [Apache Airflow Documentation: ShortCircuitOperator](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGcCU5DLgjbfuBN_Xea8-THpjUecZeTvZfsKbtbjfbVahsd-05eXoUFUso37QF8piM4I6OsGbla4_7vdJWTk7fXKzrKWfgbS5YZlLg9Ev5uuf0GBNDAcqvn3KuYCnt2h0LzjAA0YwjcGAkrYIU0NAgyVNICvaBcRQd0mqw7H4F8HPopL-t9QLedp5KiNImqknoIWHixfrY) — Apache Software Foundation (2023)
  Publisher: Apache Software Foundation
  提供了 Apache Airflow 中 ShortCircuitOperator 的具体实现细节，这是用于阻止管道执行的常见工具。
- [dbt Docs: Testing your dbt project](https://docs.getdbt.com/docs/build/data-tests#what-happens-when-a-test-fails) — dbt Labs (2023)
  Publisher: dbt Labs
  解释了如何配置 dbt 测试，通过 `severity: error` 在数据质量断言失败时停止管道执行。
- [Anomaly Detection Principles and Techniques](https://doi.org/10.1145/1541880.1541882) — Varun Chandola, Arindam Banerjee, Vipin Kumar (2009)
  Journal: ACM Computing Surveys; Publisher: Association for Computing Machinery (ACM); Volume: 41; Pages: 1-58; DOI: [10.1145/1541880.1541882](https://doi.org/10.1145/1541880.1541882)
  提供了异常检测方法的全面概述，包括 Z 分数等统计方法，这些方法与自适应熔断器相关。
