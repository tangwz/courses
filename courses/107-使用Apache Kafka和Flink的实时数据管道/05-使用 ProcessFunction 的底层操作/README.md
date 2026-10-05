# 第 5 章：使用 ProcessFunction 的底层操作

来源：[原章节](https://apxml.com/zh/courses/real-time-data-pipelines-kafka-flink/chapter-5-low-level-operations-processfunctions)

[返回课程目录](../README.md)

DataStream API 中的高级操作符能够高效处理许多标准转换任务。然而，特定的业务需求往往要求逻辑超越预定义窗口或简单聚合的能力。当您需要实现复杂事件处理、自定义状态过期或动态连接时，仅依赖标准操作符可能会受到局限。

Apache Flink 提供了 `ProcessFunction` 作为底层接口以应对这些情况。此函数让您的应用能够直接访问流处理的基本要素：状态和计时器。通过直接与这些组件交互，您可以定义任意处理行为，例如检测多个流中的模式或管理依赖于时间的工作流。

本单元侧重于 `ProcessFunction` 及其特殊变体的实现。我们将分析如何手动管理键控状态以及使用 `TimerService` 调度回调。本文将介绍 `CoProcessFunction` 用于连接不同流，以及 Broadcast State 模式用于向所有并行实例分发控制消息。最后，我们将实现异步 I/O 来执行对外部数据库的非阻塞查找，确保网络延迟不会降低管道的吞吐量。

## 小节

- 1. [ProcessFunction 结构体系](01-ProcessFunction%20%E7%BB%93%E6%9E%84%E4%BD%93%E7%B3%BB.md)
- 2. [定时器服务与事件调度](02-%E5%AE%9A%E6%97%B6%E5%99%A8%E6%9C%8D%E5%8A%A1%E4%B8%8E%E4%BA%8B%E4%BB%B6%E8%B0%83%E5%BA%A6.md)
- 3. [广播状态模式](03-%E5%B9%BF%E6%92%AD%E7%8A%B6%E6%80%81%E6%A8%A1%E5%BC%8F.md)
- 4. [外部查找的异步 I/O](04-%E5%A4%96%E9%83%A8%E6%9F%A5%E6%89%BE%E7%9A%84%E5%BC%82%E6%AD%A5%20I-O.md)
- 5. [动手实践：动态规则评估](05-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%8A%A8%E6%80%81%E8%A7%84%E5%88%99%E8%AF%84%E4%BC%B0.md)
