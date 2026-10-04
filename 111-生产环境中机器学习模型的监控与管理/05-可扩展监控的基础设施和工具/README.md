# 第 5 章：可扩展监控的基础设施和工具

来源：[原章节](https://apxml.com/zh/courses/monitoring-managing-ml-models-production/chapter-5-scalable-monitoring-infrastructure)

[返回课程目录](../README.md)

在生产环境中监控机器学习模型会产生大量数据，并需要能在负载下稳定运行的系统。在明确了*要监控什么*之后（从数据漂移到性能下降），现在将重点转向建设和管理必要的基础设施，以便高效地支持这些大规模监控活动的实际操作层面。

本章将讨论相关的工程挑战。您将学习到：

*   记录数据量大的环境中预测数据和监控输出的策略。
*   使用时序数据库（$TSDBs$），它们专门设计用于高效处理带有时间戳的指标数据。
*   设计可以水平扩展的监控管道分布式系统架构。
*   将您的监控组件与Kubeflow、MLflow和SageMaker等常见MLOps平台集成。
*   专门用于ML监控的开源和商业工具概览。
*   创建有用的仪表盘和设置有意义的警报，以便及时了解模型健康状况的方法。

我们将探讨如何选择和配置这些组件，以构建一个符合生产机器学习需求的监控系统。

## 小节

- 1. [高并发预测服务的日志记录策略](01-%E9%AB%98%E5%B9%B6%E5%8F%91%E9%A2%84%E6%B5%8B%E6%9C%8D%E5%8A%A1%E7%9A%84%E6%97%A5%E5%BF%97%E8%AE%B0%E5%BD%95%E7%AD%96%E7%95%A5.md)
- 2. [时序数据库在监控指标中的应用](02-%E6%97%B6%E5%BA%8F%E6%95%B0%E6%8D%AE%E5%BA%93%E5%9C%A8%E7%9B%91%E6%8E%A7%E6%8C%87%E6%A0%87%E4%B8%AD%E7%9A%84%E5%BA%94%E7%94%A8.md)
- 3. [监控流程的分布式架构](03-%E7%9B%91%E6%8E%A7%E6%B5%81%E7%A8%8B%E7%9A%84%E5%88%86%E5%B8%83%E5%BC%8F%E6%9E%B6%E6%9E%84.md)
- 4. [与MLOps平台（如Kubeflow、MLflow、SageMaker）的整合](04-%E4%B8%8EMLOps%E5%B9%B3%E5%8F%B0%EF%BC%88%E5%A6%82Kubeflow%E3%80%81MLflow%E3%80%81SageMaker%EF%BC%89%E7%9A%84%E6%95%B4%E5%90%88.md)
- 5. [机器学习监控专用工具和服务](05-%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%9B%91%E6%8E%A7%E4%B8%93%E7%94%A8%E5%B7%A5%E5%85%B7%E5%92%8C%E6%9C%8D%E5%8A%A1.md)
- 6. [构建有效的监控仪表盘和预警](06-%E6%9E%84%E5%BB%BA%E6%9C%89%E6%95%88%E7%9A%84%E7%9B%91%E6%8E%A7%E4%BB%AA%E8%A1%A8%E7%9B%98%E5%92%8C%E9%A2%84%E8%AD%A6.md)
- 7. [实践：使用MLflow和Grafana设置监控](07-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%BD%BF%E7%94%A8MLflow%E5%92%8CGrafana%E8%AE%BE%E7%BD%AE%E7%9B%91%E6%8E%A7.md)
