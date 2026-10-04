---
course: "introduction-to-mlops"
chapter: "foundations-of-mlops"
lesson: "production-ml-challenges"
sourceId: 7183
sourceUrl: "https://apxml.com/zh/courses/introduction-to-mlops/chapter-1-foundations-of-mlops/production-ml-challenges"
title: "生产机器学习中的常见挑战"
description: "识别部署机器学习模型时遇到的常见问题，例如数据漂移、模型陈旧和技术债。"
order: 6
plots: ["plots/7183-0.json"]
sourceHash: "e1e69912b3f924b7f98eda5db8e307385a3a02b7490aca0ccd479c384db4eb88"
sourceCorrections: []
---

成功训练机器学习 (machine learning)模型通常只是第一步；确保模型在部署后性能稳定会带来一类全新的问题。这些问题通常不是学术机器学习的重点，但却是让机器学习系统在实际操作中发挥作用的核心。理解这些挑战，可以让你明白为什么像 MLOps 这样规范的方法不仅有帮助，而且是必须的。

### 数据漂移：当一切发生变化时

生产模型最常见的故障模式之一是**数据漂移**。当模型在生产环境中接收到的数据统计特性与其训练数据发生偏离时，就会发生这种情况。简单来说，就是输入数据改变了。

想象一个通过月度支出和技术支持工单频率等特征来预测客户流失的模型。如果公司推出了新的订阅计划，客户的支出模式可能会发生剧变。该模型基于历史数据训练，现在看到了它从未遇到过的输入，导致预测准确率大幅下降。

数据漂移是无声的。模型会继续进行预测而不会报错，但这些预测的质量会变差。



![数据漂移：特征分布偏移](plots/7183-0.json)



> 平均每月支出的分布在训练期和当前的生产环境之间发生了明显偏移。模型学到的模式已不再适用。

### 概念漂移：当含义发生变化时

与数据漂移紧密相关的是**概念漂移**。在这种情况下，输入数据的统计特性可能保持不变，但输入与输出之间的关系发生了变化。你试图预测的对象其内在含义演变了。

例如，一个预测欺诈性财务交易的模型会学习与欺诈相关的模式。然而，诈骗者会不断改变策略以躲避检测。今天的“欺诈交易”特征可能与一年前大不相同。欺诈这个词本身的定义已经发生了漂移，即使交易金额和频率（输入数据）的总体分布没有变化，原始模型也会过时。

> 在数据漂移中，输入发生了变化。在概念漂移中，输入对预测的意义发生了变化。

### 环境不一致与复现危机

模型不仅仅是其训练算法；它是代码、数据和特定软件环境的组合。常见的故障源是构建模型的开发环境与运行模型的生产环境之间不匹配。

这个问题通常表现为“在我的机器上能运行”的情况。数据科学家可能使用 Python 3.9 和 scikit-learn 1.1 版本训练模型。然而，生产服务器可能运行的是 Python 3.8 或 scikit-learn 1.2。这些微小的差异可能导致模型直接运行失败，或者更糟糕的是，产生略有不同且错误的预测。如果没有对依赖项和环境进行严格控制，想要复现模型的行为几乎是不可能的。

### 机器学习 (machine learning)系统中的技术债

在软件工程中，技术债是指由于现在选择了简单的方案而没有使用耗时较长但更好的方法，从而隐含的重做成本。在机器学习中，这个问题被放大了。机器学习特有的技术债包括：

- **胶水代码：** 编写大量且脆弱的脚本来将数据从一个系统连接到另一个系统。这些系统难以测试和维护。
- **管道丛林：** 数据准备和特征工程中复杂、纠缠的工作流，这些工作流通常手动执行且文档匮乏。
- **缺乏测试：** 机器学习系统不仅需要代码单元测试。它们还需要数据验证测试、模型质量测试和基础架构测试。跳过这些会带来巨大的风险。
- **手动部署：** 如果部署新模型需要人员手动复制文件、配置服务器并重启服务，那么这个过程会很慢，容易出错，且无法扩展。

这些债务随着时间的推移不断累积，使系统变得脆弱，且极难更新或改进。

### 黑盒 (black box)问题：缺乏监控

模型部署后，你如何知道它是否仍在正常工作？如果没有完善的监控系统，你实际上是在盲目运行。部署一个没有监控的模型就像发射一颗卫星却从不检查它的轨迹或健康信号。

有效的监控不仅是检查服务器是否在线。它涉及跟踪多个层面的指标：

- **运维健康状况：** 模型 API 响应是否迅速（低延迟）？是否报错？消耗了多少 CPU 和内存？
- **数据有效性：** 输入数据是否符合预期？缺失值是否突然增加？是否发生了数据漂移？
- **模型性能：** 模型的预测是否依然准确？这通常需要反馈回路来获取近期打分数据的真实标签，以便计算随时间变化的准确率或精确率等指标。

缺乏这种可见性，模型可能会无声地失效数周或数月，提供错误的信息并损害业务价值。这些挑战表明，构建模型只是成功的机器学习 (machine learning)计划中很小的一部分。本课程随后的章节将为你提供 MLOps 原则和实践，旨在克服这些问题，让你能够构建不仅智能，而且可扩展、可复现且可靠的机器学习系统。

## 参考资料

- [A Survey on Concept Drift Adaption](https://doi.org/10.1145/2523813) — João Gama, Indrė Žliobaitė, Albert Bifet, Mykola Pechenizkiy, A. Bouchachia (2014)
  Journal: ACM Computing Surveys; Publisher: Association for Computing Machinery (ACM); Volume: 46; Pages: 1-37; DOI: [10.1145/2523813](https://doi.org/10.1145/2523813)
  这篇全面的综述文章审查了数据流中概念漂移的原因、检测和缓解策略，对数据漂移同样适用。
- [Machine Learning Design Patterns](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFroBzHQPN0Xhq4txRptokhJOAKnPz54WdUF05vC_lTR5yGtthfd2pBAiWzk4yyC52KFgrzboqwG1sfke4nRLcwxWPiSZ-GaRH0ScYUZtAqc3nKCHQW8K9BrZz8UetaJaD4DBMd_nlAWK_x3vqC8EVJLfgSdDLuOLivMpniVEfeyUssUzWUjz9lbmpHrJPLUSZVJvgwXw==) — Valliappa Lakshmanan, Sara Robinson, Michael Munn (2020)
  Publisher: O'Reilly Media; Pages: 405
  本书介绍了机器学习常见问题的设计模式，包括数据验证、模型监控以及处理数据和概念漂移等相关问题。
