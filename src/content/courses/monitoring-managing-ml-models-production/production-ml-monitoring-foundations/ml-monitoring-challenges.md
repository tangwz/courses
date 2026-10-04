---
course: "monitoring-managing-ml-models-production"
chapter: "production-ml-monitoring-foundations"
lesson: "ml-monitoring-challenges"
sourceId: 3247
sourceUrl: "https://apxml.com/zh/courses/monitoring-managing-ml-models-production/chapter-1-production-ml-monitoring-foundations/ml-monitoring-challenges"
title: "监控机器学习模型的独特挑战"
description: "说明监控机器学习模型时，与传统软件相比遇到的具体难题。"
order: 1
plots: ["plots/3247-0.json"]
sourceHash: "06cc43e1346857a0b8b67884b44bc523fa1c52180f3b6feb211e9b7fa132b555"
sourceCorrections: []
---

传统软件监控侧重于运行指标，例如延迟、错误率和资源利用率，而监控生产环境中的机器学习 (machine learning)模型则面临一套不同且更为复杂的问题。传统的应用性能管理（APM）工具对机器学习应用的健康状况提供必要但不全面的观察。机器学习模型固有的特点、它们对数据的依赖以及与动态环境的互动，都带来了需要专门方法来应对的特定监控难题。

### 静默退化问题

与传统软件不同，传统软件的错误通常表现为明确的报错、崩溃或违反固定逻辑的错误输出，而机器学习 (machine learning)模型可能静默失效。模型可能仍然生成数据类型正确且在预期范围内的预测结果，但这些预测会逐渐变得不准确或不相关。这种退化通常源于数据与模型训练时所用数据相比发生的变化。

假设有一个用于预测房价的模型。如果市场动态因训练数据中未包含的经济变化而发生显著转变，模型的预测结果，虽然看起来仍然有效（例如，正数值），但可能变得越来越不准确。没有抛出异常或生成错误代码；模型只是随着时间推移变得不那么有用。检测这种情况不仅需要监控系统健康状况，还需要监控数据的统计属性和模型的预测质量。

### 数据漂移和概念漂移

这直接引出了核心问题：数据漂移和概念漂移：

1. **数据漂移（协变量偏移）：** 这发生在输入数据 $X$ 的统计属性在训练环境和生产环境之间发生变化时。输入与输出之间的潜在关系 $P(Y|X)$ 可能保持不变，但输入分布 $P_{prod}(X)$ 与训练分布 $P_{train}(X)$ 不同。例如，一个在某个特定人群数据上训练的客户流失模型，如果部署到具有不同人群构成区域，其性能可能会下降。模型本身并不知道如何有效处理这些新的输入模式。

   
   
   ![数据漂移示例（特征分布偏移）](plots/3247-0.json)
   
   

   > 特征分布在训练数据集和实时生产数据之间发生偏移。
2. **概念漂移：** 这通常是一个更难察觉的问题，其中目标变量 $Y$ 的统计属性，或输入特征 $X$ 与目标变量 $Y$ 之间的关系，随时间发生变化。输入分布 $P(X)$ 甚至可能保持稳定，但模型学习到的潜在模式不再有效。例如，在垃圾邮件检测模型中，垃圾邮件发送者不断改变其策略（改变关键词、消息结构）。昨天被认为是垃圾邮件的内容今天可能不再是垃圾邮件的代表，导致学习到的映射 $P(Y|X)$ 过时。概念漂移要求模型进行调整或重新训练，以捕捉新的关系。

检测这些漂移需要对输入特征和模型预测进行统计监控，将生产分布与参考（通常是训练数据）进行比较。

### 真实标签延迟和缺失

评估机器学习 (machine learning)模型的真实性能（例如，准确率、精确率、召回率）需要将其预测结果与实际结果进行比较，这些实际结果通常被称为“真实标签”或“标注数据”。然而，在生产环境中获取这些真实标签可能很困难：

- **延迟：** 从做出预测到实际结果已知之间通常存在延迟。对于信用违约预测，结果（违约或不违约）可能要数月后才能得知。
- **缺失：** 在某些情况下，真实标签可能永远无法获得，或者只能通过小规模、可能存在偏差的预测样本（例如，通过昂贵的人工审查）获得。以预测设备故障为例；你只观察到故障事件，而无法观察到相同条件下的非故障事件。

这种即时、全面的真实标签的缺失意味着我们不能仅仅依靠传统性能指标进行实时监控。我们需要从数据分布、预测置信度分数或其他与性能相关但不需要标签的指标中得出代理指标。

### 非平稳环境

机器学习 (machine learning)模型在本质上是非平稳的环境中运行；条件会发生变化。用户行为演变，市场条件波动，对手调整策略，外部事件发生。在过去数据快照上训练的模型，当环境偏离该快照时，容易出现性能下降。监控必须考虑到这种非平稳性，并在模型假设不再成立时提供警示。

### 耦合和反馈循环

机器学习 (machine learning)系统常表现出复杂的相互作用。系统一部分的变化可能在其他地方产生不明显的效应。这有时被称为 CACE（改变任何东西都会改变一切）。例如，更新上游数据处理管道可能会轻微改变特征分布，影响下游模型的性能。监控需要这些依赖关系的可见性。

此外，一些机器学习系统会产生反馈循环。一个推荐系统推荐商品，用户与这些推荐互动，然后这些互动数据被用于重新训练系统。如果不仔细监控，这个循环可能会强化偏见或导致意想不到的后果。监控不仅要追踪模型的直接输出，还要追踪其潜在的下游影响以及整个系统的行为。

### 复杂性和不透明性

许多现代机器学习 (machine learning)模型，特别是深度神经网络 (neural network)，功能上是“黑箱”。虽然它们可能达到很高的预测准确性，但理解它们 *为什么* 做出特定预测可能很困难。这种不透明性使得诊断性能退化变得困难。模型失效是因为数据漂移、概念漂移，还是它未训练过的边缘情况？监控系统通常需要整合可解释性技术（本课程后续会介绍）以帮助诊断问题。

这些特殊难题凸显了对专门监控策略的必要性，这些策略要超越传统软件所使用的。有效的机器学习监控需要多方面的方法，追踪数据统计、预测行为、系统指标，以及在可能的情况下追踪实际性能，同时考虑到这些系统的动态性和复杂性。以下章节将审视应对这些挑战所需的范围和架构。

## 参考资料

- [Hidden Technical Debt in Machine Learning Systems](https://neurips.cc/paper/2015/file/86df7dcfd896fcaf26747232234ff143-Paper.pdf) — D. Sculley, Gary Holt, Daniel Golovin, Eugene Davydov, Todd Phillips, Dietmar Ebner, Vinay Chaudhary, Michael Young, Jean-François Crespo, Dan Dennison (2015)
  Journal: Advances in Neural Information Processing Systems; Publisher: NeurIPS; Volume: 28; Pages: 2503-2511
  一篇基础性论文，强调了生产机器学习系统中独特挑战和技术债务来源，包括纠缠问题。
- [A Survey on Concept Drift Adaption](https://doi.org/10.1145/2523813) — João Gama, Indrė Žliobaitė, Albert Bifet, Mykola Pechenizkiy, Abdelhamid Bouchachia (2014)
  Journal: ACM Computing Surveys (CSUR); Publisher: Association for Computing Machinery; Volume: 46; Pages: 1-37; DOI: [10.1145/2523813](https://doi.org/10.1145/2523813)
  对机器学习中概念漂移、其类型及适应策略的全面学术综述。
- [Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  一本全面指南，涵盖了设计和部署ML系统的完整生命周期，特别强调了生产挑战和监控。
