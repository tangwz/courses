---
course: "introduction-to-mlops"
chapter: "foundations-of-mlops"
lesson: "why-mlops-is-necessary"
sourceId: 7171
sourceUrl: "https://apxml.com/zh/courses/introduction-to-mlops/chapter-1-foundations-of-mlops/why-mlops-is-necessary"
title: "为什么机器学习需要 MLOps"
description: "了解 MLOps 对于可靠、高效地将机器学习项目从原型扩展到生产环境的重要性。"
order: 2
plots: ["plots/7171-0.json"]
sourceHash: "a546fedfa9d47e377aef6dcd11a1f4ebfcee553692e3c88606ee708e656cd7df"
sourceCorrections: []
---

在 Jupyter notebook 中表现出色的机器学习 (machine learning)模型只是一个好的开始，但只有在生产系统中运行，它才能产生实际的业务价值。开发环境与实际应用之间的这种差距，往往是许多机器学习项目停滞或失败的地方。臭名昭著的“在我的机器上能运行”问题在机器学习中被放大，因为这里的“机器”不仅包括代码，还包括用于训练模型的特定数据集、库和配置。

MLOps 是连接这一差距的学科。它提供的工具和实践可以解决将机器学习视为工程职能而非纯研究活动时出现的各种挑战。让我们看看为什么这种操作上的严谨性如此必要。

### 机器学习 (machine learning)系统的组成部分

在传统的软件开发中，主要管理的对象是代码。而在机器学习中，系统由三个同等重要的部分组成：代码、数据和模型本身。其中任何一个环节的变化都可能显著改变系统的行为，且通常以出人意料的方式发生。MLOps 提供了一套框架，用同样的规范来管理这三个要素。

> 机器学习系统是代码、数据和已训练模型的产物。必须管理好这三者以确保可靠性。

### 模型退化的隐形失效

与大多数传统软件不同，部署后的机器学习 (machine learning)模型即使代码没有任何改动，也可能开始失效。这种现象被称为**模型退化**或**概念漂移**，当模型在生产环境中遇到的实际数据统计特性与训练数据发生偏离时，就会发生这种情况。

例如，根据去年的数据训练的客户流失预测模型，由于客户行为、市场状况或产品功能的改变，在今天的表现可能会很差。这种性能下降通常是隐形的；应用程序不会崩溃，但其预测会变得不够准确，从而随时间推移逐渐失去效用。MLOps 建立了监控和自动再训练流水线，用于检测并应对这种退化。



![模型性能随时间的退化情况](plots/7171-0.json)



> 如果不进行再训练，随着生产数据与原始训练数据的偏离，模型的准确率通常会下降。

### 对可复现性的需求

想象一下，如果你被要求调试为什么你的模型在六个月前做出了某个错误的预测。为了调查原因，你需要重新构建当时系统的完全一致的状态。这意味着需要访问：

- 训练代码的准确版本。
- 用于训练的准确数据集。
- 准确的模型产物及其超参数 (parameter) (hyperparameter)。
- 所有软件库和依赖项的版本。

如果没有系统化的方法，重新组合出这种状态几乎是不可能的。**可复现性**是指能够可靠地重新创建模型及其结果的能力。这对于调试、合规性监管、审计以及建立对机器学习 (machine learning)系统的信任都必不可少。MLOps 通过对所有组件进行版本控制来实现这一点。

### 管理实验周期

开发一个好的机器学习 (machine learning)模型很少是一个线性过程。它是一个循环往复的实验周期。数据科学家会测试数据特征、算法和调优参数 (parameter)的无数种组合，以找到满足性能指标的模型。

如果没有结构化的流程，这可能会导致混乱。我们将很难追踪哪些实验是成功的，使用了哪些参数，或者为什么一个模型比另一个模型表现更好。MLOps 引入了**实验追踪**实践，允许团队以集中且系统的方式记录、比较和管理实验结果。这使原本可能杂乱的研究过程变成了一个有组织且可审计的工作流。

### 消除团队间的隔阂

在许多组织中，数据科学家和 IT 运维团队在各自独立的孤岛中工作。数据科学家专注于在研究环境中构建模型，而运维团队负责部署和维护稳定的基础设施。这经常会产生摩擦，因为在孤立环境中开发的模型往往无法满足生产环境的操作要求。

MLOps 创建了一个共享的框架、语言和自动化流水线，将这两个世界连接起来。它培养了一种协作文化，让数据科学家能够了解操作层面的限制，而运维团队也能理解机器学习 (machine learning)模型独特的生命周期。这种整合对于将模型顺利且高效地从原型推向生产至关重要。

> MLOps 用集成的自动化流水线取代了脱节的交付，统一了开发与运维。

总而言之，MLOps 是必不可少的，因为它将机器学习从手工制作转变为可靠且可扩展的工程学科。它提供了克服构建和维护数据学习型软件系统挑战所需的结构和自动化手段。如果没有这些实践，即使是最准确的模型也面临着仅作为孤立实验的风险，永远无法实现其全部潜在价值。

## 参考资料

- [Hidden Technical Debt in Machine Learning Systems](http://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems.pdf) — D. Sculley, Gary Holt, Daniel Golovin, Eugene Davydov, Todd Phillips, Dietmar Ebner, Vinay Chaudhary, Michael Young, Jean-François Crespo, Dan Dennison (2015)
  Journal: Advances in Neural Information Processing Systems 28; Publisher: Curran Associates, Inc.; Volume: 28; Pages: 2503-2511
  这篇基础性论文讨论了机器学习系统中常被忽视的工程和运营挑战，强调了模型开发之外对健壮实践的需求。
- [Engineering MLOps: From Model to Production](https://www.manning.com/books/engineering-mlops) — Emmanuel Raj Lakshmanan, Anurag Singh (2022)
  Publisher: Manning Publications
  一本关于在生产环境中构建和维护机器学习系统的实用指南，涵盖从实验到部署和监控的MLOps完整生命周期。
- [MLOps: A Guide to Production Machine Learning](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning) — Google Cloud (2021)
  Publisher: Google Cloud
  这份来自行业主要参与者的权威指南详细介绍了MLOps的原则和实践，用于构建稳健和可扩展的机器学习系统。
