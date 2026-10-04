---
course: "introduction-to-mlops"
chapter: "foundations-of-mlops"
lesson: "mlops-vs-devops"
sourceId: 7174
sourceUrl: "https://apxml.com/zh/courses/introduction-to-mlops/chapter-1-foundations-of-mlops/mlops-vs-devops"
title: "MLOps 与 DevOps：相似点与不同点"
description: "比较 MLOps 和 DevOps，了解它们的共同基础以及 MLOps 在机器学习生命周期中解决的独特挑战。"
order: 3
plots: []
sourceHash: "243dafb3ecb3539de7dcfc27611329ae10f391d2f69813a1189bbd9d5f881291"
sourceCorrections: []
---

如果你熟悉现代软件开发，很可能已经听过 DevOps 这个词。DevOps 是一套结合了软件开发（Dev）和 IT 运维（Ops）的实践方法，旨在缩短开发周期，并提供具有高软件质量的持续交付。其核心在于自动化、团队协作，以及提高软件发布的频率和可靠性。MLOps 将类似的原则应用于机器学习 (machine learning)，同时引入了处理这些系统特有属性的特定考量。

不要将 MLOps 看作是 DevOps 的替代品，而应将其视为扩展其原则的专门学科。DevOps 流水线旨在管理和部署代码，而 MLOps 流水线则必须管理代码、模型和数据。这种区别正是这两种实践方式之间产生相似点和不同点的根源。

### 共同的根基

MLOps 和 DevOps 有着共同的目标：提高效率、减少错误并更快地交付价值。这两个学科都高度依赖一套共同的实践方法来实现这一目标。

- **自动化：** MLOps 和 DevOps 都强调尽可能实现生命周期的自动化。这包括测试、集成和部署，这些工作通过持续集成 (CI) 和持续交付 (CD) 流水线来处理。
- **协作：** 这两个领域的一个中心思想是打破团队之间的隔阂。在 DevOps 中，这意味着开发人员和运维人员共同工作。在 MLOps 中，协作范围进一步扩大，包括了数据科学家、机器学习 (machine learning)工程师和数据工程师。
- **监控：** 这两种实践都依赖监控来确保系统部署后的运行状况和性能。这使得团队能够对问题做出快速反应，并为未来的改进收集反馈。
- **版本控制：** 使用 Git 等工具来跟踪更改是两者的基础。它提供了单一的事实来源，并确保工作是可复现且可审计的。

虽然这些共享原则构成了 MLOps 的支柱，但事情并未止步于此。机器学习引入了行为与传统软件截然不同的组件，这要求我们调整并扩展这些实践。

### MLOps 的不同之处

MLOps 与 DevOps 的主要区别源于机器学习 (machine learning)的实验性和数据依赖性。传统的软件应用程序是确定性的；给定相同的输入，它总是产生相同的输出。相比之下，机器学习模型是概率性的。它的行为是从数据中学习得来的，而不是通过显式编程设定的。这导致了几个显著的区别。

> DevOps 和 MLOps 生命周期的对比。MLOps 周期包括数据和模型特有的阶段，并且是由数据或模型性能的变化触发的，而不仅仅是代码更改。

#### 代码 vs. 代码、数据和模型

在 DevOps 中，版本控制下的主要资产是应用程序源代码。当开发人员提交新代码时，流水线就会触发。而在 MLOps 中，你需要管理三个组件：

1. **代码：** 用于数据处理、特征工程和模型训练的代码。
2. **数据：** 用于训练和评估模型的数据集。
3. **模型：** 最终部署的经过训练的模型产物。

这三个组件中任何一个发生变化都可能触发流水线的新一轮运行。例如，如果你收到了一批新的训练数据，即使一行代码都没有改动，你可能也需要重新训练并重新部署模型。这需要一个能够与代码一起对数据和模型进行版本控制的系统，而这项任务超出了传统 DevOps 工具的范围。

#### 不同的自动化触发机制

持续集成/持续交付 (CI/CD) 的概念在 DevOps 中已经非常成熟。在 MLOps 中，这一概念被扩展为一个新想法：**持续训练 (CT)**。

- **DevOps 中的 CI/CD：** 流水线通常由代码提交触发。它集成代码、运行测试并自动发布应用程序。
- **MLOps 中的 CI/CD/CT：** 流水线更加复杂。除了代码触发的运行外，流水线还可以由其他事件自动触发，例如：
  - **新数据可用：** CT 过程会自动在新数据上重新训练模型。
  - **模型性能下降：** 监控可能会发现模型的预测准确性随时间推移而降低（这一问题被称为“模型衰退”或“概念漂移”）。这可以自动触发重新训练流水线。

#### 扩展的监控范围

DevOps 监控侧重于运行指标：CPU 使用率、内存、延迟和应用程序错误。这些对 MLOps 同样有用，但还远远不够。MLOps 需要额外的一层监控，专注于模型质量。

这包括跟踪：

- **数据漂移：** 模型接收到的用于预测的实时数据，其统计属性是否与训练数据有所不同？
- **概念漂移：** 输入特征与目标变量之间的关系是否随时间发生了变化？
- **模型性能：** 分类任务的准确率、精确率和召回率，或回归任务的均方误差等指标。
- **预测偏差：** 模型对不同的用户子群体的表现是否不公平？

这种模型特有的监控对于了解模型何时不再可靠以及何时需要重新训练或更换非常有用。

### 差异总结

为了清晰地区分，以下是这两个学科在多个方面的直接对比。

| 维度 | DevOps | MLOps |
| --- | --- | --- |
| **主要产物** | 应用程序代码、二进制文件 | 代码、数据和模型 |
| **流水线触发器** | 代码更改 | 代码、数据和模型性能衰退 |
| **版本控制** | 主要是源代码版本控制（如使用 Git）。 | 对代码、数据集和模型进行版本控制。 |
| **测试** | 单元测试、集成测试、UI 测试。 | 包括数据验证、模型验证和模型质量测试。 |
| **监控** | 系统运行状况（CPU、内存、延迟）。 | 系统运行状况加上模型性能（漂移、准确性、偏差）。 |
| **核心团队** | 开发人员、运维工程师。 | 数据科学家、机器学习 (machine learning)工程师、数据工程师、开发人员和运维人员。 |
| **实践方式** | 持续集成与交付 (CI/CD)。 | CI/CD 加上持续训练 (CT)。 |

理解这些差异是构建有效 MLOps 策略的第一步。虽然 MLOps 借鉴了 DevOps 的自动化和协作思维，但它调整了实践方法，以应对机器学习系统特有的、以数据为驱动的生命周期。

## 参考资料

- [MLOps: Continuous delivery and automation pipelines in machine learning](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning) — Google Cloud (2024)
  Publisher: Google Cloud
  这份全面指南解释了 MLOps 的原则、架构模式和构建自动化机器学习管道的实践考量，展示了它如何从 DevOps 扩展而来。
- [Introducing MLOps: How to go from Model Centric to Data Centric AI](https://www.oreilly.com/library/view/introducing-mlops/9781492083298/) — Mark Treveil, Nicolas Omont, Aurélien Géron, Michel Blancard, Côme de Treglode, Gregory P. Tenten, Kevin Stumpf, Houssam A. Bakkali, and Adrien Lavoillotte (2020)
  Publisher: O'Reilly Media
  这本书对 MLOps 进行了全面介绍，涵盖了其生命周期、工具以及与 DevOps 等传统软件开发实践的区别。
- [Hidden Technical Debt in Machine Learning Systems](http://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems.pdf) — D. Sculley, Gary Holt, Daniel Golovin, Eugene Davydov, Todd Phillips, Dietmar Ebner, Vinay Chaudhary, Michael Young, Jean-François Crespo, Dan Dennison (2015)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 28; Pages: 2503-2511
  这篇有影响力的论文指出了部署和维护机器学习系统中的独特挑战，强调了 MLOps 等专业实践的必要性，并与传统软件形成了对比。
- [Towards MLOps: An Overview of the Current Tools and Future Challenges](https://link.springer.com/article/10.1186/s13677-022-00329-8) — Wael H. Gomaa and Ahmed Hamdy (2022)
  Journal: Journal of Cloud Computing; Publisher: SpringerOpen; Volume: 11; Pages: Article number 9; DOI: [10.1186/s13677-022-00329-8](https://doi.org/10.1186/s13677-022-00329-8)
  这篇论文综述了 MLOps 领域，讨论了其原则、工具和挑战，为与 DevOps 相比，MLOps 在管理机器学习系统方面的扩展范围提供了背景。
