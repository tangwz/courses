---
course: "introduction-to-mlops"
chapter: "foundations-of-mlops"
lesson: "what-is-mlops"
sourceId: 7167
sourceUrl: "https://apxml.com/zh/courses/introduction-to-mlops/chapter-1-foundations-of-mlops/what-is-mlops"
title: "什么是 MLOps？"
description: "明确定义 MLOps（机器学习运维）以及它旨在解决的 AI 开发与部署中的问题。"
order: 1
plots: []
sourceHash: "a299090aefae8b1a094802d329a5e244f1c02aad7452f02ddbd0700289d6445f"
sourceCorrections: []
---

如果你曾经训练过机器学习 (machine learning)模型，可能很熟悉在 Jupyter Notebook 中看到高准确率时的成就感。模型在测试集上表现良好，看起来已经可以上线了。然而，在本地运行的模型与在实际应用中可靠提供预测的模型之间存在巨大差距。这种差距正是机器学习运维（MLOps）发挥作用的地方。

其核心在于，**MLOps 是一套旨在可靠且高效地构建和维护生产环境机器学习系统的实践。** 它结合了机器学习、数据工程和 DevOps 的原则，以解决机器学习生命周期中的独特挑战。可以将其看作是一门运营规范，将实验性的机器学习模型转化为企业级、自动化的系统。

虽然机器学习模型看起来是核心，但它通常只是整体系统中的一小部分。一个生产级别的机器学习系统包括数据摄取管道、特征工程步骤、验证检查、服务基础设施和监控工具。模型代码本身在整个系统中所占的比例可能小得令人惊讶。

> MLOps 方法通过一套自动化和版本控制的实践将实验阶段与生产环境连接起来，形成一个持续的生命周期。

MLOps 旨在解决团队在模型落地时遇到的一系列常见且棘手的问题。

### 弥合实验与生产之间的差距

如果没有 MLOps，部署模型通常是数据科学团队与运维或工程团队之间的手动交接。这种交接往往效率低下、容易出错且阻碍重重。数据科学家可能会提供模型文件和脚本，但工程师必须想办法将其整合到可扩展、安全且稳定的应用程序中。MLOps 通过建立统一的自动化流程弥合了这一差距，让两个团队能够使用共享的工具和实践进行协作。

### 确保可复现性和一致性

机器学习 (machine learning)中的一个基本挑战是可复现性。如果你无法可靠地重现过去的结果，就无法信任你的系统。MLOps 通过对机器学习系统的所有组成部分引入版本控制来解决这个问题：

- **代码：** 用于特征工程、训练和验证的脚本。
- **数据：** 用于训练特定模型的精确数据集版本。
- **模型：** 训练出的模型工件本身及其相关的元数据。

通过对所有内容进行版本化处理，你可以将任何预测追溯到产生它的准确代码、数据和模型。这不仅是良好的实践，通常也是合规审查和故障排查的要求。

### 自动化机器学习 (machine learning)生命周期

手动流程无法扩展且会引入风险。MLOps 专注于尽可能地让机器学习生命周期自动化，从数据准备、模型训练到部署和监控。不再由人工运行训练脚本，而是由自动化流水线根据计划或在新数据可用时触发。这种自动化减少了人为错误，加快了新模型的交付速度，并让数据科学家能够专注于构建更好的模型，而不是管理基础设施。

总之，MLOps 不是单一的工具或技术。它是一种管理机器学习系统全生命周期的文化和实践方法。它将原本脱节的手动步骤转变为规范、自动化且协作的工作流。通过采用 MLOps，组织可以从在 Notebook 中构建模型，转向部署有价值且可靠的 AI 服务。

## 参考资料

- [MLOps: Continuous delivery and automation pipelines in machine learning](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning) — Valeriy Katkalo, Andrew Ferlitsch, Brian Suk, Karl Weinwurm, Alex Shterman, Alexey S. Goldin, Dmitry Ryabtsev, Evgeny Ignatenko (2023)
  Journal: Google Cloud; Publisher: Google Cloud
  概述MLOps的核心概念，展示如何使用实用框架为机器学习系统实现持续集成、交付和训练。
- [MLOps: A Survey, Taxonomy, and the Future Direction](https://dl.acm.org/doi/10.1145/3547180) — J. D. K. S. Ranathunga, Z. H. F. N. S. Ranasinghe, T. L. T. Mahesan, U. V. W. L. K. De Silva, T. G. I. N. Thilakarathne, R. H. R. C. R. Silva (2022)
  Journal: ACM Computing Surveys; Publisher: Association for Computing Machinery; Volume: 54; Pages: 1-30; DOI: [10.1145/3547180](https://doi.org/10.1145/3547180)
  对MLOps进行系统调查，提供其组成部分、挑战和未来研究方向的分类，适合理解其学术背景。
- [Practical MLOps: How to take a model from experimentation to production](https://www.oreilly.com/library/view/practical-mlops/9781098103002/) — Noah Gift, Alfredo Deza (2021)
  Publisher: O'Reilly Media
  指导读者实践MLOps的实施，侧重于构建和管理生产级机器学习系统的工具和技术。
