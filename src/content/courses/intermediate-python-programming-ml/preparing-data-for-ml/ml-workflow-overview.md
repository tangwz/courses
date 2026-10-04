---
course: "intermediate-python-programming-ml"
chapter: "preparing-data-for-ml"
lesson: "ml-workflow-overview"
sourceId: 2194
sourceUrl: "https://apxml.com/zh/courses/intermediate-python-programming-ml/chapter-5-preparing-data-for-ml/ml-workflow-overview"
title: "机器学习工作流程概览"
description: "了解机器学习项目中的典型步骤，重点介绍数据准备阶段。"
order: 1
plots: []
sourceHash: "efb490b5e802276286ec8e26ec90a40433f5ae1ec836bf56de5543c6151d1a1b"
sourceCorrections: []
---

在开始讲解数据准备的具体工具和技术之前，我们先将这项活动置于典型的机器学习 (machine learning)项目背景下。理解整个流程有助于明确数据准备为何如此重要以及它如何与其他阶段关联。虽然具体项目可能有所不同，但一组常见的步骤提供了一个有用的结构。

### 标准机器学习 (machine learning)流程

构建机器学习模型通常涉及几个不同的阶段：

1. **问题定义与构建：** 你试图解决什么问题？是分类任务（预测类别）、回归任务（预测数值）、聚类（将相似项分组）还是其他？有哪些可用数据？成功的标准是什么？明确定义目标是项目开展的前提。
2. **数据收集：** 收集解决问题所需的原始数据。数据可能来源于数据库、API、文件（如CSV或日志）、网络抓取或其他来源。
3. **探索性数据分析 (EDA)：** 熟悉数据。这包括使用统计方法和可视化（通常使用你之前学过的 Pandas, Matplotlib 和 Seaborn 等库）来了解数据分布、发现规律、找出异常或错误，并检验假设。EDA 常能提供有助于后续步骤的信息。
4. **数据准备（预处理）：** 这是本章的重点。如引言中所述，原始数据很少能直接用于建模。此阶段包括数据清洗（处理缺失值、修正错误）、数据转换（数值特征缩放、类别特征编码），并可能进行特征工程（从现有变量创建新的输入变量）。目标是将原始数据转换为所选机器学习算法适用的格式。
5. **模型选择与训练：** 根据问题类型和数据特性选择合适的机器学习算法。然后使用准备好的数据训练模型，即算法从数据中学习规律。这通常包括将数据分割为训练集和验证集。
6. **模型评估：** 评估已训练模型在未见数据（通常是独立的测试集）上的表现。计算与问题相关的指标（例如，分类的准确率，回归的均方误差），以判断模型泛化能力如何。
7. **模型部署与监控：** 如果模型表现良好，它将被部署到生产环境，以便对新的实时数据进行预测。持续监控是必要的，以确保其性能不会因输入数据变化（数据漂移）而随时间下降。

> 一个常见的机器学习工作流程，突出显示了数据准备（本章重点）的核心作用以及流程的迭代性质。评估结果不佳常会促使重新进行数据准备或模型训练的优化。

### 机器学习 (machine learning)的迭代性质

理解这个工作流程很少是严格线性的，这很重要。它具有高度迭代性。例如：

- EDA 可能发现需要收集额外数据的问题，或重新审视数据准备中的数据清洗步骤。
- 糟糕的模型评估结果可能让你回到数据准备阶段，尝试不同的特征工程技术或缩放方法。
- 有时，评估甚至可能表明需要重新定义原始问题。

### 数据准备为何如此重要

从工作流程中可以看出，数据准备充当了所收集的原始、通常混乱的数据与机器学习 (machine learning)算法所需结构化输入之间的重要桥梁。如果没有有效的数据准备：

- 算法可能因数据类型不兼容而无法运行（例如，文本而非数字）。
- 模型可能因为异常值或缺失值而学习到错误的规律。
- 尺度差异显著的特征可能对某些算法产生过大影响，导致性能不佳。
- 数据中常见的类别信息，大多数算法无法直接处理，需要进行数值编码。

数据预处理涉及使用 Pandas 和 Scikit-learn 等库的基本技术，以执行必要的数据转换。通过这些转换，可以确保数据处于最佳状态，以便成功进行模型训练和评估。具体实践包括处理不同数据类型、缩放特征、有效分割数据，以及使用 Scikit-learn Pipelines 简化流程。

## 参考资料

- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本构建机器学习系统的实用指南，涵盖从数据准备到部署的整个工作流程，并包含Python示例（第3版）。
- [Python for Data Analysis](https://www.oreilly.com/library/view/python-for-data/9781098104030/) — Wes McKinney (2022)
  Publisher: O'Reilly Media; Pages: 579
  使用Python Pandas库进行数据操作、清理和准备的权威资源，对探索性数据分析至关重要（第3版）。
- [Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html) — Scikit-learn developers (2024)
  解释Scikit-learn中各种数据预处理技术和工具的官方文档，与本章的重点直接相关。
- [Hidden Technical Debt in Machine Learning Systems](https://proceedings.neurips.cc/paper_files/paper/2015/file/86df7dcfd896fcaf267472dc0224d36d-Paper.pdf) — D. Sculley, Gary Holt, Daniel Golovin, Eugene Davydov, Todd Phillips, Dietmar Ebner, Vinay Chaudhary, Michael Young, Jean-François Crespo, Dan Dennison (2015)
  Journal: Advances in Neural Information Processing Systems; Publisher: MIT Press; Volume: 2; Pages: 2503-2511
  一篇广泛引用的论文，讨论机器学习系统的工程挑战和成本，强调基础设施、数据依赖以及工作流程中非模型部分的重要性。
