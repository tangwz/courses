---
course: "julia-for-machine-learning"
chapter: "julia-ml-pipelines-workflows"
lesson: "mlj-pipelines-introduction"
sourceId: 6886
sourceUrl: "https://apxml.com/zh/courses/julia-for-machine-learning/chapter-5-julia-ml-pipelines-workflows/mlj-pipelines-introduction"
title: "MLJ 管道介绍"
description: "理解在 MLJ.jl 中使用管道的理念及其优点，以简化机器学习工作流程。"
order: 1
plots: []
sourceHash: "48d949eff757a3318d81bff42aa9dc6b546d9c326cdb1375186d0e36a50c0652"
sourceCorrections: []
---

随着您的机器学习 (machine learning)项目复杂度增加，涉及数据准备、特征转换和模型训练等多个阶段，单独管理这些步骤会变得繁琐，并可能导致工作难以复现。MLJ.jl 提供了一种管理此类工作流程的有效抽象方式：管道。

MLJ.jl 中的管道允许您将一系列操作串联起来，将整个序列视为一个单一的复合模型。想象一下您数据的装配线：原始数据从一端进入，经过一系列转换和处理步骤，训练好的模型或预测结果从另一端输出。这种方法将机器学习任务不同部分之间常有的临时连接规范化。

使用管道具有几个重要的优点。首先，它们为您的机器学习代码带来了结构和清晰性。整个工作流程在一个地方定义，而不是分散的代码片段，这使其更易于理解和维护。其次，管道自动化了这些步骤的执行。一旦定义，您就可以使用单个命令拟合和预测整个管道，就像任何其他 MLJ 模型一样。

这种结构化方法极大地帮助了可复现性。通过封装从初始数据转换到最终预测的整个过程，管道确保每次都以相同的顺序应用相同的步骤。此外，管道促进了模块化。您可以通过简单地替换管道内的组件，轻松尝试不同的预处理技术或模型，而不扰乱整体结构。当您想使用相同的预处理设置比较不同模型架构的性能时，这特别有用。

在 MLJ.jl 中，管道通常通过组合单个操作来构建，这些操作可以包括数据预处理器（如标准化器或编码器）和机器学习模型。一个操作的输出直接成为下一个操作的输入。整个管道一旦构建，就如同一个标准的 MLJ 模型。它可以拟合数据，用于进行预测，甚至调优以获得最佳性能。

下图呈现了一个作为管道组织的通用机器学习工作流程：

> 一个典型的机器学习工作流程，被封装为管道。数据按顺序流经预处理阶段到模型训练，最终生成训练好的模型或预测结果。

在此图中，数据源自某个起点，并经过不同阶段。每个框代表一个独立的操作或一组相关操作。例如，“数据清洗”可能涉及处理缺失值和移除异常值，而“特征工程”可能包括创建新的预测变量或转换现有变量。这些预处理步骤为“模型训练与超参数 (parameter) (hyperparameter)调优”阶段准备数据。最后，管道输出一个“已训练模型”，可用于对新数据进行“预测”，或者如果应用于测试集，则可能直接输出预测结果。

MLJ 管道的优点与 MLJ 生态系统中的其他功能结合时会更明显。例如，您可以使用交叉验证评估整个管道，并调优跨越管道不同阶段的超参数，所有这些都可以在一个统一的框架内进行。这种整合方法简化了构建和优化可靠机器学习系统的过程。您不仅仅是在单独调优一个模型；您是在优化从数据到预测的整个路径。

在接下来的章节中，我们将考察构建这些管道的实际方面。您将学习如何定义预处理步骤，集成模型，并使用 MLJ.jl 的语法将它们串联起来。我们还将讲解如何管理这些管道，包括保存和加载它们，这对于部署您的工作或与他人共享很重要。

## 参考资料

- [Composing Models: Pipelines](https://mlj.sciml.ai/dev/composing_models/) — The MLJ.jl Developers (2025)
  MLJ.jl的官方文档，详细介绍了如何在框架内构建和使用机器学习管道。
- [Designing Machine Learning Systems: An Iterative Process for Production-Ready AI](https://learning.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  提供了构建健壮机器学习系统的全面视角，包含有关工作流管理、MLOps实践和可复现性的讨论。
