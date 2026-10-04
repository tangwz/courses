---
course: "data-versioning-experiment-tracking"
chapter: "tracking-experiments-mlflow"
lesson: "importance-experiment-tracking"
sourceId: 4077
sourceUrl: "https://apxml.com/zh/courses/data-versioning-experiment-tracking/chapter-3-tracking-experiments-mlflow/importance-experiment-tracking"
title: "实验跟踪的重要性"
description: "讨论模型开发过程中记录参数、指标、代码和产物的好处。"
order: 1
plots: []
sourceHash: "44d07195566e9718211d6be7963976f083189899263eddb7438583c5fe160bb2"
sourceCorrections: []
---

使用DVC等工具管理数据集是实现可复现机器学习 (machine learning)的重大一步。然而，建模过程本身会增加另一层复杂性。开发机器学习模型很少是线性的过程。它是一个迭代循环，包括：

- 尝试不同的算法（例如，线性回归与随机森林）。
- 调整模型`超参数`（例如学习率、树深度或正则化 (regularization)强度）。
- 尝试使用不同组的输入`特征`。
- 修改数据预处理或模型架构的代码。
- 使用不同版本的输入数据（由DVC跟踪）。

如果没有系统化的方法，这个过程很快就会变得混乱。想象一下训练几十甚至上百个模型。几周后，你可能会发现自己问：

- "哪种参数 (parameter)组合产生了最佳验证准确率？"
- "上个月部署的模型使用了哪个具体的代码版本？"
- "我在那次有希望的运行中使用了`normalized_data_v2`还是`normalized_data_v3`数据集？"
- "为什么我无法复现上次团队会议上展示的结果？"

手动将这些信息记录在电子表格、文件名或分散的笔记中容易出错，难以查找，并且无法有效扩展，尤其是在团队内部。这正是**实验跟踪**旨在解决的问题。

实验跟踪是指系统记录每次尝试（或“运行”）训练机器学习模型时所有相关信息的做法。这通常包括：

- **参数：** 控制运行的输入，例如超参数 (hyperparameter)（`learning_rate=0.01`，`n_estimators=100`）、特征集选择或算法选择。
- **指标：** 衡量模型性能的定量输出，如准确率、精确率、召回率、F1分数、RMSE或损失值，通常在训练时期或最终评估后记录。
- **代码版本：** 用于该次运行的训练脚本和相关代码的具体的版本（例如Git提交哈希）。
- **数据版本：** 识别用于训练和评估的准确数据集版本的信息（可能链接回DVC跟踪的版本）。
- **产物：** 运行期间生成的任何输出文件，例如训练好的模型文件本身、图表（如混淆矩阵或学习曲线）、特征重要性表或环境规格（`requirements.txt`、Conda环境文件）。

### 系统化跟踪的好处

采用结构化的方法进行实验跟踪，在机器学习 (machine learning)生命周期中提供了多项显著优点：

#### 可复现性

这可能是最直接的好处。如果你细致记录参数 (parameter)、代码版本、数据版本和环境细节，你（或同事）将更有可能准确地复现那次具体的训练运行及其结果。这为调试、验证和建立结果信任度提供了根本支持。

#### 比较与分析

记录了多次运行后，你可以轻松比较它们的参数和结果指标。你的随机森林中增加树的数量是否提升了性能？使用特征集B是否优于特征集A？跟踪工具通常提供界面来可视化这些比较，帮助你理解变化的影响，并高效地找到最有前景的模型配置。

#### 协作与知识共享

在团队中工作时，共享的实验跟踪系统充当中央日志本。团队成员可以查看彼此的实验，理解所用的设置，查看结果，并在前人的工作基础上进行，而无需不断询问细节。这促进更好的沟通并避免重复劳动。

#### 调试与审计

如果模型突然表现不佳，或者你需要理解为什么做出了特定的预测，跟踪日志会提供必要的上下文 (context)信息。你可以将有问题运行与成功运行进行比较以找出差异，或追溯已部署模型所用的准确代码、数据和参数。

#### 做出明智的决策

跟踪提供了所需的经验证据，以就模型选择、超参数 (hyperparameter)调优策略和特征工程方向做出明智的决策。它将开发从猜测转向更数据驱动的优化过程。

#### 优化工作流程

实验跟踪不是管理分散的文件和笔记，而是将日志记录直接集成到你的训练脚本中。这创建了更规范高效的工作流程，节省时间并减少记忆实验细节的精神负担。

虽然简单的跟踪可能始于print语句或基本的日志文件记录，但现代机器学习开发的复杂性很快就需要更专业的工具。本章主要介绍**MLflow Tracking**，这是一个流行的开源方案，专门为管理机器学习生命周期而设计，包括实验跟踪功能。我们将探讨如何将MLflow集成到你的训练过程中，以实现上述优点。

## 参考资料

- [MLflow Documentation](https://www.mlflow.org/docs/latest/index.html) — The MLflow Project (2024)
  MLflow 的官方指南，涵盖实验跟踪功能。
- [Introducing MLOps: How to Go from Experimentation to Production](https://www.oreilly.com/library/view/introducing-mlops/9781492062601/) — Mark Treveil, Nicolas Omont, Aurélien Simon, Paolo Cozzolino, Arnaud Menguy, Léo Dreyfus, Jean-Luc Ambrosi, Kenji Chagnot, Adrien Lavoillotte (2020)
  Publisher: O'Reilly Media
  在 MLOps 实践中，提供实验跟踪的背景信息。
- [DVC Documentation](https://dvc.org/doc) — Iterative, Inc. (2024)
  辅助理解数据版本控制，其支持完整的实验跟踪。
