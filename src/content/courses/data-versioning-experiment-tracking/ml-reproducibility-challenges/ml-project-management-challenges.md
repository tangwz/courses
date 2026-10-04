---
course: "data-versioning-experiment-tracking"
chapter: "ml-reproducibility-challenges"
lesson: "ml-project-management-challenges"
sourceId: 4048
sourceUrl: "https://apxml.com/zh/courses/data-versioning-experiment-tracking/chapter-1-ml-reproducibility-challenges/ml-project-management-challenges"
title: "机器学习项目管理中的难题"
description: "分析管理机器学习项目中数据、代码和结果时遇到的常见困难。"
order: 1
plots: []
sourceHash: "cb302aba97439addbea1140a7bb4e681ac8b1203d452edbfaa059591bc301369"
sourceCorrections: []
---

开发机器学习 (machine learning)模型常常感觉更像实验科学而非传统软件工程。构建一个Web应用程序涉及管理代码变动，而机器学习项目则增加了与数据、参数 (parameter)以及训练算法固有随机性相关的多层复杂性。仅仅使用Git管理代码不足以确保你能可靠地重现过去的结果，或理解你的模型是如何演变的。让我们来分析一下出现的具体困难。

### 依赖关系的复杂网络

一个典型的机器学习 (machine learning)项目涉及几个相互关联的组成部分：

1. **数据：** 原始数据、处理过的数据、特征、标签。数据集可能非常庞大，常达数GB或数TB，因此不适合直接存储在Git仓库中。此外，数据会随时间变化。新数据不断到来，会进行修正，或者预处理步骤会改进。每项变动都可能改变模型的行为。
2. **代码：** 用于数据清洗、特征工程、模型训练、评估以及潜在部署的脚本。代码会随着bug的修复、算法的调整或新库的采用而演变。
3. **配置与参数 (parameter)：** 超参数 (hyperparameter)（如学习率、树深）、特征选择方案、随机种子、算法选择。即使是微小的变化也可能对结果产生显著影响。
4. **环境：** 编程语言（例如Python）、库（例如scikit-learn、TensorFlow、PyTorch）、操作系统和硬件（特别是GPU）的版本。库版本或硬件的细微差异可能导致非确定性行为或不同的结果。
5. **输出产物：** 训练好的模型、评估指标、性能图表、预测文件。它们是整个过程的产物，将它们与产生它们的精确输入（数据、代码、配置、环境）关联起来非常重要。

考虑一个常见情形：你三个月前训练了一个模型，它表现良好。现在，你需要在新数据上重新训练它，或向利益相关者解释其预测结果。你可能会遇到以下问题：

- 究竟使用了数据集的哪个*精确*版本？是5月1日来自`source_A`的原始数据，还是应用`script_v2.py`后的清洗版本？
- 究竟使用了哪些*精确*的超参数？学习率是0.01还是0.001？包含哪些特征？
- 哪个版本的训练脚本生成了那个特定的模型文件？是`main`分支上的那个，还是`feature/new-loss-function`分支上的？
- 环境中安装的TensorFlow或PyTorch是哪个版本？它是在CPU上运行，还是在特定类型的GPU上运行？

如果没有系统地追踪这些元素的方法，回答这些问题就会变成耗时的“侦探”工作，常常以猜测告终，或无法重现原始结果。

> 相互关联的组成部分会影响机器学习训练过程的输出。追踪每个元素对于可重现性是必需的。

### 迭代速度与可追溯性

机器学习 (machine learning)依赖于实验。你可能会尝试几十或上百种变体：不同的算法、特征集、数据子集和超参数 (parameter) (hyperparameter)组合。这种快速迭代是有成效的，但如果管理不当，会产生混乱的历史记录。Jupyter等笔记本环境，虽然非常适合实验，但如果单元格运行顺序不当或代码在没有版本控制的情况下频繁被覆盖，这个问题就会加剧。手动在电子表格或文本文件中记录信息很快就会变得难以管理且容易出错。

### 协作障碍

当多个人协作完成一个机器学习 (machine learning)项目时，这些难题会成倍增加。你如何确保每个人都使用相同版本的数据？一个团队成员如何重现他人的实验结果？如果项目的历史和依赖关系没有清晰地记录和可重现，新成员的加入可能会很困难。可重现性不足会阻碍调试、知识共享和项目可靠移交。

这些困难突显了针对机器学习生命周期专门设计的方法和工具的必要性。我们需要比Git的代码版本控制功能更进一步的方法来处理大量数据，追踪实验参数 (parameter)和结果，并管理构建机器学习模型中固有的复杂依赖关系。接下来的章节将介绍数据版本控制和实验追踪等核心理念，它们是应对这些难题的根据。

## 参考资料

- [Machine Learning Engineering](https://www.oreilly.com/library/view/machine-learning-engineering/9781492067850/) — Hannes Hapke, Catherine Nelson, Rahul Agarwal (2020)
  Publisher: O'Reilly Media
  本书提供了构建和管理机器学习系统的全面指南，其早期章节讨论了常见挑战。
- [Reproducibility in machine learning: A critical review](https://dl.acm.org/doi/10.1145/3386266) — Joelle Pineau, Ludovic Denoyer, Matthieu Labeau, Mark R. Lee, James MacGibbon, Pascal Van Hentenryck (2020)
  Journal: ACM Computing Surveys; Publisher: Association for Computing Machinery; Volume: 53; Pages: 1-40; DOI: [10.1145/3386266](https://doi.org/10.1145/3386266)
  一篇学术综述文章，讨论了机器学习中可重现性的范围和组成部分，涵盖数据、代码和环境因素。
- [MLOps: Continuous delivery and automation for machine learning](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEoDfVZLLNObwVHPhET3yaUlxhpfic7KbEKuwtTHMFEJ5RmTS-Sf6fJWQHhSayQwX4RG53P_SVyLlBY_YF1UC9YOd0zUCP315aOSzxKiJnBUneh3SyagYPM_GL858lu2EMyYCNnv3mCfUNS5BlLUTibukm6fMnPvLD2tjcvGsdzcckTT79aDUTXFr5k_n1mlmoVDUlQ2J0FCDabbcNFWoOV8eHpclhM) — Google Cloud (2024)
  Publisher: Google Cloud
  来自领先行业提供商关于MLOps实践的权威指南，解决了管理机器学习项目生命周期的实际挑战。
