---
course: "basics-ml-deployment"
chapter: "getting-started-model-deployment"
lesson: "why-deploy-ml-models"
sourceId: 3943
sourceUrl: "https://apxml.com/zh/courses/basics-ml-deployment/chapter-1-getting-started-model-deployment/why-deploy-ml-models"
title: "为什么要部署机器学习模型？"
description: "了解部署模型的原因，例如使预测可供用户或应用程序使用。"
order: 2
plots: []
sourceHash: "e4dd37da43d192d7bea6f9f87422155c3aca47050b9c0534eb15d54c08fc70fc"
sourceCorrections: []
---

训练机器学习 (machine learning)模型是一项重要成果，通常是数据收集、准备和实验大量工作的结果。然而，一个训练好的模型存储在你的本地机器或笔记本中，并不能自行提供持续的价值。部署机器学习模型的主要原因是使其预测能力可供需要它们的用户、应用程序或系统使用。

可以这样理解：训练培养了模型的“智能”，但部署让这种智能发挥作用。没有部署，模型仍是潜在资产而非活跃资产。

以下是部署模型是必要步骤的基本原因：

1. **从预测中创造价值：** 大多数机器学习模型的核心目的是对新的、未见过的数据进行预测。部署提供了一种机制，可以将新数据输入模型并获得预测结果。这可能意味着：

   - **提供推荐：** 根据顾客的浏览历史，在电商网站上向他们推荐商品。
   - **检测异常：** 实时识别潜在的欺诈性信用卡交易。
   - **自动化任务：** 对传入的支持工单进行分类，将其分派给正确的团队。
   - **改善用户体验：** 在应用程序中提供个性化内容或功能。
     在每种情况下，模型的预测都驱动着行动或提供信息，从而创造价值。
2. **与应用程序集成：** 模型通常需要成为更大软件系统或业务流程的一部分。部署通常涉及将模型封装在接口中，例如 API（应用程序编程接口），这使其他应用程序能够轻松请求预测。这种集成允许将模型的洞察力整合到现有工作流程或面向用户的产品中。
3. **可访问性：** 部署使模型的功用可供用户、客户或其他自动化系统访问，他们可以与已部署的模型交互，而无需理解其训练或架构的深层复杂性。
4. **运行使用：** 许多模型都用于持续或频繁使用。部署建立运行模型所需的可靠基础设施，高效处理传入请求，并随时间推移一致地提供预测。这与模型开发阶段常进行的临时性执行形成对比。

> 部署使已训练模型可访问，使其能够处理新数据并为应用程序或用户生成预测。

本质上，部署是模型开发与模型实际运用之间的桥梁。它将静态分析资产转变为动态组成部分，能够提供持续的洞察并在实际环境中驱动行动。没有这一步骤，训练中识别出的潜在益处无法实现。

## 参考资料

- [Machine Learning Design Patterns](https://www.oreilly.com/library/view/machine-learning-design/9781098115777/) — Valliappa Lakshmanan, Sara Robinson, Michael Munn (2021)
  Publisher: O'Reilly Media
  提供构建和部署生产级机器学习系统的模式，强调操作化的必要性。
- [Machine Learning Engineering for Production (MLOps) Specialization](https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops) — Andrew Ng (2021)
  Publisher: DeepLearning.AI (on Coursera)
  一门全面的课程，涵盖机器学习的整个生命周期，包括部署模型以持续创造价值的操作必要性及方法。
- [Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications](https://www.oreilly.com/library/view/designing-machine-learning-systems/9781098107963/) — Chip Huyen (2022)
  Publisher: O'Reilly Media, Inc.; Pages: 386
  讨论如何设计能够有效部署和维护在生产环境中的稳健机器学习系统。
