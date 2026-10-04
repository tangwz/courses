---
course: "basics-ml-deployment"
chapter: "getting-started-model-deployment"
lesson: "what-is-ml-deployment"
sourceId: 3940
sourceUrl: "https://apxml.com/zh/courses/basics-ml-deployment/chapter-1-getting-started-model-deployment/what-is-ml-deployment"
title: "什么是机器学习部署？"
description: "理解训练后部署机器学习模型的定义和目的。"
order: 1
plots: []
sourceHash: "ea89ecf3a9d9afcfa25fad04025c356b2d0b90582bd6bf344b86db9e38ddd264"
sourceCorrections: []
---

您已成功训练了一个机器学习 (machine learning)模型。您向它输入了数据，选择了算法，微调 (fine-tuning)了参数 (parameter)，并在独立的测试集上评估了其性能。结果看起来很有前景。但下一步是什么呢？一个训练好的模型如果只停留在您的本地机器或开发环境中，其自身作用并不大。它需要被投入使用。这时，*机器学习部署*就派上用场了。

**机器学习部署**是指将您训练好的模型投入生产环境的过程，以便模型可以接收新的输入数据并返回预测结果。可以把它想象成将您的模型从实验室或个人电脑中移到实际应用中，让它真正执行其受训的任务。

一个机器学习模型通常根据房屋的面积、卧室数量和位置等特征来预测房价。

- **训练：** 这个过程使用历史房屋数据来教模型这些特征与最终销售价格之间的关系。结果是一个训练好的模型文件（例如 `.pkl` 或 `.joblib` 文件，我们稍后会介绍）。
- **部署：** 这个文件随后被集成到网页应用程序中。现在，潜在的买家或卖家可以在应用程序中输入房屋的详细信息，部署后的模型会实时提供估价。

没有部署，房价模型就只是开发过程中的一个产物。部署是连接模型创建和模型使用的桥梁，它使应用程序、用户或其他系统能够从模型的预测能力中获益。

本质上，部署包含几个实际步骤：

1. **打包：** 将训练好的模型文件、任何必要的预处理步骤（如数据缩放器或编码器）以及加载和运行模型所需的代码捆绑在一起。
2. **创建接口：** 构建一种方式，使外部系统或用户能够向模型发送数据并接收预测结果。这通常通过应用程序编程接口（API）来实现。
3. **设置基础设施：** 配置和管理必要的硬件和软件（服务器、容器、云服务），以便打包好的模型及其接口能够稳定运行并按需扩展。

这个过程将您静态的、训练好的模型转变为动态的、可操作的预测服务。这是将机器学习投入实际应用并从数据科学工作中获取实际价值的一个基本步骤。

> 典型的机器学习生命周期。部署是使模型可供最终用户或应用程序使用的重要一步。

## 参考资料

- [Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  这本权威书籍深入探讨了如何设计和构建用于生产的机器学习系统，涵盖了从数据到部署和监控的整个生命周期。
- [Practitioners Guide to MLOps: A Framework for Continuous Delivery and Automation of Machine Learning](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEVxRMSCyoZPIMkeRALHtwkuZJTxBOiD02KpilgUhsm8MkjeraALTSf3Sc9jFCQdVyGPNXOAtffYSRv1Whaqnlh11669kVBBQNSERJBKwAsuQK2BmZR2cQC3FKNkDve8KeXgoq56Grpo8lBj5WL_YmtWBbltnQ5etttsGmaST5sCKZG79eBgeHERGeXKJfqYw==) — Google Cloud (2021)
  Publisher: Google Cloud
  这份来自 Google Cloud 的官方指南提供了一个结构化的框架和最佳实践，用于实施 MLOps，详细阐述了机器学习模型持续集成、持续交付和持续训练（包括部署策略）的关键步骤。
- [MLOps: A Survey, Current Challenges, and Future Directions](https://dl.acm.org/doi/10.1145/3607068) — Tobias Kreuzberger, Nils Kühl, Sebastian Hirschl (2023)
  Journal: ACM Computing Surveys; Publisher: ACM; Volume: 56; Pages: Article 53, pp 1–40; DOI: [10.1145/3607068](https://doi.org/10.1145/3607068)
  这项全面的学术调查综合了 MLOps 的现状，定义了包括机器学习部署在内的核心概念，并讨论了该领域的当前挑战和未来的研究方向。
