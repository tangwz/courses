---
course: "introduction-to-mlops"
chapter: "automation-and-cicd-for-ml"
lesson: "introduction-to-cicd"
sourceId: 7227
sourceUrl: "https://apxml.com/zh/courses/introduction-to-mlops/chapter-4-automation-and-cicd-for-ml/introduction-to-cicd"
title: "CI/CD 简介"
description: "关于软件开发中持续集成和持续交付基本概念的入门介绍。"
order: 1
plots: []
sourceHash: "da46d9f7ab696bc4fcc2262b58c4c0f52526bd37513cf7295c25e849a67766e4"
sourceCorrections: []
---

机器学习 (machine learning)生命周期包含数据准备和模型部署等多个阶段。手动执行这些阶段既费时又容易出错。为了从原型过渡到可靠的生产系统，必须采用自动化手段。这种自动化的基石是来自软件工程的一套实践，称为持续集成和持续交付，通常简称为 CI/CD。

这些实践旨在创建一个快速、自动且可重复的过程，将开发人员机器上的新代码发布到生产环境中。让我们来详细分析每个组成部分。

### 什么是持续集成 (CI)？

持续集成是指自动将多个贡献者的代码变更合并到一个共享代码库中的实践。想象一个开发团队正在为一个应用程序的不同功能进行开发。在过去，他们可能会各自独立工作数周后再尝试合并变更，这一过程通常会导致复杂且难以解决的冲突。

CI 通过鼓励开发人员频繁地（通常每天多次）合并变更来解决这个问题。每当开发人员向代码库提交新代码时，自动化系统就会启动。该系统主要执行两个任务：

1. **构建 (Build)：** 编译代码并组装软件。如果软件构建失败，团队会立即收到通知。
2. **测试 (Test)：** 运行一系列自动化测试（如单元测试和集成测试），以验证新代码是否按预期工作，且没有破坏任何现有功能。

如果构建或测试失败，CI 系统会报告错误。这提供了即时反馈，让开发人员在记忆犹新时快速修复问题。CI 的主要目标是及早发现集成问题，从而实现更稳定的代码库和更快的开发周期。

> 持续集成循环。开发人员的代码提交触发自动构建和测试过程，以获得即时反馈。

### 什么是持续交付 (CD)？

持续交付是持续集成的逻辑延伸。CI 确保主代码库始终稳定可用，而持续交付则将经过验证的代码发布到环境中的过程自动化。

在代码变更成功通过所有自动化 CI 测试后，它会被自动打包成制品（即准备好部署的软件包）。然后，该制品会被部署到一个或多个预发布环境，例如“阶段测试 (staging)”或“质量保证 (QA)”环境。持续交付的结果是，你始终拥有一个通过了所有自动化检查、可随时投入生产的软件版本。

部署到实际生产服务器的最后一步通常是一个手动点击的操作。这允许业务部门决定发布的具体时机。持续交付使部署成为一种常规、低风险的活动，因为通往生产环境的整个过程都是自动化的且经过了彻底测试。

### 持续交付 vs. 持续部署

CD 这个术语也可以代表持续部署。两者密切相关，但有一个明显的区别：

- **持续交付 (Continuous Delivery)：** 将软件发布到生产环境是一个“手动”步骤。自动化在最终部署前停止，等待人工决策。
- **持续部署 (Continuous Deployment)：** 每一个通过 CI/CD 流水线所有阶段的变更都会“自动”部署到生产环境。无需人工干预。

对于大多数组织来说，持续交付是初步目标。持续部署是一项更高级的实践，需要对自动化测试和监控系统有极高的信心。在本课程中，当我们提到 CD 时，主要指持续交付，因为它在不取消人工监管的情况下，为自动化发布提供了坚实支持。

> 从持续集成到发布的演进过程。持续交付将直到手动生产部署前的所有步骤自动化，而持续部署则连同这最后一步也一并自动化。

通过将构建、测试和发布阶段自动化，CI/CD 流水线使团队能够更快速、更可靠地向用户交付价值。虽然这些原则提供了强有力的支持，但机器学习 (machine learning)引入了新的复杂挑战。机器学习系统不仅是代码，它是代码、数据和模型的结合。简单的代码单元测试是不够的。我们还必须验证数据、测试模型性能并跟踪实验。在后续章节中，我们将专门针对机器学习生命周期调整这些核心 CI/CD 思想，并引入一个新的组件：持续训练 (CT)。

## 参考资料

- [Continuous Delivery: Reliable Software Releases through Build, Test, and Deployment Automation](https://www.informit.com/store/continuous-delivery-reliable-software-releases-through-9780321601919) — Jez Humble and David Farley (2010)
  Publisher: Addison-Wesley Professional
  这本书是持续交付和持续集成原则和实践的主要参考资料。它解释了如何将软件交付从版本控制自动化到生产。
- [Accelerate: The Science of Lean Software and DevOps: Building and Scaling High Performing Technology Organizations](https://itrevolution.com/accelerate-book/) — Nicole Forsgren, Jez Humble, and Gene Kim (2018)
  Publisher: IT Revolution Press
  这本书为DevOps和CI/CD实践的有效性提供了实证，展示了它们对组织绩效的影响。
- [Continuous Integration](https://martinfowler.com/articles/continuousIntegration.html) — Martin Fowler (2024)
  这篇有影响力的文章介绍了并定义了持续集成，概述了其核心原则和优势。它是CI实践的奠基性文本。
- [What is CI/CD?](https://cloud.google.com/learn/what-is-ci-cd) — Google Cloud (2023)
  Publisher: Google Cloud
  这份官方指南来自一家主要云服务提供商，对持续集成和持续交付的概念提供了清晰简洁的概述。
