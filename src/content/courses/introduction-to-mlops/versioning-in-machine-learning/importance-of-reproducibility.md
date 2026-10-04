---
course: "introduction-to-mlops"
chapter: "versioning-in-machine-learning"
lesson: "importance-of-reproducibility"
sourceId: 7207
sourceUrl: "https://apxml.com/zh/courses/introduction-to-mlops/chapter-3-versioning-in-machine-learning/importance-of-reproducibility"
title: "可复现性的必要性"
description: "了解为什么可复现性是构建可靠且易于维护的机器学习系统的核心要求。"
order: 1
plots: []
sourceHash: "5569cc8c8ca2827c61ee1107e3b2a47d5459628dafecd5f70f59a264cc35cdae"
sourceCorrections: []
---

假设你的用户流失预测模型已经稳定运行了数月，突然之间准确率大幅下降。这是因为最近为了优化特征而修改了代码吗？还是因为每日导入的数据中包含了一个模型无法处理的、未曾预料到的新类别？如果无法可靠地追踪模型的来源，找到根本原因将变成一项既困难又耗时的调查。这就是可复现性要解决的核心问题。

在机器学习 (machine learning)中，可复现性是指确保模型及其结果可以被完全重新创建的实践。这不仅仅是把同一个脚本运行两次，它意味着在给定的特定版本训练代码、完全相同的数据集以及相同的软件环境下，你可以生成一个一模一样的训练模型。

### 为什么每个机器学习 (machine learning)项目都需要可复现性

起初，关注可复现性可能看起来像是额外的负担，尤其是在节奏极快的实验阶段。然而，从一开始就在工作流中建立这种规范可以防止后期出现严重问题。项目中各项资产之间的联系正是可复现性如此实用的原因。部署的模型不是一个独立的产物；它是代码、数据及其周围环境特定组合的结果。

> 特定版本的模型是其代码、数据和环境共同作用的产物。改变其中任何一项输入都可能导致生成不同的模型。

让我们看看为什么这种做法是构建专业机器学习系统基础的主要原因。

#### 调试与隔离问题

这是最直接且实际的好处。当生产环境中的模型表现异常时，可复现的工作流允许你提出精确的问题：

- **改变了什么？** 你可以将当前的“故障”状态与上一个已知的“正常”状态进行对比。
- **我能重现错误吗？** 通过获取产生故障模型的精确代码、数据和依赖项，你可以在开发环境中可靠地重现该错误以便分析。
- **是代码还是数据的问题？** 你可以用旧数据测试新代码，或者用新数据测试旧代码。这种只有通过资产版本化才能实现的系统化流程，能迅速锁定问题源头。

#### 促进协作

机器学习很少是单打独斗。数据科学家可能负责开发模型，机器学习工程师可能负责部署，而数据工程师则负责管理数据管道。可复现性充当了这些角色之间的契约。当数据科学家完成一项实验时，他们可以移交一组指针：代码的 Git 提交哈希值、数据集版本的标识符以及依赖项列表。机器学习工程师随后可以使用这些指针重新创建完全相同的模型并准备投入生产，确信他们处理的是正确的产物。这消除了经常困扰版本管理不善项目的“在我的机器上能运行”的问题。

#### 确保审计与合规性

在金融、保险和医疗等受监管行业，通常需要解释模型的决策。审计人员可能会问：“为什么模型拒绝了这个人的贷款申请？”为了回答这个问题，你必须能够将预测追溯到生成它的模型、训练该模型的数据以及处理该数据时使用的逻辑。可复现性提供了这一必不可少的审计追踪。没有它，你就无法为模型的行为辩护，也无法满足监管要求。

#### 系统地在现有工作基础上推进

科学的进步建立在能够验证和扩展先前结果的基础上。机器学习的开发也不例外。为了确定一个新想法是否有所改进，你需要一个稳定的基准进行对比。可复现性提供了那个基准。当你尝试新的模型架构或特征工程技术时，你可以确定性能的任何变化都是由你的新想法引起的，而不是因为数据中某些未跟踪的变化或环境中不同的库版本。这使模型开发从凭直觉猜测的过程转变为系统性的工程学科。

明确了可复现性的作用后，我们接下来的步骤是学习管理这些组件的实用工具和技术。我们将从开发者最熟悉的元素开始：使用 Git 进行代码版本管理。

## 参考资料

- [Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications](https://books.google.com/books/about/Designing_Machine_Learning_Systems.html?id=T6oFEAAAQBAJ) — Chip Huyen (2022)
  Publisher: O'Reilly Media; Pages: 367
  提供构建可靠机器学习系统的实用策略，特别强调在整个开发和部署过程中进行版本控制、追踪和确保模型输出一致性。
- [Machine Learning Engineering](https://www.whsmith.co.uk/products/machine-learning-engineering/andriy-burkov/paperback/9781999579579.html) — Andriy Burkov (2020)
  Publisher: True Positive Inc.; Pages: 310
  涵盖创建和部署机器学习系统的完整生命周期，涉及版本控制和实验日志记录等重要工程实践，以支持可重现性。
- [Reproducibility and Replicability in Science](https://nap.nationalacademies.org/catalog/25303/reproducibility-and-replicability-in-science) — National Academies of Sciences, Engineering, and Medicine (2019)
  Publisher: The National Academies Press; DOI: [10.17226/25303](https://doi.org/10.17226/25303)
  这份来自国家科学院机构的报告，为理解科学领域普遍的可重现性和可复制性建立了一个框架，直接适用于机器学习等计算领域。
