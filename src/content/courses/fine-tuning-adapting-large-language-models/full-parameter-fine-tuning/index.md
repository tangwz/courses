---
course: "fine-tuning-adapting-large-language-models"
sourceUrl: "https://apxml.com/zh/courses/fine-tuning-adapting-large-language-models/chapter-3-full-parameter-fine-tuning"
sourceId: 667
chapter: "full-parameter-fine-tuning"
title: "全参数微调"
order: 3
description: "全面介绍LLM全微调方法、挑战、正则化和优化策略。"
hasQuiz: false
---

在准备好适应所需的必要数据集后，我们现在转向修改预训练大型语言模型本身的方法。本章侧重于**全参数微调**，这种方式是根据新的任务特定数据更新原始模型中的每个权重（$\theta$）。

您将学习此过程的核心机制，包括：

*   建立和配置用于微调的完整训练循环。
*   选择合适的超参数（例如学习率（$\eta$）和批处理大小）的策略。
*   如正则化等技术，以防止目标数据集上的过拟合。
*   管理所需大量计算资源的方法。
*   保存训练进度检查点和恢复任务的最佳实践。

在本章结束时，您将理解全微调的完整工作流程，并通过动手实践获得实际经验，为以后理解更资源高效的方法打下基础。
