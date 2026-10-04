---
course: "how-to-build-a-large-language-model"
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-30-continuous-training-model-updates"
sourceId: 1108
chapter: "continuous-training-model-updates"
title: "持续训练与模型更新"
order: 30
description: "讨论大语言模型持续训练和更新的方法，以适应新数据或架构改进。"
hasQuiz: false
---

部署大语言模型是一个重要节点，但并非最终一步。模型存在于信息变化、数据分布转移、用户期望演变的环境中。为保持效果和适用性，大语言模型通常需要持续更新和改进。

本章讲解大语言模型初次部署后进行维护和完善的实践方面。我们将介绍持续训练、纳入新知识以及管理演进中模型生命周期的方法。

具体来说，您将了解：

*   **持续预训练**的方法，以引入新数据同时避免灾难性遗忘。
*   持续进行**有监督微调（SFT）**和**基于人类反馈的强化学习（RLHF）**的方法，以根据新数据或反馈优化对齐。
*   将**新数据源**安全地引入训练流程的技术。
*   使用**架构修改**更新模型时涉及的考量。
*   在生产环境中对持续更新的模型进行**版本管理、部署以及可能的版本回滚**的流程。

我们将审视确保模型在其运行周期内保持最新和良好表现所需的工程实践。
