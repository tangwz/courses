---
course: "how-to-build-a-large-language-model"
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-14-advanced-architectural-modifications"
sourceId: 1092
chapter: "advanced-architectural-modifications"
title: "高级架构改进"
order: 14
description: "讨论近期架构创新，例如参数高效Adapter和专家混合(MoE)层。"
hasQuiz: false
---

在熟悉了核心Transformer架构及其扩展能力后，实际应用常常需要为提高效率或进行专门适配而做更多调整。为满足每个新要求而完全重新训练大型模型，这在计算上成本很高，且常常是不必要的。

本章着重介绍对标准Transformer设计所做的修改，以满足这些需求。我们将考察：

*   **参数高效适配：** 像Adapter模块这样的技术，能够通过只修改总参数的一小部分来对特定任务进行微调。
*   **条件计算：** 像专家混合(MoE)层这样的方法，它们增加了模型中参数总数以提升能力，但对任意给定输入只激活一部分参数，从而控制计算成本。

您将了解这些方法背后的原因，明白它们与基础Transformer的结构区别，并研究MoE系统中路由机制和负载均衡等实现时的考量。
