---
course: "llm-constitutional-ai-rlaif"
sourceUrl: "https://apxml.com/zh/courses/llm-constitutional-ai-rlaif/chapter-3-implementing-constitutional-ai"
sourceId: 869
chapter: "implementing-constitutional-ai"
title: "构建宪法级人工智能系统"
order: 3
description: "构建宪法级人工智能流程的实用指南，涵盖批评意见生成与模型优化。"
hasQuiz: false
---

继上一章对宪法级人工智能进行理论探讨之后，本章将提供一份实用指南，指导如何构建CAI流程的各个组成部分。我们将从理论转入实践编码，详细说明实现CAI核心监督学习阶段所需的步骤。

您将学到如何：
*   组织并准备宪法文件($\mathcal{K}$)，以便自动化使用。
*   从基础大语言模型($M_{base}$)生成初始回应。
*   实施负责基于$\mathcal{K}$生成批评意见($C$)的人工智能系统。
*   实施负责基于$C$修改回应的人工智能系统。
*   构建监督微调(SFT)数据集，将批评意见与改进后的回应($R$)配对。
*   使用生成的CAI数据集微调大语言模型($M_{SFT}$)。

我们还将讨论常见的实施难题、调试策略，并包含一个实操练习，侧重于构建一个重要组成部分：人工智能批评意见生成步骤。本章将为您提供开始构建CAI系统的实践知识。
