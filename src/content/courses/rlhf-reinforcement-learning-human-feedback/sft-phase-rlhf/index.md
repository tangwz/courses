---
course: "rlhf-reinforcement-learning-human-feedback"
sourceUrl: "https://apxml.com/zh/courses/rlhf-reinforcement-learning-human-feedback/chapter-2-sft-phase-rlhf"
sourceId: 951
chapter: "sft-phase-rlhf"
title: "监督微调 (SFT) 阶段"
order: 2
description: "详细说明 RLHF 初始 SFT 步骤的流程和重要性。"
hasQuiz: false
---

基于人类反馈的强化学习 (RLHF) 过程通常从监督微调 (SFT) 开始。这一初始步骤使一个通用预训练大型语言模型 (LLM) 适应，以更好地符合目标任务或应用场景，*在*强化学习阶段*之前*。SFT 使用一个包含高质量提示-响应示例的数据集，为模型提供对所需行为和输出格式的扎实基础认知。

本章侧重于SFT阶段。您将学习：

*   SFT 在为 RLHF 初始化策略时的作用。
*   收集合适演示数据集的方法。
*   重要的实施方面，包括训练配置和超参数。
*   评估 SFT 模型表现的技巧。

我们将通过一个实际练习来结束本章，演示如何在语言模型上进行 SFT。理解 SFT 对于构建一个高效的 RLHF 流程非常重要。
