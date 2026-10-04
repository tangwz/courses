---
course: "adversarial-machine-learning"
sourceUrl: "https://apxml.com/zh/courses/adversarial-machine-learning/chapter-4-model-inference-privacy-attacks"
sourceId: 821
chapter: "model-inference-privacy-attacks"
title: "模型推断与隐私攻击"
order: 4
description: "介绍成员推断、属性推断、模型反演和模型窃取攻击。"
hasQuiz: false
---

在回顾了操纵模型输入（规避攻击）或训练过程（投毒攻击）的攻击后，我们现在转向从已训练模型中或关于已训练模型获取信息的方法。本章讨论针对模型本身及其训练数据机密性的攻击，这类攻击通常只需通过标准查询访问即可进行。

您将学习以下几种推断技术：
*   **成员推断：** 判断某个特定数据样本 $x$ 是否被包含在训练数据集 $D_{train}$ 中。
*   **属性推断：** 根据模型输出或行为，推断训练数据记录的敏感特征或属性。
*   **模型反演：** 重建用于训练特定类别的数据的平均或代表性样本。
*   **模型窃取（功能提取）：** 通过观察目标黑盒模型的输入-输出对 $(x, f(x))$，创建模仿其行为的代理模型。

这些攻击与数据隐私直接关联。了解这些攻击对于评估已部署模型潜在的信息泄露是必要的。我们还将讨论这些攻击如何与差分隐私等形式化隐私原理相关联。在本章结束时，您将掌握这些推断方法的原理及其安全影响。
