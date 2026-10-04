---
course: "intro-llm-red-teaming"
sourceUrl: "https://apxml.com/zh/courses/intro-llm-red-teaming/chapter-4-advanced-evasion-exfiltration-methods"
sourceId: 1138
chapter: "advanced-evasion-exfiltration-methods"
title: "高级规避与数据窃取方法"
order: 4
description: "审视LLM规避、模型窃取和数据窃取攻击中更复杂的技术。"
hasQuiz: true
---

本章介绍攻击者可能用来规避检测或从大型语言模型中提取信息的更复杂方法。我们将审视那些通常需要对模型工作原理有更透彻的认识，或对输入和系统交互进行更精密的操纵的技术。

您将了解基于梯度的攻击，这种攻击如果攻击者对模型架构有一定了解，会特别有效，通常由函数 $M(x; \theta)$ 中的参数 $\theta$ 表示。我们还将涵盖使用替代模型的迁移攻击、用于识别训练数据的成员推断攻击，以及模型窃取技术。此外，本章还将讨论如何绕过输入过滤器和输出净化器，链接多种攻击技术以增强效果，以及在低资源或黑盒情况中采用的策略。

理解这些高级攻击手段，对于开发更具弹性的防御措施，并预见大型语言模型系统面临的更广泛的潜在威胁，是很有帮助的。本章包含一个实际练习，模拟信息窃取场景，以帮助您应用这些理念。
