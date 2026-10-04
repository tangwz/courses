---
course: "llm-alignment-safety"
sourceUrl: "https://apxml.com/zh/courses/llm-alignment-safety/chapter-5-adversarial-attacks-defenses-llms"
sourceId: 691
chapter: "adversarial-attacks-defenses-llms"
title: "对抗性攻击与防御"
order: 5
description: "调查针对LLM的对抗性攻击，以及构建更强健模型的方法。"
hasQuiz: false
---

之前的章节侧重于引导大型语言模型（LLM）实现预期行为，而本章将讨论部署这些强大系统所固有的安全问题。即使是经过RLHF等方法校准的模型，在面对专门制作的、旨在规避安全措施或导致意外行为的输入时，也可能表现出弱点。了解这些潜在的失误点对于构建可靠的AI应用非常重要。

在这里，您将学习如何：

*   对针对LLM的不同类型对抗性攻击进行分类和识别，例如越狱和提示注入。
*   分析训练期间的数据投毒如何损害模型完整性。
*   识别隐私方面的影响，例如成员推理攻击。
*   实施防御策略，包括对抗训练以提升模型韧性。
*   应用输入净化和输出过滤技术作为实用的防护措施。
*   了解LLM安全形式化验证方法的当前状况和局限性。

我们将审视这些攻击背后的机制，以及构建防御所涉及的实际步骤，助您开发更安全的LLM系统。
